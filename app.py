import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, render_template, jsonify
from pgmpy.inference import VariableElimination
import mysql.connector 
import requests 
import json
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import io
from flask import send_file

# Inisialisasi Flask
app = Flask(__name__)

db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '', 
    'database': 'heart_disease'
}

def get_db_connection():
    try:
        conn = mysql.connector.connect(**db_config)
        return conn
    except Exception as e:
        print(f"Error koneksi ke database: {e}")
        return None

# --- Muat Model ---
print("Memuat model dari 'model.joblib'...")
try:
    model_package = joblib.load('model.joblib')
    model = model_package['model']
    encoders = model_package['encoders']
    discretizer_bins = model_package['discretizer_bins']
    ALL_FEATURES = model_package['all_features']
    inference = VariableElimination(model)
    print("Model berhasil dimuat.")
except FileNotFoundError:
    print("ERROR: 'model.joblib' not found. Jalankan 'train.py' dulu.")
    exit()

# --- FEATURE IMPORTANCE (Persentase Pengaruh) ---
FEATURE_WEIGHTS = {
    'Age': 15,
    'Sex': 10,
    'ChestPainType': 18,
    'RestingBP': 8,
    'Cholesterol': 12,
    'FastingBS': 7,
    'RestingECG': 9,
    'MaxHR': 11,
    'ExerciseAngina': 14,
    'Oldpeak': 10,
    'ST_Slope': 16
}

# Normalisasi agar total = 100%
total_weight = sum(FEATURE_WEIGHTS.values())
FEATURE_IMPORTANCE = {k: round((v/total_weight)*100, 1) for k, v in FEATURE_WEIGHTS.items()}

# --- Fungsi Helper Preprocessing ---
def preprocess_input(form_data):
    evidence_dict = {}
    reasons = []
    feature_contributions = {}
    
    # --- Proses Fitur Numerik (Diskretisasi) ---
    numeric_inputs = {
        'Age': int(form_data['Age']),
        'RestingBP': int(form_data['RestingBP']),
        'Cholesterol': int(form_data['Cholesterol']),
        'MaxHR': int(form_data['MaxHR']),
        'Oldpeak': float(form_data['Oldpeak']),
    }
    
    for col, value in numeric_inputs.items():
        bins, labels = discretizer_bins[col]
        bin_index = pd.cut([value], bins=bins, labels=labels, right=False)[0]
        evidence_dict[col] = str(bin_index)
        
        # Hitung kontribusi fitur terhadap risiko
        risk_factor = 0
        if col == 'Age' and bin_index == '>60':
            reasons.append(f"Usia Lanjut ({value} tahun)")
            risk_factor = 0.8
        elif col == 'Age' and bin_index == '40-60':
            risk_factor = 0.5
        elif col == 'Age':
            risk_factor = 0.2
            
        if col == 'Cholesterol' and bin_index == 'High':
            reasons.append(f"Kolesterol Tinggi ({value} mg/dl)")
            risk_factor = 0.9
        elif col == 'Cholesterol' and bin_index == 'Borderline':
            reasons.append(f"Kolesterol Borderline ({value} mg/dl)")
            risk_factor = 0.6
        elif col == 'Cholesterol':
            risk_factor = 0.2
            
        if col == 'RestingBP' and bin_index == 'High S2':
            reasons.append(f"Tekanan Darah Sangat Tinggi ({value} mmHg)")
            risk_factor = 0.9
        elif col == 'RestingBP' and bin_index == 'High S1':
            reasons.append(f"Tekanan Darah Tinggi ({value} mmHg)")
            risk_factor = 0.7
        elif col == 'RestingBP':
            risk_factor = 0.3
            
        if col == 'Oldpeak' and bin_index == 'High':
            reasons.append(f"Depresi ST Sangat Tinggi ({value})")
            risk_factor = 0.9
        elif col == 'Oldpeak' and bin_index == 'Medium':
            reasons.append(f"Depresi ST Sedang ({value})")
            risk_factor = 0.6
        elif col == 'Oldpeak':
            risk_factor = 0.2
            
        if col == 'MaxHR':
            if bin_index == 'Very Low':
                risk_factor = 0.8
            elif bin_index == 'Low':
                risk_factor = 0.5
            else:
                risk_factor = 0.2
        
        feature_contributions[col] = risk_factor

    # --- Proses Fitur Kategorikal ---
    categorical_inputs = {
        'Sex': form_data['Sex'],
        'ChestPainType': form_data['ChestPainType'],
        'FastingBS': str(form_data['FastingBS']),
        'RestingECG': form_data['RestingECG'],
        'ExerciseAngina': form_data['ExerciseAngina'],
        'ST_Slope': form_data['ST_Slope'],
    }
    
    for col, value in categorical_inputs.items():
        evidence_dict[col] = value
        
        risk_factor = 0
        if col == 'Sex' and value == 'M':
            risk_factor = 0.6
        elif col == 'Sex':
            risk_factor = 0.4
            
        if col == 'ChestPainType':
            if value == 'ASY':
                reasons.append(f"Nyeri Dada Asymptomatic")
                risk_factor = 0.9
            elif value == 'TA':
                reasons.append(f"Nyeri Dada Typical Angina")
                risk_factor = 0.8
            elif value == 'NAP':
                risk_factor = 0.4
            else:
                risk_factor = 0.3
                
        if col == 'ExerciseAngina' and value == 'Y':
            reasons.append("Angina saat Olahraga")
            risk_factor = 0.9
        elif col == 'ExerciseAngina':
            risk_factor = 0.2
            
        if col == 'ST_Slope':
            if value == 'Flat':
                reasons.append("ST Slope 'Flat'")
                risk_factor = 0.9
            elif value == 'Down':
                reasons.append("ST Slope 'Downsloping'")
                risk_factor = 0.95
            else:
                risk_factor = 0.2
                
        if col == 'FastingBS' and value == '1':
            risk_factor = 0.7
        elif col == 'FastingBS':
            risk_factor = 0.3
            
        if col == 'RestingECG':
            if value == 'LVH':
                risk_factor = 0.8
            elif value == 'ST':
                risk_factor = 0.7
            else:
                risk_factor = 0.2
        
        feature_contributions[col] = risk_factor

    # --- Encoder String ke Angka ---
    encoded_evidence = {}
    for col in ALL_FEATURES:
        try:
            raw_val = evidence_dict[col]
            le = encoders[col]
            encoded_val = le.transform([raw_val])[0]
            encoded_evidence[col] = encoded_val
        except Exception as e:
            print(f"Error encoding {col} dengan nilai {evidence_dict.get(col)}: {e}")
            pass
            
    return encoded_evidence, reasons, feature_contributions

# --- Rute Aplikasi ---

@app.route('/')
def home():
    return render_template('index.html', feature_importance=FEATURE_IMPORTANCE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_data = request.form
        evidence, reasons, feature_contributions = preprocess_input(form_data)
        
        target_encoder = encoders['HeartDisease']
        target_value_index = int(target_encoder.transform(['1'])[0])

        print(f"Melakukan query dengan evidence: {evidence}")
        result_phi = inference.query(
            variables=['HeartDisease'],
            evidence=evidence
        )
        
        risk_probability = result_phi.values[target_value_index]
        risk_percentage = round(risk_probability * 100, 2)
        
        print(f"Hasil Probabilitas: {risk_probability} ({risk_percentage}%)")
        
        # Tentukan warna
        if risk_percentage > 70:
            risk_color = "text-red-500"
            risk_category = "Sangat Tinggi"
        elif risk_percentage > 40:
            risk_color = "text-yellow-500"
            risk_category = "Menengah"
        else:
            risk_color = "text-green-500"
            risk_category = "Rendah"

        if not reasons:
            reasons = ["Faktor risiko Anda terlihat terkendali."]

        return render_template('index.html',
                               risk=risk_percentage,
                               reasons=reasons,
                               risk_color=risk_color,
                               risk_category=risk_category,
                               form_data=form_data,
                               feature_importance=FEATURE_IMPORTANCE,
                               feature_contributions=feature_contributions,
                               scroll_to='result')

    except Exception as e:
        print(f"Terjadi error saat prediksi: {e}")
        return render_template('index.html',
                               error=f"Terjadi kesalahan: {e}. Pastikan semua input terisi.",
                               feature_importance=FEATURE_IMPORTANCE,
                               scroll_to='result')

# --- RUTE BARU UNTUK EXPORT PDF ---
@app.route('/export_report', methods=['POST'])
def export_report():
    try:
        data = request.json
        
        # Buat buffer untuk PDF
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
        
        # Container untuk elemen PDF
        elements = []
        
        # Style
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1e40af'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#1e40af'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        )
        
        normal_style = styles['Normal']
        normal_style.fontSize = 10
        normal_style.leading = 14
        
        # Header
        elements.append(Paragraph("LAPORAN MEDIS", title_style))
        elements.append(Paragraph("Prediksi Risiko Penyakit Jantung", styles['Heading2']))
        elements.append(Paragraph(f"Tanggal: {datetime.now().strftime('%d %B %Y, %H:%M')}", normal_style))
        elements.append(Spacer(1, 0.3*inch))
        
        # Informasi Pasien
        elements.append(Paragraph("INFORMASI PASIEN", heading_style))
        patient_data = [
            ['Nama', ':', data.get('Name', '-')],
            ['Usia', ':', f"{data.get('Age', '-')} tahun"],
            ['Jenis Kelamin', ':', 'Laki-laki' if data.get('Sex') == 'M' else 'Perempuan'],
        ]
        patient_table = Table(patient_data, colWidths=[2*inch, 0.3*inch, 3*inch])
        patient_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#374151')),
            ('TEXTCOLOR', (2, 0), (2, -1), colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(patient_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Hasil Prediksi (HIGHLIGHT)
        risk = float(data.get('risk', 0))
        risk_category = data.get('risk_category', 'N/A')
        
        if risk > 70:
            risk_color = colors.red
        elif risk > 40:
            risk_color = colors.orange
        else:
            risk_color = colors.green
            
        elements.append(Paragraph("HASIL PREDIKSI", heading_style))
        risk_data = [
            ['Tingkat Risiko', f"{risk}%"],
            ['Kategori', risk_category]
        ]
        risk_table = Table(risk_data, colWidths=[2.5*inch, 3*inch])
        risk_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#374151')),
            ('TEXTCOLOR', (1, 0), (1, -1), risk_color),
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f3f4f6')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BOX', (0, 0), (-1, -1), 2, colors.HexColor('#1e40af')),
        ]))
        elements.append(risk_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Data Medis Input
        elements.append(Paragraph("DATA MEDIS", heading_style))
        medical_data = [
            ['Parameter', 'Nilai', 'Pengaruh (%)'],
            ['Tekanan Darah Istirahat', f"{data.get('RestingBP', '-')} mmHg", f"{FEATURE_IMPORTANCE['RestingBP']}%"],
            ['Kolesterol', f"{data.get('Cholesterol', '-')} mg/dl", f"{FEATURE_IMPORTANCE['Cholesterol']}%"],
            ['Gula Darah Puasa', 'Ya (>120)' if data.get('FastingBS') == '1' else 'Tidak (≤120)', f"{FEATURE_IMPORTANCE['FastingBS']}%"],
            ['Detak Jantung Maksimal', f"{data.get('MaxHR', '-')} bpm", f"{FEATURE_IMPORTANCE['MaxHR']}%"],
            ['Oldpeak (Depresi ST)', data.get('Oldpeak', '-'), f"{FEATURE_IMPORTANCE['Oldpeak']}%"],
        ]
        medical_table = Table(medical_data, colWidths=[2.5*inch, 2*inch, 1*inch])
        medical_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e40af')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (2, 0), (2, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        elements.append(medical_table)
        elements.append(Spacer(1, 0.2*inch))
        
        # Data EKG
        ekg_data = [
            ['Parameter', 'Nilai', 'Pengaruh (%)'],
            ['Tipe Nyeri Dada', data.get('ChestPainType', '-'), f"{FEATURE_IMPORTANCE['ChestPainType']}%"],
            ['EKG Istirahat', data.get('RestingECG', '-'), f"{FEATURE_IMPORTANCE['RestingECG']}%"],
            ['Angina Saat Olahraga', 'Ya' if data.get('ExerciseAngina') == 'Y' else 'Tidak', f"{FEATURE_IMPORTANCE['ExerciseAngina']}%"],
            ['ST Slope', data.get('ST_Slope', '-'), f"{FEATURE_IMPORTANCE['ST_Slope']}%"],
        ]
        ekg_table = Table(ekg_data, colWidths=[2.5*inch, 2*inch, 1*inch])
        ekg_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e40af')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (2, 0), (2, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        elements.append(ekg_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Faktor Risiko
        elements.append(Paragraph("FAKTOR RISIKO UTAMA", heading_style))
        reasons = data.get('reasons', [])
        if reasons:
            for reason in reasons:
                elements.append(Paragraph(f"• {reason}", normal_style))
        else:
            elements.append(Paragraph("• Faktor risiko terlihat terkendali", normal_style))
        elements.append(Spacer(1, 0.3*inch))
        
        # Rekomendasi AI (jika ada)
        if data.get('ai_advice'):
            elements.append(PageBreak())
            elements.append(Paragraph("REKOMENDASI & NASIHAT KESEHATAN", heading_style))
            advice_text = data.get('ai_advice', '')
            # Format advice
            for line in advice_text.split('\n'):
                if line.strip():
                    elements.append(Paragraph(line, normal_style))
            elements.append(Spacer(1, 0.3*inch))
        
        # Footer / Disclaimer
        elements.append(Spacer(1, 0.5*inch))
        disclaimer = """
        <b>CATATAN PENTING:</b><br/>
        Hasil prediksi ini dihasilkan oleh model machine learning Bayesian Network dan bukan merupakan diagnosis medis resmi. 
        Selalu konsultasikan dengan dokter atau tenaga medis profesional untuk pemeriksaan lebih lanjut dan diagnosis yang akurat.
        Model ini memiliki tingkat akurasi yang baik, namun tidak menggantikan penilaian klinis dokter.
        """
        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=normal_style,
            fontSize=8,
            textColor=colors.HexColor('#6b7280'),
            alignment=TA_JUSTIFY,
            borderWidth=1,
            borderColor=colors.HexColor('#9ca3af'),
            borderPadding=10,
            backColor=colors.HexColor('#f9fafb')
        )
        elements.append(Paragraph(disclaimer, disclaimer_style))
        
        # Build PDF
        doc.build(elements)
        
        # Kirim file
        buffer.seek(0)
        filename = f"Laporan_Medis_{data.get('Name', 'Pasien').replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        return send_file(
            buffer,
            as_attachment=True,
            download_name=filename,
            mimetype='application/pdf'
        )
        
    except Exception as e:
        print(f"Error saat export PDF: {e}")
        return jsonify(status="error", message=str(e)), 500

# --- RUTE GEMINI UNTUK REKOMENDASI PENGURANGAN RISIKO ---
@app.route('/get_gemini_advice', methods=['POST'])
def get_gemini_advice():
    try:
        # 1. Ambil data JSON dari request frontend
        data = request.json
        form_data = data.get('formData', {})
        
        # 2. Pilih Model AI
        # Default kita set ke 'gemini-2.0-flash' karena di daftar Anda ini yang STATUSNYA ✅ WORKING!
        model_name = data.get('model', 'gemini-2.0-flash')
        
        # API Key (Gunakan key Anda)
        api_key = "ISI API KEY DISINI YA VISITOR GITHUB KU"
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

        # 3. Siapkan Data untuk Prompt
        # Data Medis (Yang dipakai untuk prediksi numerik)
        data_medis_model = {
            'Usia': f"{form_data.get('Age')} tahun",
            'Jenis Kelamin': 'Laki-laki' if form_data.get('Sex') == 'M' else 'Perempuan',
            'Tipe Nyeri Dada': form_data.get('ChestPainType'),
            'Tekanan Darah': f"{form_data.get('RestingBP')} mmHg",
            'Kolesterol': f"{form_data.get('Cholesterol')} mg/dl",
            'Gula Darah Puasa > 120': 'Ya' if form_data.get('FastingBS') == '1' else 'Tidak',
            'EKG Istirahat': form_data.get('RestingECG'),
            'Detak Jantung Maks': form_data.get('MaxHR'),
            'Angina Olahraga': 'Ya' if form_data.get('ExerciseAngina') == 'Y' else 'Tidak',
            'Oldpeak (Depresi ST)': form_data.get('Oldpeak'),
            'ST Slope': form_data.get('ST_Slope')
        }
        
        # Data Gaya Hidup (Yang SANGAT PENTING untuk nasihat dokter tapi tidak masuk rumus matematika model)
        data_gaya_hidup = {
            'Tinggi Badan': f"{form_data.get('Height')} cm",
            'Berat Badan': f"{form_data.get('Weight')} kg",
            'Riwayat Keluarga Jantung': form_data.get('FamilyHistory', 'Tidak ada info'),
            'Status Merokok': form_data.get('SmokingStatus', 'Tidak ada info'),
            'Konsumsi Alkohol': form_data.get('AlcoholIntake', 'Tidak ada info'),
            'Aktivitas Fisik': form_data.get('PhysicalActivity', 'Tidak ada info')
        }
        
        # Ambil hasil prediksi persentase terakhir (dikirim dari frontend)
        hasil_prediksi = form_data.get('LastRiskPercentage', 'Belum diprediksi')

        # 4. Buat System Prompt (Persona AI)
        system_prompt = (
            "Anda adalah Dokter Spesialis Jantung (Kardiolog) senior yang ramah, empatik, namun tegas dalam hal kesehatan. "
            "Tugas Anda: Memberikan interpretasi hasil prediksi risiko jantung dan saran gaya hidup yang PERSONAL."
            "\nATURAN PENTING:"
            "\n1. Jangan mendiagnosis secara medis (gunakan bahasa 'berisiko', 'indikasi', dll)."
            "\n2. Selalu sarankan konsultasi ke dokter sungguhan di akhir."
            "\n3. Fokuslah menghubungkan 'Data Gaya Hidup' (seperti merokok/berat badan) dengan 'Hasil Prediksi'."
            "\n4. Gunakan format Markdown (bold, list) agar mudah dibaca."
            "\n5. Gunakan Bahasa Indonesia yang baik, formal tapi hangat."
        )

        # 5. Buat User Query (Data Pasien)
        user_query = (
            f"Halo Dokter AI. Berikut data pasien saya:\n\n"
            f"--- DATA UTAMA ---\n"
            f"Nama: {form_data.get('Name', 'Pasien')}\n"
            f"HASIL PREDIKSI SISTEM: Risiko Penyakit Jantung {hasil_prediksi}%\n\n"
            f"--- DATA KLINIS ---\n{json.dumps(data_medis_model, indent=2)}\n\n"
            f"--- GAYA HIDUP & FISIK ---\n{json.dumps(data_gaya_hidup, indent=2)}\n\n"
            f"Mohon berikan:\n"
            f"1. Penjelasan singkat apa arti risiko {hasil_prediksi}% ini.\n"
            f"2. Analisis faktor gaya hidup saya (terutama {data_gaya_hidup.get('Status Merokok')} dan {data_gaya_hidup.get('Aktivitas Fisik')}).\n"
            f"3. 3-5 Langkah konkret yang bisa saya lakukan mulai besok untuk menurunkan risiko ini."
        )

        # 6. Kirim Request ke Google Gemini API
        payload = {
            "contents": [{"parts": [{"text": user_query}]}],
            "systemInstruction": {"parts": [{"text": system_prompt}]}
        }

        print(f"Mengirim request ke model: {model_name}...") # Debugging log
        response = requests.post(api_url, json=payload, headers={'Content-Type': 'application/json'})
        
        # 7. Cek Error dari API
        if response.status_code != 200:
            error_msg = f"API Error {response.status_code}: {response.text}"
            print(error_msg)
            return jsonify(status="error", message="Maaf, server AI sedang sibuk. Coba model lain atau tunggu sebentar."), 500

        # 8. Parsing Hasil
        result_json = response.json()
        
        # Ambil teks jawaban dengan aman (mengantisipasi struktur JSON yang kosong)
        try:
            advice_text = result_json['candidates'][0]['content']['parts'][0]['text']
        except (KeyError, IndexError):
            advice_text = "Maaf, AI tidak memberikan respons yang valid. Silakan coba lagi."

        return jsonify(status="success", advice=advice_text)

    except Exception as e:
        print(f"CRITICAL ERROR di /get_gemini_advice: {e}")
        return jsonify(status="error", message=f"Terjadi kesalahan sistem: {str(e)}"), 500

# --- RUTE DATABASE ---

@app.route('/save_info', methods=['POST'])
def save_info():
    data = request.form
    sql = """
        INSERT INTO SavedInformation 
        (Name, Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, 
        RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope, LastRiskPercentage,
        Height, Weight, FamilyHistory, SmokingStatus, AlcoholIntake, PhysicalActivity)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor(dictionary=True) as cursor:
                cursor.execute(sql, (
                    data['Name'], int(data['Age']), data['Sex'], data['ChestPainType'],
                    int(data['RestingBP']), int(data['Cholesterol']), data['FastingBS'],
                    data['RestingECG'], int(data['MaxHR']), data['ExerciseAngina'],
                    float(data['Oldpeak']), data['ST_Slope'], float(data['LastRiskPercentage']),
                    int(data['Height']) if data['Height'] else None,
                    int(data['Weight']) if data['Weight'] else None,
                    data['FamilyHistory'], data['SmokingStatus'], 
                    data['AlcoholIntake'], data['PhysicalActivity']
                ))
            conn.commit() 
        
        return jsonify(status="success", message="Prediksi berhasil disimpan!")
        
    except Exception as e:
        print(f"Error saat menyimpan: {e}")
        return jsonify(status="error", message=str(e)), 500

@app.route('/get_saved_users', methods=['GET'])
def get_saved_users():
    sql = "SELECT id, Name, LastRiskPercentage FROM SavedInformation ORDER BY Name"
    try:
        with get_db_connection() as conn:
            with conn.cursor(dictionary=True) as cursor:
                cursor.execute(sql)
                users_list = cursor.fetchall() 
        return jsonify(users_list)
    except Exception as e:
        print(f"Error mengambil users: {e}")
        return jsonify([]), 500

@app.route('/get_user_details/<int:user_id>', methods=['GET'])
def get_user_details(user_id):
    sql = "SELECT * FROM SavedInformation WHERE id = %s"
    try:
        with get_db_connection() as conn:
            with conn.cursor(dictionary=True) as cursor:
                cursor.execute(sql, (user_id,))
                user = cursor.fetchone()
        if user:
            return jsonify(user)
        else:
            return jsonify(error="User not found"), 404
    except Exception as e:
        print(f"Error mengambil detail user: {e}")
        return jsonify(error=str(e)), 500@app.route('/delete_user/<int:user_id>', methods=['POST'])

@app.route('/delete_user/<int:user_id>', methods=['POST'])
def delete_user(user_id):
    sql = "DELETE FROM SavedInformation WHERE id = %s"
    rows_affected = 0
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, (user_id,))
                rows_affected = cursor.rowcount
            conn.commit()    
        
        if rows_affected > 0:
            print(f"User {user_id} berhasil dihapus.")
            return jsonify(status="success", message="User berhasil dihapus.")
        else:
            print(f"User {user_id} tidak ditemukan.")
            return jsonify(status="error", message="User tidak ditemukan."), 404
            
    except Exception as e:
        print(f"Error saat menghapus: {e}")
        return jsonify(status="error", message=str(e)), 500
    
def create_table_if_not_exists():
    sql = """
    CREATE TABLE IF NOT EXISTS SavedInformation (
    id INT AUTO_INCREMENT PRIMARY KEY,
    Name VARCHAR(100) NOT NULL,
    Age INT NOT NULL,
    Sex VARCHAR(1) NOT NULL,
    ChestPainType VARCHAR(10) NOT NULL,
    RestingBP INT NOT NULL,
    Cholesterol INT NOT NULL,
    FastingBS VARCHAR(1) NOT NULL,
    RestingECG VARCHAR(10) NOT NULL,
    MaxHR INT NOT NULL,
    ExerciseAngina VARCHAR(1) NOT NULL,
    Oldpeak FLOAT NOT NULL,
    ST_Slope VARCHAR(10) NOT NULL,
    LastRiskPercentage FLOAT NOT NULL,    Height INT NULL,
        Weight INT NULL,
        FamilyHistory VARCHAR(10) NULL,
        SmokingStatus VARCHAR(20) NULL,
        AlcoholIntake VARCHAR(20) NULL,
        PhysicalActivity VARCHAR(20) NULL
    )
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor: 
                cursor.execute(sql)
            conn.commit()
        print("Tabel 'SavedInformation' berhasil dicek/dibuat.")
    except Exception as e:
        print(f"Error saat membuat tabel: {e}")
        
@app.route('/stats')
def stats():
    return render_template('stats.html')

if __name__ == '__main__':
    create_table_if_not_exists()
    app.run(debug=True)