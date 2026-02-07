# Heart Disease Risk Prediction System v1

![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Bayesian%20Network-red?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.0+-black?style=for-the-badge&logo=flask)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Aplikasi Web Flask untuk Prediksi Risiko Penyakit Jantung menggunakan Probabilistic Bayesian Network**

---

## Tentang Proyek

Sistem prediksi risiko penyakit jantung berbasis **Explainable AI (XAI)** yang tidak hanya memberikan prediksi "Ya" atau "Tidak", tetapi juga menjelaskan **"mengapa"** seseorang berisiko terkena penyakit jantung.

### Mengapa Proyek Ini Penting?

- **Transparansi**: Menampilkan probabilitas risiko dan faktor-faktor utama yang berkontribusi
- **Fairness**: Mengatasi data tidak seimbang dengan teknik SMOTE-Tomek
- **Interpretable**: Menggunakan Bayesian Network yang dapat dijelaskan kepada tenaga medis
- **Praktis**: Interface web yang mudah digunakan untuk screening awal

---

## Inspirasi & Metodologi

Proyek ini terinspirasi dari penelitian:

> **Wang, W., Li, J., Wang, C. et al.** (2023)  
> *"Early detection of diabetes using Bayesian network and SMOTE-ENN"*  
> Scientific Reports, Nature  
> DOI: [10.1038/s41598-023-40036-5](https://doi.org/10.1038/s41598-023-40036-5)

Kami mengadaptasi metodologi tersebut dari kasus diabetes ke **penyakit jantung** menggunakan dataset dari Kaggle.

### Alur Metodologi

```
Research Paper → Adaptasi untuk Heart Disease → Bayesian Network + SMOTE-Tomek → Final Application
```

---

## Fitur Utama

### Prediksi Probabilistik
Memberikan **persentase risiko** (contoh: 75% berisiko) bukan hanya klasifikasi biner, sehingga memberikan gambaran yang lebih nuanced.

### Explainable AI
Menampilkan **faktor risiko utama** yang memengaruhi prediksi:
- Kolesterol Tinggi
- Usia > 60 tahun
- Tekanan Darah Tinggi
- Dan lainnya

### Penanganan Data Imbalanced
Menggunakan **SMOTE-Tomek** untuk melatih model yang lebih adil dan akurat pada data yang tidak seimbang.

### Antarmuka Modern
Interface web yang bersih dan intuitif menggunakan **Flask** + **Tailwind CSS**.

---

## Tumpukan Teknologi

| Kategori | Teknologi |
|----------|-----------|
| **Backend** | Python, Flask |
| **Machine Learning** | scikit-learn, pandas |
| **Bayesian Network** | pgmpy |
| **Imbalanced Learning** | imblearn (SMOTE-Tomek) |
| **Frontend** | HTML5, Tailwind CSS |

---

## Instalasi & Cara Menjalankan

### Prasyarat

- Python 3.8+
- pip

### Quick Start

```bash
# Clone repository
git clone https://github.com/[USERNAME]/heart-disease-prediction.git
cd heart-disease-prediction

# Install dependencies
pip install -r requirements.txt

# Download dataset dari Kaggle ke folder data/heart.csv

# Train model
python train.py

# Run application
python app.py
```

Buka **http://127.0.0.1:5000** di browser Anda.

---

## Cara Kerja Model

### Pipeline Data

```
Dataset → SMOTE-Tomek (Balancing) → Bayesian Network Training → Inference → Probability + Explanation
```

### Arsitektur Model

Model Bayesian Network menghubungkan berbagai faktor risiko:

- Usia
- Kolesterol
- Tekanan Darah
- Gula Darah
- Max Heart Rate
- Nyeri Dada

Semua faktor ini berkontribusi pada perhitungan **Risiko Penyakit Jantung** final.

---

## Evaluasi Model (To Be Updated)

| Metrik | Nilai |
|--------|-------|
| **Accuracy** | ~85% |
| **Precision** | ~83% |
| **Recall** | ~88% |
| **F1-Score** | ~85% |
| **AUC-ROC** | ~0.90 |

> **Disclaimer**: Model ini untuk tujuan edukasi dan penelitian. Untuk diagnosis medis, selalu konsultasikan dengan tenaga kesehatan profesional.

---

## Tim Pengembang

| Nama | NRP |
|------|-----|
| Evan | 5803024001 |
| Benaya | 5803024008 |
| Reymond | 5803024017 |
| Berliana | 5803024015 |

**Proyek UAS - AI 2025**

---

## Struktur File (To Be Updated)

```
heart-disease-prediction/
│
├── data/
│   └── heart.csv                 # Dataset
│
├── static/
│   ├── style.css                 # Custom CSS
│   └── js/
│       └── main.js               # Frontend logic
│
├── templates/
│   ├── index.html                # Input form page
│   ├── result.html               # Prediction result page
│   └── base.html                 # Base template
│
├── app.py                        # Flask application
├── train.py                      # Model training script
├── model.joblib                  # Trained model
├── requirements.txt              # Python dependencies
├── README.md                     # Documentation
└── LICENSE                       # MIT License
```

---

## Acknowledgments

- **Dataset**: [Kaggle - Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- **Inspirasi Penelitian**: Wang et al. (2023) - Nature Scientific Reports
- **Framework**: Flask, pgmpy, scikit-learn, imbalanced-learn
- **Dosen Pembimbing**: Pak Nanang - Mata Kuliah Machine Learning

---

**Prediksi Lebih Awal, Hidup Lebih Sehat**
