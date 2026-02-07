# ♡ Heart Disease Risk Prediction System v1 (10/11/2025)


<div align="center">

![Heart Disease Prediction](https://img.shields.io/badge/Machine%20Learning-Bayesian%20Network-red?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.0+-black?style=for-the-badge&logo=flask)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Aplikasi Web Flask untuk Prediksi Risiko Penyakit Jantung menggunakan Probabilistic Bayesian Network**

[Demo](#-demo) • [Fitur](#-fitur-utama) • [Instalasi](#-instalasi--cara-menjalankan) • [Tim](#-tim-pengembang)

---

<img src="https://media.giphy.com/media/3o7btPCcdNniyf0ArS/giphy.gif" width="300" alt="Heart Animation"/>

</div>

## 📋 Tentang Proyek

Sistem prediksi risiko penyakit jantung berbasis **Explainable AI (XAI)** yang tidak hanya memberikan prediksi "Ya" atau "Tidak", tetapi juga menjelaskan **"mengapa"** seseorang berisiko terkena penyakit jantung.

### 🎯 Mengapa Proyek Ini Penting?

- **🔍 Transparansi**: Menampilkan probabilitas risiko dan faktor-faktor utama yang berkontribusi
- **⚖️ Fairness**: Mengatasi data tidak seimbang dengan teknik SMOTE-Tomek
- **🧠 Interpretable**: Menggunakan Bayesian Network yang dapat dijelaskan kepada tenaga medis
- **🚀 Praktis**: Interface web yang mudah digunakan untuk screening awal

---

## 💡 Inspirasi & Metodologi

<div align="center">

```mermaid
graph LR
    A[📚 Research Paper] -->|Adaptasi| B[🫀 Heart Disease]
    B --> C[Bayesian Network]
    B --> D[SMOTE-Tomek]
    C --> E[Explainable Prediction]
    D --> E
    E --> F[🎯 Final Application]
```

</div>

Proyek ini terinspirasi dari penelitian:

> **Wang, W., Li, J., Wang, C. et al.** (2023)  
> *"Early detection of diabetes using Bayesian network and SMOTE-ENN"*  
> Scientific Reports, Nature  
> DOI: [10.1038/s41598-023-40036-5](https://doi.org/10.1038/s41598-023-40036-5)

Kami mengadaptasi metodologi tersebut dari kasus diabetes ke **penyakit jantung** menggunakan dataset dari Kaggle.

---

## ✨ Fitur Utama

<table>
<tr>
<td width="50%">

### 🎲 Prediksi Probabilistik
Memberikan **persentase risiko** (contoh: 75% berisiko) bukan hanya klasifikasi biner, sehingga memberikan gambaran yang lebih nuanced.

</td>
<td width="50%">

### 🔬 Explainable AI
Menampilkan **faktor risiko utama** yang memengaruhi prediksi:
- Kolesterol Tinggi
- Usia > 60 tahun
- Tekanan Darah Tinggi
- dll.

</td>
</tr>
<tr>
<td width="50%">

### ⚖️ Penanganan Data Imbalanced
Menggunakan **SMOTE-Tomek** untuk melatih model yang lebih adil dan akurat pada data yang tidak seimbang.

</td>
<td width="50%">

### 🎨 Antarmuka Modern
Interface web yang bersih dan intuitif menggunakan **Flask** + **Tailwind CSS**.

</td>
</tr>
</table>

---

## 🛠️ Tumpukan Teknologi

<div align="center">

| Kategori | Teknologi |
|----------|-----------|
| **Backend** | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white) |
| **Machine Learning** | ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) ![pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) |
| **Bayesian Network** | pgmpy |
| **Imbalanced Learning** | imblearn (SMOTE-Tomek) |
| **Frontend** | ![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=flat&logo=html5&logoColor=white) ![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=flat&logo=tailwind-css&logoColor=white) |

</div>

---

## 🚀 Instalasi & Cara Menjalankan

### 📦 Prasyarat

- Python 3.8+
- pip

### ⚡ Quick Start

```bash
# Clone & Navigate
git clone https://github.com/[USERNAME]/heart-disease-prediction.git
cd heart-disease-prediction

# Install Dependencies
pip install -r requirements.txt

# Download dataset dari Kaggle ke folder data/heart.csv

# Train Model
python train.py

# Run App
python app.py
```

Buka **http://127.0.0.1:5000** di browser Anda.

---

## 📸 Demo

<div align="center">

<img src="https://media.giphy.com/media/xT9IgzoKnwFNmISR8I/giphy.gif" width="500" alt="Coming Soon"/>

### 🎬 Screenshots Coming Soon!

*Demo lengkap akan ditambahkan sebelum hari H UAS*

</div>

---

## 🧪 Cara Kerja Model

<div align="center">

```mermaid
graph LR
    A[📊 Dataset] -->|SMOTE-Tomek| B[⚖️ Balanced Data]
    B -->|Training| C[🧠 Bayesian Network]
    C -->|Inference| D[📈 Probability + Explanation]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style D fill:#9f9,stroke:#333,stroke-width:2px
```

### Model Architecture

```mermaid
graph TD
    Age[Usia] --> Risk[Risiko Penyakit Jantung]
    Cholesterol[Kolesterol] --> Risk
    BP[Tekanan Darah] --> Risk
    Sugar[Gula Darah] --> Risk
    MaxHR[Max Heart Rate] --> Risk
    ChestPain[Nyeri Dada] --> Risk
    
    style Risk fill:#ff6b6b,stroke:#333,stroke-width:3px
```

</div>

---

## 📊 Contoh Hasil Evaluasi Model (TO BE UPDATED)

| Metrik | Nilai |
|--------|-------|
| **Accuracy** | ~85% |
| **Precision** | ~83% |
| **Recall** | ~88% |
| **F1-Score** | ~85% |
| **AUC-ROC** | ~0.90 |

> ⚠️ **Disclaimer**: Model ini untuk tujuan edukasi dan penelitian. Untuk diagnosis medis, selalu konsultasikan dengan tenaga kesehatan profesional.

---

## 👥 Tim Pengembang

<div align="center">

<table>
<tr>
<td align="center" width="20%">
<img src="https://ui-avatars.com/api/?name=Evan&background=0D8ABC&color=fff&size=128" width="100px;" alt="Evan"/><br />
<sub><b>Evan</b></sub><br />
<sub>5803024001</sub>
</td>
<td align="center" width="20%">
<img src="https://ui-avatars.com/api/?name=Benaya&background=6366F1&color=fff&size=128" width="100px;" alt="Benaya"/><br />
<sub><b>Benaya</b></sub><br />
<sub>5803024008</sub>
</td>
<td align="center" width="20%">
<img src="https://ui-avatars.com/api/?name=Reymond&background=8B5CF6&color=fff&size=128" width="100px;" alt="Reymond"/><br />
<sub><b>Reymond</b></sub><br />
<sub>5803024017</sub>
</td>
<td align="center" width="20%">
<img src="https://ui-avatars.com/api/?name=Berliana&background=EC4899&color=fff&size=128" width="100px;" alt="Berliana"/><br />
<sub><b>Berliana</b></sub><br />
<sub>5803024015</sub>
</td>
</tr>
</table>

**Proyek UAS - AI 2025**

</div>

---

## 📚 Struktur File (TO BE UPDATED)

```
heart-disease-prediction/
│
├── 📁 data/
│   └── heart.csv                 # Dataset
│
├── 📁 static/
│   ├── style.css                 # Custom CSS
│   └── js/
│       └── main.js               # Frontend logic
│
├── 📁 templates/
│   ├── index.html                # Input form page
│   ├── result.html               # Prediction result page
│   └── base.html                 # Base template
│
├── 📄 app.py                     # Flask application
├── 📄 train.py                   # Model training script
├── 📄 model.joblib               # Trained model
├── 📄 requirements.txt           # Python dependencies
├── 📄 README.md                  # Documentation (you are here!)
└── 📄 LICENSE                    # MIT License
```

---

## 🙏 Acknowledgments

- **Dataset**: [Kaggle - Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- **Inspirasi Penelitian**: Wang et al. (2023) - Nature Scientific Reports
- **Framework**: Flask, pgmpy, scikit-learn, imbalanced-learn
- **Dosen Pembimbing**: [Nama Dosen] - Mata Kuliah Machine Learning

---

<div align="center">

### 💖 Made with Love & Python

<img src="https://media.giphy.com/media/LnQjpWaON8nhr21vNW/giphy.gif" width="60" alt="Love"/>

**⭐ Jangan lupa beri bintang jika proyek ini bermanfaat! ⭐**

</div>

---

<div align="center">

**🫀 Prediksi Lebih Awal, Hidup Lebih Sehat 🫀**

</div>
