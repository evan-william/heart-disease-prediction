import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from imblearn.combine import SMOTETomek
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import HillClimbSearch, BayesianEstimator

print("Memulai skrip training model...")

# --- Konfigurasi & Definisi ---

# Fitur kategorikal dan numerik berdasarkan dataset
# (HeartDisease adalah target)
CATEGORICAL_FEATURES = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
NUMERIC_FEATURES = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
TARGET = 'HeartDisease'

# Kamus untuk menyimpan encoder
encoders = {}
# Kamus untuk menyimpan bin (kelompok) untuk data numerik
discretizer_bins = {}

# --- Fungsi Preprocessing ---

def preprocess(df):
    print("Memulai preprocessing data...")
    df_processed = df.copy()
    
    # Ganti nilai 0 yang tidak logis (sesuai deskripsi data Kaggle)
    # RestingBP 0 tidak mungkin, ganti dengan median
    median_bp = df_processed['RestingBP'].median()
    df_processed['RestingBP'] = df_processed['RestingBP'].replace(0, median_bp)
    
    # Cholesterol 0 tidak mungkin, ganti dengan median
    median_chol = df_processed['Cholesterol'].median()
    df_processed['Cholesterol'] = df_processed['Cholesterol'].replace(0, median_chol)

    # --- Diskretisasi Fitur Numerik ---
    # Mengubah fitur numerik menjadi kategori (penting untuk Bayesian Network)
    # Ini sesuai dengan slide Anda (misal "Umur >= 60")
    print("Melakukan diskretisasi fitur numerik...")
    
    # Age
    bins = [0, 40, 60, 100]
    labels = ['<40', '40-60', '>60']
    df_processed['Age'] = pd.cut(df_processed['Age'], bins=bins, labels=labels, right=False)
    discretizer_bins['Age'] = (bins, labels)

    # RestingBP
    bins = [0, 120, 140, 160, 300]
    labels = ['Normal', 'Elevated', 'High S1', 'High S2']
    df_processed['RestingBP'] = pd.cut(df_processed['RestingBP'], bins=bins, labels=labels, right=False)
    discretizer_bins['RestingBP'] = (bins, labels)

    # Cholesterol
    bins = [0, 200, 240, 600]
    labels = ['Normal', 'Borderline', 'High']
    df_processed['Cholesterol'] = pd.cut(df_processed['Cholesterol'], bins=bins, labels=labels, right=False)
    discretizer_bins['Cholesterol'] = (bins, labels)

    # MaxHR
    bins = [0, 100, 140, 170, 250]
    labels = ['Very Low', 'Low', 'Normal', 'High']
    df_processed['MaxHR'] = pd.cut(df_processed['MaxHR'], bins=bins, labels=labels, right=False)
    discretizer_bins['MaxHR'] = (bins, labels)
    
    # Oldpeak
    bins = [-np.inf, 0, 1, 2.5, np.inf]
    labels = ['Normal', 'Low', 'Medium', 'High']
    df_processed['Oldpeak'] = pd.cut(df_processed['Oldpeak'], bins=bins, labels=labels, right=False)
    discretizer_bins['Oldpeak'] = (bins, labels)
    
    # --- Encoding Fitur Kategorikal ---
    # Mengubah semua fitur (termasuk yang baru didiskretisasi) menjadi angka
    print("Melakukan encoding fitur...")
    
    all_features_to_encode = CATEGORICAL_FEATURES + NUMERIC_FEATURES + [TARGET]
    
    for col in all_features_to_encode:
        # Pastikan kolom adalah string sebelum encoding (LabelEncoder menangani ini)
        df_processed[col] = df_processed[col].astype(str)
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        encoders[col] = le # Simpan encoder
        
    print("Preprocessing selesai.")
    return df_processed

# --- Muat & Proses Data ---
try:
    data = pd.read_csv('data/heart.csv')
    processed_data = preprocess(data)
except FileNotFoundError:
    print("ERROR: File 'data/heart.csv' tidak ditemukan.")
    print("Silakan download dataset dari Kaggle dan letakkan di folder 'data/'.")
    exit()

# --- SMOTE-Tomek ---
print("Memisahkan X dan y...")
X = processed_data.drop(TARGET, axis=1)
y = processed_data[TARGET]

print("Ukuran data sebelum SMOTE-Tomek:")
print(y.value_counts())

print("Menerapkan SMOTE-Tomek... (Ini mungkin butuh beberapa saat)")
smt = SMOTETomek(random_state=42)
X_resampled, y_resampled = smt.fit_resample(X, y)

print("Ukuran data setelah SMOTE-Tomek:")
print(y_resampled.value_counts())

# Gabungkan kembali data untuk training Bayesian Network
resampled_data = pd.concat([X_resampled, y_resampled], axis=1)
# Kembalikan nama kolom
resampled_data.columns = processed_data.columns

# --- Training Bayesian Network ---
print("Memulai training Bayesian Network...")

# a. Belajar Struktur (Mencari hubungan sebab-akibat / DAG)
print("Step 5a: Belajar struktur model (HillClimbSearch)...")
hc = HillClimbSearch(data=resampled_data)
best_structure = hc.estimate()
print("Struktur model (edges):")
print(best_structure.edges())

model = DiscreteBayesianNetwork(best_structure.edges())

# b. Belajar Parameter (Menghitung probabilitas)
print("Step 5b: Belajar parameter model (BayesianEstimator)...")
model.fit(resampled_data, estimator=BayesianEstimator)
print("Model berhasil dilatih.")

# --- Simpan Model & Objek Preprocessing ---
print("Menyimpan model dan objek preprocessing ke 'model.joblib'...")
# Simpan semua yang kita perlukan untuk prediksi dalam satu file
save_package = {
    'model': model,
    'encoders': encoders,
    'discretizer_bins': discretizer_bins,
    'all_features': NUMERIC_FEATURES + CATEGORICAL_FEATURES # Urutan fitur
}
joblib.dump(save_package, 'model.joblib')

print("--- TRAINING SELESAI ---")
print("Model berhasil disimpan sebagai 'model.joblib'.")
print("Sekarang bisa menjalankan 'flask run' untuk memulai web app!")