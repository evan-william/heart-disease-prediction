import joblib
import warnings

try:
    from pgmpy.models import DiscreteBayesianNetwork
    from sklearn.preprocessing import LabelEncoder
except ImportError:
    print("ERROR: Pustaka 'pgmpy' atau 'scikit-learn' tidak ditemukan.")
    print("Pastikan Anda sudah menjalankan 'pip install -r requirements.txt' di lingkungan Anda.")
    exit()

# Nonaktifkan peringatan yang mungkin muncul dari versi library
warnings.filterwarnings("ignore", category=UserWarning)

# Nama file yang ingin Anda periksa
filename = 'model.joblib'

print(f"Mencoba memuat dan memeriksa isi dari '{filename}'...")
print("=" * 60)

try:
    # 1. Muat "koper" joblib
    package = joblib.load(filename)

    print("\n--- ✅ Loaded :D ---")
    
    # 2. Tampilkan struktur paket (keys)
    print(f"\nIsi utama paket (keys): {list(package.keys())}")
    
    # 3. Tampilkan 'encoders' (Kamus Penerjemah String -> Angka)
    print("\n--- 1. 'encoders' (Kamus Penerjemah String -> Angka) ---")
    if 'encoders' in package:
        for feature, encoder in package['encoders'].items():
            print(f"  > Fitur: {feature}")
            print(f"    Kelas/Kategori: {list(encoder.classes_)}")
    else:
        print("  > 'encoders' tidak ditemukan.")
        
    # 4. Tampilkan 'discretizer_bins' (Aturan Pengelompokan Angka)
    print("\n--- 2. 'discretizer_bins' (Aturan Pengelompokan Angka) ---")
    if 'discretizer_bins' in package:
        for feature, (bins, labels) in package['discretizer_bins'].items():
            print(f"  > Fitur: {feature}")
            print(f"    Batasan (Bins): {bins}")
            print(f"    Label Kategori: {labels}")
    else:
        print("  > 'discretizer_bins' tidak ditemukan.")
        
    # 5. Tampilkan 'all_features' (Daftar Cek Fitur Model)
    print("\n--- 3. 'all_features' (Daftar Cek Fitur Model) ---")
    if 'all_features' in package:
        print(f"  {package['all_features']}")
    else:
        print("  > 'all_features' tidak ditemukan.")
    
    # 6. Tampilkan info 'model' (Otak Bayesian Network)
    print("\n--- 4. 'model' (Otak Bayesian Network) ---")
    if 'model' in package:
        model = package['model']
        print(f"  Tipe Objek: {type(model)}")
        print("\n  Struktur Model (Edges/Panah):")
        # Mencetak semua panah/hubungan yang dipelajari
        print(model.edges())
        print("\n  (Note: Tabel probabilitas (CPDs) tidak ditampilkan karena terlalu besar.)")
    else:
        print("  > 'model' tidak ditemukan.")

except FileNotFoundError:
    print(f"\n--- ERROR ---")
    print(f"File '{filename}' tidak ditemukan.")
    print("Pastikan Anda sudah menjalankan 'python train.py' terlebih dahulu.")
except Exception as e:
    print(f"\n--- ERROR LAIN ---")
    print(f"Terjadi error saat memuat file: {e}")

print("\n" + "=" * 60)
print("Pemeriksaan selesai.")