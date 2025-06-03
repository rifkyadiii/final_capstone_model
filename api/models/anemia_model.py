# models/anemia_model.py
import os
import json
from ..config import Config
from .loader import load_artifacts # Asumsi loader.py ada di package 'models'

print("\n--- Memuat Model Anemia ---")

# Load Keras model, scaler, dan label encoders dict menggunakan loader generik
# Path target_le tidak diperlukan karena target (0/1) ditangani oleh output model.
anemia_model, anemia_scaler, anemia_label_encoders, _ = load_artifacts(
    Config.ANEMIA_MODEL_PATH,
    Config.ANEMIA_SCALER_PATH,
    Config.ANEMIA_LABEL_ENCODERS_PATH
)

# Load daftar fitur input yang diurutkan
anemia_input_features_ordered = None
if Config.ANEMIA_INPUT_FEATURES_ORDERED_PATH and os.path.exists(Config.ANEMIA_INPUT_FEATURES_ORDERED_PATH):
    try:
        with open(Config.ANEMIA_INPUT_FEATURES_ORDERED_PATH, 'r') as f:
            anemia_input_features_ordered = json.load(f)
        print(f"Daftar fitur input Anemia dari {Config.ANEMIA_INPUT_FEATURES_ORDERED_PATH} berhasil dimuat: {anemia_input_features_ordered}")
    except Exception as e:
        print(f"ERROR memuat daftar fitur input Anemia: {e}")
        anemia_input_features_ordered = None
else:
    if Config.ANEMIA_INPUT_FEATURES_ORDERED_PATH:
         print(f"PERINGATAN: File daftar fitur input Anemia tidak ditemukan di {Config.ANEMIA_INPUT_FEATURES_ORDERED_PATH}")
    else:
         print(f"PERINGATAN: Path untuk ANEMIA_INPUT_FEATURES_ORDERED_PATH tidak dikonfigurasi.")


# Turunkan fitur kategorikal dan numerik secara dinamis
anemia_categorical_features = []
anemia_numeric_features_to_scale = []

if anemia_label_encoders and isinstance(anemia_label_encoders, dict) and anemia_input_features_ordered:
    anemia_categorical_features = list(anemia_label_encoders.keys())
    # Pastikan hanya fitur yang ada di input_features_ordered yang dipertimbangkan
    anemia_categorical_features = [col for col in anemia_categorical_features if col in anemia_input_features_ordered]
    
    anemia_numeric_features_to_scale = [
        feature for feature in anemia_input_features_ordered if feature not in anemia_categorical_features
    ]
    print(f"Fitur kategorikal Anemia (dari label encoders & input order): {anemia_categorical_features}")
    print(f"Fitur numerik Anemia (untuk scaling): {anemia_numeric_features_to_scale}")
elif anemia_input_features_ordered:
    print("Peringatan: Label encoders Anemia tidak dimuat atau bukan dictionary. Mengasumsikan semua fitur (jika ada) adalah numerik atau tidak memerlukan encoding spesifik dari file.")
    if isinstance(anemia_label_encoders, dict) and not anemia_label_encoders: # Dict kosong
        anemia_numeric_features_to_scale = list(anemia_input_features_ordered) # Semua numerik
        print("Tidak ada label encoder Anemia yang dimuat (dictionary kosong), semua fitur input dianggap numerik.")

# Cek apakah semua artefak yang dibutuhkan berhasil dimuat
ANEMIA_ARTIFACTS_LOADED = all([
    anemia_model is not None,
    anemia_scaler is not None,
    anemia_label_encoders is not None, # Harus berupa dictionary (bisa kosong jika tidak ada categorical)
    anemia_input_features_ordered is not None
])

if not ANEMIA_ARTIFACTS_LOADED:
    print("PERINGATAN: Tidak semua artefak esensial model Anemia berhasil dimuat. Prediksi mungkin gagal.")
else:
    print("Semua artefak esensial model Anemia berhasil dimuat.")

# Nama kelas target untuk Anemia
ANEMIA_TARGET_CLASSES = {
    0: "Tidak Terkena Anemia",
    1: "Terkena Anemia"
}