# models/obesity_model.py
from config import Config  # Menggunakan relative import
from .loader import load_artifacts # Menggunakan relative import

print("--- Memuat Model Obesitas ---")
obesity_model, obesity_scaler, obesity_le_dict, obesity_target_le = load_artifacts(
    Config.OBESITY_MODEL_PATH,
    Config.OBESITY_SCALER_PATH,
    Config.OBESITY_LE_DICT_PATH,
    Config.OBESITY_TARGET_LE_PATH
)
print(f"Path Config.OBESITY_MODEL_PATH (absolute): {Config.OBESITY_MODEL_PATH}")

# Cek apakah semua artefak berhasil dimuat
OBESITY_ARTIFACTS_LOADED = all([obesity_model, obesity_scaler, obesity_le_dict, obesity_target_le])
if not OBESITY_ARTIFACTS_LOADED:
    print("PERINGATAN: Tidak semua artefak model obesitas berhasil dimuat. Prediksi mungkin gagal.")
else:
    print("Semua artefak model obesitas berhasil dimuat.")