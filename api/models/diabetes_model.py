# models/diabetes_model.py
from ..config import Config
from .loader import load_artifacts

print("\n--- Memuat Model Diabetes ---")
diabetes_model, diabetes_scaler, diabetes_feature_encoders, diabetes_target_encoder = load_artifacts(
    Config.DIABETES_MODEL_PATH,
    Config.DIABETES_SCALER_PATH,
    Config.DIABETES_FEATURE_ENCODERS_PATH,
    Config.DIABETES_TARGET_ENCODER_PATH
)
print(f"Path Config.DIABETES_MODEL_PATH (absolute): {Config.DIABETES_MODEL_PATH}")

DIABETES_ARTIFACTS_LOADED = all([diabetes_model, diabetes_scaler, diabetes_feature_encoders, diabetes_target_encoder])
if not DIABETES_ARTIFACTS_LOADED:
    print("PERINGATAN: Tidak semua artefak model diabetes berhasil dimuat. Prediksi mungkin gagal.")
else:
    print("Semua artefak model diabetes berhasil dimuat.")

print("\n--- Pemuatan Artefak Selesai (dari semua modul model) ---")