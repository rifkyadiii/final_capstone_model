# models/stroke_model.py
from ..config import Config
from .loader import load_artifacts

print("\n--- Memuat Model Stroke ---")
stroke_model, stroke_scaler, stroke_le_dict, _ = load_artifacts(
    Config.STROKE_MODEL_PATH,
    Config.STROKE_SCALER_PATH,
    Config.STROKE_LE_DICT_PATH
)
print(f"Path Config.STROKE_MODEL_PATH (absolute): {Config.STROKE_MODEL_PATH}")

STROKE_ARTIFACTS_LOADED = all([stroke_model, stroke_scaler, stroke_le_dict])
if not STROKE_ARTIFACTS_LOADED:
    print("PERINGATAN: Tidak semua artefak model stroke berhasil dimuat. Prediksi mungkin gagal.")
else:
    print("Semua artefak model stroke berhasil dimuat.")