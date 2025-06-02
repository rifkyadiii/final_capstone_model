# models/cardiovascular_model.py
from ..config import Config
from .loader import load_artifacts

print("\n--- Memuat Model Kardiovaskular ---")
cardiovascular_model, cardiovascular_scaler, _, _ = load_artifacts(
    Config.CARDIOVASCULAR_MODEL_PATH,
    Config.CARDIOVASCULAR_SCALER_PATH
)
print(f"Path Config.CARDIOVASCULAR_MODEL_PATH (absolute): {Config.CARDIOVASCULAR_MODEL_PATH}")

CARDIOVASCULAR_ARTIFACTS_LOADED = all([cardiovascular_model, cardiovascular_scaler])
if not CARDIOVASCULAR_ARTIFACTS_LOADED:
    print("PERINGATAN: Tidak semua artefak model kardiovaskular berhasil dimuat. Prediksi mungkin gagal.")
else:
    print("Semua artefak model kardiovaskular berhasil dimuat.")