# models/loader.py
import os
import joblib
from tensorflow.keras.models import load_model
import traceback

def load_artifacts(model_path, scaler_path=None, le_dict_path=None, target_le_path=None):
    """Memuat model Keras dan objek preprocessor."""
    model, scaler, le_dict, target_le = None, None, None, None
    abs_model_path = None # Untuk debugging
    try:
        if model_path:
            abs_model_path = os.path.abspath(model_path)
            if os.path.exists(abs_model_path):
                model = load_model(abs_model_path)
                print(f"Model dari {abs_model_path} berhasil dimuat.")
            else:
                print(f"PERINGATAN: File model tidak ditemukan di {abs_model_path}")
        
        abs_scaler_path = None
        if scaler_path:
            abs_scaler_path = os.path.abspath(scaler_path)
            if os.path.exists(abs_scaler_path):
                scaler = joblib.load(abs_scaler_path)
                print(f"Scaler dari {abs_scaler_path} berhasil dimuat.")
            else:
                print(f"PERINGATAN: File scaler tidak ditemukan di {abs_scaler_path}")

        abs_le_dict_path = None
        if le_dict_path: # Untuk feature encoders
            abs_le_dict_path = os.path.abspath(le_dict_path)
            if os.path.exists(abs_le_dict_path):
                le_dict = joblib.load(abs_le_dict_path)
                print(f"Feature/Label encoders dari {abs_le_dict_path} berhasil dimuat.")
            else:
                print(f"PERINGATAN: File feature/label encoders tidak ditemukan di {abs_le_dict_path}")
        
        abs_target_le_path = None
        if target_le_path: # Untuk target encoder
            abs_target_le_path = os.path.abspath(target_le_path)
            if os.path.exists(abs_target_le_path):
                target_le = joblib.load(abs_target_le_path)
                print(f"Target encoder dari {abs_target_le_path} berhasil dimuat.")
            else:
                print(f"PERINGATAN: File Target encoder tidak ditemukan di {abs_target_le_path}")
                
        return model, scaler, le_dict, target_le
        
    except Exception as e:
        # Tentukan path dasar yang relevan untuk pesan error
        base_dir_debug = "N/A"
        if abs_model_path: base_dir_debug = os.path.dirname(abs_model_path)
        elif abs_scaler_path: base_dir_debug = os.path.dirname(abs_scaler_path)
        elif abs_le_dict_path: base_dir_debug = os.path.dirname(abs_le_dict_path)
        elif abs_target_le_path: base_dir_debug = os.path.dirname(abs_target_le_path)
        
        print(f"ERROR memuat artefak (path dasar sekitar {base_dir_debug}): {e}")
        print(traceback.format_exc())
        return None, None, None, None