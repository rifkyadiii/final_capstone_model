# api/models/heart_attack_model.py

import os
import json
from config import Config  # Relative import
from .loader import load_artifacts # Relative import

print("\n--- Memuat Model Heart Attack ---")

# Load Keras model, scaler, feature label encoders dict, and target label encoder
ha_model, ha_scaler, ha_feature_le_dict, ha_target_le = load_artifacts(
    Config.HEART_ATTACK_MODEL_PATH,
    Config.HEART_ATTACK_SCALER_PATH,
    Config.HEART_ATTACK_FEATURE_LE_DICT_PATH,
    Config.HEART_ATTACK_TARGET_LE_PATH
)

# Load the ordered list of input features
ha_input_features_ordered = None
if Config.HEART_ATTACK_INPUT_FEATURES_ORDERED_PATH and os.path.exists(Config.HEART_ATTACK_INPUT_FEATURES_ORDERED_PATH):
    try:
        with open(Config.HEART_ATTACK_INPUT_FEATURES_ORDERED_PATH, 'r') as f:
            ha_input_features_ordered = json.load(f)
        print(f"Daftar fitur input Heart Attack dari '{Config.HEART_ATTACK_INPUT_FEATURES_ORDERED_PATH}' berhasil dimuat.")
    except Exception as e:
        print(f"ERROR memuat daftar fitur input Heart Attack: {e}")
else:
    missing_path_msg = Config.HEART_ATTACK_INPUT_FEATURES_ORDERED_PATH if Config.HEART_ATTACK_INPUT_FEATURES_ORDERED_PATH else "Path tidak dikonfigurasi"
    print(f"PERINGATAN: File daftar fitur input Heart Attack tidak ditemukan di '{missing_path_msg}' atau path tidak dikonfigurasi.")

# Define features based on the notebook preprocessing
# ONLY these 3 features were scaled during training
ha_numeric_features_to_scale = ['age', 'waist_circumference', 'triglycerides']

# These were encoded during training
ha_categorical_features = ['gender', 'smoking_status', 'alcohol_consumption']

# Binary features (not scaled, just converted to int)
ha_binary_features = ['hypertension', 'diabetes', 'obesity', 'previous_heart_disease', 
                      'medication_usage', 'participated_in_free_screening']

print(f"Fitur kategorikal Heart Attack: {ha_categorical_features}")
print(f"Fitur numerik Heart Attack (untuk scaling): {ha_numeric_features_to_scale}")
print(f"Fitur binary Heart Attack: {ha_binary_features}")

# Check if all essential artifacts are loaded
HEART_ATTACK_ARTIFACTS_LOADED = all([
    ha_model is not None,
    ha_scaler is not None,
    ha_feature_le_dict is not None, # Must be a dictionary (can be empty if no categorical features)
    ha_target_le is not None,
    ha_input_features_ordered is not None
])

if not HEART_ATTACK_ARTIFACTS_LOADED:
    print("PERINGATAN: Tidak semua artefak esensial model Heart Attack berhasil dimuat. Prediksi akan gagal.")
else:
    print("Semua artefak esensial model Heart Attack berhasil dimuat.")
    if ha_target_le:
        print(f"Kelas target Heart Attack (dari target LE): {list(ha_target_le.classes_)}")

# Target class names/mapping (can be derived from ha_target_le if loaded)
HEART_ATTACK_TARGET_CLASSES_MAP = {}
if ha_target_le:
    for i, class_name in enumerate(ha_target_le.classes_):
        HEART_ATTACK_TARGET_CLASSES_MAP[i] = str(class_name) # Store original class name as string
else:
    # Fallback if target_le fails to load, though ARTIFACTS_LOADED should catch this
    HEART_ATTACK_TARGET_CLASSES_MAP = {0: "Class 0 (No Heart Attack - Placeholder)", 1: "Class 1 (Heart Attack - Placeholder)"}