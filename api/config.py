# config.py
import os

class Config:
    """Menyimpan konfigurasi path dan nama fitur."""
    # Path relatif dari folder yang berisi 'app.py' (root aplikasi modular)
    # ke folder model.
    # Asumsi: Struktur folder aplikasi (app.py, config.py, models/, routes/, utils/)
    # berada di satu direktori (misalnya, 'api/'), dan folder model
    # (misal, 'obesity-model') adalah sibling dari direktori tersebut.
    BASE_PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # Naik satu level ke root proyek
    
    OBESITY_MODEL_DIR = os.path.join(BASE_PROJECT_DIR, 'obesity-model', 'saved_model_obesity')
    STROKE_MODEL_DIR = os.path.join(BASE_PROJECT_DIR, 'stroke-model', 'saved_model_stroke')
    CARDIOVASCULAR_MODEL_DIR = os.path.join(BASE_PROJECT_DIR, 'cardiovascular-model', 'saved_model_cardiovascular')
    DIABETES_MODEL_DIR = os.path.join(BASE_PROJECT_DIR, 'diabetes-model', 'saved_model_diabetes')

    OBESITY_MODEL_PATH = os.path.join(OBESITY_MODEL_DIR, 'obesity_model.keras')
    OBESITY_SCALER_PATH = os.path.join(OBESITY_MODEL_DIR, 'obesity_scaler.pkl')
    OBESITY_LE_DICT_PATH = os.path.join(OBESITY_MODEL_DIR, 'obesity_le_dict.pkl')
    OBESITY_TARGET_LE_PATH = os.path.join(OBESITY_MODEL_DIR, 'obesity_target_le.pkl')

    STROKE_MODEL_PATH = os.path.join(STROKE_MODEL_DIR, 'stroke_model.keras')
    STROKE_SCALER_PATH = os.path.join(STROKE_MODEL_DIR, 'stroke_scaler.pkl')
    STROKE_LE_DICT_PATH = os.path.join(STROKE_MODEL_DIR, 'stroke_le_dict.pkl')

    CARDIOVASCULAR_MODEL_PATH = os.path.join(CARDIOVASCULAR_MODEL_DIR, 'cardiovascular_model.keras')
    CARDIOVASCULAR_SCALER_PATH = os.path.join(CARDIOVASCULAR_MODEL_DIR, 'cardiovascular_scaler.pkl')

    DIABETES_MODEL_PATH = os.path.join(DIABETES_MODEL_DIR, 'diabetes_model.keras')
    DIABETES_SCALER_PATH = os.path.join(DIABETES_MODEL_DIR, 'diabetes_scaler.pkl')
    DIABETES_FEATURE_ENCODERS_PATH = os.path.join(DIABETES_MODEL_DIR, 'diabetes_feature_encoders.pkl')
    DIABETES_TARGET_ENCODER_PATH = os.path.join(DIABETES_MODEL_DIR, 'diabetes_target_encoder.pkl')

    # Fitur Obesitas
    OBESITY_NUMERIC_FEATURES = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    OBESITY_CATEGORICAL_FEATURES = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC',
                                    'SMOKE', 'SCC', 'CALC', 'MTRANS']
    OBESITY_ALL_FEATURES_ORDERED = [
        'Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
        'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS'
    ]

    # Fitur Stroke
    STROKE_NUMERIC_FEATURES = ['age', 'avg_glucose_level', 'bmi']
    STROKE_CATEGORICAL_FEATURES = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    STROKE_ALL_FEATURES_ORDERED = [
        'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
        'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'
    ]
     # Fitur biner untuk stroke (yang tidak dikategorikan secara eksplisit tapi perlu diubah jadi numerik)
    STROKE_BINARY_FEATURES = ['hypertension', 'heart_disease']


    # Fitur Kardiovaskular
    CARDIOVASCULAR_RAW_INPUT_FEATURES = [
        'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
        'cholesterol', 'gluc', 'smoke', 'alco', 'active'
    ]
    CARDIOVASCULAR_MODEL_FEATURES_ORDERED = [
        'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
        'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi'
    ]
    CARDIOVASCULAR_NUMERIC_FEATURES_TO_SCALE = CARDIOVASCULAR_MODEL_FEATURES_ORDERED
    # Fitur biner untuk cardiovascular (jika ada yg perlu perlakuan khusus, contoh: 'gender' diubah jadi 0/1 jika model mengharapkannya)
    # gender, smoke, alco, active biasanya sudah numerik (0/1) dalam dataset kardiovaskular yang umum
    # cholesterol dan gluc bisa jadi kategorikal yang diubah numerik (1,2,3) tapi di-scale
    CARDIOVASCULAR_BINARY_LIKE_FEATURES = ['gender', 'smoke', 'alco', 'active'] # Ini adalah contoh jika perlu konversi eksplisit

    # Fitur Diabetes
    DIABETES_RAW_INPUT_FEATURES = [
        'gender', 'age', 'hypertension', 'heart_disease',
        'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level'
    ]
    DIABETES_MODEL_FEATURES_ORDERED = DIABETES_RAW_INPUT_FEATURES # Asumsi urutan sama
    DIABETES_NUMERIC_FEATURES = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    DIABETES_CATEGORICAL_FEATURES = ['gender', 'smoking_history']
    DIABETES_BINARY_FEATURES = ['hypertension', 'heart_disease']