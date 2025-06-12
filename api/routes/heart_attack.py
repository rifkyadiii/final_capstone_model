# api/routes/heart_attack.py
from flask import Blueprint, request, jsonify
import numpy as np
import traceback
from config import Config
from utils.preprocessing import preprocess_general_input_data
from models.heart_attack_model import (
    ha_model, ha_scaler, ha_feature_le_dict, ha_target_le,
    ha_input_features_ordered, ha_categorical_features,
    ha_numeric_features_to_scale, ha_binary_features,
    HEART_ATTACK_ARTIFACTS_LOADED, HEART_ATTACK_TARGET_CLASSES_MAP
)

heart_attack_bp = Blueprint('heart_attack_bp', __name__)

@heart_attack_bp.route('/predict/heart_attack', methods=['POST'])
def predict_heart_attack_route():
    if not HEART_ATTACK_ARTIFACTS_LOADED:
        return jsonify({"error": "Model Heart Attack atau preprocessor tidak berhasil dimuat. Periksa log server."}), 500
    
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Tidak ada data JSON yang diterima."}), 400

        if not ha_input_features_ordered:
             return jsonify({"error": "Konfigurasi fitur input untuk Heart Attack tidak tersedia di server."}), 500

        missing_features = [
            feature for feature in ha_input_features_ordered if feature not in data
        ]
        if missing_features:
            return jsonify({"error": f"Fitur yang hilang dalam input: {missing_features}. Fitur yang diharapkan: {ha_input_features_ordered}"}), 400

        # Preprocess data with correct feature definitions
        processed_df = preprocess_general_input_data(
            data_dict=data,
            all_features_ordered=ha_input_features_ordered,
            numeric_features_to_scale=ha_numeric_features_to_scale,
            scaler=ha_scaler,
            categorical_features=ha_categorical_features,
            le_dict=ha_feature_le_dict,
            binary_features=ha_binary_features
        )

        # Make prediction
        prediction_proba_raw = ha_model.predict(processed_df)[0]
        
        # Tentukan indeks kelas prediksi (0 atau 1)
        predicted_class_idx = int(np.argmax(prediction_proba_raw))

        # Asumsi: Indeks 1 adalah kelas positif ('more chance of heart attack')
        # Ambil probabilitas untuk kelas positif tersebut
        # Pastikan model Anda memiliki 2 output probabilitas
        prob_heart_attack = 0.0
        if len(prediction_proba_raw) > 1:
            prob_heart_attack = float(prediction_proba_raw[1])
        else:
            # Handle kasus jika model hanya mengeluarkan satu probabilitas
            prob_heart_attack = float(prediction_proba_raw[0])

        return jsonify({
            "prediction": str(predicted_class_idx),
            "prediction_label_numeric": predicted_class_idx,
            "probability_of_heart_attack": prob_heart_attack
        })

    except ValueError as ve:
        return jsonify({"error": f"Kesalahan pada data input atau preprocessing: {str(ve)}"}), 400
    except RuntimeError as re:
        return jsonify({"error": f"Kesalahan runtime saat prediksi heart attack: {str(re)}"}), 500
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": "Terjadi kesalahan internal pada server saat prediksi heart attack.", "details": str(e)}), 500