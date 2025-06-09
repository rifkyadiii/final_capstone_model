# routes/diabetes.py
from flask import Blueprint, request, jsonify
import traceback
from config import Config
from utils.preprocessing import preprocess_general_input_data
from models.diabetes_model import (
    diabetes_model, diabetes_scaler, diabetes_feature_encoders, 
    diabetes_target_encoder, DIABETES_ARTIFACTS_LOADED
)

diabetes_bp = Blueprint('diabetes_bp', __name__)

@diabetes_bp.route('/predict/diabetes', methods=['POST'])
def predict_diabetes_route():
    if not DIABETES_ARTIFACTS_LOADED:
        return jsonify({"error": "Model Diabetes atau preprocessor tidak berhasil dimuat. Periksa log server."}), 500
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Tidak ada data JSON yang diterima."}), 400
        
        for feature in Config.DIABETES_RAW_INPUT_FEATURES:
            if feature not in data:
                return jsonify({"error": f"Fitur yang hilang dalam input: '{feature}' untuk prediksi diabetes."}), 400

        processed_df = preprocess_general_input_data(
            data_dict=data,
            all_features_ordered=Config.DIABETES_MODEL_FEATURES_ORDERED,
            numeric_features_to_scale=Config.DIABETES_NUMERIC_FEATURES,
            scaler=diabetes_scaler,
            categorical_features=Config.DIABETES_CATEGORICAL_FEATURES,
            le_dict=diabetes_feature_encoders, # Ini adalah dictionary of encoders
            binary_features=Config.DIABETES_BINARY_FEATURES
        )
        
        prediction_prob = diabetes_model.predict(processed_df)[0][0]
        prediction_prob = float(prediction_prob) # Konversi ke float standar

        prediction_binary_label = 1 if prediction_prob > 0.5 else 0 # Asumsi threshold 0.5
        
        # Pastikan diabetes_target_encoder dimuat dan digunakan dengan benar
        if diabetes_target_encoder:
            predicted_class_str = diabetes_target_encoder.inverse_transform([prediction_binary_label])[0]
        else:
            # Fallback jika target encoder tidak ada (seharusnya tidak terjadi jika DIABETES_ARTIFACTS_LOADED true)
            predicted_class_str = "Diabetes" if prediction_binary_label == 1 else "No Diabetes"

        return jsonify({
            'probability_of_diabetes': prediction_prob,
            'prediction_label_numeric': prediction_binary_label,
            'prediction': str(predicted_class_str) # Hasil dari inverse_transform
        })

    except ValueError as ve:
        return jsonify({"error": f"Kesalahan validasi atau preprocessing data diabetes: {str(ve)}"}), 400
    except RuntimeError as re:
        return jsonify({"error": f"Kesalahan runtime saat prediksi diabetes: {str(re)}"}), 500
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": "Terjadi kesalahan internal saat prediksi diabetes.", "details": str(e)}), 500