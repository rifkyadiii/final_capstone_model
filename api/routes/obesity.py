# routes/obesity.py
from flask import Blueprint, request, jsonify
import numpy as np
import traceback
from ..config import Config
from ..utils.preprocessing import preprocess_general_input_data
from ..models.obesity_model import obesity_model, obesity_scaler, obesity_le_dict, obesity_target_le, OBESITY_ARTIFACTS_LOADED

obesity_bp = Blueprint('obesity_bp', __name__)

@obesity_bp.route('/predict/obesity', methods=['POST'])
def predict_obesity_route():
    if not OBESITY_ARTIFACTS_LOADED:
        return jsonify({"error": "Model Obesitas atau preprocessor tidak berhasil dimuat. Periksa log server."}), 500
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Tidak ada data JSON yang diterima."}), 400
        
        # Validasi input mentah (semua fitur obesitas diharapkan ada di input)
        # preprocess_general_input_data akan menangani jika ada yg hilang dari all_features_ordered
        # tapi cek awal ini bisa lebih spesifik untuk payload.
        for feature in Config.OBESITY_ALL_FEATURES_ORDERED:
            if feature not in data:
                return jsonify({"error": f"Fitur yang hilang dalam input: '{feature}' untuk prediksi obesitas."}), 400

        processed_df = preprocess_general_input_data(
            data_dict=data,
            all_features_ordered=Config.OBESITY_ALL_FEATURES_ORDERED,
            numeric_features_to_scale=Config.OBESITY_NUMERIC_FEATURES,
            scaler=obesity_scaler,
            categorical_features=Config.OBESITY_CATEGORICAL_FEATURES,
            le_dict=obesity_le_dict
        )
        prediction_proba = obesity_model.predict(processed_df)
        predicted_class_idx = np.argmax(prediction_proba[0])
        predicted_class_label = obesity_target_le.inverse_transform([predicted_class_idx])[0]
        probabilities = {obesity_target_le.inverse_transform([i])[0]: float(prob) 
                         for i, prob in enumerate(prediction_proba[0])}
        
        return jsonify({"predicted_class": str(predicted_class_label), "probabilities": probabilities})

    except ValueError as ve:
        return jsonify({"error": f"Kesalahan validasi atau preprocessing data obesitas: {str(ve)}"}), 400
    except RuntimeError as re:
        return jsonify({"error": f"Kesalahan runtime saat prediksi obesitas: {str(re)}"}), 500
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": "Terjadi kesalahan internal saat prediksi obesitas.", "details": str(e)}), 500