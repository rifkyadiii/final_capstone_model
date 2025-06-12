# routes/stroke.py
from flask import Blueprint, request, jsonify
import numpy as np
import traceback
from config import Config
from utils.preprocessing import preprocess_general_input_data
from models.stroke_model import stroke_model, stroke_scaler, stroke_le_dict, STROKE_ARTIFACTS_LOADED

stroke_bp = Blueprint('stroke_bp', __name__)

@stroke_bp.route('/predict/stroke', methods=['POST'])
def predict_stroke_route():
    if not STROKE_ARTIFACTS_LOADED:
        return jsonify({"error": "Model Stroke atau preprocessor tidak berhasil dimuat. Periksa log server."}), 500
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Tidak ada data JSON yang diterima."}), 400

        for feature in Config.STROKE_ALL_FEATURES_ORDERED:
            if feature not in data:
                return jsonify({"error": f"Fitur yang hilang dalam input: '{feature}' untuk prediksi stroke."}), 400

        processed_df = preprocess_general_input_data(
            data_dict=data,
            all_features_ordered=Config.STROKE_ALL_FEATURES_ORDERED,
            numeric_features_to_scale=Config.STROKE_NUMERIC_FEATURES,
            scaler=stroke_scaler,
            categorical_features=Config.STROKE_CATEGORICAL_FEATURES,
            le_dict=stroke_le_dict,
            binary_features=Config.STROKE_BINARY_FEATURES # Tambahkan ini
        )
        prediction_proba_arr = stroke_model.predict(processed_df)[0]
        # Output dari model Keras biasanya float32, konversi ke float standar untuk JSON
        prob_no_stroke = float(prediction_proba_arr[0])
        prob_stroke = float(prediction_proba_arr[1])
        
        predicted_class_label = 1 if prob_stroke >= 0.5 else 0 # Asumsi threshold 0.5
        
        return jsonify({
            "prediction": str(predicted_class_label), 
            "prediction_label_numeric": predicted_class_label,
            "probability_of_stroke": prob_stroke 
        })

    except ValueError as ve:
        return jsonify({"error": f"Kesalahan validasi atau preprocessing data stroke: {str(ve)}"}), 400
    except RuntimeError as re:
        return jsonify({"error": f"Kesalahan runtime saat prediksi stroke: {str(re)}"}), 500
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": "Terjadi kesalahan internal saat prediksi stroke.", "details": str(e)}), 500