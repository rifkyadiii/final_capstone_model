# routes/anemia.py
from flask import Blueprint, request, jsonify
import numpy as np
import traceback
from config import Config # Relatif import dari package yang sama
from utils.preprocessing import preprocess_general_input_data
from models.anemia_model import (
    anemia_model, anemia_scaler, anemia_label_encoders,
    anemia_input_features_ordered, anemia_categorical_features,
    anemia_numeric_features_to_scale, ANEMIA_ARTIFACTS_LOADED,
    ANEMIA_TARGET_CLASSES
)

anemia_bp = Blueprint('anemia_bp', __name__)

@anemia_bp.route('/predict/anemia', methods=['POST'])
def predict_anemia_route():
    if not ANEMIA_ARTIFACTS_LOADED:
        return jsonify({"error": "Model Anemia atau preprocessor tidak berhasil dimuat. Periksa log server."}), 500
    
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Tidak ada data JSON yang diterima."}), 400

        # Validasi input mentah: pastikan semua fitur yang dibutuhkan ada
        if not anemia_input_features_ordered: # Jika daftar fitur gagal dimuat
             return jsonify({"error": "Konfigurasi fitur input Anemia tidak tersedia."}), 500

        missing_features = [
            feature for feature in anemia_input_features_ordered if feature not in data
        ]
        if missing_features:
            return jsonify({"error": f"Fitur yang hilang dalam input: {missing_features} untuk prediksi anemia."}), 400

        # Preprocess data
        processed_df = preprocess_general_input_data(
            data_dict=data,
            all_features_ordered=anemia_input_features_ordered,
            numeric_features_to_scale=anemia_numeric_features_to_scale,
            scaler=anemia_scaler,
            categorical_features=anemia_categorical_features,
            le_dict=anemia_label_encoders
            # Tidak ada 'binary_features' atau 'special_handling' khusus untuk anemia sejauh ini
        )

        # Prediksi
        prediction_proba = anemia_model.predict(processed_df)[0]
        
        predicted_class_idx = int(np.argmax(prediction_proba))

        prob_anemia = 0.0
        if len(prediction_proba) > 1:
            prob_anemia = float(prediction_proba[1])
        else:
            # Penanganan jika model hanya mengeluarkan satu probabilitas
            prob_anemia = float(prediction_proba[0])

        # Kembalikan respons JSON dalam format standar yang diinginkan
        return jsonify({
            "prediction": str(predicted_class_idx),
            "prediction_label_numeric": predicted_class_idx,
            "probability_of_anemia": prob_anemia
        })

    except ValueError as ve:
        return jsonify({"error": f"Kesalahan validasi atau preprocessing data anemia: {str(ve)}"}), 400
    except RuntimeError as re:
        return jsonify({"error": f"Kesalahan runtime saat prediksi anemia: {str(re)}"}), 500
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": "Terjadi kesalahan internal saat prediksi anemia.", "details": str(e)}), 500