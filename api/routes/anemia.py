# routes/anemia.py
from flask import Blueprint, request, jsonify
import numpy as np
import traceback
from ..config import Config # Relatif import dari package yang sama
from ..utils.preprocessing import preprocess_general_input_data
from ..models.anemia_model import (
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
        # 'le_dict' adalah anemia_label_encoders
        # 'categorical_features' adalah anemia_categorical_features (diturunkan di anemia_model.py)
        # 'numeric_features_to_scale' adalah anemia_numeric_features_to_scale (diturunkan)
        # 'all_features_ordered' adalah anemia_input_features_ordered
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
        prediction_proba = anemia_model.predict(processed_df) # Outputnya adalah probabilitas per kelas
        
        # Ambil probabilitas untuk kelas 0 dan kelas 1
        prob_class_0 = float(prediction_proba[0][0]) # Probabilitas "Tidak Terkena Anemia"
        prob_class_1 = float(prediction_proba[0][1]) # Probabilitas "Terkena Anemia"

        predicted_class_idx = np.argmax(prediction_proba[0]) # 0 atau 1
        predicted_class_label = ANEMIA_TARGET_CLASSES.get(predicted_class_idx, "Label tidak diketahui")

        return jsonify({
            "predicted_class_label": predicted_class_label,
            "predicted_class_index": int(predicted_class_idx),
            "probabilities": {
                ANEMIA_TARGET_CLASSES[0]: prob_class_0,
                ANEMIA_TARGET_CLASSES[1]: prob_class_1
            }
        })

    except ValueError as ve:
        return jsonify({"error": f"Kesalahan validasi atau preprocessing data anemia: {str(ve)}"}), 400
    except RuntimeError as re:
        return jsonify({"error": f"Kesalahan runtime saat prediksi anemia: {str(re)}"}), 500
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": "Terjadi kesalahan internal saat prediksi anemia.", "details": str(e)}), 500