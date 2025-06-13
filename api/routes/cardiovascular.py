# routes/cardiovascular.py
from flask import Blueprint, request, jsonify
import traceback
from config import Config
from utils.preprocessing import preprocess_general_input_data, handle_cardiovascular_features
from models.cardiovascular_model import cardiovascular_model, cardiovascular_scaler, CARDIOVASCULAR_ARTIFACTS_LOADED

cardiovascular_bp = Blueprint('cardiovascular_bp', __name__)

@cardiovascular_bp.route('/predict/cardiovascular', methods=['POST'])
def predict_cardiovascular_route():
    if not CARDIOVASCULAR_ARTIFACTS_LOADED:
        return jsonify({"error": "Model Kardiovaskular atau scaler tidak berhasil dimuat. Periksa log server."}), 500
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Tidak ada data JSON yang diterima."}), 400
        
        # Validasi input mentah (sebelum BMI dihitung)
        for feature in Config.CARDIOVASCULAR_RAW_INPUT_FEATURES:
            if feature not in data:
                return jsonify({"error": f"Fitur yang hilang dalam input: '{feature}' untuk prediksi kardiovaskular."}), 400
        
        processed_df = preprocess_general_input_data(
            data_dict=data, # data mentah akan diproses oleh handle_cardiovascular_features dulu
            all_features_ordered=Config.CARDIOVASCULAR_MODEL_FEATURES_ORDERED,
            numeric_features_to_scale=Config.CARDIOVASCULAR_NUMERIC_FEATURES_TO_SCALE,
            scaler=cardiovascular_scaler,
            # Tidak ada categorical_features atau le_dict eksplisit untuk model ini di contoh awal
            # Jika ada (misal, gender perlu di-encode), tambahkan di Config dan di sini
            binary_features=Config.CARDIOVASCULAR_BINARY_LIKE_FEATURES, # jika perlu konversi eksplisit 0/1
            special_handling=handle_cardiovascular_features
        )
        
        probability = cardiovascular_model.predict(processed_df)[0][0] # Ambil probabilitas dari output model
        probability = float(probability) # Konversi ke float standar

        prediction_numeric_label = 1 if probability > 0.5 else 0

        # Tentukan label teks (High/Medium/Low) berdasarkan rentang probabilitas
        if probability > 0.7:
            prediction_text_label = 'High Risk'
        elif probability > 0.3:
            prediction_text_label = 'Medium Risk'
        else:
            prediction_text_label = 'Low Risk'

        # Kembalikan respons sesuai format spesifik yang terakhir diminta
        return jsonify({
            "prediction": prediction_text_label,
            "prediction_label_numeric": prediction_numeric_label,
            "probability_of_cardiovascular_disease": probability
        })


    except ValueError as ve:
        return jsonify({"error": f"Kesalahan validasi atau preprocessing data kardiovaskular: {str(ve)}"}), 400
    except RuntimeError as re:
        return jsonify({"error": f"Kesalahan runtime saat prediksi kardiovaskular: {str(re)}"}), 500
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": "Terjadi kesalahan internal saat prediksi kardiovaskular.", "details": str(e)}), 500