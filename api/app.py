# api/app.py
from flask import Flask, jsonify
import os

# Import Blueprints using explicit relative import
from .routes.obesity import obesity_bp
from .routes.stroke import stroke_bp
from .routes.cardiovascular import cardiovascular_bp
from .routes.diabetes import diabetes_bp
from .routes.anemia import anemia_bp
from .routes.heart_attack import heart_attack_bp
from .config import Config

app = Flask(__name__)

# Register Blueprints
app.register_blueprint(obesity_bp)
app.register_blueprint(stroke_bp)
app.register_blueprint(cardiovascular_bp)
app.register_blueprint(diabetes_bp)
app.register_blueprint(anemia_bp)
app.register_blueprint(heart_attack_bp)
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Selamat Datang di API Prediksi Kesehatan Modular!",
        "endpoints": {
            "prediksi_obesitas": "/predict/obesity (POST)",
            "prediksi_stroke": "/predict/stroke (POST)",
            "prediksi_kardiovaskular": "/predict/cardiovascular (POST)",
            "prediksi_diabetes": "/predict/diabetes (POST)",
            "prediksi_anemia": "/predict/anemia (POST)",
            "prediksi_heart_attack": "/predict/heart_attack (POST)"
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))