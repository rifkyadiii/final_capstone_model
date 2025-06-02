#!/usr/bin/env python
# coding: utf-8

# app.py (Main Application File)
from flask import Flask, jsonify
import os

# Import Blueprints
from .routes.obesity import obesity_bp
from .routes.stroke import stroke_bp
from .routes.cardiovascular import cardiovascular_bp
from .routes.diabetes import diabetes_bp
from .config import Config

app = Flask(__name__)

# Register Blueprints
app.register_blueprint(obesity_bp)
app.register_blueprint(stroke_bp)
app.register_blueprint(cardiovascular_bp)
app.register_blueprint(diabetes_bp)

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Selamat Datang di API HealtGuard!",
        "endpoints": {
            "prediksi_obesitas": "/predict/obesity (POST)",
            "prediksi_stroke": "/predict/stroke (POST)",
            "prediksi_kardiovaskular": "/predict/cardiovascular (POST)",
            "prediksi_diabetes": "/predict/diabetes (POST)"
        }
    })

if __name__ == '__main__':
    # Path di Config sudah diatur relatif terhadap lokasi config.py
    # Cetak CWD dan path model untuk verifikasi saat startup jika perlu
    print(f"Current working directory (app.py): {os.getcwd()}")
    
    # Contoh path yang bisa dicek dari Config (path sudah absolut di dalam Config)
    print(f"Resolved Config.OBESITY_MODEL_PATH: {Config.OBESITY_MODEL_PATH}")
    print(f"Does OBESITY_MODEL_PATH exist? {os.path.exists(Config.OBESITY_MODEL_PATH)}")

    # Menjalankan aplikasi
    # Ganti host dan port sesuai kebutuhan production/deployment
    # Untuk development, debug=True berguna
    app.run(host='0.0.0.0', port=5000, debug=True)