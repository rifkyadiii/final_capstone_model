# 📊 Proyek Deep Learning untuk Capstone CodingCamp by DBS 2025

Repositori ini berisi dua proyek utama yang berfokus pada penerapan deep learning untuk prediksi kondisi kesehatan berdasarkan data medis dan gaya hidup:

1. **Prediksi Risiko Stroke**
2. **Klasifikasi Tingkat Obesitas**

Keduanya menggunakan dataset dari Kaggle dan ditujukan untuk edukasi, eksplorasi machine learning, serta demonstrasi kemampuan deployment model dalam format TensorFlow dan ONNX.

## 📁 Struktur Proyek

```
.
├── stroke-model/                   # Proyek klasifikasi risiko stroke
│   ├── stroke-model.py
│   ├── stroke-model.ipynb
│   ├── saved_model_stroke/
│   ├── model_stroke.onnx
│   └── README.md
│
├── obesity-model/                  # Proyek klasifikasi tingkat obesitas
│   ├── obesity-model.py
│   ├── obesity-model.ipynb
│   ├── saved_model_obesity/
│   ├── model_obesity.onnx
│   └── README.md
│
├── dataset/                        # Folder berisi dataset
│   ├── stroke_data.csv
│   └── obesity_data.csv
│
├── reference/                      # Folder referensi dan sumber pustaka
│   ├── stroke-paper.pdf
│   └── obesity-study.pdf
│
├── README.md                       # Dokumentasi utama
│
└── requirements.txt                # File daftar library yang digunakan
```

## 🧠 Proyek 1: Prediksi Risiko Stroke

* **Tujuan**: Memprediksi apakah seseorang berisiko terkena stroke berdasarkan data demografis, medis, dan gaya hidup.
* **Teknologi**: Deep Neural Network (DNN) dengan regularisasi (Dropout, L2, BatchNorm).
* **Evaluasi**: Akurasi \~77%, AUC \~0.82
* **Dataset**: [Stroke Prediction Dataset - Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
* **Output**: Kelas stroke (0 atau 1)
* **Deployment**: TensorFlow dan ONNX

📄 [Baca selengkapnya → README Stroke](./stroke/README.md)

## 🧬 Proyek 2: Klasifikasi Tingkat Obesitas

* **Tujuan**: Mengklasifikasikan tingkat obesitas (7 kelas) berdasarkan kebiasaan makan, aktivitas fisik, dan kondisi fisik.
* **Teknologi**: Deep Neural Network dengan aktivasi ReLU dan Softmax, Dropout, EarlyStopping
* **Evaluasi**: Akurasi \~95%
* **Dataset**: [Obesity Levels - Kaggle](https://www.kaggle.com/datasets/fatemehmehrparvar/obesity-levels)
* **Output**: Salah satu dari 7 kelas obesitas
* **Deployment**: TensorFlow dan ONNX

📄 [Baca selengkapnya → README Obesity](./obesity/README.md)

## 🚀 Cara Menjalankan Proyek

1. **Kloning repositori ini**:

   ```bash
   git clone <URL-repo-anda>
   cd nama-folder
   ```

2. **Pasang dependensi**:

   ```bash
   pip install -r requirements.txt
   # atau install manual
   pip install pandas numpy matplotlib seaborn scikit-learn tensorflow tf2onnx imblearn
   ```

3. **Masuk ke folder proyek dan jalankan notebook atau script**:

   ```bash
   cd stroke
   python stroke_prediction.py
   # atau
   cd ../obesity
   python obesitas_classification.py
   ```

## ⚠️ Catatan

* Model dalam repositori ini hanya untuk tujuan **edukasi dan penelitian**.
* Tidak dimaksudkan untuk digunakan dalam diagnosis atau pengambilan keputusan medis tanpa konsultasi profesional.

## 🧾 Lisensi

Proyek ini dilisensikan di bawah [MIT License](LICENSE) — silakan gunakan, modifikasi, dan kontribusi sesuai kebutuhan.
