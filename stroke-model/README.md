# Dokumentasi Model Prediksi Stroke

Repository ini berisi implementasi model klasifikasi deep learning untuk memprediksi risiko stroke berdasarkan faktor kesehatan dan gaya hidup.

## Deskripsi Dataset

Dataset stroke prediction berisi data pasien dengan informasi medis dan demografis yang digunakan untuk memprediksi apakah seseorang berisiko terkena stroke. Data diambil dari [Dataset Stroke Prediction](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset).


### Atribut

Dataset mencakup atribut berikut:

| **Kategori**        | **Nama Fitur (Kolom)** | **Deskripsi**                        | **Tipe Data** | **Cara Input / Keterangan**                                                 |
| ------------------- | ---------------------- | ------------------------------------ | ------------- | --------------------------------------------------------------------------- |
| **Data Demografis** | `age`                  | Usia pasien                          | Integer       | Input langsung. Contoh: 16                                                  |
|                     | `gender`               | Jenis kelamin                        | String        | Pilihan: `Male`, `Female`, `Other`                                          |
|                     | `ever_married`         | Status pernikahan                    | String        | Pilihan: `Yes`, `No`                                                        |
|                     | `work_type`            | Jenis pekerjaan                      | String        | Pilihan: `Private`, `Self-employed`, `Govt_job`, `children`, `Never_worked` |
|                     | `Residence_type`       | Tipe tempat tinggal                  | String        | Pilihan: `Urban`, `Rural`                                                   |
| **Kondisi Medis**   | `hypertension`         | Riwayat hipertensi                   | Integer       | 0: Tidak, 1: Ya                                                             |
|                     | `heart_disease`        | Riwayat penyakit jantung             | Integer       | 0: Tidak, 1: Ya                                                             |
|                     | `avg_glucose_level`    | Rata-rata kadar gula darah           | Float         | Dalam mg/dL – dari hasil tes darah. Contoh 228.69                                |
|                     | `bmi`                  | Body Mass Index (Indeks Massa Tubuh) | Float         | Dalam kg/m² – dihitung dari tinggi dan berat badan. Contoh: 35.2                    |
| **Gaya Hidup**      | `smoking_status`       | Status merokok                       | String        | Pilihan: `never smoked`, `formerly smoked`, `smokes`, `Unknown`             |
| **Target**          | `stroke`               | Riwayat stroke                       | Integer       | 0: Tidak Stroke, 1: Stroke                                                  |

## Alur Kerja

Implementasi mencakup langkah-langkah berikut:

### 1. Eksplorasi Data (EDA)
- Analisis statistik deskriptif
- Visualisasi distribusi data
- Analisis korelasi antar fitur
- Pemeriksaan nilai yang hilang
- Analisis ketidakseimbangan kelas target

### 2. Preprocessing Data
- Penghapusan kolom ID yang tidak relevan
- Penanganan nilai yang hilang pada fitur BMI
- Encoding fitur kategorikal menggunakan LabelEncoder
- Standarisasi fitur numerik dengan StandardScaler
- Pembagian data menjadi set pelatihan dan pengujian
- Penggunaan SMOTE untuk mengatasi ketidakseimbangan kelas

### 3. Pemodelan
- Arsitektur: Deep Neural Network dengan 5 layer (256, 128, 64, 32, dan output)
- Implementasi teknik regularisasi:
  - Dropout layers dengan tingkat berbeda
  - L2 regularization pada setiap layer
  - Batch Normalization
- Penggunaan AdamOptimizer dengan learning rate rendah
- Callbacks: Early Stopping dan ReduceLROnPlateau

### 4. Evaluasi Model
- Metrik: Accuracy, Precision, Recall, F1-Score
- Visualisasi: Confusion Matrix, ROC Curve dengan AUC
- Analisis Learning Curves (akurasi dan loss)

### 5. Implementasi Fungsi Prediksi
- Fungsi untuk memprediksi risiko stroke pasien baru
- Pengembalian kelas prediksi dan nilai probabilitas

### 6. Deployment
- Penyimpanan model dalam format TensorFlow SavedModel
- Konversi model ke format ONNX untuk deployment

## Cara Menggunakan

1. **Persiapan Lingkungan**:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn tensorflow imblearn tf2onnx
   ```

2. **Menjalankan Kode**:
   - Jalankan notebook `Prediksi_Stroke.ipynb` 
   - Atau jalankan script Python:
     ```bash
     python stroke_prediction.py
     ```

3. **Prediksi dengan Data Baru**:
   ```python
   hasil = predict_stroke(
       age=65,              # Usia
       gender='Male',       # Jenis kelamin
       hypertension=1,      # Memiliki hipertensi
       heart_disease=1,     # Memiliki penyakit jantung
       ever_married='Yes',  # Status pernikahan
       work_type='Private', # Jenis pekerjaan
       residence_type='Urban', # Tipe tempat tinggal
       avg_glucose_level=200,  # Kadar glukosa rata-rata
       bmi=28,                 # BMI
       smoking_status='formerly smoked' # Status merokok
   )
   
   print(f"Hasil: {'Stroke' if hasil['predicted_class'] == 1 else 'Tidak Stroke'}")
   print(f"Probabilitas: {hasil['probability']:.4f}")
   ```

## Struktur File

```
.
├── stroke-model.ipynb              # Jupyter notebook
├── saved_model_stroke/              # Model TensorFlow tersimpan
└── README.md                        # Dokumentasi
```

## Hasil dan Performa

Model mencapai akurasi sekitar 77% pada data pengujian, dengan performa yang baik pada semua kelas stroke. Detail lengkap performa dapat dilihat pada laporan klasifikasi dan confusion matrix dalam kode.
## Referensi

Gupta, A., Mishra, N., Jatana, N., Malik, S., Gepreel, K. A., Asmat, F., & Mohanty, S. N. (2025). Predicting stroke risk: An effective stroke prediction model based on neural networks. Journal of Neurorestoratology, 13(1), 100156. https://doi.org/10.1016/j.jnrt.2024.100156

## Catatan

- Model ini mengimplementasikan berbagai teknik regularisasi untuk menangani ketidakseimbangan kelas dan mencegah overfitting.
- Model dimaksudkan untuk tujuan edukasi dan penelitian, bukan untuk diagnosis medis.
- Penggunaan model dalam konteks medis harus selalu melibatkan konsultasi dengan profesional kesehatan.
