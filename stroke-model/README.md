# Dokumentasi Model Prediksi Stroke

Repository ini berisi implementasi model klasifikasi deep learning untuk memprediksi risiko stroke berdasarkan faktor kesehatan dan gaya hidup.

## Deskripsi Dataset

Dataset stroke prediction berisi data pasien dengan informasi medis dan demografis yang digunakan untuk memprediksi apakah seseorang berisiko terkena stroke. Data diambil dari [Dataset Stroke Prediction](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset).


### Atribut

Dataset mencakup atribut berikut:

1. **Demografis**:
   - `id`: ID unik pasien (dihapus dalam preprocessing)
   - `gender`: Jenis kelamin (Male/Female/Other)
   - `age`: Usia pasien
   - `Residence_type`: Tipe tempat tinggal (Rural/Urban)

2. **Kondisi Medis**:
   - `hypertension`: Memiliki hipertensi (0=tidak, 1=ya)
   - `heart_disease`: Memiliki penyakit jantung (0=tidak, 1=ya)
   - `avg_glucose_level`: Rata-rata kadar glukosa dalam darah
   - `bmi`: Body Mass Index

3. **Gaya Hidup**:
   - `ever_married`: Status pernikahan (Yes/No)
   - `work_type`: Jenis pekerjaan (Private, Self-employed, Govt_job, children, Never_worked)
   - `smoking_status`: Status merokok (formerly smoked, never smoked, smokes, Unknown)

4. **Target**:
   - `stroke`: Pernah mengalami stroke (1) atau tidak (0)

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