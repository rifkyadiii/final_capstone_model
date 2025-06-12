# Sistem Prediksi Penyakit Diabetes

Sistem machine learning yang memprediksi risiko penyakit diabetes menggunakan neural network. Proyek ini mengimplementasikan model deep learning untuk menganalisis data kesehatan pasien dan memberikan penilaian risiko penyakit diabetes.

## Gambaran Proyek

Sistem ini menggunakan neural network untuk memprediksi risiko penyakit diabetes berdasarkan berbagai parameter kesehatan termasuk usia, jenis kelamin, BMI, riwayat merokok, hipertensi, penyakit jantung, kadar HbA1c, dan kadar glukosa darah.

## Informasi Dataset

| Kategori              | DataFitur            | Deskripsi                              | Tipe Data | Cara Input / Keterangan                                         |
|-----------------------|----------------------|-----------------------------------------|-----------|------------------------------------------------------------------|
| Mudah Diketahui       | gender               | Jenis kelamin                           | String    | Pilihan: Female, Male, Other                                     |
|                       | age                  | Usia dalam tahun                        | Integer   | Input langsung. Contoh: 20                                       |
|                       | smoking_history      | Riwayat merokok                         | String    | Pilihan: never, No Info, current, former, ever, not current      |
| Kondisi Kesehatan     | hypertension         | Riwayat hipertensi                      | Integer   | Input: Ya/Tidak (0: Tidak, 1: Ya)                                |
|                       | heart_disease        | Riwayat penyakit jantung                | Integer   | Input: Ya/Tidak (0: Tidak, 1: Ya)                                |
| Pengukuran Fisik      | bmi                  | Body Mass Index (kg/m²)                 | Float     | Bisa dihitung dari tinggi dan berat badan. Contoh: 28.5          |
|                       | HbA1c_level          | Kadar HbA1c (%)                         | Float     | Perlu tes darah HbA1c. Contoh:  6.2                              |
|                       | blood_glucose_level  | Kadar glukosa darah (mg/dL)             | Integer   | Perlu tes gula darah. Contoh: 145                                             |
| Target Variable       | diabetes             | Status diabetes                         | Integer   | 0: Tidak Diabetes, 1: Diabetes                                   |

## Output Prediksi

1. Probability (Probabilitas)
   
Tingkat kepercayaan model bahwa pasien memiliki risiko diabetes (0.0 - 1.0).

2. Prediction (Prediksi)

  - Tidak Diabetes: Probabilitas ≤ 0.5
  - Diabetes: Probabilitas > 0.5

3. Risk Level (Tingkat Risiko)

  - Low: Probabilitas < 0.3 - Risiko minimal
  - Medium: Probabilitas 0.3-0.7 - Perlu pemantauan
  - High: Probabilitas > 0.7 - Perlu tindakan segera

## Alur Kerja

1. Import Library & Load Data
   - Menggunakan pustaka: pandas, numpy, seaborn, matplotlib, scikit-learn, tensorflow, dan tf2onnx
   - Dataset diimpor dari file CSV lokal 
2. Eksplorasi Data (EDA)
   - Analisis informasi dataset dan missing values
   - Statistik deskriptif untuk memahami distribusi data
   - Pemeriksaan shape dan struktur data
3. Preprocessing Data
   - Encoding fitur kategorikal (gender, smoking_history) menggunakan LabelEncoder
   - Normalisasi fitur numerik (age, bmi, HbA1c_level, blood_glucose_level) dengan StandardScaler
   - Split data menjadi training dan testing (80:20)
4. Modeling Data
   - Arsitektur Neural Network
   - Optimizer: Adam
   - Loss function: Binary Crossentropy
   - Early Stopping callback untuk mencegah overfitting
5. Evaluasi Model
   - Metrik evaluasi:
     - Accuracy
     - Precision
     - Recall
     - F1-score
     - Confusion Matrix
   - Visualisasi:
     - Training/Validation Accuracy dan Loss curves
     - Confusion Matrix heatmap
6. Inferensi (Prediksi Baru)
    - Fungsi prediksi untuk data pasien baru
    - Input berupa parameter klinis: jenis kelamin, usia, BMI, riwayat kesehatan, dll.
    - Output:
      - Probabilitas diabetes
      - Prediksi kategori (Diabetes/Tidak Diabetes)
7. Deployment
   - Model disimpan dalam format .h5 (diabetes_prediction_model.h5)
   - Scaler disimpan sebagai .pkl (diabetes_scaler.pkl)
   - Model dikonversi ke format ONNX (diabetes_model.onnx)

## Hasil dan Performa

![image](https://github.com/user-attachments/assets/72646464-f374-4915-8867-82ac015cde31)

## Referensi

- Kaggle: https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset
- Jurnal: 


