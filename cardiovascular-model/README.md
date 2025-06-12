# Sistem Prediksi Penyakit Kardiovaskular

Sistem machine learning yang memprediksi risiko penyakit kardiovaskular menggunakan neural network. Proyek ini mengimplementasikan model deep learning untuk menganalisis data kesehatan pasien dan memberikan penilaian risiko penyakit kardiovaskular.

## Gambaran Proyek
Sistem ini menggunakan neural network untuk memprediksi risiko penyakit kardiovaskular berdasarkan berbagai parameter kesehatan termasuk usia, jenis kelamin, pengukuran fisik, tekanan darah, kadar kolesterol, dan faktor gaya hidup.

## Informasi Dataset

| Kategori Data              | Fitur       | Deskripsi                        | Tipe Data      | Cara Input / Keterangan                                                |
| -------------------------- | ----------- | -------------------------------- | -------------- | ---------------------------------------------------------------------- |
| Mudah Diketahui          | age         | Usia dalam tahun                 | Integer        | Input langsung. Contoh: 20                                                   |
|                            | gender      | Jenis kelamin                    | Integer | Pilihan (0: Perempuan, 1: Laki-laki)                                   |
|                            | height      | Tinggi badan (cm)                | Integer        | Bisa diukur sendiri. Contoh: 163                                            |
|                            | weight      | Berat badan (kg)                 | Float  | Bisa ditimbang sendiri. Contoh: 58                                                 |
| Perlu Pemeriksaan Medis | ap\_hi      | Tekanan darah sistolik           | Integer        | Perlu tensimeter. Contoh 115                                                  |
|                            | ap\_lo      | Tekanan darah diastolik          | Integer        | Perlu tensimeter. Contoh 75                                                       |
|                            | cholesterol | Kadar kolesterol                 | Integer | Perlu tes darah (1: Kolesterol normal, 2: Kolesterol diatas normal)                                                        |
|                            | gluc        | Kadar gula darah                 | Integer | Perlu tes gula (1: Glukosa normal, 2: Glukosa diatas normal)                                                         |
| Gaya Hidup              | smoke       | Status merokok                   | Integer | Input: Ya/Tidak (0: Tidak, 1: Ya)                                      |
|                            | alco        | Konsumsi alkohol                 | Integer | Input: Ya/Tidak (0: Tidak, 1: Ya)                                      |
|                            | active      | Aktivitas fisik rutin            | Integer | Input: Ya/Tidak (0: Tidak aktif, 1: Aktif); olahraga ≥150 menit/minggu |
| Target Variable         | cardio      | Prediksi penyakit kardiovaskular | Integer        | 0: Risiko Rendah, 1: Risiko Tinggi                                     |

## Output Prediksi

1. Probability (Probabilitas)

Tingkat kepercayaan model bahwa pasien memiliki risiko penyakit kardiovaskular (persen).

2. Prediction (Prediksi)

  - Low Risk: Probabilitas ≤ 50%
  - High Risk: Probabilitas > 50%

3. Risk Level (Tingkat Risiko)

  - Low: Probabilitas < 30% - Risiko minimal
  - Medium: Probabilitas 30-70% - Perlu pemantauan
  - High: Probabilitas > 70% - Perlu tindakan segera

## Alur Kerja

1. Import Library & Load Data
   - Menggunakan pustaka: pandas, seaborn, matplotlib, scikit-learn, tensorflow, dan tf2onnx.
   - Dataset diimpor dari file CSV lokal.
3. Eksplorasi Data (EDA)
   - Analisis distribusi target (cardio)
   - Visualisasi hubungan usia, tekanan darah, kolesterol, dan BMI dengan status penyakit
   - Heatmap korelasi antar fitur
5. Preprocessing Data
   - Menghapus outlier pada fitur ap_hi, ap_lo, dan bmi
   - Normalisasi fitur numerik dengan StandardScaler
   - Split data menjadi training dan testing dengan stratifikasi
7. Modeling Data
   - Arsitektur Neural Network dengan beberapa layer Dense dan Dropout
   - Fungsi aktivasi ReLU dan Sigmoid (untuk klasifikasi biner)
   - Optimizer: Adam
   - Early Stopping dan ReduceLROnPlateau sebagai callbacks
   - Validasi menggunakan validation_split
9. Evaluasi Data
    - Metrik evaluasi:
      - Accuracy
      - Precision
      - Recall
      - F1-score
      - Confusion Matrix
      - ROC Curve & AUC Score
    - Visualisasi akurasi, loss, dan kurva ROC
11. Inferensi (Prediksi Baru)
    - Fungsi predict_cardiovascular_disease() memungkinkan prediksi untuk data pasien baru
    - Input berupa parameter klinis: usia, jenis kelamin, tekanan darah, kolesterol, kebiasaan merokok, dll.
    - Output:
      - Probabilitas terjadinya penyakit
      - Prediksi (High/Low Risk)
      - Tingkat risiko (Low / Medium / High)
13. Deployment
    - Model disimpan dalam format .h5 (cardio_prediction_model.h5)
    - Scaler disimpan sebagai .pkl (cardio_scaler.pkl)
    - Model dikonversi ke format ONNX (cardiovascular_model.onnx).
14. Hasil dan Performa
    
    ![image](https://github.com/user-attachments/assets/aee05841-3743-4069-8175-a73a9ab70751)

16. Referensi
    - Kaggle: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset
    - Jurnal: https://pmc.ncbi.nlm.nih.gov/articles/PMC10621606/
