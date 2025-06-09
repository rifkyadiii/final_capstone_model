# Model Klasifikasi Tingkat Obesitas

Repository ini berisi implementasi model klasifikasi deep learning untuk memprediksi tingkat obesitas berdasarkan atribut fisik dan kebiasaan hidup seseorang.

## Deskripsi Dataset

Dataset yang digunakan berisi data tentang kebiasaan hidup dan kondisi fisik individu yang dikaitkan dengan 7 tingkat obesitas berbeda. Data diambil dari [Dataset Obesity Levels](https://www.kaggle.com/datasets/fatemehmehrparvar/obesity-levels).

### Atribut

Dataset mencakup atribut berikut:

| **Kategori**         | **Nama Fitur (Kolom)**           | **Deskripsi**                                         | **Tipe Data** | **Input**                                                    |
| -------------------- | -------------------------------- | ----------------------------------------------------- | ------------- | ------------------------------------------------------------------------------ |
| **Data Fisik**       | `Age`                            | Usia dalam tahun                                      | Integer       | Input langsung. Contoh: 16                                                                 |
|                      | `Gender`                         | Jenis kelamin                                         | String        | Pilihan: `Male`, `Female`                                                      |
|                      | `Height`                         | Tinggi badan dalam meter                              | Float         | Bisa diukur sendiri. Contoh: 163                                                            |
|                      | `Weight`                         | Berat badan dalam kilogram                            | Float         | Bisa ditimbang sendiri: Contoh: 68                                                         |
| **Riwayat Keluarga** | `family_history_with_overweight` | Riwayat keluarga dengan kelebihan berat badan         | String        | Pilihan: `yes`, `no`                                                           |
| **Kebiasaan Makan**  | `FAVC`                           | Konsumsi makanan berkalori tinggi                     | String        | Sering mengonsumsi? Pilihan: `yes`, `no`                                       |
|                      | `FCVC`                           | Frekuensi konsumsi sayuran                            | Float         | Skala 1 (jarang) - 3 (sering)                                                  |
|                      | `NCP`                            | Jumlah makan utama per hari                           | Float         | Antara 1 - 4 kali                                                              |
|                      | `CAEC`                           | Konsumsi makanan di antara jam makan                  | String        | Pilihan: `no`, `Sometimes`, `Frequently`, `Always`                             |
|                      | `CH2O`                           | Konsumsi air putih per hari (liter)                   | Float         | Antara 1 - 3 liter                                                             |
| **Gaya Hidup**       | `SMOKE`                          | Status merokok                                        | String        | Pilihan: `yes`, `no`                                                           |
|                      | `SCC`                            | Memantau konsumsi kalori                              | String        | Pilihan: `yes`, `no`                                                           |
|                      | `FAF`                            | Frekuensi aktivitas fisik per minggu                  | Float         | Antara 0 - 3 kali                                                              |
|                      | `TUE`                            | Waktu penggunaan teknologi (komputer/gadget) per hari | Float         | Antara 0 - 2 jam                                                               |
|                      | `CALC`                           | Konsumsi alkohol                                      | String        | Pilihan: `no`, `Sometimes`, `Frequently`, `Always`                             |
|                      | `MTRANS`                         | Moda transportasi                                     | String        | Pilihan: `Automobile`, `Bike`, `Motorbike`, `Public_Transportation`, `Walking` |
| **Target**           | `NObeyesdad`                     | Tingkat obesitas                                      | String        | 7 kategori (Contoh: `Insufficient_Weight`, `Normal_Weight`, dst.)              |

## Alur Kerja

Implementasi mencakup langkah-langkah berikut:

### 1. Eksplorasi Data (EDA)
- Analisis statistik deskriptif
- Visualisasi distribusi data
- Analisis korelasi antar fitur
- Pemeriksaan nilai yang hilang

### 2. Preprocessing Data
- Encoding fitur kategorikal menggunakan LabelEncoder
- Standarisasi fitur numerik menggunakan StandardScaler
- Pembagian data menjadi set pelatihan dan pengujian

### 3. Pemodelan
- Arsitektur: Deep Neural Network dengan 4 layer (128, 64, 32, dan output)
- Dropout layers untuk regularisasi
- Aktivasi ReLU pada hidden layers dan Softmax pada output layer
- Early stopping untuk mencegah overfitting

### 4. Evaluasi Model
- Metrik: Accuracy, Precision, Recall, F1-Score
- Visualisasi: Confusion Matrix, Learning Curves

### 5. Inferensi
- Implementasi prediksi untuk data baru
- Menampilkan probabilitas setiap kelas

### 6. Deployment
- Konversi model ke format ONNX untuk deployment

## Cara Menggunakan

1. **Persiapan Lingkungan**:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn tensorflow tf2onnx
   ```

2. **Menjalankan Kode**:
   - Jalankan notebook `Klasifikasi_Tingkat_Obesitas.ipynb` 
   - Atau jalankan script Python:
     ```bash
     python obesitas_classification.py
     ```

3. **Prediksi dengan Data Baru**:
   ```python
   # Contoh kode untuk prediksi
   new_data = pd.DataFrame([{
       'Age': 22,
       'Gender': le_dict['Gender'].transform(['Male'])[0],
       'Height': 1.75,
       'Weight': 85,
       # Tambahkan atribut lainnya
   }])
   
   # Preprocessing
   new_data[fitur_numerik] = scaler.transform(new_data[fitur_numerik])
   
   # Prediksi
   prediction = model.predict(new_data)
   predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])
   ```

## Struktur File

```
.
├── obesity-model.ipynb                 # Jupyter notebook
├── saved_model_obesity/                # Model TensorFlow tersimpan
└── README.md                           # Dokumentasi
```

## Hasil dan Performa

Model mencapai akurasi sekitar 95% pada data pengujian, dengan performa yang baik pada semua kelas obesitas. Detail lengkap performa dapat dilihat pada laporan klasifikasi dan confusion matrix dalam kode.

## Referensi

Mendoza-Palechor, F., & de la Hoz-Manotas, A. (2019). Dataset for estimation of obesity levels based on eating habits and physical condition in individuals from Colombia, Peru and Mexico. Data in Brief, 25, 104344. https://doi.org/10.1016/j.dib.2019.104344

## Catatan

- Model ini dimaksudkan untuk tujuan edukasi dan penelitian, bukan untuk diagnosis medis.
- Penggunaan model dalam konteks medis harus selalu melibatkan konsultasi dengan profesional kesehatan.
