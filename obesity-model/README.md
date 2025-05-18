# Model Klasifikasi Tingkat Obesitas

Repository ini berisi implementasi model klasifikasi deep learning untuk memprediksi tingkat obesitas berdasarkan atribut fisik dan kebiasaan hidup seseorang.

## Deskripsi Dataset

Dataset yang digunakan berisi data tentang kebiasaan hidup dan kondisi fisik individu yang dikaitkan dengan 7 tingkat obesitas berbeda. Data diambil dari [Dataset Obesity Levels](https://www.kaggle.com/datasets/fatemehmehrparvar/obesity-levels).

### Atribut

Dataset mencakup atribut berikut:

1. **Fitur Demografis dan Fisik**:
   - `Age`: Usia dalam tahun
   - `Gender`: Jenis kelamin (Male/Female)
   - `Height`: Tinggi badan dalam meter
   - `Weight`: Berat badan dalam kilogram
   - `family_history_with_overweight`: Riwayat keluarga dengan kelebihan berat badan (yes/no)

2. **Kebiasaan Makan**:
   - `FAVC`: Konsumsi makanan tinggi kalori (yes/no)
   - `FCVC`: Frekuensi konsumsi sayuran
   - `NCP`: Jumlah makanan utama per hari
   - `CAEC`: Konsumsi makanan antara waktu makan (Sometimes/Frequently/Always/no)
   - `CH2O`: Konsumsi air per hari
   - `CALC`: Konsumsi alkohol (Sometimes/Frequently/Always/no)

3. **Aktivitas Fisik dan Gaya Hidup**:
   - `SCC`: Menghitung kalori yang dikonsumsi (yes/no)
   - `FAF`: Frekuensi aktivitas fisik
   - `TUE`: Waktu penggunaan perangkat teknologi
   - `SMOKE`: Status merokok (yes/no)
   - `MTRANS`: Transportasi utama yang digunakan

4. **Target**:
   - `NObeyesdad`: Tingkat obesitas (7 kelas)

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
├── obesity-model.py          # Script Python utama
├── obesity-model.ipynb  # Jupyter notebook
├── saved_model_obesity/                # Model TensorFlow tersimpan
├── model_obesity.onnx                  # Model dalam format ONNX
└── README.md                           # Dokumentasi
```

## Hasil dan Performa

Model mencapai akurasi sekitar 95% pada data pengujian, dengan performa yang baik pada semua kelas obesitas. Detail lengkap performa dapat dilihat pada laporan klasifikasi dan confusion matrix dalam kode.

## Referensi

Mendoza-Palechor, F., & de la Hoz-Manotas, A. (2019). Dataset for estimation of obesity levels based on eating habits and physical condition in individuals from Colombia, Peru and Mexico. Data in Brief, 25, 104344. https://doi.org/10.1016/j.dib.2019.104344

## Catatan

- Model ini dimaksudkan untuk tujuan edukasi dan penelitian, bukan untuk diagnosis medis.
- Penggunaan model dalam konteks medis harus selalu melibatkan konsultasi dengan profesional kesehatan.