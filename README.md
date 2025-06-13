# API Prediksi Penyakit Multi-Model

Proyek ini adalah sebuah REST API yang dibangun menggunakan Flask untuk menyediakan prediksi berbagai macam kondisi kesehatan. API ini melayani beberapa model *machine learning* yang telah dilatih untuk memprediksi penyakit berikut:

  * Anemia
  * Penyakit Kardiovaskular
  * Diabetes
  * Serangan Jantung (*Heart Attack*)
  * Obesitas
  * Stroke

Setiap model dikembangkan secara terpisah dan diintegrasikan ke dalam satu API untuk kemudahan akses dan penggunaan.

-----

## 🚀 Akses API (Live)

Aplikasi ini telah di-deploy dan dapat diakses secara publik. Ini adalah cara yang direkomendasikan untuk menggunakan dan menguji API.

**Base URL:** `https://web-production-9468.up.railway.app/`

-----

## Fitur Utama

  - **Prediksi Multi-Penyakit**: Menyediakan endpoint terpisah untuk 6 kondisi kesehatan yang berbeda.
  - **Arsitektur Modular**: Kode API diorganisir dengan rapi menggunakan Flask Blueprints untuk setiap model.
  - **Pra-pemrosesan Otomatis**: Input dari pengguna secara otomatis di-scaling dan di-encode agar sesuai dengan format yang dibutuhkan oleh model.
  - **Siap Deploy**: Dilengkapi dengan `Procfile` dan `requirements.txt` untuk kemudahan deployment di platform seperti Heroku atau Railway.

-----

## Struktur Proyek

```
.
├── anemia-model/              # Notebook dan model Anemia
├── api/                       # Kode sumber Flask API
│   ├── app.py                 # Entry point aplikasi
│   ├── models/                # Logika untuk memuat & menjalankan model
│   ├── routes/                # Definisi endpoint (Blueprints)
│   └── utils/                 # Fungsi bantuan (preprocessing)
├── cardiovascular-model/      # Notebook dan model Kardiovaskular
├── dataset/                   # Semua file dataset
├── diabetes-model/            # Notebook dan model Diabetes
├── heart-attack-model/        # Notebook dan model Serangan Jantung
├── obesity-model/             # Notebook dan model Obesitas
├── stroke-model/              # Notebook dan model Stroke
├── Procfile                   # Konfigurasi untuk deployment
├── requirements.txt           # Daftar dependensi Python
└── README.md                  # File ini
```

-----

## Teknologi yang Digunakan

  - **Backend**: Python, Flask
  - **Machine Learning**: TensorFlow (Keras), Scikit-learn
  - **Data Manipulation**: Pandas, NumPy
  - **Deployment**: Gunicorn, Railway

-----

## 👨‍💻 Tim Pengembang

Proyek ini dibangun sebagai bagian dari **Tugas Akhir (Capstone Project) DBS Coding Camp 2024** untuk jalur pembelajaran **Machine Learning Engineer**.

Proyek ini dikerjakan oleh tim yang terdiri dari 3 anggota:

  * **[MC222D5Y1146 Moch Rifky Aulia Adikusumah]** - [GitHub Profile](https://github.com/rifkyadiii)

      * Model Prediksi Obesitas [https://github.com/rifkyadiii/final_capstone_model/tree/main/obesity-model]
      * Model Prediksi Stroke [https://github.com/rifkyadiii/final_capstone_model/tree/main/stroke-model]

  * **[MC222D5X1148 Gevira Zahra Shofa]** - [GitHub Profile](https://github.com/gevirazahrashofa)

      * Model Prediksi Diabetes [https://github.com/gevirazahrashofa/proyek-capstone-diabetes]
      * Model Prediksi Penyakit Kardiovaskular [https://github.com/gevirazahrashofa/proyek-capstone-cardiovascular]

  * **[MC222D5Y1149 Ghani Husna Darmawan]** - [GitHub Profile](https://github.com/GhaniHD)

      * Model Prediksi Anemia [https://github.com/GhaniHD/Model_Anemia]
      * Model Prediksi Serangan Jantung [https://github.com/GhaniHD/Model_Heart_Attack]

-----

## 💻 Instalasi (Untuk Development Lokal)

Bagian ini ditujukan untuk developer yang ingin memeriksa kode sumber atau mengembangkannya lebih lanjut.

> **PENTING:**
> Menjalankan aplikasi ini secara lokal **tidak disarankan untuk penggunaan biasa** karena memerlukan penyesuaian pada path file model. Path untuk memuat model (`.keras`, `.pkl`) diatur untuk lingkungan produksi di Railway. Jika dijalankan secara lokal, mungkin perlu mengubah cara pemanggilan path di dalam direktori agar sesuai dengan struktur direktori di komputer.

Jika Anda memahami risiko di atas dan tetap ingin melanjutkan, ikuti langkah-langkah berikut:

1.  **Clone Repositori**

    ```bash
    https://github.com/rifkyadiii/final_capstone_model.git
    cd nama-repositori
    ```

2.  **Buat dan Aktifkan Virtual Environment**

    ```bash
    # Untuk Windows
    python -m venv venv
    .\venv\Scripts\activate

    # Untuk macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependensi**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Jalankan Aplikasi**

    ```bash
    # Menggunakan server development Flask
    python -m api.app
    ```

    Aplikasi akan berjalan di `http://127.0.0.1:5000`. Ingat, fungsionalitas prediksi mungkin gagal tanpa penyesuaian path.

-----

## Penggunaan API

Untuk berinteraksi dengan API yang sudah di-deploy, kirim request `POST` ke endpoint yang sesuai dengan body JSON.

### Endpoint Prediksi Anemia

  * **URL**: `https://web-production-9468.up.railway.app/predict/anemia`
  * **Method**: `POST`
  * **Content-Type**: `application/json`
  * **Body (Contoh)**:
    ```json
    {
        "HGB": 11.0,      
        "HCT": 33.0,      
        "RBC": 4.5,       
        "RDW": 15.0,      
        "MCH": 26.0, 
        "MCHC": 30.0,     
        "MCV": 80.0,      
        "SD": 3.5,        
        "TSD": 40.0
    }
    ```
    * **Response Sukses**:
    ```json
    {
        "prediction": "1",
        "prediction_label_numeric": 1,
        "probability_of_anemia": 0.9999997615814209
    }
    ```

### Endpoint Prediksi Kardiovaskular

  * **URL**: `https://web-production-9468.up.railway.app/predict/cardiovascular`
  * **Method**: `POST`
  * **Content-Type**: `application/json`
  * **Body (Contoh)**:
    ```json
    {
        "age": 55,
        "gender": 1, // (0: Perempuan, 1: Laki-laki)  
        "height": 175,
        "weight": 80,
        "ap_hi": 140,
        "ap_lo": 90,
        "cholesterol": 0, // (1: Kolesterol normal, 2: Kolesterol diatas normal)
        "gluc": 0, // (1: Glukosa normal, 2: Glukosa diatas normal)
        "smoke": 0, // (0: Tidak, 1: Ya) 
        "alco": 0, // (0: Tidak, 1: Ya)
        "active": 1 // (0: Tidak aktif, 1: Aktif)
    }
    ```
    * **Response Sukses**:
    ```json
    {
        "prediction": "High Risk", //probability > 7 = High Risk, probability > 3 = Medium Risk, probability > 7 = Low Risk
        "prediction_label_numeric": 1,
        "probability_of_cardiovascular_disease": 0.85
    }
    ```

### Endpoint Prediksi Diabetes

  * **URL**: `https://web-production-9468.up.railway.app/predict/diabetes`
  * **Method**: `POST`
  * **Content-Type**: `application/json`
  * **Body (Contoh)**:
    ```json
    {
        "gender": "Female", // (Female, Male, Other)
        "age": 45,
        "hypertension": 0, // (0: Tidak, 1: Ya) 
        "heart_disease": 0, // (0: Tidak, 1: Ya) 
        "smoking_history": "never", // (never, No Info, current, former, ever, not current)
        "bmi": 28.5,
        "HbA1c_level": 6.2,
        "blood_glucose_level": 145
    }
    ```
    * **Response Sukses**:
    ```json
    {
        "prediction": "0",
        "prediction_label_numeric": 0,
        "probability_of_diabetes": 0.04034673422574997
    }
    ```

### Endpoint Prediksi Serangan Jantung

  * **URL**: `https://web-production-9468.up.railway.app/predict/heart_attack`
  * **Method**: `POST`
  * **Content-Type**: `application/json`
  * **Body (Contoh)**:
    ```json
    {
        "age": 65,
        "gender": "Male", // (Female, Male)
        "hypertension": 1, // (0: Tidak, 1: Ya)
        "diabetes": 0, // (0: Tidak, 1: Ya)
        "obesity": 1, // (0: Tidak, 1: Ya)
        "waist_circumference": 102.5,
        "smoking_status": "Past", // (Current, Never, Past)
        "alcohol_consumption": "High", // (Hight, Moderate)
        "triglycerides": 180.0,
        "previous_heart_disease": 1, // (0: Tidak, 1: Ya)
        "medication_usage": 0, // (0: Tidak, 1: Ya)
        "participated_in_free_screening": 1 // (0: Tidak, 1: Ya)
    }
    ```
    * **Response Sukses**:
    ```json
    {
        "prediction": "0",
        "prediction_label_numeric": 0,
        "probability_of_heart_attack": 0.44310927391052246
    }
    ```

### Endpoint Prediksi Obesitas

  * **URL**: `https://web-production-9468.up.railway.app/predict/obesity`
  * **Method**: `POST`
  * **Content-Type**: `application/json`
  * **Body (Contoh)**:
    ```json
    {
        "Gender": "Male", // (Female, Male)
        "Age": 30,
        "Height": 1.75,
        "Weight": 85,
        "family_history_with_overweight": "yes", // (yes, no)
        "FAVC": "yes", // (yes, no)
        "FCVC": 2.5, // (Skala 1-3)
        "NCP": 3.0, // (Skala 1-4)
        "CAEC": "Sometimes", // (no, Sometimes, Frequently, Always)
        "SMOKE": "no", // (yes, no)
        "CH2O": 2.0, // (Skala 1-3)
        "SCC": "no", // (yes, no)
        "FAF": 1.5, // (Skala 1-3)
        "TUE": 0.5, // (Skala 0-2)
        "CALC": "Sometimes", // (no, Sometimes)
        "MTRANS": "Public_Transportation" // (Automobile, Bike, Motorbike, Public_Transportation, Walking)
    }
    ```
    * **Response Sukses**:
    ```json
    {
        "prediction": "Overweight_Level_II",
        "prediction_label_numeric": 3,
        "probabilities": 0.55
    }
    ```

### Endpoint Prediksi Stroke

  * **URL**: `https://web-production-9468.up.railway.app/predict/stroke`
  * **Method**: `POST`
  * **Content-Type**: `application/json`
  * **Body (Contoh)**:
    ```json
    {
        "gender": "Female", // (Male, Female)
        "age": 65,
        "hypertension": 1, // (0: Tidak, 1: Ya)
        "heart_disease": 0, // 0: Tidak, 1: Ya
        "ever_married": "Yes", // (Yes, No)
        "work_type": "Private", // (Private, Self-employed, Govt_job, children, Never_worked)
        "Residence_type": "Urban", // (Urban, Rural)
        "avg_glucose_level": 180.50,
        "bmi": 32.1,
        "smoking_status": "formerly smoked" // (never smoked, formerly smoked, smokes, Unknown)
    }
    ```
    * **Response Sukses**:
    ```json
    {
        "prediction": "1",
        "prediction_label_numeric": 1,
        "probability_of_stroke": 0.9158374667167664
    }
    ```

-----

## Model dan Dataset

Setiap model dilatih menggunakan dataset yang spesifik dan disimpan bersama dengan artifak pra-pemrosesannya (scaler, encoder).

| Model                  | Dataset Sumber              | Direktori Model          |
| ---------------------- | --------------------------- | ------------------------ |
| **Anemia** | `anemia_data.xlsx`          | `anemia-model/`          |
| **Kardiovaskular** | `cardiovascular_data.csv`   | `cardiovascular-model/`  |
| **Diabetes** | `diabetes_data.csv`         | `diabetes-model/`        |
| **Serangan Jantung** | `heart_attack_data.csv`     | `heart-attack-model/`    |
| **Obesitas** | `obesity_data.csv`          | `obesity-model/`         |
| **Stroke** | `stroke_data.csv`           | `stroke-model/`          |

Notebook Jupyter (`.ipynb`) yang berisi proses eksplorasi data, pelatihan, dan evaluasi model juga tersedia di dalam setiap direktori model.
