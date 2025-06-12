# Prediksi Serangan Jantung (Heart Attack Prediction)

Model ini dikembangkan untuk memprediksi kemungkinan serangan jantung berdasarkan beberapa fitur kesehatan dan gaya hidup. Model ini telah melalui proses *preprocessing* data yang komprehensif, pelatihan menggunakan Neural Network (TensorFlow/Keras), dan diekspor ke format ONNX untuk inferensi yang lebih efisien di lingkungan produksi.

---

### Input Data yang Dibutuhkan

Model membutuhkan data dalam format spesifik. Berikut adalah fitur-fitur yang diharapkan beserta tipe datanya:

| Fitur                            | Tipe Data Asli       | Contoh                                   | Keterangan                                              | Cara Input                                         |
| :------------------------------- | :------------------- | :--------------------------------------- | :------------------------------------------------------ | :------------------------------------------------- |
| `age`                            | Numerik (Integer)    | `55`                                     | Usia pasien.                                            | Input angka                                        |
| `gender`                         | Kategorikal (String) | `Male`, `Female`                         | Jenis kelamin pasien.                                   | Dropdown: `Male`, `Female`                         |
| `hypertension`                   | Numerik (Binary)     | `0`, `1`                                 | Riwayat hipertensi (0: Tidak, 1: Ya).                   | Checkbox / Toggle                                  |
| `diabetes`                       | Numerik (Binary)     | `0`, `1`                                 | Riwayat diabetes (0: Tidak, 1: Ya).                     | Checkbox / Toggle                                  |
| `obesity`                        | Numerik (Binary)     | `0`, `1`                                 | Riwayat obesitas (0: Tidak, 1: Ya).                     | Checkbox / Toggle                                  |
| `waist_circumference`            | Numerik (Float)      | `90.5`                                   | Lingkar pinggang (dalam cm).                            | Input angka desimal                                |
| `smoking_status`                 | Kategorikal (String) | 'Current', 'Never', 'Past' | Status merokok pasien.                                  | Dropdown: 'Current', 'Never', 'Past' |
| `alcohol_consumption`            | Kategorikal (String) | 'High', 'Moderate'             | Tingkat konsumsi alkohol pasien.                        | Dropdown: 'High', 'Moderate'             |
| `triglycerides`                  | Numerik (Float)      | `150.2`                                  | Tingkat trigliserida dalam darah (mg/dL).               | Input angka desimal                                |
| `previous_heart_disease`         | Numerik (Binary)     | `0`, `1`                                 | Riwayat penyakit jantung sebelumnya (0: Tidak, 1: Ya).  | Checkbox / Toggle                                  |
| `medication_usage`               | Numerik (Binary)     | `0`, `1`                                 | Penggunaan obat-obatan (0: Tidak, 1: Ya).               | Checkbox / Toggle                                  |
| `participated_in_free_screening` | Numerik (Binary)     | `0`, `1`                                 | Partisipasi dalam *screening* gratis (0: Tidak, 1: Ya). | Checkbox / Toggle                                  |


---

### Output Prediksi

*Backend* akan mengembalikan hasil prediksi yang bisa berupa:
* **Probabilitas:** Angka antara 0 dan 1 untuk setiap kelas (misalnya, probabilitas tidak serangan jantung, probabilitas serangan jantung).
* **Label Prediksi:** `Yes` (kemungkinan serangan jantung tinggi) atau `No` (kemungkinan serangan jantung rendah).

Disarankan untuk menampilkan probabilitas dan label prediksi untuk memberikan informasi yang lebih lengkap kepada pengguna.

---
