# Prediksi Anemia (Anemia Prediction)

Model ini dikembangkan untuk memprediksi status anemia berdasarkan beberapa parameter darah yang relevan. Model telah melalui proses *preprocessing* data yang komprehensif, pelatihan menggunakan Neural Network (TensorFlow/Keras), dan diekspor ke format ONNX untuk inferensi yang efisien di lingkungan produksi.

---

### Input Data yang Dibutuhkan

Model membutuhkan data dalam format spesifik. Berikut adalah fitur-fitur yang diharapkan beserta tipe datanya:

| Fitur     | Tipe Data Asli       | Contoh       | Keterangan                                      | Cara Input              |
| :-------- | :------------------- | :----------- | :---------------------------------------------- | :---------------------- |
| `HGB`     | Numerik (Float)      | `14.5`       | Kadar Hemoglobin (g/dL).                        | Input angka desimal     |
| `HCT`     | Numerik (Float)      | `42.0`       | Hematokrit (%).                                 | Input angka desimal     |
| `RBC`     | Numerik (Float)      | `5.0`        | Jumlah Sel Darah Merah (juta/µL).               | Input angka desimal     |
| `RDW`     | Numerik (Float)      | `11.5`       | Distribusi Lebar Sel Darah Merah (%).           | Input angka desimal     |
| `MCH`     | Numerik (Float)      | `29.0`       | Kadar Hemoglobin Mean Corpuscular (pg).         | Input angka desimal     |
| `MCHC`    | Numerik (Float)      | `33.0`       | Kadar Hemoglobin Mean Corpuscular Konsentrasi (g/dL). | Input angka desimal |
| `MCV`     | Numerik (Float)      | `89.0`       | Volume Sel Darah Merah Mean Corpuscular (fL).   | Input angka desimal     |
| `SD`      | Numerik (Float)      | `40.0`       | Standar Deviasi (terkait RDW atau parameter lain). | Input angka desimal |
| `TSD`     | Numerik (Float)      | `50.0`       | Total Standar Deviasi (terkait distribusi).     | Input angka desimal     |

---

### Output Prediksi

*Backend* akan mengembalikan hasil prediksi yang bisa berupa:
* **Probabilitas:** Angka antara 0 dan 1 untuk setiap kelas (misalnya, probabilitas tidak terkena anemia, probabilitas terkena anemia).
* **Label Prediksi:** `Tidak Terkena Anemia` atau `Terkena Anemia`.

Disarankan untuk menampilkan probabilitas dan label prediksi untuk memberikan informasi yang lebih lengkap kepada pengguna.

---
