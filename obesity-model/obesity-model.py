#!/usr/bin/env python
# coding: utf-8

# Import Library

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

get_ipython().system('pip install tf2onnx')
import tf2onnx


# Load Data

# In[2]:


# Memuat data obesitas dari file CSV
df = pd.read_csv('/kaggle/input/obesity-levels/ObesityDataSet_raw_and_data_sinthetic.csv')

# Menampilkan beberapa baris awal dataset
print("Data awal:")
df.head()


# Exploratory Data Analysis (EDA)

# In[3]:


# Deskripsi statistik data numerik
print("\nStatistik deskriptif:")
df.describe()


# In[4]:


# Informasi tentang tipe data dan nilai non-null
print("\nInformasi dataset:")
df.info()


# In[5]:


# Menampilkan ukuran dataset (jumlah baris dan kolom)
print(f"\nUkuran dataset: {df.shape[0]} baris x {df.shape[1]} kolom")


# In[6]:


# Memeriksa nilai yang hilang
nilai_null = df.isnull().sum()
print("\nJumlah nilai yang hilang pada setiap kolom:")
print(nilai_null)

if nilai_null.sum() == 0:
    print("\nDataset tidak memiliki nilai yang hilang!")


# In[7]:


# Visualisasi distribusi kelas target (tingkat obesitas)
plt.figure(figsize=(12, 6))
distribusi_kelas = df['NObeyesdad'].value_counts()
distribusi_kelas.plot(kind='bar')
plt.title('Distribusi Tingkat Obesitas')
plt.ylabel('Jumlah')
plt.xlabel('Tingkat Obesitas')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[8]:


print("\nDistribusi kelas target:")
print(distribusi_kelas)


# In[9]:


# Visualisasi korelasi antar fitur numerik
plt.figure(figsize=(12, 10))
korelasi = df.corr(numeric_only=True)
sns.heatmap(korelasi, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Korelasi antar Fitur Numerik')
plt.tight_layout()
plt.show()


# In[10]:


# Distribusi fitur numerik
fitur_numerik = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

plt.figure(figsize=(15, 12))
for i, kolom in enumerate(fitur_numerik):
    plt.subplot(3, 3, i+1)
    sns.histplot(df[kolom], kde=True)
    plt.title(f'Distribusi {kolom}')
plt.tight_layout()
plt.show()


# In[11]:


# Hubungan antara berat dan tinggi berdasarkan kelas obesitas
plt.figure(figsize=(10, 8))
sns.scatterplot(x='Height', y='Weight', hue='NObeyesdad', data=df, palette='viridis')
plt.title('Hubungan Tinggi dan Berat Badan berdasarkan Tingkat Obesitas')
plt.tight_layout()
plt.show()


# Preprocessing

# In[12]:


# === ENCODE FITUR KATEGORIKAL ===
# Kolom kategorikal yang perlu dienkode menjadi numerik
fitur_kategorikal = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 
                  'SMOKE', 'SCC', 'CALC', 'MTRANS']

print("\nFitur kategorikal yang akan dienkode:")
for kolom in fitur_kategorikal:
    print(f"{kolom}: {df[kolom].unique()}")

# Encode setiap kolom kategorikal ke angka menggunakan LabelEncoder
# Menyimpan encoder untuk digunakan pada data baru nanti
le_dict = {}
for kolom in fitur_kategorikal:
    le = LabelEncoder()
    df[kolom] = le.fit_transform(df[kolom])
    le_dict[kolom] = le
    print(f"Encoding {kolom}: {list(le.classes_)} -> {list(range(len(le.classes_)))}")

# Encode label/target (kelas obesitas)
label_encoder = LabelEncoder()
df['NObeyesdad'] = label_encoder.fit_transform(df['NObeyesdad'])
print("\nKelas target setelah encoding:")
for i, kelas in enumerate(label_encoder.classes_):
    print(f"{kelas} -> {i}")


# In[13]:


# === NORMALISASI FITUR NUMERIK ===
# Standarisasi fitur-fitur numerik untuk meningkatkan kinerja model
print("\nMelakukan standarisasi fitur numerik...")
scaler = StandardScaler()
df[fitur_numerik] = scaler.fit_transform(df[fitur_numerik])

print("\nData setelah preprocessing:")
print(df.head())


# In[14]:


# === PISAH FITUR & LABEL ===
# Memisahkan fitur (X) dan target (y)
X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']


# In[15]:


# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nPembagian data: {X_train.shape[0]} data latih, {X_test.shape[0]} data uji")


# Pemodelan dengan Neural Network

# In[16]:


# Mendapatkan jumlah kelas output
num_classes = len(y.unique())
print(f"Jumlah kelas target: {num_classes}")


# In[17]:


# Membuat arsitektur model neural network
print("\nMembuat arsitektur model...")
model = Sequential([
    tf.keras.Input(shape=(X_train.shape[1],)),
    Dense(128, activation='relu'),  # Menambah kompleksitas model
    Dropout(0.3),  # Regularisasi untuk menghindari overfitting
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')  # Output layer untuk klasifikasi multi-kelas
])


# In[18]:


# Menampilkan ringkasan model
model.summary()


# In[19]:


# Compile model dengan optimizer Adam dan loss function untuk klasifikasi multi-kelas
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# In[20]:


# Implementasi early stopping untuk menghindari overfitting
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)


# In[21]:


# Latih model
print("\nMelatih model...")
history = model.fit(
    X_train, y_train,
    epochs=100,  # Maksimum epoch
    batch_size=32,
    validation_split=0.2,  # 20% dari data training digunakan sebagai validasi
    callbacks=[early_stop],
    verbose=1
)


# Evaluasi Model

# In[22]:


# Evaluasi model pada data test
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f'Akurasi pada data uji: {test_acc:.4f}')
print(f'Loss pada data uji: {test_loss:.4f}')


# In[23]:


# Prediksi pada data test
y_pred_proba = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_proba, axis=1)

# Akurasi menggunakan sklearn
accuracy = accuracy_score(y_test, y_pred_classes)
print(f'Akurasi (sklearn): {accuracy:.4f}')


# In[24]:


# Confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[25]:


# Classification report
print("\nLaporan klasifikasi:")
print(classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_))


# In[26]:


# Visualisasi metrik training
plt.figure(figsize=(12, 5))

# Plot akurasi
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Akurasi Model')
plt.xlabel('Epoch')
plt.ylabel('Akurasi')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Model')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


# Inferensi dengan Data Baru

# In[27]:


# Contoh data baru
new_data = pd.DataFrame([{
    'Age': 22,
    'Gender': le_dict['Gender'].transform(['Male'])[0],
    'Height': 1.75,
    'Weight': 85,
    'family_history_with_overweight': le_dict['family_history_with_overweight'].transform(['yes'])[0],
    'FAVC': le_dict['FAVC'].transform(['yes'])[0],
    'FCVC': 2.0,
    'NCP': 3.0,
    'CAEC': le_dict['CAEC'].transform(['Sometimes'])[0],
    'SMOKE': le_dict['SMOKE'].transform(['no'])[0],
    'CH2O': 2.0,
    'SCC': le_dict['SCC'].transform(['no'])[0],
    'FAF': 1.0,
    'TUE': 1.0,
    'CALC': le_dict['CALC'].transform(['Sometimes'])[0],
    'MTRANS': le_dict['MTRANS'].transform(['Public_Transportation'])[0]
}])

print("Data inferensi sebelum preprocessing:")
print(new_data)


# In[28]:


# Normalisasi fitur numerik
new_data[fitur_numerik] = scaler.transform(new_data[fitur_numerik])

print("\nData inferensi setelah preprocessing:")
print(new_data)


# In[29]:


# Prediksi
prediction_proba = model.predict(new_data)
predicted_class_idx = np.argmax(prediction_proba[0])
predicted_class = label_encoder.inverse_transform([predicted_class_idx])[0]

print(f"\nPrediksi kelas obesitas: {predicted_class}")
print(f"\nProbabilitas per kelas: {prediction_proba[0]}")
for i, kelas in enumerate(label_encoder.classes_):
    print(f"{kelas}: {prediction_proba[0][i]:.4f}")


# Simpan Model

# In[30]:


# Simpan model dalam format TensorFlow SavedModel
model_dir = "saved_model_obesity"
tf.saved_model.save(model, model_dir)
print(f"Model TensorFlow telah disimpan di: {model_dir}")


# In[31]:


# Konversi model ke format ONNX
get_ipython().system('python -m tf2onnx.convert --saved-model {model_dir} --output model_obesity.onnx')
print("\nProses selesai!")

