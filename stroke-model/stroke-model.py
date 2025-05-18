#!/usr/bin/env python
# coding: utf-8

# Import Library

# In[358]:


# Import Library
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Preprocessing dan model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Deep learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# Over-sampling SMOTE
from imblearn.over_sampling import SMOTE

# Konversi model ke ONNX
import tf2onnx


# Load Data

# In[359]:


# Memuat dataset stroke dari file CSV
df = pd.read_csv('/kaggle/input/stroke-prediction-dataset/healthcare-dataset-stroke-data.csv')

# Menampilkan beberapa baris awal dataset
print("Data awal:")
df.head()


# Exploratory Data Analysis (EDA)

# In[360]:


# Menampilkan ukuran dataset (jumlah baris dan kolom)
print(f"\nUkuran dataset: {df.shape[0]} baris x {df.shape[1]} kolom")


# In[361]:


# Informasi tentang tipe data dan nilai non-null
print("\nInformasi dataset:")
df.info()


# In[362]:


# Memeriksa nilai yang hilang
nilai_null = df.isnull().sum()
print("\nJumlah nilai yang hilang pada setiap kolom:")
print(nilai_null)


# In[363]:


# Distribusi target (stroke/tidak stroke)
print("\nDistribusi kelas target:")
distribusi_stroke = df['stroke'].value_counts()
print(distribusi_stroke)
print(f"Persentase kasus stroke: {distribusi_stroke[1]/len(df)*100:.2f}%")


# In[364]:


# Visualisasi distribusi kelas target
plt.figure(figsize=(10, 6))
distribusi_stroke.plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Distribusi Kasus Stroke')
plt.xlabel('Stroke')
plt.ylabel('Jumlah')
plt.xticks([0, 1], ['Tidak Stroke (0)', 'Stroke (1)'])
plt.grid(axis='y', alpha=0.3)
plt.show()


# In[365]:


# Analisis statistik deskriptif
print("\nStatistik deskriptif:")
df.describe()


# In[366]:


# Visualisasi korelasi antar fitur numerik
numeric_df = df.select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=(12, 10))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Korelasi antar Fitur Numerik')
plt.tight_layout()
plt.show()


# In[367]:


# Distribusi fitur numerik
fitur_numerik = ['age', 'avg_glucose_level', 'bmi']
plt.figure(figsize=(15, 5))
for i, kolom in enumerate(fitur_numerik):
    plt.subplot(1, 3, i+1)
    sns.histplot(df[kolom], kde=True)
    plt.title(f'Distribusi {kolom}')
plt.tight_layout()
plt.show()


# In[368]:


# Distribusi fitur numerik berdasarkan target
plt.figure(figsize=(18, 6))
for i, kolom in enumerate(fitur_numerik):
    plt.subplot(1, 3, i+1)
    sns.boxplot(x='stroke', y=kolom, data=df)
    plt.title(f'{kolom} berdasarkan Stroke')
    plt.xlabel('Stroke')
plt.tight_layout()
plt.show()


# In[369]:


# Analisis fitur kategorikal
fitur_kategorikal = ['gender', 'hypertension', 'heart_disease',
                     'ever_married', 'work_type', 'Residence_type',
                     'smoking_status']

# Visualisasi distribusi fitur kategorikal berdasarkan target
plt.figure(figsize=(20, 15))
for i, kolom in enumerate(fitur_kategorikal):
    ax = plt.subplot(3, 3, i+1)  # Buat axes eksplisit
    pd.crosstab(df[kolom], df['stroke']).plot(kind='bar', stacked=True, ax=ax)
    ax.set_title(f'Distribusi {kolom} berdasarkan Stroke')
    ax.set_xlabel(kolom)
    ax.set_ylabel('Jumlah')
    ax.legend(['Tidak Stroke', 'Stroke'])

plt.tight_layout()
plt.show()


# Preprocessing

# In[370]:


# Menghapus kolom ID yang tidak relevan
print("\nMenghapus kolom ID...")
df = df.drop('id', axis=1)


# In[371]:


# Menangani nilai yang hilang
print("\nMenangani nilai yang hilang...")
print(f"Jumlah nilai BMI yang hilang: {df['bmi'].isnull().sum()}")
print(f"Median BMI: {df['bmi'].median():.2f}")


# In[372]:


# Mengisi nilai BMI yang hilang dengan median
df.fillna({'bmi': df['bmi'].median()}, inplace=True)


# In[373]:


# Verifikasi tidak ada lagi nilai yang hilang
print("\nNilai yang hilang setelah diisi:")
print(df.isnull().sum())


# In[374]:


# Encoding fitur kategorikal
print("\nEncoding fitur kategorikal...")
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
le_dict = {}


# In[375]:


# Simpan nilai unik untuk setiap kolom kategorikal
for kolom in categorical_cols:
    print(f"{kolom}: {df[kolom].unique()}")
    le = LabelEncoder()
    df[kolom] = le.fit_transform(df[kolom])
    le_dict[kolom] = le
    print(f"Encoding {kolom}: {list(le.classes_)} -> {list(range(len(le.classes_)))}")


# In[376]:


#Normalisasi fitur numerik
print("\nMelakukan standarisasi fitur numerik...")
numerical_cols = ['age', 'avg_glucose_level', 'bmi']
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])


# In[377]:


print("\nData setelah preprocessing:")
print(df.head())


# In[378]:


# Memisahkan fitur dan target
X = df.drop('stroke', axis=1)
y = df['stroke']


# In[379]:


# Membagi data menjadi set pelatihan dan pengujian
# Stratify digunakan untuk mempertahankan proporsi kelas target
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
print(f"\nPembagian data: {X_train.shape[0]} data latih, {X_test.shape[0]} data uji")


# In[380]:


# Menggunakan SMOTE untuk mengatasi ketidakseimbangan kelas
print("\nMenerapkan SMOTE untuk mengatasi ketidakseimbangan kelas...")
print(f"Distribusi kelas sebelum SMOTE: {y_train.value_counts()}")

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"Distribusi kelas setelah SMOTE: {pd.Series(y_train_resampled).value_counts()}")


# In[381]:


# One-hot encode target
y_train_categorical = to_categorical(y_train_resampled, num_classes=2)
y_test_categorical = to_categorical(y_test, num_classes=2)


# Pemodelan dengan Neural Network

# In[382]:


model = Sequential([
    Dense(16, activation='relu', input_dim=X_train_resampled.shape[1]),
    Dense(8, activation='relu'),
    Dense(2, activation='softmax')
])


model.summary()


# In[383]:


# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[384]:


# Melatih model
print("\nMelatih model...")
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("\nMelatih model...")
history = model.fit(
    X_train_resampled,
    y_train_categorical, 
    validation_split=0.2,
    epochs=700,
    batch_size=32,
    callbacks=[early_stop],
    verbose=2
)


# Evaluasi Model

# In[385]:


# Evaluasi model pada data test
loss, acc = model.evaluate(X_test, y_test_categorical, verbose=0)
print(f"\nAkurasi Uji: {acc:.4f}")
print(f"Loss Uji: {loss:.4f}")


# In[386]:


# Prediksi pada data test
y_pred_proba = model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)


# In[387]:


# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Tidak Stroke', 'Stroke'], yticklabels=['Tidak Stroke', 'Stroke'])
plt.title("Confusion Matrix")
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.tight_layout()
plt.show()


# In[388]:


# Classification report
print("\nLaporan klasifikasi:")
print(classification_report(y_test, y_pred, target_names=["Tidak Stroke", "Stroke"]))


# In[389]:


#  Ambil probabilitas untuk kelas 1 (stroke)
y_pred_proba = model.predict(X_test)[:, 1]  # Ambil kolom ke-1 (kelas stroke)

# Ubah y_test_categorical ke bentuk label asli (0 atau 1)
y_test_label = np.argmax(y_test_categorical, axis=1)

# Hitung ROC-AUC Score
auc_score = roc_auc_score(y_test_label, y_pred_proba)
print(f"\nROC-AUC Score: {auc_score:.4f}")

# Hitung nilai fpr dan tpr untuk plot ROC
fpr, tpr, _ = roc_curve(y_test_label, y_pred_proba)

# Plot ROC Curve
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'AUC = {auc_score:.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# Inferensi dengan Data Baru

# In[390]:


def predict_stroke(age, gender, hypertension, heart_disease, ever_married,
                   work_type, residence_type, avg_glucose_level, bmi, smoking_status):
    new_data = pd.DataFrame([{
        'gender': le_dict['gender'].transform([gender])[0],
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'ever_married': le_dict['ever_married'].transform([ever_married])[0],
        'work_type': le_dict['work_type'].transform([work_type])[0],
        'Residence_type': le_dict['Residence_type'].transform([residence_type])[0],
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'smoking_status': le_dict['smoking_status'].transform([smoking_status])[0]
    }])
    
    new_data[numerical_cols] = scaler.transform(new_data[numerical_cols])
    prob = model.predict(new_data)[0][0]
    
    return {
        'predicted_class': int(prob >= 0.5),
        'prob_stroke': prob,
        'prob_tidak_stroke': 1 - prob
    }


# In[391]:


result = predict_stroke(
    age=65,
    gender='Male',
    hypertension=1,
    heart_disease=1,
    ever_married='Yes',
    work_type='Private',
    residence_type='Urban',
    avg_glucose_level=200,
    bmi=28,
    smoking_status='formerly smoked'
)

print("\nHasil Prediksi Contoh:")
print(f"Prediksi: {'Stroke' if result['predicted_class'] == 1 else 'Tidak Stroke'}")
print(f"Probabilitas Stroke: {result['prob_stroke']:.4f}")
print(f"Probabilitas Tidak Stroke: {result['prob_tidak_stroke']:.4f}")


# Simpan Model

# In[392]:


# Simpan model dalam format TensorFlow SavedModel
model_dir = "saved_model_stroke"
tf.saved_model.save(model, model_dir)
print(f"Model TensorFlow telah disimpan di: {model_dir}")


# In[393]:


# Konversi model ke format ONNX
get_ipython().system('python -m tf2onnx.convert --saved-model {model_dir} --output model_obesity.onnx')
print("\nProses selesai!")

