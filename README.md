# Sistem Klasifikasi Surat Masuk & Keluar (Internship Project)

Project ini bertujuan untuk mengklasifikasikan arsip surat secara otomatis menggunakan perbandingan algoritma **Machine Learning (SVM)** dan **Deep Learning (LSTM)**.

## 📂 Struktur Folder
* `data/`: Menyimpan dataset mentah (2023-2025) dan hasil preprocessing.
* `models/`: Menyimpan model AI yang sudah dilatih (.pkl & .h5).
* `app.py`: Dashboard utama berbasis Streamlit.

## 🚀 Cara Menjalankan Program (Urut)

Untuk mereproduksi hasil penelitian, jalankan perintah berikut di terminal secara berurutan:

**1. Data Preprocessing (Pembersihan Data)**
```bash
python 1_data_processing.py
- Output: data/data_clean.csv
python 2_train_svm.py
- Output: Akurasi Training SVM & File Model
python 3_train_lstm.py
- Output: Akurasi Training LSTM & File Model
python 4_evaluation.py
- Output: Tabel Perbandingan (Precision, Recall, F1-Score) & Grafik png
streamlit run app.py

Library yang Digunakan
Python 3.10+
Pandas, NumPy (Pengolahan Data)
Scikit-Learn (SVM)
TensorFlow/Keras (LSTM)
Streamlit (Interface)
Sastrawi (Stemming Bahasa Indonesia)