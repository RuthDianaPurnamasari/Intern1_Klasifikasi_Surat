# Sistem Klasifikasi Surat Masuk & Keluar (Internship Project)

Project ini bertujuan untuk mengklasifikasikan arsip surat secara otomatis menggunakan perbandingan algoritma **Machine Learning (SVM)** dan **Deep Learning (LSTM, Bi-LSTM, CNN)**.

## 📂 Struktur Folder
Agar lebih terstruktur, kode dibagi menjadi modul terpisah:

* `data/`: Menyimpan dataset mentah (2023-2025) dan hasil preprocessing.
* `models/`: Menyimpan model AI yang sudah dilatih (`.pkl` untuk SVM/Tokenizer & `.h5` untuk Deep Learning).
* `src/`: Berisi *source code* modular (preprocessing, training, evaluasi).
* `app.py`: Dashboard utama berbasis Streamlit.

## 🚀 Cara Menjalankan Program (Urut)

Untuk mereproduksi hasil penelitian tanpa menjalankan skrip satu per satu, gunakan perintah berikut di terminal:

### 1. Persiapan Environment
Pastikan library terinstal:
```bash
pip install -r requirements.txt

2. Pipeline Otomatis (Preprocessing & Training)
Jalankan satu perintah ini untuk melakukan pembersihan data (preprocessing) sekaligus melatih model (SVM & Deep Learning):

python run_pipeline.py
Proses yang terjadi:

Menjalankan src/preprocessing.py -> Output: data/data_clean.csv

Melatih Model SVM & Deep Learning -> Output: File model tersimpan di folder models/

3. Evaluasi & Analisis Grafik
Untuk menampilkan tabel perbandingan (Precision, Recall, F1-Score) dan grafik analisis:

python run_analysis.py
4. Menjalankan Aplikasi (Dashboard)
Untuk mencoba klasifikasi surat menggunakan antarmuka visual:

streamlit run app.py
🛠️ Library yang Digunakan
Python 3.10+
Pandas, NumPy: Pengolahan Data
Scikit-Learn: Algoritma SVM & Evaluasi
TensorFlow/Keras: Algoritma Deep Learning (LSTM, Bi-LSTM, CNN)
Sastrawi: Stemming Bahasa Indonesia
Streamlit: Interface / GUI