# src/feature_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from src.config import *
import os

def analyze_feature_correlation():
    print("\n🔍 [ANALYSIS] Menghitung Korelasi Fitur (Kata Kunci)...")
    
    # 1. Cek File Data
    if not os.path.exists(DATA_CLEAN_PATH):
        print("File data_clean_dl.csv tidak ditemukan! jalankan run_pipeline.py terlebih dahulu.")
        return
    
    # 2. Load Data Bersih
    df = pd.read_csv(DATA_CLEAN_PATH)
    df = df.dropna(subset=['Teks_Input_Gabungan', 'Kategori_Target'])
    
    # 3. TF-IDF
    print("Menghitung bobot kata(TF-IDF) dan Chi-Square...")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,1))
    X_tfidf = tfidf.fit_transform(df['Teks_Input_Gabungan'].astype(str))
    y = df['Kategori_Target']
    
    # 4. Hitung Chi-Square (Korelasi Kata vs Label)
    print("Menghitung skor Chi-Square...")
    chi2score = chi2(X_tfidf, y)[0]
    
    # 5. Visualisasi Top 20 Kata Paling Berpengaruh (Global)
    plt.figure(figsize=(12, 8))
    wscores = zip(tfidf.get_feature_names_out(), chi2score)
    wchi2 = sorted(wscores, key=lambda x: x[1])
    topchi2 = list(zip(*wchi2[-20:]))
    
    x = range(len(topchi2[1]))
    labels = topchi2[0]
    
    plt.barh(x, topchi2[1], align='center', alpha=0.8, color='#1565C0')
    plt.yticks(x, labels)
    plt.xlabel('Nilai Chi-Square (Kekuatan Korelasi)')
    plt.title('20 Kata Paling Kuat Membedakan Jenis Surat')
    plt.tight_layout()
    plt.savefig('grafik_korelasi_fitur.png') # PENTING: Disimpan agar bisa dibaca app.py
    print("Grafik korelasi tersimpan: grafik_korelasi_fitur.png")
    
    # 6. Ekstrak Top Keywords per Kategori
    print("\nKATA KUNCI UTAMA PER KATEGORI:")
    
    unique_labels = sorted(df['Kategori_Target'].unique())
    correlation_data = []
    
    for label in unique_labels:
        features_chi2 = chi2(X_tfidf, df['Kategori_Target'] == label)
        indices = np.argsort(features_chi2[0])
        feature_names = np.array(tfidf.get_feature_names_out())[indices]
        
        # Ambil 10 kata teratas
        top_words = [v for v in feature_names[-10:]] 
        
        correlation_data.append({
            'Kategori': label,
            'Kata Kunci': ", ".join(top_words)
        })
        
    # Simpan ke CSV untuk Dashboard
    pd.DataFrame(correlation_data).to_csv('data/feature_correlation.csv', index=False)
    print("Data korelasi tersimpan: data/feature_correlation.csv")
    print("🔍 [ANALYSIS] Selesai.\n")

if __name__ == "__main__":
    analyze_feature_correlation()



# # src/feature_analysis.py
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_selection import chi2
# from src.config import *
# import os

# def analyze_feature_correlation():
#     print("\n🔍 [ANALYSIS] Menghitung Korelasi Fitur (Kata Kunci)...")
    
#     # 1. Cek File Data
#     if not os.path.exists(DATA_CLEAN_PATH):
#         print("File data_clean_dl.csv tidak ditemukan! jalankan run_pipeline.py terlebih dahulu.")
#         return
    
#     # 2. Load Data Bersih
#     df = pd.read_csv(DATA_CLEAN_PATH)
#     df = df.dropna(subset=['Teks_Input_Gabungan', 'Kategori_Target'])
    
#     # 3. TF-IDF untuk Analisis Statistik (Bukan untuk training DL, tapi untuk lihat korelasi fitur Chi-Square)
#     print("Menghitung bobot kata(TF-IDF) dan Chi-Square...")
#     tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,1))
#     X_tfidf = tfidf.fit_transform(df['Teks_Input_Gabungan'].astype(str))
#     y = df['Kategori_Target']
    
#     # 4. Hitung Chi-Square (Korelasi Kata vs Label)
#     print("Menghitung skor Chi-Square...")
#     chi2score = chi2(X_tfidf, y)[0]
    
#     # 5. Visualisasi Top 20 Kata Paling Berpengaruh (Global)
#     plt.figure(figsize=(12, 8))
#     wscores = zip(tfidf.get_feature_names_out(), chi2score)
#     wchi2 = sorted(wscores, key=lambda x: x[1])
#     topchi2 = list(zip(*wchi2[-20:]))
    
#     x = range(len(topchi2[1]))
#     labels = topchi2[0]
    
#     plt.barh(x, topchi2[1], align='center', alpha=0.8, color='#1565C0')
#     plt.yticks(x, labels)
#     plt.xlabel('Nilai Chi-Square (Kekuatan Korelasi)')
#     plt.title('20 Kata Paling Kuat Membedakan Jenis Surat')
#     plt.tight_layout()
#     plt.savefig('grafik_korelasi_fitur.png')
#     print("Grafik korelasi tersimpan: grafik_korelasi_fitur.png")
    
#     # 6. Ekstrak Top Keywords per Kategori
#     print("\nKATA KUNCI UTAMA PER KATEGORI:")
    
#     unique_labels = sorted(df['Kategori_Target'].unique())
#     correlation_data = []
    
#     for label in unique_labels:
#         features_chi2 = chi2(X_tfidf, df['Kategori_Target'] == label)
#         indices = np.argsort(features_chi2[0])
#         feature_names = np.array(tfidf.get_feature_names_out())[indices]
        
#         # Ambil 10 kata teratas
#         top_words = [v for v in feature_names[-10:]] # Ambil dari belakang (score tertinggi)
        
#         # Simpan ke list
#         correlation_data.append({
#             'Kategori': label,
#             'Kata Kunci': ", ".join(top_words)
#         })
        
#         print(f"{label}: {', '.join(top_words)}")
        
#     # Simpan ke CSV untuk Dashboard
#     pd.DataFrame(correlation_data).to_csv('data/feature_correlation.csv', index=False)
#     print("Data korelasi tersimpan: data/feature_correlation.csv")
#     print("🔍 [ANALYSIS] Selesai menghitung korelasi fitur.\n")

# if __name__ == "__main__":
#     analyze_feature_correlation()