# src/clustering.py

import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ============================================================
# 0. Normalisasi Kategori (Masalah Utama Kamu)
# ============================================================

def normalize_category(text):
    """Menghapus angka di depan kategori dan merapikan format."""
    if pd.isna(text):
        return text

    text = str(text)

    # Hapus angka + tanda '-' atau '.'
    text = re.sub(r"^\s*\d+\s*[-.]*\s*", "", text)

    # Hapus spasi berlebihan
    text = re.sub(r"\s+", " ", text).strip()

    return text.title()


# ============================================================
# 1. ELBOW METHOD
# ============================================================

def calculate_elbow(df, max_k=10):
    texts = df["Teks_Input_Gabungan"].astype(str)

    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(texts)

    k_values = list(range(2, max_k + 1))
    inertias = []

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)

    return k_values, inertias


# ============================================================
# 2. K-MEANS & PCA
# ============================================================

def run_kmeans_analysis(df, n_clusters=3):
    df = df.copy()

    # NORMALISASI KATEGORI DI SINI
    df["Tipe_Normalized"] = df["Tipe"].apply(normalize_category)
    df["Dari_Normalized"] = df["Dari/Untuk"].apply(normalize_category)

    texts = df["Teks_Input_Gabungan"].astype(str)

    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(texts)

    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X)

    # PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X.toarray())

    df["x"] = X_pca[:, 0]
    df["y"] = X_pca[:, 1]

    # Extract keywords
    feature_names = np.array(tfidf.get_feature_names_out())
    keywords = []

    for cid in range(n_clusters):
        centroid = kmeans.cluster_centers_[cid]
        idx = centroid.argsort()[-10:][::-1]
        top_words = feature_names[idx]
        keywords.append({"Cluster": cid, "Keywords": ", ".join(top_words)})

    return df, pd.DataFrame(keywords)


# ============================================================
# 3. Cluster Summary
# ============================================================

def analyze_cluster_distribution(df_clustered):
    result = (
        df_clustered.groupby("Cluster")
        .agg({
            "Teks_Input_Gabungan": "count",
            "Tipe_Normalized": lambda x: x.mode()[0],
            "Dari_Normalized": lambda x: x.mode()[0],
        })
        .rename(columns={
            "Teks_Input_Gabungan": "Jumlah Data",
            "Tipe_Normalized": "Tipe Dominan",
            "Dari_Normalized": "Pengirim Dominan"
        })
    )

    return result.reset_index()


# ============================================================
# 4. Breakdown per Kolom
# ============================================================

def get_cluster_breakdown(df_clustered, col_name):
    temp = (
        df_clustered.groupby(["Cluster", col_name])
        .size()
        .reset_index(name="Jumlah")
        .sort_values(["Cluster", "Jumlah"], ascending=[True, False])
    )
    return temp


# ============================================================
# 5. Deteksi Kategori Asli vs Normalisasi (Fitur Tambahan)
# ============================================================

def detect_category_variations(df):
    """
    Menghasilkan tabel:
    - Bentuk Asli
    - Hasil Normalisasi
    - Jumlah
    """
    df_temp = df.copy()
    df_temp["Tipe_Normalized"] = df_temp["Tipe"].apply(normalize_category)

    return (
        df_temp.groupby(["Tipe", "Tipe_Normalized"])
        .size()
        .reset_index(name="Jumlah")
        .sort_values("Tipe_Normalized")
    )



# # src/clustering.py
# import pandas as pd
# import numpy as np
# from sklearn.cluster import KMeans
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import PCA
# from src.config import *

# def get_vectorizer_and_features(df):
#     """Mengubah teks menjadi angka (Vektorisasi) menggunakan TF-IDF"""
#     tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
#     # Pastikan data string dan tidak kosong
#     texts = df['Teks_Input_Gabungan'].fillna("").astype(str).tolist()
#     matrix = tfidf.fit_transform(texts)
#     return matrix, tfidf

# def calculate_elbow(df, max_k=10):
#     """Menghitung inertia untuk grafik Elbow (Mencari K Optimal)"""
#     matrix, _ = get_vectorizer_and_features(df)
#     inertias = []
#     k_range = range(1, max_k + 1)
    
#     print(f"[CLUSTERING] Menghitung Elbow untuk K=1 s.d {max_k}...")
#     for k in k_range:
#         # n_init=10 agar hasil stabil
#         kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
#         kmeans.fit(matrix)
#         inertias.append(kmeans.inertia_)
        
#     return list(k_range), inertias

# def get_top_keywords(tfidf, kmeans, n_clusters, n_terms=10):
#     """Mencari kata kunci (Centroid) untuk setiap cluster"""
#     data = []
#     order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
#     terms = tfidf.get_feature_names_out()
    
#     for i in range(n_clusters):
#         top_terms = [terms[ind] for ind in order_centroids[i, :n_terms]]
#         data.append({
#             'Cluster ID': i,
#             'Kata Kunci Dominan': ", ".join(top_terms)
#         })
#     return pd.DataFrame(data)

# def analyze_cluster_distribution(df):
#     """
#     Fitur Profiling: Membedah isi setiap cluster.
#     Menghitung dominasi Tipe Surat dan Pengirim di setiap kelompok.
#     """
#     stats = []
#     unique_clusters = sorted(df['Cluster'].unique())
    
#     for k in unique_clusters:
#         subset = df[df['Cluster'] == k]
        
#         # 1. Top 3 Tipe Surat
#         top_tipe = subset['Tipe'].value_counts().head(3)
#         top_tipe_str = ", ".join([f"{idx} ({val})" for idx, val in top_tipe.items()])
        
#         # 2. Top 3 Pengirim (Dari/Untuk)
#         top_dari = subset['Dari/Untuk'].value_counts().head(3)
#         top_dari_str = ", ".join([f"{idx} ({val})" for idx, val in top_dari.items()])

#         # 3. Top Jenis (Masuk/Keluar)
#         top_jenis = subset['Jenis_Surat'].value_counts().head(1)
#         top_jenis_str = ", ".join([f"{idx}" for idx, val in top_jenis.items()])
        
#         stats.append({
#             'Cluster ID': k,
#             'Jumlah Data': len(subset),
#             'Jenis Dominan': top_jenis_str,
#             'Tipe Surat Terbanyak': top_tipe_str,
#             'Pengirim Terbanyak': top_dari_str
#         })
        
#     return pd.DataFrame(stats)

# def get_cluster_breakdown(df, col_name):
#     """Helper untuk grafik batang distribusi"""
#     counts = df.groupby(['Cluster', col_name]).size().reset_index(name='Jumlah')
#     # Ambil Top 5 kategori per cluster agar grafik rapi
#     counts = counts.sort_values('Jumlah', ascending=False).groupby('Cluster').head(5)
#     return counts

# def run_kmeans_analysis(df, n_clusters=3):
#     """Pipeline Utama Clustering"""
#     print(f"[CLUSTERING] Menjalankan K-Means dengan K={n_clusters}...")
#     matrix, tfidf = get_vectorizer_and_features(df)
    
#     # 1. K-Means
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
#     clusters = kmeans.fit_predict(matrix)
    
#     # 2. PCA (Proyeksi ke 2D untuk Visualisasi Scatter Plot)
#     pca = PCA(n_components=2, random_state=42)
#     coords = pca.fit_transform(matrix.toarray())
    
#     # 3. Gabung Data
#     df_res = df.copy()
#     df_res['Cluster'] = clusters
#     df_res['x'] = coords[:, 0]
#     df_res['y'] = coords[:, 1]
    
#     # 4. Ekstrak Keywords
#     df_keywords = get_top_keywords(tfidf, kmeans, n_clusters)
    
#     return df_res, df_keywords