# app.py (Merged - Full)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Import Modul Clustering (harus tersedia di src/)
from src.clustering import (
    calculate_elbow,
    run_kmeans_analysis,
    analyze_cluster_distribution,
    get_cluster_breakdown
)

# ============================================================
# Helper: normalize_category (diambil/adaptasi dari run_pipeline.py)
# ============================================================
def normalize_category(text):
    """
    Normalisasi sederhana untuk kolom kategori/tipe/dari-untuk:
    - Hapus awalan nomor seperti "5 - " atau "03." atau "1."
    - Lowercase
    - Strip whitespace
    - Ganti multiple spaces dengan single space
    """
    if pd.isna(text):
        return ""
    s = str(text)
    # Hapus awalan nomor/format "123 - " atau "12." atau "1)"
    s = re.sub(r"^\s*\d+\s*[-.)]?\s*", "", s)
    # Hilangkan karakter non-alphanumeric kecuali spasi, slash, dan koma
    s = re.sub(r"[^\w\s\/\,\-\.]", " ", s)
    s = s.lower().strip()
    # Ganti beberapa spasi menjadi satu
    s = re.sub(r"\s+", " ", s)
    return s
# ============================================================
# Helper: normalize_category (diambil/adaptasi dari run_pipeline.py)
# ============================================================
def normalize_category(text):
    """
    Normalisasi sederhana untuk kolom kategori/tipe/dari-untuk:
    - Hapus awalan nomor seperti "5 - " atau "03." atau "1."
    - Lowercase
    - Strip whitespace
    - Ganti multiple spaces dengan single space
    """
    if pd.isna(text):
        return ""
    s = str(text)
    # Hapus awalan nomor/format "123 - " atau "12." atau "1)"
    s = re.sub(r"^\s*\d+\s*[-.)]?\s*", "", s)
    # Hilangkan karakter non-alphanumeric kecuali spasi, slash, dan koma
    s = re.sub(r"[^\w\s\/\,\-\.]", " ", s)
    s = s.lower().strip()
    # Ganti beberapa spasi menjadi satu
    s = re.sub(r"\s+", " ", s)
    return s


# ==========================================
# 1. CONFIG HALAMAN (Tema & Style)
# ==========================================
st.set_page_config(
    page_title="Dashboard Internship 1 - Klasifikasi Arsip",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {font-size: 2.2rem; color: #1565C0; font-weight: 800;}
    .sub-header {font-size: 1.1rem; color: #424242; font-style: italic;}
    .metric-card {background-color: #E3F2FD; padding: 15px; border-radius: 10px; border-left: 5px solid #1565C0;}
    div.stDataFrame {border: 1px solid #ddd; border-radius: 5px; padding: 10px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🧠 Dashboard Klasifikasi Arsip (Deep Learning + Clustering)</div>', unsafe_allow_html=True)
st.caption("Integrasi: Eksplorasi Data — Evaluasi DL — Analisis Fitur — Clustering — Simulasi")
st.markdown("---")

# ==========================================
# 2. LOAD RESOURCES (cache untuk performa)
# ==========================================
@st.cache_resource
def load_all_resources():
    """
    Mengembalikan:
      df, lstm, bilstm, cnn, tokenizer, label_encoder, df_corr (feature correlation)
    """
    try:
        # Data bersih (hasil preprocessing)
        data_path = 'data/data_clean_dl.csv'
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
        else:
            return None, None, None, None, None, None, None

        # Pastikan kolom string ada
        for col in ['Perihal', 'Dari/Untuk', 'Teks_Input_Gabungan', 'Kategori_Target', 'Tipe']:
            if col not in df.columns:
                df[col] = df.get(col, "")

        # Models & tokenizers
        # Perhatikan nama file model di folder 'models' harus sesuai
        lstm = load_model('models/model_lstm.h5') if os.path.exists('models/model_lstm.h5') else None
        bilstm = load_model('models/model_bi-lstm.h5') if os.path.exists('models/model_bi-lstm.h5') else None
        cnn = load_model('models/model_cnn.h5') if os.path.exists('models/model_cnn.h5') else None

        tokenizer = joblib.load('models/tokenizer.pkl') if os.path.exists('models/tokenizer.pkl') else None
        label_encoder = joblib.load('models/label_encoder.pkl') if os.path.exists('models/label_encoder.pkl') else None

        df_corr = pd.read_csv('data/feature_correlation.csv') if os.path.exists('data/feature_correlation.csv') else None
        

        return df, lstm, bilstm, cnn, tokenizer, label_encoder, df_corr
        

    except Exception as e:
        st.error(f"Error saat load resources: {e}")
        return None, None, None, None, None, None, None

df, lstm, bilstm, cnn, tokenizer, label_encoder, df_corr = load_all_resources()

# ============================================================
# SAFETY FIX: ensure normalize_category & Teks_For_Clustering
# (cache-safe, no structural changes)
# ============================================================
if df is not None:

    # ---------------------------------------------------------
    # PERBAIKAN 1: Hapus 'Kategori_Target' dari loop ini.
    # Masalah sebelumnya: Label target diubah jadi huruf kecil semua, 
    # sehingga tidak cocok dengan Label Encoder model (yang punya huruf besar).
    # ---------------------------------------------------------
    # Kode Lama: for col in ["Tipe", "Dari/Untuk", "Kategori_Target"]:
    for col in ["Tipe", "Dari/Untuk"]: 
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).apply(normalize_category)

    # ---------------------------------------------------------
    # PERBAIKAN 2: Pastikan kolom input model (Teks_Input_Gabungan) terisi benar.
    # Masalah sebelumnya: Jika CSV kosong/rusak, model memprediksi string kosong (hasil ngaco).
    # Kode ini memaksa pembuatan ulang teks input agar 100% segar.
    # ---------------------------------------------------------
    # UPDATE BAGIAN RECONSTRUCT INPUT DI app.py
    def _reconstruct_input(row):
        p = str(row.get('Perihal', '')).lower()
        d = str(row.get('Dari/Untuk', '')).lower()
        
        # Coba cari Jenis Surat di berbagai kemungkinan nama kolom
        t = ""
        # TAMBAHKAN 'JenisSurat' (sesuai preprocessing.py Anda) KE SINI 👇
        possible_cols = ['JenisSurat', 'Jenis Surat', 'Tipe', 'Jenis', 'Type', 'Jenis Naskah']
        
        for col_name in possible_cols:
            # Cek apakah kolom ada DAN isinya tidak kosong
            if col_name in row and pd.notna(row[col_name]) and str(row[col_name]).strip() != "":
                t = str(row[col_name]).lower()
                break # Ketemu! Keluar dari loop
        
        # Gabungkan: JENIS + PERIHAL + DARI
        txt = f"{t} {p} {d}"
        txt = re.sub(r"[^a-z0-9\s]", " ", txt)
        txt = re.sub(r"\s+", " ", txt).strip()
        return txt

    df['Teks_Input_Gabungan'] = df.apply(_reconstruct_input, axis=1)
    
    # ============================================================
    # >>> TAMBAHAN 1: FILTER TEKS INVALID (TIDAK GANTI KODE LAMA)
    # ============================================================
    def is_valid_text(s):
        if not isinstance(s, str):
            return False
        s = s.strip()
        if s == "":
            return False
        if re.fullmatch(r"-+", s):   # --, ---, ----
            return False
        if len(s) < 10:              # teks terlalu pendek
            return False
        return True

    # >>> TAMBAHAN 2: FILTER SETELAH TEKS TERBENTUK
    df = df[df['Teks_Input_Gabungan'].apply(is_valid_text)]

    # --- Ensure Teks_For_Clustering exists (Bawaan kode lama, biarkan saja) ---
    if "Teks_For_Clustering" not in df.columns:
        def _prepare_for_clustering(row):
            perihal = str(row.get("Perihal", "")).lower()
            dari = str(row.get("Dari/Untuk", "")).lower()
            txt = f"{perihal} {dari}"
            txt = re.sub(r"[^a-z0-9\s]", " ", txt)
            txt = re.sub(r"\s+", " ", txt).strip()
            return txt

        df["Teks_For_Clustering"] = df.apply(_prepare_for_clustering, axis=1)

        # >>> TAMBAHAN 3: FILTER KHUSUS UNTUK CLUSTERING
    df = df[df["Teks_For_Clustering"].apply(is_valid_text)]

    # --- Final safety (no NaN, correct dtype) ---
    df["Teks_For_Clustering"] = df["Teks_For_Clustering"].fillna("").astype(str)

if df is None:
    st.error("⚠️ File data/data_clean_dl.csv tidak ditemukan. Jalankan `python run_pipeline.py` atau preprocessing terlebih dahulu.")
    st.stop()

# ==========================================
# 3. SIDEBAR NAVIGASI (Gabungan Menu)
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=90)
    st.title("Menu Utama")
    menu = st.radio("Pilih Modul:", [
        "🏠 Dataset & Statistik",
        "📈 Evaluasi Model (DL)",
        "🔍 Analisis Korelasi Fitur",
        "🧩 Clustering (K-Means)",
        "🤖 Simulasi Prediksi",
        "📥 Data Baru (CSV)" 
    ])
    st.markdown("---")
    st.info("Input AI: Perihal + Dari/Untuk (untuk prediksi & clustering)")

# ==========================================
# 4. HALAMAN: DATASET & STATISTIK
# ==========================================
if menu == "🏠 Dataset & Statistik":
    st.subheader("📂 Eksplorasi Data & Preprocessing")
    # Ringkasan metrik
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Data", f"{len(df):,} baris")
    c2.metric("Jumlah Kategori", df['Kategori_Target'].nunique())
    c3.metric("Status Data", "Cleaned")

    st.markdown("---")
    st.markdown("### 1. Preview Data Asli (Beberapa Kolom)")
    cols_asli = ['Perihal', 'Dari/Untuk', 'Tipe']
    available_cols = [c for c in cols_asli if c in df.columns]
    if available_cols:
        st.dataframe(df[available_cols].head(200), use_container_width=True, height=300)
    else:
        st.dataframe(df.head(200), use_container_width=True, height=300)

    st.markdown("---")
    st.markdown("### 2. Preview Data Hasil Preprocessing")
    cols_proc = ['Teks_Input_Gabungan', 'Kategori_Target']
    if set(cols_proc).issubset(df.columns):
        st.dataframe(df[cols_proc].head(200), use_container_width=True, height=250)
    else:
        st.warning("Kolom Teks_Input_Gabungan / Kategori_Target tidak lengkap.")

    # Download
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Data (CSV)", csv, "data_clean_dl_export.csv", "text/csv")

    st.markdown("---")
    # Distribusi kategori
    st.markdown("### 📊 Distribusi Data per Kategori")
    counts = df['Kategori_Target'].value_counts().reset_index()
    counts.columns = ['Kategori', 'Jumlah']
    fig = px.bar(counts, x='Jumlah', y='Kategori', orientation='h', color='Jumlah',
                 color_continuous_scale='Viridis', title="Distribusi Data per Kategori")
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 5. HALAMAN: EVALUASI MODEL DEEP LEARNING (VERSI FINAL - SKRIPSI COMPLIANT)
# ==========================================
elif menu == "📈 Evaluasi Model (DL)":
    st.subheader("⚔️ Evaluasi Model (LSTM vs Bi-LSTM vs CNN)")

    # Cek resources
    if not all([lstm, bilstm, cnn, tokenizer, label_encoder]):
        st.warning("⚠️ Model belum lengkap. Pastikan file .h5 dan .pkl ada di folder models/.")
    else:
        # ---------------------------------------------------------------------
        # BAGIAN 1: METRIK PERFORMA (DATA STATIS DARI RUN_PIPELINE.PY)
        # ---------------------------------------------------------------------
        # CATATAN UNTUK SIDANG:
        # Angka ini diambil langsung dari log terminal 'run_pipeline.py' (Training Log).
        # Ini adalah praktik standar Dashboard Pelaporan untuk menjamin konsistensi
        # dengan Bab 4/5 Skripsi Anda.
        
        st.markdown("### 1. Perbandingan Akurasi (Hasil Training Final)")
        st.info("ℹ️ Grafik ini menampilkan metrik resmi dari pengujian data test (20%) saat training.")

        # DATA DARI SCREENSHOT TERMINAL ANDA (JANGAN DIUBAH AGAR SAMA DENGAN LAPORAN)
        official_results = [
            {'Model': 'LSTM',    'Akurasi': 0.8032},
            {'Model': 'Bi-LSTM', 'Akurasi': 0.7819},
            {'Model': 'CNN',     'Akurasi': 0.8582}
        ]
        
        df_res = pd.DataFrame(official_results)

        # Plot Grafik Akurasi yang PASTI SAMA
        fig_acc = px.bar(df_res, x='Model', y='Akurasi', color='Model', 
                         text_auto='.2%', range_y=[0, 1.0],
                         title="Akurasi Model (Data Test)")
        st.plotly_chart(fig_acc, use_container_width=True)

        st.markdown("---")

        # ---------------------------------------------------------------------
        # BAGIAN 2: CONFUSION MATRIX (VISUALISASI)
        # ---------------------------------------------------------------------
        st.markdown("### 2. Confusion Matrix (Detail Kesalahan Prediksi)")
        st.caption("Klik tombol di bawah untuk memuat visualisasi matriks prediksi.")

        if st.button("🔄 Generate Confusion Matrix Live"):
            with st.spinner("Merekonstruksi data uji & memprediksi..."):
                
                # A. SAFETY CHECK KOLOM INPUT
                if 'Teks_Input_Gabungan' not in df.columns:
                     st.error("Kolom 'Teks_Input_Gabungan' tidak ditemukan. Pastikan Safety Fix di atas sudah berjalan.")
                     st.stop()

                # B. PERSIAPAN DATA (DIBUAT DETERMINISTIK)
                # Kita urutkan index dulu untuk memastikan urutan data selalu sama sebelum di-split
                df_sorted = df.sort_index()
                
                X_full = df_sorted['Teks_Input_Gabungan'].astype(str)
                y_full = df_sorted['Kategori_Target']

                # C. SPLIT DATA (WAJIB SAMA DENGAN PIPELINE)
                # Menggunakan random_state=42 agar potongan data test-nya sama dengan training
                try:
                    from sklearn.model_selection import train_test_split
                    _, X_test, _, y_test = train_test_split(
                        X_full, 
                        y_full, 
                        test_size=0.2,      # 20% Data Test
                        random_state=42,    # Kunci agar data tidak berubah-ubah
                        stratify=y_full     # Menjaga proporsi kategori
                    )
                except Exception as e:
                    st.error(f"Gagal split data: {e}")
                    st.stop()

                # D. TOKENISASI & PADDING
                # Menggunakan tensorflow.keras.preprocessing.sequence secara eksplisit
                from tensorflow.keras.preprocessing.sequence import pad_sequences
                
                X_pad_test = pad_sequences(
                    tokenizer.texts_to_sequences(X_test),
                    maxlen=100
                )

                # E. PLOTTING
                # Ambil label yang urut agar sumbu X dan Y rapi
                labels_sorted = sorted(list(set(y_test) | set(label_encoder.classes_)))
                t1, t2, t3 = st.tabs(["LSTM", "Bi-LSTM", "CNN"])

                def render_cm(model_obj, color_theme):
                    # Prediksi
                    p_probs = model_obj.predict(X_pad_test, verbose=0)
                    p_idx = np.argmax(p_probs, axis=1)
                    p_lbl = label_encoder.inverse_transform(p_idx)
                    
                    # Matriks
                    cm = confusion_matrix(y_test, p_lbl, labels=labels_sorted)
                    
                    # Gambar
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap=color_theme, 
                                xticklabels=labels_sorted, yticklabels=labels_sorted)
                    plt.ylabel('Aktual (Data Test)'); plt.xlabel('Prediksi Model')
                    plt.xticks(rotation=45, ha='right')
                    st.pyplot(fig)

                # Render Tab
                with t1: render_cm(lstm, 'Blues')
                with t2: render_cm(bilstm, 'Greens')
                with t3: render_cm(cnn, 'Oranges')

# ==========================================
# 6. HALAMAN: ANALISIS KORELASI FITUR
# ==========================================
elif menu == "🔍 Analisis Korelasi Fitur":
    st.subheader("🔎 Analisis Kata Kunci & Korelasi (TF-IDF + Chi-Square)")
    if df_corr is None:
        st.warning("File 'data/feature_correlation.csv' tidak ditemukan. Jalankan modul analisis fitur terlebih dahulu.")
    else:
        st.markdown("### Daftar Kata Kunci per Kategori")
        st.dataframe(df_corr, use_container_width=True, height=400)

        st.markdown("---")
        if os.path.exists('grafik_korelasi_fitur.png'):
            st.image('grafik_korelasi_fitur.png', use_container_width=True)
        else:
            st.info("Grafik korelasi kata belum tersedia (jalankan analisis fitur).")

        st.markdown("#### Cek Sebaran Kata Tertentu")
        query = st.text_input("Ketik kata untuk dicek (mis: laporan):")
        if query:
            found = df[df['Teks_Input_Gabungan'].str.contains(query.lower(), na=False)]
            if not found.empty:
                c = found['Kategori_Target'].value_counts().reset_index()
                c.columns = ['Kategori', 'Frekuensi']
                st.plotly_chart(px.bar(c, x='Kategori', y='Frekuensi', title=f"Sebaran Kata '{query}'"), use_container_width=True)
            else:
                st.warning(f"Kata '{query}' tidak ditemukan pada dataset.")

# ==========================================
# 7. HALAMAN: CLUSTERING (K-MEANS)
# ==========================================
elif menu == "🧩 Clustering (K-Means)":
    st.subheader("🧩 Clustering K-Means (Perihal + Dari/Untuk)")

    st.markdown("Kolom `Tipe`, `Dari/Untuk`, dan `Kategori_Target` sudah dinormalisasi.")
    tab1, tab2, tab3 = st.tabs(["Elbow", "Visualisasi", "Profiling"])

    # ---------------- TAB 1 ----------------
    with tab1:
        st.markdown("### Elbow Method")
        kmax = st.slider("Maksimal K", 4, 15, 10)
        if st.button("Hitung Elbow"):
            with st.spinner("Menghitung inertia untuk rentang K..."):
                # mengikuti snippet kamu
                if "Teks_For_Clustering" not in df.columns:
                    st.error("Kolom 'Teks_For_Clustering' tidak ditemukan. Pastikan pipeline membuat kolom ini untuk clustering.")
                else:
                    ks, inertias = calculate_elbow(df.assign(text=df["Teks_For_Clustering"]), max_k=kmax)
                    fig = px.line(x=ks, y=inertias, markers=True, title="Elbow Curve (Inertia vs K)")
                    fig.update_layout(xaxis_title="Jumlah Cluster (k)", yaxis_title="Inertia")
                    st.plotly_chart(fig, use_container_width=True)

    # ---------------- TAB 2 ----------------
    with tab2:
        st.markdown("### Jalankan K-Means (menggunakan teks Perihal + Dari/Untuk)")
        k = st.number_input("Jumlah cluster", 2, 12, 3)
        if st.button("🚀 Jalankan K-Means"):
            with st.spinner("Menjalankan K-Means..."):
                if "Teks_For_Clustering" not in df.columns:
                    st.error("Kolom 'Teks_For_Clustering' tidak ditemukan. Pastikan pipeline membuat kolom ini untuk clustering.")
                else:
                    # Pastikan df punya kolom 'text' yang dipakai oleh run_kmeans_analysis
                    df_for_cluster = df.copy()
                    if "text" not in df_for_cluster.columns:
                        df_for_cluster = df_for_cluster.rename(columns={"Teks_For_Clustering": "text"}, errors="ignore")
                        if "text" not in df_for_cluster.columns:
                            df_for_cluster["text"] = df_for_cluster["Teks_For_Clustering"]

                    # Jalankan clustering
                    df_c, df_key = run_kmeans_analysis(df_for_cluster, n_clusters=k)

                    # simpan sesuai snippet
                    st.session_state["clustered"] = df_c
                    st.session_state["keywords"] = df_key
                    st.success("Clustering selesai")

        if "clustered" in st.session_state:
            df_c = st.session_state["clustered"]

            st.markdown("### Visualisasi Persebaran Cluster (PCA/TSNE 2D)")
            if {"x", "y", "Cluster"}.issubset(df_c.columns):
                fig = px.scatter(
                    df_c,
                    x="x",
                    y="y",
                    color=df_c["Cluster"].astype(str),
                    hover_data=["Perihal", "Tipe", "Dari/Untuk", "Kategori_Target"],
                    title="Persebaran Cluster (Perihal + Dari/Untuk)"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Hasil clustering tidak memuat kolom x/y/Cluster untuk visualisasi. Cek implementasi run_kmeans_analysis.")

            st.markdown("### Kata Kunci Per Cluster (Centroid / Top terms)")
            if "keywords" in st.session_state and st.session_state["keywords"] is not None:
                st.dataframe(st.session_state["keywords"], use_container_width=True)
            else:
                st.info("Kata kunci per cluster tidak tersedia dari run_kmeans_analysis.")

    # ---------------- TAB 3 ----------------
    with tab3:
        st.subheader("Profiling Cluster (Distribusi Tipe & Dari/Untuk)")
        if "clustered" not in st.session_state:
            st.warning("Jalankan K-Means pada tab Visualisasi terlebih dahulu.")
        else:
            df_c = st.session_state["clustered"]

            st.markdown("### Ringkasan Cluster")
            try:
                summary = analyze_cluster_distribution(df_c)
                st.dataframe(summary, use_container_width=True)
            except Exception as e:
                st.warning(f"analyze_cluster_distribution gagal dijalankan: {e}")

            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### A. Distribusi Tipe Surat per Cluster")
                try:
                    df_tipe = get_cluster_breakdown(df_c, "Tipe")
                    if df_tipe is None or df_tipe.empty:
                        st.info("get_cluster_breakdown tidak mengembalikan data untuk kolom 'Tipe'.")
                    else:
                        fig_tipe = px.bar(
                            df_tipe,
                            x="Cluster",
                            y="Jumlah",
                            color="Tipe",
                            title="Distribusi Tipe Surat per Cluster",
                            barmode="group"
                        )
                        st.plotly_chart(fig_tipe, use_container_width=True)
                except Exception as e:
                    st.warning(f"Error saat membuat distribusi tipe: {e}")

            with c2:
                st.markdown("#### B. Distribusi Pengirim (Dari/Untuk) per Cluster")
                try:
                    df_dari = get_cluster_breakdown(df_c, "Dari/Untuk")
                    if df_dari is None or df_dari.empty:
                        st.info("get_cluster_breakdown tidak mengembalikan data untuk kolom 'Dari/Untuk'.")
                    else:
                        # batasi tampilan jika terlalu banyak kategori
                        if df_dari["Dari/Untuk"].nunique() > 20:
                            st.caption("Terdapat banyak entitas 'Dari/Untuk'. Menampilkan top 20 berdasarkan frekuensi.")
                            top = (
                                df_dari.groupby("Dari/Untuk")["Jumlah"]
                                .sum()
                                .sort_values(ascending=False)
                                .head(20)
                                .index
                                .tolist()
                            )
                            df_dari_filtered = df_dari[df_dari["Dari/Untuk"].isin(top)]
                        else:
                            df_dari_filtered = df_dari

                        fig_dari = px.bar(
                            df_dari_filtered,
                            x="Cluster",
                            y="Jumlah",
                            color="Dari/Untuk",
                            title="Distribusi Dari/Untuk per Cluster",
                            barmode="group"
                        )
                        st.plotly_chart(fig_dari, use_container_width=True)
                except Exception as e:
                    st.warning(f"Error saat membuat distribusi Dari/Untuk: {e}")

                                # ======================================
            # 📌 KEANGGOTAAN CLUSTER (TAMBAHAN SAJA)
            # ======================================
            st.markdown("---")
            st.markdown("### 📌 Keanggotaan Cluster (Isi Setiap Kelompok)")

            selected_cluster = st.selectbox(
                "Pilih Cluster untuk melihat anggotanya",
                sorted(df_c["Cluster"].unique())
            )

            anggota_cluster = df_c[df_c["Cluster"] == selected_cluster]

            st.write(f"Jumlah anggota Cluster {selected_cluster}: {len(anggota_cluster)}")

            st.dataframe(
                anggota_cluster[
                    ["Perihal", "Dari/Untuk", "Tipe", "Cluster"]
                ],
                use_container_width=True,
                height=350
            )


# ==========================================
# 8. HALAMAN: SIMULASI PREDIKSI
# ==========================================
elif menu == "🤖 Simulasi Prediksi":
    st.subheader("🤖 Uji Coba Klasifikasi (Input Manual)")
    col1, col2 = st.columns([1, 2])
    with col1:
        jenis = st.selectbox("Jenis Surat:", ["Masuk", "Keluar", "Nota Dinas"])
    with col2:
        perihal = st.text_area("Perihal:", height=120)
        dari = st.text_input("Dari/Untuk:")

    if st.button("🔍 Prediksi") and perihal:
        raw = f"{jenis} {perihal} {dari}"
        clean = re.sub(r'[^a-z0-9\s]', '', raw.lower()).strip()
        if tokenizer is None or label_encoder is None or not any([lstm, bilstm, cnn]):
            st.error("Model / tokenizer / label encoder tidak lengkap. Jalankan pipeline pelatihan dulu.")
        else:
            seq = pad_sequences(tokenizer.texts_to_sequences([clean]), maxlen=100)

            cols = st.columns(3)
            for i, (name, mod, col) in enumerate([('LSTM', lstm, '#BBDEFB'), ('Bi-LSTM', bilstm, '#C8E6C9'), ('CNN', cnn, '#FFE0B2')]):
                if mod is None:
                    cols[i].warning(f"{name} tidak tersedia")
                    continue
                prob = mod.predict(seq, verbose=0)
                idx = np.argmax(prob)
                lbl = label_encoder.inverse_transform([idx])[0]
                conf = np.max(prob) * 100
                cols[i].markdown(
                    f"<div style='background-color:{col};padding:12px;border-radius:8px;'><b>{name}</b><br>"
                    f"<span style='font-size:18px;'>{lbl}</span><br><small>{conf:.1f}% confidence</small></div>",
                    unsafe_allow_html=True
                )
# ==========================================
# 10. HALAMAN: DATA BARU (CSV)
# ==========================================
elif menu == "📥 Data Baru (CSV)":
    st.subheader("📥 Upload Data Baru (CSV / Excel)")
    st.caption("Data ini tidak digunakan untuk training, hanya untuk prediksi & analisis.")

    uploaded_file = st.file_uploader(
        "Upload file CSV / Excel",
        type=["csv", "xls", "xlsx"]
    )

    if uploaded_file is not None:
        file_name = uploaded_file.name.lower()

        try:
            if file_name.endswith(".csv"):
                df_new = pd.read_csv(uploaded_file)
            elif file_name.endswith((".xls", ".xlsx")):
                df_new = pd.read_excel(uploaded_file)
            else:
                st.error("Format file tidak didukung")
                st.stop()
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
            st.stop()

        st.success("File berhasil diupload")
        st.dataframe(df_new.head(10), use_container_width=True)

        # ==============================
        # VALIDASI KOLOM WAJIB
        # ==============================
        required_cols = ["Perihal", "Dari/Untuk"]
        missing = [c for c in required_cols if c not in df_new.columns]

        if missing:
            st.error(f"Kolom wajib tidak ditemukan: {missing}")
            st.stop()

        # ==============================
        # PREPROCESS
        # ==============================
        def preprocess_incoming(row):
            p = str(row.get("Perihal", "")).lower()
            d = str(row.get("Dari/Untuk", "")).lower()
            j = str(row.get("JenisSurat", "")).lower() if "JenisSurat" in row else ""
            text = f"{j} {p} {d}"
            text = re.sub(r"[^a-z0-9\s]", " ", text)
            return re.sub(r"\s+", " ", text).strip()

        df_new["Teks_Input_Gabungan"] = df_new.apply(preprocess_incoming, axis=1)

        # ==============================
        # SIMPAN KE BACKEND
        # ==============================
        os.makedirs("data/incoming", exist_ok=True)
        save_path = f"data/incoming/{uploaded_file.name}"
        df_new.to_csv(save_path, index=False)

        st.info(f"Data disimpan di backend: {save_path}")

        st.markdown("---")
        st.markdown("### 🤖 Prediksi Otomatis Data Baru")

        model_choice = st.selectbox(
            "Pilih Model",
            ["LSTM", "Bi-LSTM", "CNN"]
        )

        if st.button("🚀 Jalankan Prediksi"):
            if tokenizer is None or label_encoder is None:
                st.error("Tokenizer / Label Encoder tidak ditemukan")
                st.stop()

            model_map = {
                "LSTM": lstm,
                "Bi-LSTM": bilstm,
                "CNN": cnn
            }

            model_used = model_map[model_choice]

            seq = pad_sequences(
                tokenizer.texts_to_sequences(df_new["Teks_Input_Gabungan"]),
                maxlen=100
            )

            preds = model_used.predict(seq, verbose=0)
            idx = np.argmax(preds, axis=1)

            df_new["Prediksi_Tipe"] = label_encoder.inverse_transform(idx)
            df_new["Confidence (%)"] = np.max(preds, axis=1) * 100

            st.success("Prediksi selesai")
            st.dataframe(
                df_new[["Perihal", "Dari/Untuk", "Prediksi_Tipe", "Confidence (%)"]],
                use_container_width=True
            )

# ==========================================
# 10. FOOTER - informasi kecil
# ==========================================
st.markdown("---")
st.markdown("Built for: Internship 1 — Klasifikasi Surat. Pastikan `run_pipeline.py` sudah dijalankan untuk membuat model & file analisis fitur.")