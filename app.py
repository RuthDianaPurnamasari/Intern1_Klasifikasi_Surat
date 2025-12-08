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

# 1. KONFIGURASI HALAMAN
st.set_page_config(
    page_title="Dashboard Internship 1 Deep Learning",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #1565C0; font-weight: 800;}
    .sub-header {font-size: 1.2rem; color: #424242; font-style: italic;}
    div.stDataFrame {border: 1px solid #ddd; border-radius: 5px; padding: 10px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🧠 Dashboard Klasifikasi Arsip (Deep Learning)</div>', unsafe_allow_html=True)
st.caption("Analisis Komparasi Arsitektur: LSTM vs Bi-LSTM vs CNN dengan Input Multikolom")
st.markdown("---")

# 2. LOAD RESOURCES
@st.cache_resource
def load_all_resources():
    try:
        if os.path.exists('data/data_clean_dl.csv'):
            df = pd.read_csv('data/data_clean_dl.csv')
        else:
            return None, None, None, None, None, None, None, None

        lstm = load_model('models/model_lstm.h5')
        bilstm = load_model('models/model_bi-lstm.h5')
        cnn = load_model('models/model_cnn.h5')
        tok = joblib.load('models/tokenizer.pkl')
        le = joblib.load('models/label_encoder.pkl')
        df_corr = pd.read_csv('data/feature_correlation.csv') if os.path.exists('data/feature_correlation.csv') else None
            
        return df, lstm, bilstm, cnn, tok, le, df_corr
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None, None, None, None, None, None, None

df, lstm, bilstm, cnn, tokenizer, label_encoder, df_corr = load_all_resources()

if df is None:
    st.error("⚠️ File sistem tidak lengkap! Jalankan 'python run_pipeline.py' dulu.")
    st.stop()

# 3. SIDEBAR NAVIGASI
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
    st.title("Menu Utama")
    menu = st.radio("Pilih Modul:", 
        ["🏠 Dataset & Statistik", 
         "📈 Evaluasi Model (DL)", 
         "🔍 Analisis Korelasi Fitur", 
         "🤖 Simulasi Prediksi"]
    )
    st.markdown("---")
    st.info("Input Data: **Perihal** + **Dari/Untuk**")

# ==========================================
# 4. HALAMAN 1: DATASET (TAMPILAN EXCEL-STYLE)
# ==========================================
if menu == "🏠 Dataset & Statistik":
    st.subheader("📂 Eksplorasi Data & Preprocessing")
    
    # Metrik Ringkas
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Data", f"{len(df):,} Baris")
    c2.metric("Jumlah Kategori", df['Kategori_Target'].nunique()) 
    c3.metric("Status Data", "Siap Olah (Cleaned)")
    
    st.markdown("---")

    # --- TABEL 1: DATA ASLI (STYLE RAPI) ---
    st.markdown("### 1. Preview Data Asli (Format Excel)")
    st.info("Data ini adalah gabungan dari tahun 2023-2025 yang belum diubah menjadi huruf kecil.")

    cols_asli = ['Perihal', 'Dari/Untuk', 'Tipe']
    available_cols = [c for c in cols_asli if c in df.columns]
    
    if available_cols:
        st.dataframe(
            df[available_cols].style.set_properties(**{'text-align': 'left'}),
            use_container_width=True, 
            height=400,
            hide_index=True 
        )
    else:
        st.warning("Kolom asli tidak ditemukan. Menampilkan semua data.")
        st.dataframe(df, use_container_width=True, height=400)

    st.markdown("---")

    # --- TABEL 2: DATA HASIL PREPROCESSING ---
    st.markdown("### 2. Preview Data Hasil Preprocessing (Input AI)")
    st.success("Data di bawah ini sudah melalui proses: Case Folding (huruf kecil), Gabung Kolom, dan Tokenizing.")
    
    cols_process = ['Teks_Input_Gabungan', 'Kategori_Target']
    
    if set(cols_process).issubset(df.columns):
        st.dataframe(
            df[cols_process].style.set_properties(**{'background-color': '#f9f9f9', 'color': 'black'}),
            use_container_width=True,
            height=300,
            hide_index=True
        )
    else:
        st.error(f"Kolom {cols_process} tidak ditemukan. Kolom tersedia: {list(df.columns)}")
    
    # Tombol Download
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Semua Data (.csv)", csv, "data_internship1_full.csv", "text/csv", type='primary')
    
    st.markdown("---")
    
    # Grafik Distribusi
    st.markdown("### 📊 Statistik Sebaran Data")
    counts = df['Kategori_Target'].value_counts().reset_index()
    counts.columns = ['Kategori', 'Jumlah']
    
    fig = px.bar(counts, x='Jumlah', y='Kategori', orientation='h', 
                 color='Jumlah', text='Jumlah', title="Distribusi Data per Kategori",
                 color_continuous_scale='Viridis')
    fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=600)
    st.plotly_chart(fig, use_container_width=True)

# 5. HALAMAN 2: EVALUASI
elif menu == "📈 Evaluasi Model (DL)":
    st.subheader("⚔️ Hasil Pengujian Model")
    
    if st.button("🚀 Jalankan Evaluasi Live"):
        with st.spinner("Menguji 3 Model..."):
            _, X_test, _, y_test = train_test_split(df['Teks_Input_Gabungan'], df['Kategori_Target'], test_size=0.2, random_state=42, stratify=df['Kategori_Target'])
            X_pad = pad_sequences(tokenizer.texts_to_sequences(X_test.astype(str)), maxlen=100)
            
            res = []
            preds = {}
            for name, model in [('LSTM', lstm), ('Bi-LSTM', bilstm), ('CNN', cnn)]:
                p_idx = np.argmax(model.predict(X_pad, verbose=0), axis=1)
                p_lbl = label_encoder.inverse_transform(p_idx)
                res.append({'Model': name, 'Akurasi': accuracy_score(y_test, p_lbl)})
                preds[name] = p_lbl
            
            st.markdown("### 1. Perbandingan Akurasi")
            st.plotly_chart(px.bar(pd.DataFrame(res), x='Model', y='Akurasi', color='Model', text_auto='.2%'), use_container_width=True)
            
            st.markdown("### 2. Confusion Matrix")
            labels = sorted(y_test.unique())
            t1, t2, t3 = st.tabs(["LSTM", "Bi-LSTM", "CNN"])
            def plot_cm(p, cmap):
                cm = confusion_matrix(y_test, p, labels=labels)
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=labels, yticklabels=labels)
                plt.xticks(rotation=45, ha='right'); plt.ylabel('Aktual'); plt.xlabel('Prediksi')
                st.pyplot(fig)
            with t1: plot_cm(preds['LSTM'], 'Blues')
            with t2: plot_cm(preds['Bi-LSTM'], 'Greens')
            with t3: plot_cm(preds['CNN'], 'Oranges')


# ==========================================
# 6. HALAMAN 3: ANALISIS FITUR (LAYOUT DIPERBAIKI)
# ==========================================
elif menu == "🔍 Analisis Korelasi Fitur":
    st.subheader("Analisis Kata Kunci")
    
    if df_corr is not None:
        # BAGIAN ATAS: TABEL KATA KUNCI (FULL WIDTH)
        st.markdown("#### 📋 Daftar Kata Kunci per Kategori")
        st.dataframe(df_corr, use_container_width=True, height=400)
        
        st.markdown("---")
        
        # BAGIAN BAWAH: GRAFIK & CEK KATA
        st.markdown("#### 📊 Grafik Kekuatan Korelasi Kata")
        
        # Tampilkan Gambar Grafik
        if os.path.exists('grafik_korelasi_fitur.png'):
            st.image('grafik_korelasi_fitur.png', use_container_width=True)
            
        st.markdown("<br>", unsafe_allow_html=True) # Spasi dikit
        
        # Input Cek Kata
        txt = st.text_input("🔍 Cek Distribusi Kata Tertentu:", placeholder="Ketik kata disini (contoh: laporan)...")
        if txt:
            f = df[df['Teks_Input_Gabungan'].str.contains(txt.lower(), na=False)]
            if not f.empty:
                c = f['Kategori_Target'].value_counts().reset_index()
                c.columns = ['Kategori', 'Frekuensi']
                st.plotly_chart(px.bar(c, x='Kategori', y='Frekuensi', title=f"Sebaran Kata '{txt}'"), use_container_width=True)
            else: 
                st.warning(f"Kata '{txt}' tidak ditemukan dalam dataset.")
    else: 
        st.error("File 'feature_correlation.csv' tidak ditemukan. Jalankan 'python run_analysis.py' dulu.")


# 7. HALAMAN 4: SIMULASI
elif menu == "🤖 Simulasi Prediksi":
    st.subheader("Uji Coba Manual")
    col1, col2 = st.columns(2)
    with col1: p = st.text_area("Perihal:", height=100)
    with col2: s = st.text_input("Dari/Untuk:")
    if st.button("Prediksi") and p:
        raw = p + " " + s
        clean = re.sub(r'[^a-z0-9\s]', '', raw.lower()).strip()
        seq = pad_sequences(tokenizer.texts_to_sequences([clean]), maxlen=100)
        
        cols = st.columns(3)
        for i, (name, mod, col) in enumerate([('LSTM', lstm, '#BBDEFB'), ('Bi-LSTM', bilstm, '#C8E6C9'), ('CNN', cnn, '#FFE0B2')]):
            prob = mod.predict(seq)
            lbl = label_encoder.inverse_transform([np.argmax(prob)])[0]
            conf = np.max(prob)*100
            cols[i].markdown(f"<div style='background-color:{col};padding:15px;border-radius:10px;'><b>{name}</b><br><span style='font-size:20px;'>{lbl}</span><br><small>{conf:.1f}%</small></div>", unsafe_allow_html=True)