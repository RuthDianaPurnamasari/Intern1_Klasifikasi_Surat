import pandas as pd
import numpy as np
import re
import os
import joblib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from src.config import *

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text) 
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_preprocess():
    print("[PREPROCESSING] Memuat data dan mengamankan format CSV...")
    
    # 1. Load Data
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(('.csv', '.xlsx')) and 'clean' not in f]
    dfs = []
    for f in files:
        try:
            path = os.path.join(DATA_DIR, f)
            if f.endswith('.csv'): df = pd.read_csv(path, header=1)
            else: df = pd.read_excel(path, header=1)
            dfs.append(df)
        except: pass
    
    if not dfs: raise ValueError("Data tidak ditemukan!")
    df = pd.concat(dfs, ignore_index=True)
    
    # 2. Deteksi & Rename Kolom (Standar Excel)
    cols = [c.lower() for c in df.columns]
    
    idx_tipe = next((i for i, c in enumerate(cols) if 'tipe' in c), 1)
    idx_perihal = next((i for i, c in enumerate(cols) if 'perihal' in c), 2)
    idx_dari = next((i for i, c in enumerate(cols) if 'dari' in c or 'untuk' in c), None)

    col_tipe_asli = df.columns[idx_tipe]
    col_perihal_asli = df.columns[idx_perihal]
    
    rename_map = {
        col_tipe_asli: 'Tipe',       
        col_perihal_asli: 'Perihal'
    }
    
    if idx_dari is not None:
        col_dari_asli = df.columns[idx_dari]
        rename_map[col_dari_asli] = 'Dari/Untuk'
        df = df.rename(columns=rename_map)
        # Gabung Teks untuk AI
        df['raw_gabungan'] = df['Perihal'].astype(str) + " " + df['Dari/Untuk'].astype(str)
    else:
        df = df.rename(columns=rename_map)
        df['Dari/Untuk'] = "-"
        df['raw_gabungan'] = df['Perihal'].astype(str)

    # 3. Cleaning untuk AI
    df['Teks_Input_Gabungan'] = df['raw_gabungan'].apply(clean_text)
    
    def clean_lbl(l):
        if not isinstance(l, str): return None
        l = re.sub(r'^\d+\s*-\s*', '', l).replace('\n', ' ').strip().upper()
        return None if l in ['-', '--', '', 'NAN'] else l
    
    df['Kategori_Target'] = df['Tipe'].apply(clean_lbl)
    
    # Hapus data tidak valid
    df = df.dropna(subset=['Teks_Input_Gabungan', 'Kategori_Target'])
    df = df[df['Teks_Input_Gabungan'] != ""]
    
    counts = df['Kategori_Target'].value_counts()
    df = df[df['Kategori_Target'].isin(counts[counts >= 10].index)]
    
    
    # FITUR PENTING: PEMBERSIHAN KHUSUS AGAR CSV TIDAK PECAH
    # Kita ganti Enter (\n), Return (\r), dan Koma (,) dengan Spasi
    # agar struktur CSV tetap terjaga (karena CSV dipisah koma/baris).
    
    target_cols = ['Perihal', 'Dari/Untuk', 'Tipe', 'Teks_Input_Gabungan', 'Kategori_Target']
    
    print("Membersihkan karakter Enter & Koma agar data CSV rapi...")
    for col in target_cols:
        df[col] = df[col].astype(str).str.replace(r'[\n\r,]+', ' ', regex=True).str.strip()

    # Simpan CSV yang sudah rapi
    df[target_cols].to_csv(DATA_CLEAN_PATH, index=False)
    print(f" Data Rapi tersimpan di: {DATA_CLEAN_PATH}")
    
    # 4. Tokenizing & Encoding
    le = LabelEncoder()
    y = le.fit_transform(df['Kategori_Target'])
    
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
    tokenizer.fit_on_texts(df['Teks_Input_Gabungan'])
    X_seq = tokenizer.texts_to_sequences(df['Teks_Input_Gabungan'])
    X_pad = pad_sequences(X_seq, maxlen=MAX_SEQ_LENGTH)
    
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
    joblib.dump(le, os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    joblib.dump(tokenizer, os.path.join(MODEL_DIR, 'tokenizer.pkl'))
    
    # 5. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_pad, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, len(le.classes_)