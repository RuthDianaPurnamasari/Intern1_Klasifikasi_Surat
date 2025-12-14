# ============================================
# preprocessing.py — FINAL FIX STREAMLIT VERSION
# ============================================

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


# ----------------------------------------------------
# CLEAN TEKS
# ----------------------------------------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ----------------------------------------------------
# LOAD & PREPROCESS
# ----------------------------------------------------
def load_and_preprocess():
    print("[PREPROCESSING] Memuat dan membersihkan dataset...")

    # ==================================================
    # 1. LOAD SEMUA FILE DATA
    # ==================================================
    files = [
        f for f in os.listdir(DATA_DIR)
        if f.endswith((".csv", ".xlsx")) and "clean" not in f
    ]

    dfs = []
    for f in files:
        path = os.path.join(DATA_DIR, f)
        try:
            if f.endswith(".csv"):
                df = pd.read_csv(path, header=1)
            else:
                df = pd.read_excel(path, header=1)
            dfs.append(df)
        except Exception as e:
            print(f"⚠ Gagal membaca {f}: {e}")

    if not dfs:
        raise ValueError("Tidak ada dataset ditemukan.")

    df = pd.concat(dfs, ignore_index=True)


    # ==================================================
    # 2. DETEKSI KOLOM ASLI (TIDAK DIHAPUS)
    # ==================================================
    cols = [c.lower() for c in df.columns]

    tipe_col = next((c for c in df.columns if "tipe" in c.lower()), None)
    perihal_col = next((c for c in df.columns if "perihal" in c.lower()), None)
    dari_col = next((c for c in df.columns if "dari" in c.lower() or "untuk" in c.lower()), None)

    jenis_patterns = ["jns", "jenis", "jenis surat", "jns surat"]
    jenis_col = next((c for c in df.columns if any(p in c.lower() for p in jenis_patterns)), None)

    if tipe_col is None or perihal_col is None:
        raise ValueError("Kolom Tipe/Perihal tidak ditemukan!")

    rename_map = {
        tipe_col: "Tipe",
        perihal_col: "Perihal"
    }

    if dari_col: rename_map[dari_col] = "Dari/Untuk"
    if jenis_col: rename_map[jenis_col] = "JenisSurat"

    df = df.rename(columns=rename_map)

    # Jika kolom tidak ada → isi default
    if "Dari/Untuk" not in df.columns:
        df["Dari/Untuk"] = "-"

    if "JenisSurat" not in df.columns:
        df["JenisSurat"] = "umum"


    # ==================================================
    # 3. FILTER PERIHAL KOSONG
    # ==================================================
    df["Perihal"] = df["Perihal"].astype(str)
    df = df[df["Perihal"].str.strip() != ""]
    df = df[df["Perihal"].str.lower() != "nan"]


    # ==================================================
    # 4. GABUNG TEKS (untuk model)
    # ==================================================
    df["raw_gabungan"] = (
        df["JenisSurat"].astype(str) + " " +
        df["Perihal"].astype(str) + " " +
        df["Dari/Untuk"].astype(str)
    )

    df["Teks_Input_Gabungan"] = df["raw_gabungan"].apply(clean_text)


    # ==================================================
    # 5. CLEAN LABEL (TIDAK MENGHAPUS KOLOM TIPE)
    # ==================================================
    def clean_label(x):
        if not isinstance(x, str):
            return None
        x = re.sub(r"^\d+\s*[-.]*\s*", "", x).strip().upper()
        return None if x in ["", "-", "NAN"] else x

    df["Kategori_Target"] = df["Tipe"].apply(clean_label)
    df = df.dropna(subset=["Kategori_Target"])


    # ==================================================
    # 6. FILTER LABEL <10 DATA
    # ==================================================
    counts = df["Kategori_Target"].value_counts()
    valid_labels = counts[counts >= 10].index
    df = df[df["Kategori_Target"].isin(valid_labels)]

    print(f"📌 Data valid setelah filter: {len(df)}")


    # ==================================================
    # 7. SIMPAN CLEAN CSV (KOLUM ASLI + PREPROCESSING)
    # ==================================================
    target_cols = [
        "JenisSurat",
        "Perihal",
        "Dari/Untuk",
        "Tipe",
        "Teks_Input_Gabungan",
        "Kategori_Target"
    ]

    for c in target_cols:
        df[c] = df[c].astype(str).replace(r"[\n\r,]+", " ", regex=True).str.strip()

    df[target_cols].to_csv(DATA_CLEAN_PATH, index=False)
    print(f"✅ Data clean tersimpan: {DATA_CLEAN_PATH}")


    # ==================================================
    # 8. TOKENIZER + LABEL ENCODER
    # ==================================================
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
    tokenizer.fit_on_texts(df["Teks_Input_Gabungan"])

    X = pad_sequences(tokenizer.texts_to_sequences(df["Teks_Input_Gabungan"]), maxlen=MAX_SEQ_LENGTH)

    le = LabelEncoder()
    y = le.fit_transform(df["Kategori_Target"])

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    joblib.dump(tokenizer, os.path.join(MODEL_DIR, "tokenizer.pkl"))
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.pkl"))


    # ==================================================
    # 9. SPLIT
    # ==================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    return X_train, X_test, y_train, y_test, len(le.classes_)


# # src/preprocessing.py
# # src/preprocessing.py

# import pandas as pd
# import numpy as np
# import re
# import os
# import joblib
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from src.config import *

# def clean_text(text):
#     if not isinstance(text, str): 
#         return ""
#     text = text.lower()
#     text = re.sub(r'[^a-z0-9\s]', '', text)
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text


# def load_and_preprocess():
#     print("[PREPROCESSING] Memuat data dan mengamankan format CSV...")

#     # -----------------------------------------------------------
#     # 1. LOAD DATASET
#     # -----------------------------------------------------------
#     files = [f for f in os.listdir(DATA_DIR) 
#              if f.endswith(('.csv', '.xlsx')) and 'clean' not in f]

#     dfs = []
#     for f in files:
#         try:
#             path = os.path.join(DATA_DIR, f)
#             if f.endswith('.csv'):
#                 df = pd.read_csv(path, header=1)
#             else:
#                 df = pd.read_excel(path, header=1)
#             dfs.append(df)
#         except:
#             pass

#     if not dfs:
#         raise ValueError("Data tidak ditemukan!")

#     df = pd.concat(dfs, ignore_index=True)

#     # -----------------------------------------------------------
#     # 2. DETEKSI & RENAME KOLOM
#     # -----------------------------------------------------------
#     cols = [c.lower() for c in df.columns]

#     idx_tipe = next((i for i, c in enumerate(cols) if 'tipe' in c), 1)
#     idx_perihal = next((i for i, c in enumerate(cols) if 'perihal' in c), 2)
#     idx_dari = next((i for i, c in enumerate(cols) if 'dari' in c or 'untuk' in c), None)
#     idx_jenis = next((i for i, c in enumerate(cols) if 'jenis' in c), None)

#     # Kolom asli
#     col_tipe_asli = df.columns[idx_tipe]
#     col_perihal_asli = df.columns[idx_perihal]

#     # Rename dasar
#     rename_map = {
#         col_tipe_asli: 'Tipe',
#         col_perihal_asli: 'Perihal'
#     }

#     # Tambahkan kolom JENIS
#     if idx_jenis is not None:
#         col_jenis_asli = df.columns[idx_jenis]
#         rename_map[col_jenis_asli] = 'Jenis'
#     else:
#         df["Jenis"] = "surat"

#     # ⚠️ PATCH PENTING: TERAPKAN RENAME MAP SEBELUM DIPAKAI
#     df = df.rename(columns=rename_map)

#     # Handle kolom Dari/Untuk
#     if idx_dari is not None:
#         col_dari_asli = df.columns[idx_dari]
#         df = df.rename(columns={col_dari_asli: "Dari/Untuk"})

#         df['raw_gabungan'] = (
#             df['Jenis'].astype(str) + " " +
#             df['Perihal'].astype(str) + " " +
#             df['Dari/Untuk'].astype(str)
#         )
#     else:
#         df["Dari/Untuk"] = "-"
#         df['raw_gabungan'] = (
#             df['Jenis'].astype(str) + " " +
#             df['Perihal'].astype(str)
#         )

#     # -----------------------------------------------------------
#     # 3. CLEANING TEKS
#     # -----------------------------------------------------------
#     df['Teks_Input_Gabungan'] = df['raw_gabungan'].apply(clean_text)

#     def clean_lbl(l):
#         if not isinstance(l, str):
#             return None
#         l = re.sub(r'^\d+\s*-\s*', '', l).replace('\n', ' ').strip().upper()
#         return None if l in ['-', '--', '', 'NAN'] else l

#     df['Kategori_Target'] = df['Tipe'].apply(clean_lbl)

#     # Filter data kosong
#     df = df.dropna(subset=['Teks_Input_Gabungan', 'Kategori_Target'])
#     df = df[df['Teks_Input_Gabungan'] != ""]

#     # Hapus kategori jarang
#     counts = df['Kategori_Target'].value_counts()
#     df = df[df['Kategori_Target'].isin(counts[counts >= 10].index)]

#     # -----------------------------------------------------------
#     # 4. BERSIHKAN KARAKTER BAHAYA CSV
#     # -----------------------------------------------------------
#     target_cols = ['Perihal', 'Dari/Untuk', 'Tipe', 'Jenis',
#                    'Teks_Input_Gabungan', 'Kategori_Target']

#     print("Membersihkan karakter Enter & Koma agar data CSV rapi...")
#     for col in target_cols:
#         df[col] = df[col].astype(str).replace(r'[\n\r,]+', ' ', regex=True).str.strip()

#     # Simpan CSV
#     df[target_cols].to_csv(DATA_CLEAN_PATH, index=False)
#     print(f" Data Rapi tersimpan di: {DATA_CLEAN_PATH}")

#     # -----------------------------------------------------------
#     # 5. TOKENIZER & LABEL ENCODER
#     # -----------------------------------------------------------
#     le = LabelEncoder()
#     y = le.fit_transform(df['Kategori_Target'])

#     tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
#     tokenizer.fit_on_texts(df['Teks_Input_Gabungan'])
#     X_seq = tokenizer.texts_to_sequences(df['Teks_Input_Gabungan'])
#     X_pad = pad_sequences(X_seq, maxlen=MAX_SEQ_LENGTH)

#     if not os.path.exists(MODEL_DIR):
#         os.makedirs(MODEL_DIR)

#     joblib.dump(le, os.path.join(MODEL_DIR, 'label_encoder.pkl'))
#     joblib.dump(tokenizer, os.path.join(MODEL_DIR, 'tokenizer.pkl'))

#     # -----------------------------------------------------------
#     # 6. TRAIN/TEST SPLIT
#     # -----------------------------------------------------------
#     X_train, X_test, y_train, y_test = train_test_split(
#         X_pad,
#         y,
#         test_size=TEST_SIZE,
#         random_state=RANDOM_STATE,
#         stratify=y
#     )

#     return X_train, X_test, y_train, y_test, len(le.classes_)


# import pandas as pd
# import numpy as np
# import re
# import os
# import joblib
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from src.config import *

# def clean_text(text):
#     if not isinstance(text, str): return ""
#     text = text.lower()
#     text = re.sub(r'[^a-z0-9\s]', '', text) 
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text

# def load_and_preprocess():
#     print("[PREPROCESSING] Memuat data dan mengamankan format CSV...")
    
#     # 1. Load Data
#     files = [f for f in os.listdir(DATA_DIR) if f.endswith(('.csv', '.xlsx')) and 'clean' not in f]
#     dfs = []
#     for f in files:
#         try:
#             path = os.path.join(DATA_DIR, f)
#             if f.endswith('.csv'): df = pd.read_csv(path, header=1)
#             else: df = pd.read_excel(path, header=1)
#             dfs.append(df)
#         except: pass
    
#     if not dfs: raise ValueError("Data tidak ditemukan!")
#     df = pd.concat(dfs, ignore_index=True)
    
#     # 2. Deteksi & Rename Kolom (Standar Excel)
#     cols = [c.lower() for c in df.columns]
    
#     idx_tipe = next((i for i, c in enumerate(cols) if 'tipe' in c), 1)
#     idx_perihal = next((i for i, c in enumerate(cols) if 'perihal' in c), 2)
#     idx_dari = next((i for i, c in enumerate(cols) if 'dari' in c or 'untuk' in c), None)

#     col_tipe_asli = df.columns[idx_tipe]
#     col_perihal_asli = df.columns[idx_perihal]
    
#     rename_map = {
#         col_tipe_asli: 'Tipe',       
#         col_perihal_asli: 'Perihal'
#     }
    
#     if idx_dari is not None:
#         col_dari_asli = df.columns[idx_dari]
#         rename_map[col_dari_asli] = 'Dari/Untuk'
#         df = df.rename(columns=rename_map)
#         # Gabung Teks untuk AI
#         df['raw_gabungan'] = df['Perihal'].astype(str) + " " + df['Dari/Untuk'].astype(str)
#     else:
#         df = df.rename(columns=rename_map)
#         df['Dari/Untuk'] = "-"
#         df['raw_gabungan'] = df['Perihal'].astype(str)

#     # 3. Cleaning untuk AI
#     df['Teks_Input_Gabungan'] = df['raw_gabungan'].apply(clean_text)
    
#     def clean_lbl(l):
#         if not isinstance(l, str): return None
#         l = re.sub(r'^\d+\s*-\s*', '', l).replace('\n', ' ').strip().upper()
#         return None if l in ['-', '--', '', 'NAN'] else l
    
#     df['Kategori_Target'] = df['Tipe'].apply(clean_lbl)
    
#     # Hapus data tidak valid
#     df = df.dropna(subset=['Teks_Input_Gabungan', 'Kategori_Target'])
#     df = df[df['Teks_Input_Gabungan'] != ""]
    
#     counts = df['Kategori_Target'].value_counts()
#     df = df[df['Kategori_Target'].isin(counts[counts >= 10].index)]
    
    
#     # FITUR PENTING: PEMBERSIHAN KHUSUS AGAR CSV TIDAK PECAH
#     # Kita ganti Enter (\n), Return (\r), dan Koma (,) dengan Spasi
#     # agar struktur CSV tetap terjaga (karena CSV dipisah koma/baris).
    
#     target_cols = ['Perihal', 'Dari/Untuk', 'Tipe', 'Teks_Input_Gabungan', 'Kategori_Target']
    
#     print("Membersihkan karakter Enter & Koma agar data CSV rapi...")
#     for col in target_cols:
#         df[col] = df[col].astype(str).str.replace(r'[\n\r,]+', ' ', regex=True).str.strip()

#     # Simpan CSV yang sudah rapi
#     df[target_cols].to_csv(DATA_CLEAN_PATH, index=False)
#     print(f" Data Rapi tersimpan di: {DATA_CLEAN_PATH}")
    
#     # 4. Tokenizing & Encoding
#     le = LabelEncoder()
#     y = le.fit_transform(df['Kategori_Target'])
    
#     tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
#     tokenizer.fit_on_texts(df['Teks_Input_Gabungan'])
#     X_seq = tokenizer.texts_to_sequences(df['Teks_Input_Gabungan'])
#     X_pad = pad_sequences(X_seq, maxlen=MAX_SEQ_LENGTH)
    
#     if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
#     joblib.dump(le, os.path.join(MODEL_DIR, 'label_encoder.pkl'))
#     joblib.dump(tokenizer, os.path.join(MODEL_DIR, 'tokenizer.pkl'))
    
#     # 5. Split
#     X_train, X_test, y_train, y_test = train_test_split(
#         X_pad, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
#     )
    
#     return X_train, X_test, y_train, y_test, len(le.classes_)