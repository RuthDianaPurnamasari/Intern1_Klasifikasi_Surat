# src/train.py
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from src.model_architectures import get_lstm_model, get_bilstm_model, get_cnn_model
from src.config import *

# --- TAMBAHAN PENTING: KUNCI ANGKA ACAK ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

# Panggil fungsi ini sebelum melakukan apapun!
set_seed(42) 

def run_training(X_train, X_test, y_train, y_test, num_classes):
    print("\n[TRAINING] Memulai pelatihan 3 Arsitektur DL (Mode Stabil)...")
    
    # Kita inisialisasi ulang seed di dalam fungsi untuk kepastian ganda
    set_seed(42)
    
    models_map = {
        'LSTM': get_lstm_model(num_classes),
        'Bi-LSTM': get_bilstm_model(num_classes),
        'CNN': get_cnn_model(num_classes)
    }
    
    for name, model in models_map.items():
        print(f"\nTraining Model: {name}...")
        
        # Callback: Berhenti kalau tidak tambah pintar
        callback = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        
        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_test, y_test),
            callbacks=[callback],
            verbose=1,
            shuffle=False # Matikan shuffle agar urutan belajar konsisten
        )
        
        save_path = os.path.join(MODEL_DIR, f'model_{name.lower()}.h5')
        model.save(save_path)
        print(f"Model tersimpan: {save_path}")