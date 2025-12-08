# src/evaluate.py
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tensorflow.keras.models import load_model
from src.config import *
import os

def run_evaluation(X_test, y_test):
    print("\n[EVALUATION] Menguji performa model...")
    
    models = ['LSTM', 'Bi-LSTM', 'CNN']
    results = []
    
    for m in models:
        path = os.path.join(MODEL_DIR, f'model_{m.lower()}.h5')
        model = load_model(path)
        
        pred_prob = model.predict(X_test, verbose=0)
        pred = np.argmax(pred_prob, axis=1)
        
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average='weighted', zero_division=0)
        prec = precision_score(y_test, pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, pred, average='weighted', zero_division=0)
        
        results.append({'Model': m, 'Akurasi': acc, 'F1-Score': f1, 'Precision': prec, 'Recall': rec})
    
    df_res = pd.DataFrame(results)
    print("\nHASIL AKHIR:")
    print(df_res.round(4).to_string(index=False))
    
    # Save chart
    df_melt = df_res.melt(id_vars='Model', var_name='Metrik', value_name='Skor')
    plt.figure(figsize=(10,6))
    sns.barplot(data=df_melt, x='Metrik', y='Skor', hue='Model', palette='viridis')
    plt.ylim(0, 1.1)
    plt.title("Komparasi Model Deep Learning")
    plt.savefig('grafik_evaluasi_dl.png')
    print("Grafik tersimpan: grafik_evaluasi_dl.png")