# run_pipeline.py
from src.preprocessing import load_and_preprocess
from src.train import run_training
from src.evaluate import run_evaluation

if __name__ == "__main__":
    # 1. Preprocessing
    X_train, X_test, y_train, y_test, num_classes = load_and_preprocess()
    
    # 2. Training (3 Model)
    run_training(X_train, X_test, y_train, y_test, num_classes)
    
    # 3. Evaluasi
    run_evaluation(X_test, y_test)
    
    print("\nPIPELINE SELESAI! Silakan jalankan 'streamlit run app.py'")