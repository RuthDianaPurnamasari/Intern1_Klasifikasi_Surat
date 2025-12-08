# src/config.py
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_CLEAN_PATH = os.path.join(DATA_DIR, 'data_clean_dl.csv')

# Hyperparameters
MAX_VOCAB_SIZE = 5000   # Jumlah kata unik maksimal
MAX_SEQ_LENGTH = 100    # Panjang kalimat (padding)
EMBEDDING_DIM = 100     # Dimensi vektor kata
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
TEST_SIZE = 0.2
RANDOM_STATE = 42