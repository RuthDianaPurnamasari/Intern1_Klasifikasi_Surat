# src/model_architectures.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Bidirectional, Conv1D, GlobalMaxPooling1D, Dropout
from src.config import *

def get_lstm_model(num_classes):
    model = Sequential([
        Embedding(MAX_VOCAB_SIZE, EMBEDDING_DIM),
        SpatialDropout1D(0.2),
        LSTM(100, dropout=0.2, recurrent_dropout=0.2),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def get_bilstm_model(num_classes):
    model = Sequential([
        Embedding(MAX_VOCAB_SIZE, EMBEDDING_DIM),
        SpatialDropout1D(0.2),
        Bidirectional(LSTM(64, return_sequences=True)),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def get_cnn_model(num_classes):
    model = Sequential([
        Embedding(MAX_VOCAB_SIZE, EMBEDDING_DIM),
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model