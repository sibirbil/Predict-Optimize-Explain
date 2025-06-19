#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main training pipeline
"""
# main.py

from src.data_preprocessing import load_and_preprocess_data
from src.train import train_fnn
from src.evaluate import evaluate_model
from config import config

if __name__ == '__main__':
    # Load and preprocess data - data can be found in https://drive.google.com/drive/folders/1TrYzMUA_xLID5-gXOy_as8sH2ahLwz-l  (datasets.zip)
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_and_preprocess_data(
        '/Users/batuhanatas/Downloads/datasets/char/Char_train.npz',
        '/Users/batuhanatas/Downloads/datasets/char/Char_valid.npz',
        '/Users/batuhanatas/Downloads/datasets/char/Char_test.npz'
    )

    # config
    hidden_dims, dropout, lr, batch_size, epochs, weight_decay= config

    # Train model 
    model_path = train_fnn(
        X_train, y_train * 100, 
        X_valid, y_valid * 100,
        hidden_dims=hidden_dims,
        dropout=dropout,
        lr=lr,
        batch_size=batch_size,
        epochs=epochs,
        weight_decay=weight_decay
    )

    # Evaluate on test data
    evaluate_model(model_path, X_test, y_test * 100, hidden_dims, dropout)
