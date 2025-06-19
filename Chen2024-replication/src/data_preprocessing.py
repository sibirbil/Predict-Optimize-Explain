#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 22:46:43 2025

@author: batuhanatas
"""

# src/data_preprocessing.py

import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(train_path, valid_path, test_path):
    train_data = np.load(train_path)
    valid_data = np.load(valid_path)
    test_data = np.load(test_path)

    # Preprocess data (masking)
    X_train_full = train_data['data']
    X_valid_full = valid_data['data']
    X_test_full = test_data['data']

    mask_train = np.all(X_train_full != -99.99, axis=2)
    X_train_clean = X_train_full[mask_train]
    mask_valid = np.all(X_valid_full != -99.99, axis=2)
    X_valid_clean = X_valid_full[mask_valid]
    mask_test = np.all(X_test_full != -99.99, axis=2)
    X_test_clean = X_test_full[mask_test]

    y_train = X_train_clean[:, 0]
    X_train = X_train_clean[:, 1:]
    y_valid = X_valid_clean[:, 0]
    X_valid = X_valid_clean[:, 1:]
    y_test = X_test_clean[:, 0]
    X_test = X_test_clean[:, 1:]

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train, X_valid_scaled, y_valid, X_test_scaled, y_test
