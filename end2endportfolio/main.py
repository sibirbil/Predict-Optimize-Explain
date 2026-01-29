#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 17:25:55 2025

@author: batuhanatas
"""

import numpy as np
import torch
from src.data_utils import clean_missing, split_features_and_targets, scale_features
from src.model import FNN
from src.train import train_regret_model

# Load dataset
train_npz = np.load('/Users/batuhanatas/Downloads/FinancialDatasets/char/Char_train.npz')
valid_npz = np.load('/Users/batuhanatas/Downloads/FinancialDatasets/char/Char_valid.npz')
test_npz  = np.load('/Users/batuhanatas/Downloads/FinancialDatasets/char/Char_test.npz')

# Extract raw arrays
train_data, train_dates = train_npz['data'], train_npz['date']
valid_data, valid_dates = valid_npz['data'], valid_npz['date']
test_data, test_dates   = test_npz['data'], test_npz['date']

# Clean missing values
train_clean = clean_missing(train_data, train_dates, name='train')
valid_clean = clean_missing(valid_data, valid_dates, name='valid')
test_clean  = clean_missing(test_data, test_dates, name='test')

# Split features and targets
X_train_list, y_train_list = split_features_and_targets(train_clean)
X_valid_list, y_valid_list = split_features_and_targets(valid_clean)
X_test_list,  y_test_list  = split_features_and_targets(test_clean)

# Scale features
X_train_list, X_valid_list, X_test_list, scaler = scale_features(
    X_train_list, X_valid_list, X_test_list
)

# Initialize model
model = FNN(input_dim=46)

# Set hyperparameters
lambda_ = 0.5
epochs = 15
learning_rate = 5e-4
batch_size = 30
batches_per_timestamp = 10

# Train model with regret loss
train_regret_model(
    model=model,
    lambda_=lambda_,
    X_train_list=X_train_list,
    y_train_list=y_train_list,
    X_valid_list=X_valid_list,
    y_valid_list=y_valid_list,
    X_test_list=X_test_list,
    y_test_list=y_test_list,
    epochs=epochs,
    lr=learning_rate,
    batch_size=batch_size,
    batches_per_timestamp=batches_per_timestamp
)
