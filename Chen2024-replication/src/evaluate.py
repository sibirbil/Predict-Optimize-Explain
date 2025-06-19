#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 22:54:19 2025

@author: batuhanatas
"""

# src/evaluate.py
import torch
import numpy as np
from sklearn.metrics import r2_score
from src.model import FNN

def evaluate_model(model_path, X_test, y_test, hidden_dims, dropout):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FNN(input_dim=X_test.shape[1], hidden_dims=hidden_dims, dropout=dropout).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        predictions = model(X_test_tensor).cpu().numpy()
        
        
    def r2_oos(y_true, y_pred):
        sspe = np.sum((y_test - predictions)**2)
        ss_total = np.sum(y_test**2)
        return 1 - sspe / ss_total
    r2_oos_score = r2_oos(y_test, predictions)
    print(f"Out-of-sample R²: %{100*r2_oos_score:.4f} (2020 paper has %0.70)")
    
    
    print("Return Predictions (safety check):", predictions)
    return r2_oos_score

