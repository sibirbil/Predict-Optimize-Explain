#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 15:49:10 2025

@author: batuhanatas
"""

import numpy as np
import torch
from torch import func as func
from src.model import FNN
from src.data_utils import clean_missing, split_features_and_targets, scale_features
from src.portfolio_layer import build_layer  # Used by G function
from src.G_function import G_function_benchmark  # Your benchmark G function
from src.langevin import torch_MALA_chain  # Your custom PyTorch MALA

# 1. Load trained model
model = FNN(input_dim=46)
model.load_state_dict(torch.load("trained_model.pth"))
model.eval()
trained_params = dict(model.named_parameters())

# 2. Load validation data
valid_npz = np.load('/Users/batuhanatas/Downloads/FinancialDatasets/char/Char_valid.npz')
valid_data, valid_dates = valid_npz['data'], valid_npz['date']
valid_clean = clean_missing(valid_data, valid_dates, name='valid')
X_valid_list, y_valid_list = split_features_and_targets(valid_clean)

# Scale with train scaler
train_npz = np.load('/Users/batuhanatas/Downloads/FinancialDatasets/char/Char_train.npz')
train_clean = clean_missing(train_npz['data'], train_npz['date'], name='train')
X_train_list, _ = split_features_and_targets(train_clean)
_, X_valid_list, _, _ = scale_features(X_train_list, X_valid_list, [])

# 3. Use full timestamp data (all assets at time t)
timestamp_idx = 0  # Change if you want other timestamps
x0_np = X_valid_list[timestamp_idx]
y0_np = y_valid_list[timestamp_idx]
A = x0_np.shape[0]  # Full asset count at this timestamp

# Convert to torch tensors
x0 = torch.tensor(x0_np, dtype=torch.float32)
y0 = torch.tensor(y0_np, dtype=torch.float32)
x0_full = torch.cat([y0.view(-1, 1), x0], dim=1)

# 4. Define G and gradG for full data
benchmark_return = y0.mean().item()
Sigma = torch.eye(A)  # Optional: empirical covariance if feasible
lambda_ = 0.1
G, gradG = G_function_benchmark(model, trained_params, x0, benchmark_return, Sigma, lambda_)

# Adjust step size for stability
eta = 0.005
hypsG = (G, gradG, eta)

# 5. Run MALA on full data
x0_full.requires_grad_(True)
x_final, x_traj = torch_MALA_chain(x0_full, hypsG, NSteps=500)

# 6. Save or analyze output
torch.save(x_traj, f"generated_scenarios_full_{A}_assets.pt")
print(f"Scenario generation complete. Saved {A} assets to generated_scenarios_full_{A}_assets.pt")
