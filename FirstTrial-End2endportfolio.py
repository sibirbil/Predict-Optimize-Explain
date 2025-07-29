#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 00:41:13 2025

@author: batuhanatas
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
import cvxpy as cp
import torch
import torch.nn as nn
import torch.optim as optim
from cvxpylayers.torch import CvxpyLayer
from torch import func as func

def clean_missing(data_3d, dates, name='dataset'):
    T, N, D = data_3d.shape
    clean_data = []
    for t in range(T):
        snapshot = data_3d[t]  
        mask = np.all(snapshot != -99.99, axis=1)  
        clean_snapshot =snapshot[mask]
        clean_data.append(clean_snapshot)
        date_str = dates[t] if isinstance(dates[t], str) else str(dates[t])
    return clean_data
  
train_npz = np.load('/content/drive/MyDrive/Colab Notebooks/Char_train.npz')
valid_npz = np.load('/content/drive/MyDrive/Colab Notebooks/Char_valid.npz')
test_npz  = np.load('/content/drive/MyDrive/Colab Notebooks/Char_test.npz')

train_data = train_npz['data']
valid_data = valid_npz['data']
test_data  = test_npz['data']

train_dates = train_npz['date']
valid_dates = valid_npz['date']
test_dates  = test_npz['date']

feature_names = train_npz['variable']

train_clean = clean_missing(train_data, train_dates, name='train')
valid_clean = clean_missing(valid_data, valid_dates, name='valid')
test_clean  = clean_missing(test_data, test_dates, name='test')

class FNN(nn.Module):
    def __init__(self, input_dim=46, hidden_dims=[32, 16, 8], dropout=0.5):
        super(FNN, self).__init__()
        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim,1)) 

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (N_t, D)
        returns: shape (N_t,) - predicted returns per firm
        """
        return self.net(x).squeeze(-1)


def split_features_and_targets(clean_data):
    """
    Splits cleaned data into X_t (features) and y_t (returns) per timestamp.
    Returns two lists: X_list and y_list.
    """
    X_list, y_list = [], []
    for snapshot in clean_data:
        y_t = snapshot[:, 0]        
        X_t = snapshot[:, 1:]     
        X_list.append(X_t)
        y_list.append(y_t)
    return X_list, y_list

X_train_list, y_train_list = split_features_and_targets(train_clean)
X_valid_list, y_valid_list = split_features_and_targets(valid_clean)
X_test_list,  y_test_list  = split_features_and_targets(test_clean)

scaler = StandardScaler()
all_train_X = np.vstack(X_train_list)
scaler.fit(all_train_X)

X_train_list = [scaler.transform(X) for X in X_train_list]
X_valid_list = [scaler.transform(X) for X in X_valid_list]
X_test_list  = [scaler.transform(X) for X in X_test_list]


def build_layer(n_assets, lambda_, preds):
    """
    Constructs the cvxpylayer per timestamp using preds to build sigma.
    """
    w = cp.Variable(n_assets)
    b = cp.Parameter(n_assets)  

    # Set b to the predicted returns (passing preds directly)
    b.value = preds.detach().cpu().numpy()  # Detach the tensor to avoid gradients

    sigma = np.diag(preds.detach().cpu().numpy() ** 2).astype(np.float32) 
    objective = cp.Maximize(w.T @ b - (lambda_ / 2) * cp.quad_form(w, sigma))
    constraints = [cp.sum(w) == 1, w >= 0]  
    problem = cp.Problem(objective, constraints)
    assert problem.is_dpp()  
    return CvxpyLayer(problem, parameters=[b], variables=[w])



def regret_loss(model, lambda_, params, x_t, y_t):
    """
    Computes differentiable regret: how far the predicted portfolio is from oracle.
    """
    B = len(y_t)
    device = x_t.device
    preds = func.functional_call(model, params, x_t) 
    cvxpylayer = build_layer(B, lambda_, preds)

    y_t_tensor = y_t.detach().clone().requires_grad_(False)

    # Oracle solution: best portfolio under true returns
    y_solution, = cvxpylayer(y_t_tensor)

    # Model predictions
    p_solution, = cvxpylayer(preds)

    # Checking timestamps current portfolio optimization weights
    # print("y_solution min:", y_solution.min().item(), "sum:", y_solution.sum().item(), "y_solution max:", y_solution.max().item())
    # print("p_solution min:", p_solution.min().item(), "sum:", p_solution.sum().item(), "p_solution max:", p_solution.max().item())

    # Regret: compare the portfolio returns (loss function for now it can be changed to Sharpe ratio regret etc.)
    regret = torch.dot(y_t, p_solution) - torch.dot(y_t, y_solution)
    return regret.pow(2)  # Squared regret loss



def train_regret_model(model, lambda_, X_train_list, y_train_list, X_valid_list, y_valid_list, X_test_list, y_test_list, epochs=5, lr=1e-3):
    """
    Train the model using the regret loss function and the Adam optimizer.
    Adds validation after each epoch to check performance on unseen data and tests after training.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    params = dict(model.named_parameters())

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        count = 0

        # Training Loop
        for batch_idx, (X_t_np, y_t_np) in enumerate(zip(X_train_list, y_train_list)):
            if len(X_t_np) < 10:
                continue  # Skip short timestamps

            X_t = torch.tensor(X_t_np, dtype=torch.float32).to(device)
            y_t = torch.tensor(y_t_np, dtype=torch.float32).to(device)

            loss_fn = regret_loss(model, lambda_, params, X_t, y_t)

            optimizer.zero_grad()  
            loss_fn.backward()  
            optimizer.step()

            total_loss += loss_fn.item()
            count += 1

            # Print progress within epoch
            print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(X_train_list)}")
            print(f"Current Loss: {loss_fn.item():.6f}")

        avg_loss = total_loss / count
        print(f"Epoch {epoch+1}: Avg Regret Loss (Train) = {avg_loss:.6f}")

        # Validation Loop
        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for X_valid_t_np, y_valid_t_np in zip(X_valid_list, y_valid_list):
                X_valid_t = torch.tensor(X_valid_t_np, dtype=torch.float32).to(device)
                y_valid_t = torch.tensor(y_valid_t_np, dtype=torch.float32).to(device)
                val_loss_fn = regret_loss(model, lambda_, params, X_valid_t, y_valid_t)
                val_loss += val_loss_fn.item()
                val_count += 1

        avg_val_loss = val_loss / val_count
        print(f"Epoch {epoch+1}: Avg Regret Loss (Validation) = {avg_val_loss:.6f}")

    print("Training complete!")

    # Evaluate on test data after training
    test_loss = 0.0
    test_count = 0
    with torch.no_grad():
        model.eval()
        for X_test_t_np, y_test_t_np in zip(X_test_list, y_test_list):
            X_test_t = torch.tensor(X_test_t_np, dtype=torch.float32).to(device)
            y_test_t = torch.tensor(y_test_t_np, dtype=torch.float32).to(device)
            test_loss_fn = regret_loss(model, lambda_, params, X_test_t, y_test_t)
            test_loss += test_loss_fn.item()
            test_count += 1

    avg_test_loss = test_loss / test_count
    print(f"Test Loss (Regret) = {avg_test_loss:.6f}")



lambda_ = 0.5  
model = FNN(input_dim=46)

train_regret_model(
    model, 
    lambda_, 
    X_train_list, y_train_list, 
    X_valid_list, y_valid_list, 
    X_test_list, y_test_list, 
    epochs=10
)
