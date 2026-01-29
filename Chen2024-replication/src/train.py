#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 22:47:46 2025

@author: batuhanatas
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from src.model import FNN

def train_fnn(X_train, y_train, X_valid, y_valid, hidden_dims, dropout, lr, epochs=10, batch_size=256, weight_decay=0.001, model_dir='./models'):
    

    # Use CPU explicitly (macOS)
    device = torch.device('cpu')

    # Convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_valid = torch.tensor(X_valid, dtype=torch.float32).to(device)
    y_valid = torch.tensor(y_valid, dtype=torch.float32).to(device)

    model = FNN(input_dim=X_train.shape[1], hidden_dims=hidden_dims, dropout=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train.size(0))
        epoch_loss = 0.0

        for i in range(0, X_train.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            x_batch = X_train[indices]
            y_batch = y_train[indices]

            optimizer.zero_grad()
            preds = model(x_batch)
            loss = loss_fn(preds, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(x_batch)

        model.eval()
        with torch.no_grad():
            val_preds = model(X_valid)
            val_loss = loss_fn(val_preds, y_valid)
            val_r2 = r2_score(y_valid.cpu(), val_preds.cpu())

        print(f"Epoch {epoch+1}: Train Loss = {epoch_loss / len(X_train):.4f}, "
              f"Valid Loss = {val_loss.item():.4f}, Valid R² = {val_r2:.4f}")

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_model_state = model.state_dict()

    # Save best model
    if best_model_state:
        model.load_state_dict(best_model_state)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        path = os.path.join(model_dir, 'fnn_model_best.pth')
        torch.save(model.state_dict(), path)
        print(f"Best model saved to {path}")
        return path

    return None
