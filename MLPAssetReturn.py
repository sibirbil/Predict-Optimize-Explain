#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 23 11:58:55 2025

@author: batuhanatas
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import joblib

# ----------------------------
# Dataset
# ----------------------------
class ReturnDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

# ----------------------------
# MLP Model
# ----------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, layers):
        super(MLP, self).__init__()
        net = []
        prev = input_dim
        for h in layers:
            net.append(nn.Linear(prev, h))
            net.append(nn.BatchNorm1d(h))
            net.append(nn.ReLU())
            prev = h
        net.append(nn.Linear(prev, 1))
        self.model = nn.Sequential(*net)

    def forward(self, x):
        return self.model(x)

# ----------------------------
# Training Function
# ----------------------------
def train_model(model, train_loader, val_loader, epochs=100, patience=5, use_huber=False):
    best_model = None
    best_loss = float('inf')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    stop_counter = 0

    def huber_loss(pred, target, delta=1.0):
        error = target - pred
        is_small = torch.abs(error) <= delta
        squared = 0.5 * error**2
        linear = delta * (torch.abs(error) - 0.5 * delta)
        return torch.mean(torch.where(is_small, squared, linear))

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = huber_loss(output, y_batch) if use_huber else nn.MSELoss()(output, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        all_preds, all_targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                output = model(X_batch)
                loss = huber_loss(output, y_batch) if use_huber else nn.MSELoss()(output, y_batch)
                val_losses.append(loss.item())
                all_preds.extend(output.view(-1).tolist())
                all_targets.extend(y_batch.view(-1).tolist())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_r2 = r2_score(all_targets, all_preds)
        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val R²: {val_r2:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model.state_dict()
            stop_counter = 0
        else:
            stop_counter += 1
            if stop_counter >= patience:
                print(f"⏹️ Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_model)
    return model

# ----------------------------
# Rolling Training Loop
# ----------------------------
if __name__ == "__main__":
    df = pd.read_csv("/content/drive/MyDrive/subdf_200firms.csv")
    if 'year' not in df.columns:
        df['year'] = pd.to_datetime(df['date']).dt.year

    exclude = ['target', 'PERMNO', 'date']
    features = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    input_dim = len(features)

    test_r2_scores = []

    for test_year in range(2005, 2025):
        val_year = test_year - 1
        train_years = [y for y in df['year'].unique() if y < val_year]

        train_df = df[df['year'].isin(train_years)]
        val_df   = df[df['year'] == val_year]
        test_df  = df[df['year'] == test_year]

        print(f"\n📆 Year {test_year}: Train {min(train_years)}–{max(train_years)}, Val {val_year}, Test {test_year}")
        print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")

        # Fill and scale
        X_train_df = train_df[features].replace([np.inf, -np.inf], np.nan).ffill().bfill()
        X_val_df   = val_df[features].replace([np.inf, -np.inf], np.nan).ffill().bfill()
        X_test_df  = test_df[features].replace([np.inf, -np.inf], np.nan).ffill().bfill()

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_df)
        X_val   = scaler.transform(X_val_df)
        X_test  = scaler.transform(X_test_df)

        y_train = train_df['target'].values
        y_val   = val_df['target'].values
        y_test  = test_df['target'].values

        train_loader = DataLoader(ReturnDataset(X_train, y_train), batch_size=1024, shuffle=True)
        val_loader = DataLoader(ReturnDataset(X_val, y_val), batch_size=1024)

        model = MLP(input_dim=input_dim, layers=[32, 16, 8])
        model = train_model(model, train_loader, val_loader)

        # Test evaluation
        model.eval()
        with torch.no_grad():
            y_pred = model(torch.tensor(X_test, dtype=torch.float32)).view(-1).numpy()
        test_r2 = r2_score(y_test, y_pred)
        test_r2_scores.append({'year': test_year, 'r2': test_r2})
        print(f"✅ Test R² ({test_year}): {test_r2:.4f}")

        # Optional: save
        torch.save(model.state_dict(), f"mlp_year{test_year}.pt")
        joblib.dump(scaler, f"scaler_year{test_year}.pkl")

    # Final summary
    results_df = pd.DataFrame(test_r2_scores)
    print("\n📈 Year-by-Year R²:")
    print(results_df)
