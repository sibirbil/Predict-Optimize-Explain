#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 17:34:07 2025

@author: batuhanatas
"""

import numpy as np
import torch
import torch.optim as optim
from torch.nn import Module
from typing import List, Optional

from src.data_utils import ReturnHistory, covariance_from_history
from src.regret_loss import regret_loss
from src.portfolio_layer import build_portfolio_layer


def train_regret_model(
    model: Module,
    lambda_: float,
    X_train_list: List[np.ndarray],
    y_train_list: List[np.ndarray],
    X_valid_list: List[np.ndarray],
    y_valid_list: List[np.ndarray],
    X_test_list: List[np.ndarray],
    y_test_list: List[np.ndarray],
    asset_ids_train_list: Optional[List[np.ndarray]] = None,
    asset_ids_valid_list: Optional[List[np.ndarray]] = None,
    asset_ids_test_list: Optional[List[np.ndarray]] = None,
    return_history: Optional[ReturnHistory] = None,
    epochs: int = 5,
    lr: float = 1e-3,
    batch_size: int = 20,
    batches_per_timestamp: int = 10
) -> None:
    """
    Trains a model using differentiable regret loss and portfolio optimization.

    Args:
        model (nn.Module): The neural network model.
        lambda_ (float): Risk aversion parameter.
        X_train_list, y_train_list: Training data (features and returns per timestamp).
        X_valid_list, y_valid_list: Validation data.
        X_test_list, y_test_list: Test data.
        asset_ids_train_list (List[np.ndarray], optional): Asset identifiers aligned with training snapshots.
        asset_ids_valid_list (List[np.ndarray], optional): Asset identifiers for validation snapshots.
        asset_ids_test_list (List[np.ndarray], optional): Asset identifiers for test snapshots.
        return_history (ReturnHistory, optional): Mapping asset_id -> {date -> return} used to estimate covariances.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        batch_size (int): Number of assets per batch.
        batches_per_timestamp (int): How many random batches per timestamp.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    params = dict(model.named_parameters())

    portfolio_layer = build_portfolio_layer(batch_size, lambda_)

    def _sigma_for_indices(asset_ids_seq: Optional[List[np.ndarray]], t_idx: int, sample_idx: np.ndarray) -> Optional[torch.Tensor]:
        if asset_ids_seq is None or return_history is None:
            return None
        if t_idx >= len(asset_ids_seq) or asset_ids_seq[t_idx] is None:
            return None
        ids_snapshot = asset_ids_seq[t_idx]
        if ids_snapshot is None or len(ids_snapshot) == 0:
            return None
        batch_ids = np.asarray(ids_snapshot, dtype=str)[sample_idx]
        cov = covariance_from_history(batch_ids, return_history)
        return torch.from_numpy(cov.astype(np.float64))

    for epoch in range(epochs):
        model.train()
        total_loss, count = 0.0, 0

        for batch_idx, (X_t_np, y_t_np) in enumerate(zip(X_train_list, y_train_list)):
            A_t = len(X_t_np)
            if A_t < batch_size:
                continue  # Skip small timestamps

            for _ in range(batches_per_timestamp):
                indices = np.random.choice(A_t, batch_size, replace=False)
                X_batch = torch.tensor(X_t_np[indices], dtype=torch.float32).to(device)
                y_batch = torch.tensor(y_t_np[indices], dtype=torch.float32).to(device)
                sigma_override = _sigma_for_indices(asset_ids_train_list, batch_idx, indices)

                loss = regret_loss(
                    model,
                    lambda_,
                    params,
                    X_batch,
                    y_batch,
                    portfolio_layer=portfolio_layer,
                    Sigma_override=sigma_override,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                count += 1

            print(f"Epoch {epoch+1}/{epochs}, Timestamp {batch_idx+1}/{len(X_train_list)} completed")

        avg_loss = total_loss / count if count else float("nan")
        print(f"Epoch {epoch+1}: Avg Regret Loss (Train) = {avg_loss:.6f}")

        # Validation
        model.eval()
        val_loss, val_count = 0.0, 0
        with torch.no_grad():
            for val_idx, (X_valid_t_np, y_valid_t_np) in enumerate(zip(X_valid_list, y_valid_list)):
                A_val_t = len(X_valid_t_np)
                if A_val_t < batch_size:
                    continue

                indices = np.random.choice(A_val_t, batch_size, replace=False)
                X_valid_t = torch.tensor(X_valid_t_np[indices], dtype=torch.float32).to(device)
                y_valid_t = torch.tensor(y_valid_t_np[indices], dtype=torch.float32).to(device)
                sigma_override = _sigma_for_indices(asset_ids_valid_list, val_idx, indices)

                loss = regret_loss(
                    model,
                    lambda_,
                    params,
                    X_valid_t,
                    y_valid_t,
                    portfolio_layer=portfolio_layer,
                    Sigma_override=sigma_override,
                )
                val_loss += loss.item()
                val_count += 1

        avg_val_loss = val_loss / val_count if val_count else float("nan")
        print(f"Epoch {epoch+1}: Avg Regret Loss (Validation) = {avg_val_loss:.6f}")

    print("Training complete!")

    # Test evaluation
    model.eval()
    test_loss, test_count = 0.0, 0
    with torch.no_grad():
        for test_idx, (X_test_t_np, y_test_t_np) in enumerate(zip(X_test_list, y_test_list)):
            A_test_t = len(X_test_t_np)
            if A_test_t < batch_size:
                continue

            indices = np.random.choice(A_test_t, batch_size, replace=False)
            X_test_t = torch.tensor(X_test_t_np[indices], dtype=torch.float32).to(device)
            y_test_t = torch.tensor(y_test_t_np[indices], dtype=torch.float32).to(device)
            sigma_override = _sigma_for_indices(asset_ids_test_list, test_idx, indices)

            loss = regret_loss(
                model,
                lambda_,
                params,
                X_test_t,
                y_test_t,
                portfolio_layer=portfolio_layer,
                Sigma_override=sigma_override,
            )
            test_loss += loss.item()
            test_count += 1

    avg_test_loss = test_loss / test_count if test_count else float("nan")
    print(f"Test Loss (Regret) = {avg_test_loss:.6f}")
