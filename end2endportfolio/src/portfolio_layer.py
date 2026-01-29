#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 17:30:28 2025

@author: batuhanatas
"""

import numpy as np
import cvxpy as cp
import torch
from typing import Optional, Union, Tuple
from cvxpylayers.torch import CvxpyLayer


def _sample_covariance(X: torch.Tensor, ridge: float = 1e-6) -> np.ndarray:
    """Compute a sample covariance matrix for assets in *rows* of X."""
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    n_assets = X.shape[0]
    if n_assets < 2:
        return np.eye(n_assets, dtype=np.float64)
    X_np = X.detach().cpu().double().numpy()
    X_centered = X_np - X_np.mean(axis=0, keepdims=True)
    cov = (X_centered @ X_centered.T) / max(X_np.shape[1] - 1, 1)
    cov += ridge * np.eye(n_assets)
    return cov


def _stable_cholesky(cov: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """Compute a stable Cholesky factor for a covariance matrix."""
    cov = np.asarray(cov, dtype=np.float64)
    cov = 0.5 * (cov + cov.T)
    n = cov.shape[0]
    jitter = ridge
    for _ in range(10):
        try:
            L = np.linalg.cholesky(cov + jitter * np.eye(n, dtype=np.float64))
            return L
        except np.linalg.LinAlgError:
            jitter *= 10
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 1e-8, None)
    roots = np.sqrt(eigvals)
    return eigvecs @ np.diag(roots)


def build_portfolio_layer(n_assets: int, lambda_: float) -> CvxpyLayer:
    """Create a reusable CVXPY layer parameterised by predicted returns and covariance factorizations."""
    w = cp.Variable(n_assets)
    b = cp.Parameter(n_assets)
    L = cp.Parameter((n_assets, n_assets))
    b.value = np.zeros(n_assets, dtype=np.float64)
    L.value = np.eye(n_assets, dtype=np.float64)
    objective = cp.Maximize(w.T @ b - (lambda_ / 2) * cp.sum_squares(L @ w))
    constraints = [cp.sum(w) == 1, w >= 0]
    problem = cp.Problem(objective, constraints)
    if not problem.is_dpp():
        raise RuntimeError("Portfolio optimisation problem must satisfy DPP")
    return CvxpyLayer(problem, parameters=[b, L], variables=[w])


def prepare_layer_inputs(
    preds: torch.Tensor,
    X_batch: Optional[torch.Tensor] = None,
    Sigma_override: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ridge: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare CPU/double parameters for the portfolio layer."""
    preds_cpu = preds.to(dtype=torch.double, device="cpu")
    if Sigma_override is not None:
        if isinstance(Sigma_override, torch.Tensor):
            cov_np = Sigma_override.detach().cpu().numpy()
        else:
            cov_np = np.asarray(Sigma_override, dtype=np.float64)
    elif X_batch is not None:
        cov_np = _sample_covariance(X_batch, ridge=ridge)
    else:
        n = preds_cpu.shape[0]
        cov_np = np.eye(n, dtype=np.float64)
    L_np = _stable_cholesky(cov_np, ridge=ridge)
    L_cpu = torch.from_numpy(L_np.astype(np.float64))
    return preds_cpu, L_cpu
