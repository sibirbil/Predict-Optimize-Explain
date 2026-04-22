from __future__ import annotations

import numpy as np


def make_psd(matrix: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    eigenvalues = np.maximum(eigenvalues, eps)
    return (eigenvectors * eigenvalues) @ eigenvectors.T


def ewma_covariance(
    returns_matrix: np.ndarray,
    ewma_lambda: float,
    shrinkage: float,
    ridge: float,
) -> np.ndarray | None:
    returns_matrix = np.asarray(returns_matrix, dtype=float)
    if np.isnan(returns_matrix).any():
        raise ValueError("ewma_covariance expects a dense matrix without NaNs.")

    n_time, n_assets = returns_matrix.shape
    if n_time < 2 or n_assets < 2:
        return None

    exponents = np.arange(n_time - 1, -1, -1)
    weights = (1.0 - ewma_lambda) * np.power(ewma_lambda, exponents)
    weights = weights / weights.sum()

    weighted_mean = np.sum(returns_matrix * weights[:, None], axis=0)
    centered = returns_matrix - weighted_mean[None, :]
    covariance = centered.T @ (centered * weights[:, None])
    covariance = 0.5 * (covariance + covariance.T)

    diagonal = np.diag(np.diag(covariance))
    covariance = (1.0 - shrinkage) * covariance + shrinkage * diagonal
    covariance = covariance + ridge * np.eye(n_assets)
    covariance = make_psd(covariance)
    return 0.5 * (covariance + covariance.T)
