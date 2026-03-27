"""
Risk computation utilities: covariance estimation and factorization.
"""
import numpy as np


def make_psd(Sigma: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Project symmetric matrix to positive semi-definite via eigenvalue clipping.

    Args:
        Sigma: Input matrix (should be symmetric)
        eps: Minimum eigenvalue threshold (default: 1e-10)

    Returns:
        PSD matrix with eigenvalues >= eps
    """
    # Symmetrize
    S = 0.5 * (Sigma + Sigma.T)

    # Eigendecomposition
    vals, vecs = np.linalg.eigh(S)

    # Clip negative eigenvalues
    vals = np.maximum(vals, eps)

    # Reconstruct
    return (vecs * vals) @ vecs.T


def ewma_cov_full_history_matrix(
    R: np.ndarray,
    lam: float = 0.94,
    shrink: float = 0.10,
    ridge: float = 1e-6,
    psd_proj: bool = True
) -> np.ndarray:
    """
    Exponentially Weighted Moving Average (EWMA) covariance matrix.

    Processing steps:
    1. Compute EWMA weights (most recent data gets highest weight)
    2. Calculate weighted covariance
    3. Shrink towards diagonal
    4. Add ridge regularization
    5. Optional PSD projection

    Args:
        R: Returns matrix (T × N) with NO NaNs
        lam: Decay factor for exponential weighting (default: 0.94)
        shrink: Shrinkage towards diagonal (default: 0.10)
        ridge: Ridge regularization term (default: 1e-6)
        psd_proj: Whether to project to PSD (default: True)

    Returns:
        Covariance matrix (N × N), or None if insufficient data

    Raises:
        ValueError: If R contains NaN values
    """
    R = np.asarray(R, float)

    if np.isnan(R).any():
        raise ValueError("ewma_cov_full_history_matrix expects NO NaNs (full-history).")

    T, N = R.shape
    if T < 2 or N < 2:
        return None

    # Exponential weights (newest data gets highest weight)
    exponents = np.arange(T - 1, -1, -1)
    w = (1.0 - lam) * (lam ** exponents)
    w = w / np.sum(w)

    # Weighted mean and centered returns
    mu = np.sum(R * w[:, None], axis=0)
    Xc = R - mu[None, :]

    # Weighted covariance
    Sigma = (Xc.T @ (Xc * w[:, None]))
    Sigma = 0.5 * (Sigma + Sigma.T)

    # Shrink to diagonal
    diag = np.diag(np.diag(Sigma))
    Sigma = (1.0 - shrink) * Sigma + shrink * diag

    # Add ridge regularization
    Sigma = Sigma + ridge * np.eye(N)
    Sigma = 0.5 * (Sigma + Sigma.T)

    # Optional PSD projection
    if psd_proj:
        Sigma = make_psd(Sigma, eps=1e-10)
        Sigma = 0.5 * (Sigma + Sigma.T)

    return Sigma


def sigma_vol_from_cov(Sigma: np.ndarray) -> np.ndarray:
    """
    Extract individual asset volatilities from covariance matrix.

    Args:
        Sigma: Covariance matrix (N × N)

    Returns:
        Vector of volatilities (N,) = sqrt(diag(Sigma))
    """
    diag = np.diag(Sigma)
    return np.sqrt(np.maximum(diag, 1e-12)).astype(np.float32)


def compute_sigma_factor_for_risk(Sigma: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Compute risk factor matrix U such that w'Σw = ||Uw||².

    Uses Cholesky factorization with eigen fallback:
    - If Σ = LL^T (Cholesky), returns U = L^T
    - If Cholesky fails, uses symmetric sqrt via eigendecomposition

    Args:
        Sigma: Covariance matrix (N × N)
        eps: Regularization for numerical stability (default: 1e-8)

    Returns:
        Factor matrix U (N × N) where ||Uw||² = w'Σw
    """
    S = np.asarray(Sigma, dtype=np.float64)

    # Symmetrize and regularize
    S = 0.5 * (S + S.T)
    S = S + eps * np.eye(S.shape[0], dtype=np.float64)

    # Try Cholesky factorization first
    try:
        L = np.linalg.cholesky(S)  # S = L L^T
        return (L.T).astype(np.float32)

    except np.linalg.LinAlgError:
        # Fallback: symmetric square root via eigendecomposition
        vals, vecs = np.linalg.eigh(S)
        vals = np.maximum(vals, eps)
        sqrtS = vecs @ np.diag(np.sqrt(vals)) @ vecs.T
        return sqrtS.astype(np.float32)
