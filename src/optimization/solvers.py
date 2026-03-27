"""
Portfolio optimization solvers using CVXPY.
"""
import numpy as np
import cvxpy as cp
from typing import Optional, Tuple

from .risk import make_psd


def solve_mvo_long_only(
    mu: np.ndarray,
    Sigma: np.ndarray,
    lambda_: float = 5.0
) -> Optional[np.ndarray]:
    """
    Solve standard Markowitz Mean-Variance Optimization (long-only).

    Problem:
        max μ'w - (λ/2) w'Σw
        s.t. Σw = 1, w ≥ 0

    Args:
        mu: Expected returns (N,)
        Sigma: Covariance matrix (N × N)
        lambda_: Risk aversion coefficient (default: 5.0)

    Returns:
        Optimal weights (N,), or None if solver fails
    """
    mu = np.asarray(mu, float).reshape(-1)
    N = mu.shape[0]

    if N < 2:
        return None

    # Ensure PSD
    Sigma = make_psd(np.asarray(Sigma, float), eps=1e-10)

    # Formulate problem
    w = cp.Variable(N)
    obj = cp.Maximize(mu @ w - (lambda_ / 2.0) * cp.quad_form(w, Sigma))
    cons = [cp.sum(w) == 1, w >= 0]
    prob = cp.Problem(obj, cons)

    # Solve with OSQP (natural for QP), fallback to ECOS
    try:
        prob.solve(
            solver=cp.OSQP,
            verbose=False,
            eps_abs=1e-6,
            eps_rel=1e-6,
            max_iter=20000
        )
    except Exception:
        prob.solve(solver=cp.ECOS, verbose=False)

    # Check solution
    if w.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
        return None

    # Clip negative weights and renormalize
    w_hat = np.asarray(w.value).reshape(-1)
    w_hat = np.clip(w_hat, 0, None)
    s = w_hat.sum()

    return (w_hat / s) if s > 0 else None


def solve_robust_longonly(
    mu_hat: np.ndarray,
    Sigma: np.ndarray,
    lambda_: float,
    kappa: float,
    omega_mode: str = "diagSigma",
    solver_chain: Tuple[str, ...] = ("CLARABEL", "SCS", "ECOS")
) -> Optional[np.ndarray]:
    """
    Solve robust Mean-Variance Optimization with uncertainty penalty.

    Problem:
        max μ'w - κ||Aw||₂ - (λ/2) w'Σw
        s.t. Σw = 1, w ≥ 0

    where A is the uncertainty set matrix:
    - "diagSigma": A = diag(σ), uncertainty proportional to volatility
    - "identity": A = I, equal uncertainty across assets

    Args:
        mu_hat: Expected returns (N,)
        Sigma: Covariance matrix (N × N)
        lambda_: Risk aversion coefficient
        kappa: Robustness penalty (0 = standard MVO, >0 = robust)
        omega_mode: Uncertainty set mode ("diagSigma" or "identity")
        solver_chain: Sequence of solvers to try (default: CLARABEL → SCS → ECOS)

    Returns:
        Optimal weights (N,), or None if all solvers fail

    Raises:
        ValueError: If omega_mode is unknown
    """
    mu_hat = np.asarray(mu_hat, float).reshape(-1)
    N = mu_hat.shape[0]

    if N < 2:
        return None

    # Ensure PSD
    Sigma = make_psd(np.asarray(Sigma, float), eps=1e-10)

    # Compute volatilities
    diagS = np.maximum(np.diag(Sigma), 0.0)
    vol = np.sqrt(np.maximum(diagS, 1e-12))

    # Build uncertainty set matrix A
    if omega_mode == "diagSigma":
        A = np.diag(vol)
    elif omega_mode == "identity":
        A = np.eye(N)
    else:
        raise ValueError(f"Unknown omega_mode={omega_mode}")

    # Formulate problem
    w = cp.Variable(N)
    obj = cp.Maximize(
        mu_hat @ w - kappa * cp.norm(A @ w, 2) - (lambda_ / 2.0) * cp.quad_form(w, Sigma)
    )
    cons = [cp.sum(w) == 1, w >= 0]
    prob = cp.Problem(obj, cons)

    # Try solvers in sequence
    last_err = None
    for solver_name in solver_chain:
        try:
            prob.solve(solver=solver_name, warm_start=True, verbose=False)

            if w.value is not None and prob.status in ("optimal", "optimal_inaccurate"):
                w_hat = np.asarray(w.value).reshape(-1)
                w_hat = np.clip(w_hat, 0, None)
                ss = w_hat.sum()

                # Return normalized weights or equal-weight fallback
                return (w_hat / ss) if ss > 0 else (np.ones(N) / N)

        except Exception as e:
            last_err = e

    # All solvers failed
    return None
