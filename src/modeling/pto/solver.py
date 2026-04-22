from __future__ import annotations

import numpy as np

from src.modeling.pto.covariance import make_psd


def solve_robust_long_only(
    mu: np.ndarray,
    sigma: np.ndarray,
    lambda_: float,
    kappa: float,
    omega_mode: str,
    max_iter: int = 300,
    tol: float = 1e-7,
) -> np.ndarray | None:
    mu = np.asarray(mu, dtype=float).reshape(-1)
    n_assets = mu.shape[0]
    if n_assets < 2:
        return None

    sigma = make_psd(np.asarray(sigma, dtype=float))
    omega = build_omega_matrix(sigma=sigma, omega_mode=omega_mode)
    weights = np.full(n_assets, 1.0 / n_assets, dtype=float)

    lipschitz = float(lambda_ * np.linalg.norm(sigma, ord=2))
    if kappa > 0.0:
        lipschitz += float(kappa * np.linalg.norm(omega, ord=2) ** 2 / 1e-4)
    step = 1.0 / max(lipschitz, 1.0)

    prev_obj = objective_value(weights, mu, sigma, omega, lambda_, kappa)
    for _ in range(max_iter):
        gradient = objective_gradient(weights, mu, sigma, omega, lambda_, kappa)
        candidate = project_to_simplex(weights + step * gradient)
        candidate_obj = objective_value(candidate, mu, sigma, omega, lambda_, kappa)

        if candidate_obj + 1e-12 < prev_obj:
            step *= 0.5
            if step < 1e-10:
                break
            continue

        if np.linalg.norm(candidate - weights, ord=2) < tol:
            weights = candidate
            break

        weights = candidate
        prev_obj = candidate_obj

    weights = np.clip(weights, 0.0, None)
    total_weight = weights.sum()
    if total_weight <= 0:
        return None
    return weights / total_weight


def objective_value(
    weights: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    omega: np.ndarray,
    lambda_: float,
    kappa: float,
) -> float:
    return float(
        mu @ weights
        - float(kappa) * np.linalg.norm(omega @ weights, ord=2)
        - (float(lambda_) / 2.0) * weights @ sigma @ weights
    )


def objective_gradient(
    weights: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    omega: np.ndarray,
    lambda_: float,
    kappa: float,
) -> np.ndarray:
    gradient = mu - float(lambda_) * (sigma @ weights)
    if float(kappa) > 0.0:
        omega_w = omega @ weights
        norm_omega_w = float(np.linalg.norm(omega_w, ord=2))
        if norm_omega_w > 1e-10:
            gradient = gradient - float(kappa) * (omega.T @ omega_w) / norm_omega_w
    return gradient


def build_omega_matrix(sigma: np.ndarray, omega_mode: str) -> np.ndarray:
    normalized = omega_mode.lower()
    if normalized == "identity":
        return np.eye(sigma.shape[0], dtype=float)
    if normalized == "diag_sigma":
        diagonal = np.maximum(np.diag(sigma), 1e-12)
        return np.diag(np.sqrt(diagonal))
    raise ValueError(f"Unknown omega_mode={omega_mode}")


def project_to_simplex(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=float)
    if values.ndim != 1:
        raise ValueError("project_to_simplex expects a 1D vector.")

    sorted_values = np.sort(values)[::-1]
    cssv = np.cumsum(sorted_values) - 1.0
    indices = np.arange(1, len(values) + 1)
    valid = sorted_values - cssv / indices > 0
    if not np.any(valid):
        return np.full(len(values), 1.0 / len(values), dtype=float)
    rho = indices[valid][-1]
    theta = cssv[valid][-1] / rho
    return np.maximum(values - theta, 0.0)
