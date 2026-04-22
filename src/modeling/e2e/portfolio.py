from __future__ import annotations

import warnings
from typing import Any

import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer

_LAYER_CACHE: dict[tuple[int, float, float, str], CvxpyLayer] = {}


def build_omega_matrix(sigma: torch.Tensor, omega_mode: str) -> torch.Tensor:
    normalized = omega_mode.lower()
    if normalized == "identity":
        return torch.eye(sigma.shape[0], device=sigma.device, dtype=sigma.dtype)
    if normalized == "diag_sigma":
        diagonal = torch.clamp(torch.diag(sigma), min=1e-12)
        return torch.diag(torch.sqrt(diagonal))
    raise ValueError(f"Unknown omega_mode={omega_mode}")


def build_omega_diagonal(sigma: torch.Tensor, omega_mode: str) -> torch.Tensor:
    normalized = omega_mode.lower()
    if normalized == "identity":
        return torch.ones(sigma.shape[0], device=sigma.device, dtype=sigma.dtype)
    if normalized == "diag_sigma":
        return torch.sqrt(torch.clamp(torch.diag(sigma), min=1e-12))
    raise ValueError(f"Unknown omega_mode={omega_mode}")


def build_covariance_factor(sigma: torch.Tensor) -> torch.Tensor:
    eigenvalues, eigenvectors = torch.linalg.eigh(sigma)
    clipped = torch.clamp(eigenvalues, min=1e-12)
    return torch.diag(torch.sqrt(clipped)) @ eigenvectors.transpose(0, 1)


def build_exact_cvxpy_layer(
    n_assets: int,
    lambda_: float,
    kappa: float,
    omega_mode: str,
) -> CvxpyLayer:
    cache_key = (n_assets, float(lambda_), float(kappa), str(omega_mode))
    if cache_key in _LAYER_CACHE:
        return _LAYER_CACHE[cache_key]

    weights = cp.Variable(n_assets)
    mu = cp.Parameter(n_assets)
    covariance_factor = cp.Parameter((n_assets, n_assets))
    omega_diagonal = cp.Parameter(n_assets, nonneg=True)
    upper_bounds = cp.Parameter(n_assets, nonneg=True)

    risk_term = 0.5 * float(lambda_) * cp.sum_squares(covariance_factor @ weights)
    robustness_term = float(kappa) * cp.norm(cp.multiply(omega_diagonal, weights), 2)
    objective = cp.Maximize(mu @ weights - robustness_term - risk_term)
    problem = cp.Problem(objective, [weights >= 0, weights <= upper_bounds, cp.sum(weights) == 1])
    if not problem.is_dpp():
        raise ValueError("Exact E2E portfolio layer is not DPP-compliant.")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Your problem has too many parameters for efficient DPP compilation.*")
        layer = CvxpyLayer(
            problem,
            parameters=[mu, covariance_factor, omega_diagonal, upper_bounds],
            variables=[weights],
        )
    _LAYER_CACHE[cache_key] = layer
    return layer


def robust_mirror_descent_weights(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    lambda_: float,
    kappa: float,
    omega_mode: str,
    steps: int,
    step_size: float,
) -> torch.Tensor:
    mu = mu.reshape(-1)
    sigma = sigma.to(dtype=mu.dtype, device=mu.device)
    n_assets = mu.shape[0]
    weights = torch.full((n_assets,), 1.0 / n_assets, dtype=mu.dtype, device=mu.device)
    omega = build_omega_matrix(sigma=sigma, omega_mode=omega_mode)

    for _ in range(steps):
        sigma_w = sigma @ weights
        gradient = mu - float(lambda_) * sigma_w
        if float(kappa) > 0.0:
            omega_w = omega @ weights
            norm = torch.linalg.norm(omega_w, ord=2)
            if float(norm.detach().cpu()) > 1e-10:
                gradient = gradient - float(kappa) * (omega.transpose(0, 1) @ omega_w) / norm.clamp_min(1e-10)
        logits = torch.log(weights.clamp_min(1e-12)) + float(step_size) * gradient
        weights = torch.softmax(logits, dim=0)
    return weights


def project_to_simplex_torch(vector: torch.Tensor) -> torch.Tensor:
    if vector.ndim != 1:
        raise ValueError("project_to_simplex_torch expects a 1D vector.")
    sorted_values, _ = torch.sort(vector, descending=True)
    cssv = torch.cumsum(sorted_values, dim=0) - 1.0
    indices = torch.arange(1, vector.numel() + 1, device=vector.device, dtype=vector.dtype)
    valid = sorted_values - cssv / indices > 0
    if not bool(valid.any()):
        return torch.full_like(vector, 1.0 / vector.numel())
    rho = int(torch.nonzero(valid, as_tuple=False)[-1].item() + 1)
    theta = cssv[rho - 1] / indices[rho - 1]
    return torch.clamp(vector - theta, min=0.0)


def robust_projected_gradient_weights(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    lambda_: float,
    kappa: float,
    omega_mode: str,
    steps: int,
    step_scale: float,
    tol: float,
) -> torch.Tensor:
    mu = mu.reshape(-1)
    sigma = sigma.to(dtype=mu.dtype, device=mu.device)
    n_assets = mu.shape[0]
    weights = torch.full((n_assets,), 1.0 / n_assets, dtype=mu.dtype, device=mu.device)
    omega = build_omega_matrix(sigma=sigma, omega_mode=omega_mode)
    omega_diag = build_omega_diagonal(sigma=sigma, omega_mode=omega_mode)

    lipschitz = float(lambda_) * float(torch.linalg.norm(sigma, ord=2).detach().cpu())
    if float(kappa) > 0.0:
        lipschitz += float(kappa) * float(torch.max(omega_diag.square()).detach().cpu()) / 1e-4
    base_step = float(step_scale) / max(lipschitz, 1.0)
    previous_objective = robust_preference(
        weights=weights,
        realized_excess_returns=mu,
        sigma=sigma,
        lambda_=lambda_,
        kappa=kappa,
        omega_mode=omega_mode,
    )

    for _ in range(steps):
        sigma_w = sigma @ weights
        gradient = mu - float(lambda_) * sigma_w
        if float(kappa) > 0.0:
            omega_w = omega @ weights
            norm = torch.linalg.norm(omega_w, ord=2).clamp_min(1e-10)
            gradient = gradient - float(kappa) * (omega.transpose(0, 1) @ omega_w) / norm

        step = base_step
        candidate = project_to_simplex_torch(weights + step * gradient)
        candidate_objective = robust_preference(
            weights=candidate,
            realized_excess_returns=mu,
            sigma=sigma,
            lambda_=lambda_,
            kappa=kappa,
            omega_mode=omega_mode,
        )
        halving_steps = 0
        while bool(candidate_objective + 1e-12 < previous_objective) and halving_steps < 8:
            step *= 0.5
            candidate = project_to_simplex_torch(weights + step * gradient)
            candidate_objective = robust_preference(
                weights=candidate,
                realized_excess_returns=mu,
                sigma=sigma,
                lambda_=lambda_,
                kappa=kappa,
                omega_mode=omega_mode,
            )
            halving_steps += 1

        if torch.linalg.norm(candidate - weights, ord=2) < float(tol):
            weights = candidate
            break
        weights = candidate
        previous_objective = candidate_objective
    return weights


def robust_exact_cvxpy_weights(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    lambda_: float,
    kappa: float,
    omega_mode: str,
    fixed_n_assets: int | None = None,
) -> torch.Tensor:
    mu = mu.reshape(-1)
    original_device = mu.device
    sigma_cpu = sigma.to(device="cpu", dtype=torch.float32)
    mu_cpu = mu.to(device="cpu", dtype=torch.float32)
    n_active = int(mu_cpu.shape[0])
    target_n = int(fixed_n_assets) if fixed_n_assets is not None else n_active
    if target_n < n_active:
        raise ValueError(f"fixed_n_assets={fixed_n_assets} is smaller than active asset count {n_active}.")

    if target_n > n_active:
        mu_padded = torch.zeros(target_n, dtype=torch.float32)
        mu_padded[:n_active] = mu_cpu
        sigma_padded = torch.zeros((target_n, target_n), dtype=torch.float32)
        sigma_padded[:n_active, :n_active] = sigma_cpu
        sigma_padded[n_active:, n_active:] = torch.eye(target_n - n_active, dtype=torch.float32) * 1e-6
        upper_bounds = torch.zeros(target_n, dtype=torch.float32)
        upper_bounds[:n_active] = 1.0
    else:
        mu_padded = mu_cpu
        sigma_padded = sigma_cpu
        upper_bounds = torch.ones(target_n, dtype=torch.float32)

    factor_cpu = build_covariance_factor(sigma_padded)
    omega_diag_cpu = build_omega_diagonal(sigma_padded, omega_mode=omega_mode)
    layer = build_exact_cvxpy_layer(
        n_assets=target_n,
        lambda_=lambda_,
        kappa=kappa,
        omega_mode=omega_mode,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Your problem has too many parameters for efficient DPP compilation.*")
        warnings.filterwarnings("ignore", message="Converting A to a CSC \\(compressed sparse column\\) matrix; may take a while\\.")
        solution, = layer(
            mu_padded,
            factor_cpu,
            omega_diag_cpu,
            upper_bounds,
            solver_args={"max_iters": 100000, "eps": 1e-8},
        )
    return solution[:n_active].to(device=original_device, dtype=mu.dtype)


def robust_portfolio_weights(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    lambda_: float,
    kappa: float,
    omega_mode: str,
    solver_mode: str,
    steps: int,
    step_size: float,
    fixed_n_assets: int | None = None,
) -> torch.Tensor:
    normalized = solver_mode.lower()
    if normalized == "mirror_descent":
        return robust_mirror_descent_weights(
            mu=mu,
            sigma=sigma,
            lambda_=lambda_,
            kappa=kappa,
            omega_mode=omega_mode,
            steps=steps,
            step_size=step_size,
        )
    if normalized == "exact_cvxpy":
        return robust_exact_cvxpy_weights(
            mu=mu,
            sigma=sigma,
            lambda_=lambda_,
            kappa=kappa,
            omega_mode=omega_mode,
            fixed_n_assets=fixed_n_assets,
        )
    if normalized == "projected_gradient":
        return robust_projected_gradient_weights(
            mu=mu,
            sigma=sigma,
            lambda_=lambda_,
            kappa=kappa,
            omega_mode=omega_mode,
            steps=max(steps, 50),
            step_scale=step_size,
            tol=1e-7,
        )
    raise ValueError(f"Unknown solver_mode={solver_mode}")


def weights_to_numpy(weights: torch.Tensor) -> Any:
    return weights.detach().cpu().numpy()


def robust_preference(
    weights: torch.Tensor,
    realized_excess_returns: torch.Tensor,
    sigma: torch.Tensor,
    lambda_: float,
    kappa: float,
    omega_mode: str,
) -> torch.Tensor:
    portfolio_excess = torch.dot(weights, realized_excess_returns)
    omega = build_omega_matrix(sigma=sigma, omega_mode=omega_mode)
    robustness_penalty = float(kappa) * torch.linalg.norm(omega @ weights, ord=2)
    risk_penalty = (float(lambda_) / 2.0) * (weights @ sigma @ weights)
    return portfolio_excess - robustness_penalty - risk_penalty
