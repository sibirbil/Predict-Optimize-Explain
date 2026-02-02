"""
Differentiable optimization layers for predict-and-optimize portfolio learning.
"""
import torch
import torch.nn as nn
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer


class DifferentiableMVOLayer(nn.Module):
    """
    Differentiable Markowitz Mean-Variance Optimization layer.

    Problem:
        min -μ'w + (λ/2) ||Uw||²
        s.t. Σw = 1, w ≥ 0

    where ||Uw||² = w'Σw (U is the Cholesky/sqrt factor of Σ).

    Enables backpropagation through portfolio optimization for predict-and-optimize learning.
    """

    def __init__(self, n_assets: int, lambda_: float):
        """
        Initialize differentiable MVO layer.

        Args:
            n_assets: Number of assets in universe
            lambda_: Risk aversion coefficient
        """
        super().__init__()
        n = int(n_assets)

        # Define optimization variables and parameters
        w = cp.Variable(n, nonneg=True)
        mu = cp.Parameter(n)
        U = cp.Parameter((n, n))

        # Formulate problem
        risk = cp.sum_squares(U @ w)  # = w'Σw
        obj = cp.Minimize(-mu @ w + (lambda_ / 2.0) * risk)
        cons = [cp.sum(w) == 1]
        prob = cp.Problem(obj, cons)

        # Create differentiable layer
        self.layer = CvxpyLayer(prob, parameters=[mu, U], variables=[w])
        self.n_assets = n

    def forward(self, mu: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        """
        Solve MVO and return optimal weights.

        Args:
            mu: Expected returns (n_assets,)
            U: Cholesky factor where U'U = Σ (n_assets, n_assets)

        Returns:
            Optimal portfolio weights (n_assets,)
        """
        try:
            # Solve with double precision for numerical stability
            (w,) = self.layer(mu.double(), U.double())
            w = w.float()

            # Clip negative weights and renormalize
            w = torch.clamp(w, min=0.0)
            return w / (w.sum() + 1e-12)

        except Exception as e:
            # Fallback to equal weights if solver fails (rare, not differentiable)
            print(f"[MVO solver fail] {e}")
            return torch.ones(self.n_assets, device=mu.device) / self.n_assets


class DifferentiableRobustMVOLayer(nn.Module):
    """
    Differentiable Robust Mean-Variance Optimization layer with uncertainty penalty.

    Problem:
        min -μ'w + κ||Aw||₂ + (λ/2) ||Uw||²
        s.t. Σw = 1, w ≥ 0

    where:
    - ||Uw||² = w'Σw (risk term)
    - ||Aw||₂ = uncertainty penalty (A = diag(σ) or I)
    """

    def __init__(self, n_assets: int, lambda_: float, kappa: float):
        """
        Initialize differentiable robust MVO layer.

        Args:
            n_assets: Number of assets in universe
            lambda_: Risk aversion coefficient
            kappa: Robustness penalty coefficient
        """
        super().__init__()
        n = int(n_assets)

        # Define optimization variables and parameters
        w = cp.Variable(n, nonneg=True)
        mu = cp.Parameter(n)
        U = cp.Parameter((n, n))  # U'U = Σ
        A = cp.Parameter((n, n))  # Uncertainty set matrix

        # Formulate problem
        risk = cp.sum_squares(U @ w)
        obj = cp.Minimize(-mu @ w + kappa * cp.norm(A @ w, 2) + (lambda_ / 2.0) * risk)
        cons = [cp.sum(w) == 1]
        prob = cp.Problem(obj, cons)

        # Create differentiable layer
        self.layer = CvxpyLayer(prob, parameters=[mu, U, A], variables=[w])
        self.n_assets = n

    def forward(
        self,
        mu: torch.Tensor,
        U: torch.Tensor,
        A: torch.Tensor
    ) -> torch.Tensor:
        """
        Solve robust MVO and return optimal weights.

        Args:
            mu: Expected returns (n_assets,)
            U: Cholesky factor where U'U = Σ (n_assets, n_assets)
            A: Uncertainty set matrix (n_assets, n_assets)

        Returns:
            Optimal portfolio weights (n_assets,)
        """
        try:
            # Solve with double precision for numerical stability
            (w,) = self.layer(mu.double(), U.double(), A.double())
            w = w.float()

            # Clip negative weights and renormalize
            w = torch.clamp(w, min=0.0)
            return w / (w.sum() + 1e-12)

        except Exception as e:
            # Fallback to equal weights if solver fails (rare, not differentiable)
            print(f"[Robust MVO solver fail] {e}")
            return torch.ones(self.n_assets, device=mu.device) / self.n_assets
