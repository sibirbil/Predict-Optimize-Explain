"""
Predict-and-Optimize Portfolio Model combining prediction and optimization.
"""
import torch
import torch.nn as nn
from typing import Tuple

try:
    from .score_network import PAOScoreNetwork
    from ..optimization.layers import DifferentiableMVOLayer, DifferentiableRobustMVOLayer
except ImportError:
    # Fallback for direct imports
    from models.score_network import PAOScoreNetwork
    from optimization.layers import DifferentiableMVOLayer, DifferentiableRobustMVOLayer


class PAOPortfolioModel(nn.Module):
    """
    Complete E2E portfolio optimization model.

    Combines:
    1. Score network: X → μ_raw
    2. Mu transformation: μ_raw → μ (zscore/tanh_cap/raw)
    3. Differentiable optimization: (μ, Σ, A) → w

    Supports both standard MVO (kappa=0) and robust MVO (kappa>0).
    """

    def __init__(
        self,
        input_dim: int,
        n_assets: int,
        lambda_: float,
        kappa: float,
        omega_mode: str,
        hidden_dims: Tuple[int, ...] = (32, 16, 8),
        dropout: float = 0.5,
        mu_transform: str = "zscore",
        mu_scale: float = 0.01,
        mu_cap: float = 0.05
    ):
        """
        Initialize E2E portfolio model.

        Args:
            input_dim: Number of input features per asset
            n_assets: Number of assets in universe (topK)
            lambda_: Risk aversion coefficient
            kappa: Robustness penalty (0 = standard MVO, >0 = robust)
            omega_mode: Uncertainty set mode ("identity" or "diagSigma")
            hidden_dims: Score network architecture (default: (32, 16, 8))
            dropout: Dropout rate (default: 0.5)
            mu_transform: Return transformation ("raw", "zscore", "tanh_cap")
            mu_scale: Scale for zscore transform (default: 0.01)
            mu_cap: Cap for tanh transform (default: 0.05)
        """
        super().__init__()
        self.n_assets = int(n_assets)
        self.lambda_ = float(lambda_)
        self.kappa = float(kappa)
        self.omega_mode = str(omega_mode)

        self.mu_transform = str(mu_transform)
        self.mu_scale = float(mu_scale)
        self.mu_cap = float(mu_cap)

        # Score network for predicting returns
        self.predictor = PAOScoreNetwork(input_dim, hidden_dims, dropout)

        # Differentiable optimization layer
        if self.kappa > 0:
            self.opt = DifferentiableRobustMVOLayer(n_assets=n_assets, lambda_=lambda_, kappa=kappa)
        else:
            self.opt = DifferentiableMVOLayer(n_assets=n_assets, lambda_=lambda_)

    def _build_A(self, sigma_vol: torch.Tensor) -> torch.Tensor:
        """
        Build uncertainty set matrix A.

        Args:
            sigma_vol: Individual asset volatilities (n_assets,)

        Returns:
            Uncertainty matrix A (n_assets, n_assets)
            - "identity": A = I
            - "diagSigma": A = diag(σ)

        Raises:
            ValueError: If omega_mode is unknown
        """
        if self.omega_mode == "identity":
            return torch.eye(self.n_assets, dtype=torch.float32, device=sigma_vol.device)
        if self.omega_mode == "diagSigma":
            return torch.diag(sigma_vol)
        raise ValueError(f"Unknown omega_mode={self.omega_mode}")

    def _transform_mu(self, mu_raw: torch.Tensor) -> torch.Tensor:
        """
        Transform raw predicted returns.

        Three modes:
        1. "raw": No transformation (mu = mu_raw)
        2. "zscore": Z-score normalization (mu = scale × (mu_raw - mean) / std)
        3. "tanh_cap": Bounded transform (mu = cap × tanh(mu_raw))

        Args:
            mu_raw: Raw predictions from score network (n_assets,)

        Returns:
            Transformed returns (n_assets,)

        Raises:
            ValueError: If mu_transform is unknown
        """
        if self.mu_transform == "raw":
            return mu_raw

        if self.mu_transform == "zscore":
            m = mu_raw.mean()
            s = mu_raw.std(unbiased=True) + 1e-12
            z = (mu_raw - m) / s
            return self.mu_scale * z

        if self.mu_transform == "tanh_cap":
            return self.mu_cap * torch.tanh(mu_raw)

        raise ValueError(f"Unknown mu_transform={self.mu_transform}")

    def forward(
        self,
        X: torch.Tensor,
        Sigma_factor: torch.Tensor,
        sigma_vol: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: predict returns and optimize portfolio.

        Args:
            X: Features (n_assets, input_dim)
            Sigma_factor: Risk factor matrix U where w'Σw = ||Uw||² (n_assets, n_assets)
            sigma_vol: Individual volatilities (n_assets,)

        Returns:
            Tuple of (w, mu_raw):
            - w: Optimal portfolio weights (n_assets,)
            - mu_raw: Raw predictions before transformation (n_assets,)
        """
        # Predict raw returns
        mu_raw = self.predictor(X)  # (n_assets,)

        # Transform returns
        mu = self._transform_mu(mu_raw)  # (n_assets,)

        # Solve optimization
        if self.kappa > 0:
            A = self._build_A(sigma_vol)
            w = self.opt(mu, Sigma_factor, A)
        else:
            w = self.opt(mu, Sigma_factor)

        return w, mu_raw
