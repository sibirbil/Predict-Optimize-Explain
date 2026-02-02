"""
Score network for PAO portfolio optimization.
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple


class PAOScoreNetwork(nn.Module):
    """
    Neural network for predicting expected returns in E2E framework.

    Flexible architecture with configurable hidden dimensions.
    Uses Kaiming initialization for better gradient flow.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (32, 16, 8),
        dropout: float = 0.5
    ):
        """
        Initialize score network.

        Args:
            input_dim: Number of input features
            hidden_dims: Tuple of hidden layer dimensions (default: (32, 16, 8))
            dropout: Dropout probability (default: 0.5)
        """
        super().__init__()

        # Build sequential network
        layers = []
        prev = input_dim

        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h

        # Output layer (no activation)
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

        # Kaiming initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, input_dim)

        Returns:
            Predicted mu values (batch_size,) - squeezed output
        """
        return self.net(x).squeeze(-1)


def compute_mu_reference(
    y_train: np.ndarray,
    mu_ref_mode: str = "winsor_std",
    mu_abs_quantile: float = 0.75,
    mu_winsor_p_low: float = 0.01,
    mu_winsor_p_high: float = 0.99
) -> float:
    """
    Compute reference scale for mu transformation.

    Three modes:
    1. raw_std: Standard deviation of returns
    2. abs_quantile: Quantile of absolute values
    3. winsor_std: Std of winsorized returns (recommended)

    Args:
        y_train: Training returns array
        mu_ref_mode: Reference mode ("raw_std", "abs_quantile", "winsor_std")
        mu_abs_quantile: Quantile for abs_quantile mode (default: 0.75)
        mu_winsor_p_low: Lower percentile for winsorization (default: 0.01)
        mu_winsor_p_high: Upper percentile for winsorization (default: 0.99)

    Returns:
        Reference scale (float, minimum 1e-12)

    Raises:
        ValueError: If mu_ref_mode is unknown
    """
    y = np.asarray(y_train, float)

    if mu_ref_mode == "raw_std":
        ref = float(np.std(y, ddof=1))
    elif mu_ref_mode == "abs_quantile":
        ref = float(np.quantile(np.abs(y), mu_abs_quantile))
    elif mu_ref_mode == "winsor_std":
        lo = float(np.quantile(y, mu_winsor_p_low))
        hi = float(np.quantile(y, mu_winsor_p_high))
        yw = np.clip(y, lo, hi)
        ref = float(np.std(yw, ddof=1))
    else:
        raise ValueError(f"Unknown mu_ref_mode={mu_ref_mode}")

    return max(ref, 1e-12)
