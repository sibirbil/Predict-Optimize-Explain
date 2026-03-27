"""
Feedforward Neural Network (FNN) for asset return prediction.
"""
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple, List, Dict, Any

logger = logging.getLogger(__name__)


class AssetPricingFNN(nn.Module):
    """
    Feedforward Neural Network for predicting asset returns.

    Architecture:
        Input → [32, BN, ReLU, Dropout] →
        [16, BN, ReLU, Dropout] →
        [8, BN, ReLU, Dropout] →
        Output (1)
    """

    def __init__(self, input_dim: int, dropout_rate: float = 0.5):
        """
        Initialize FNN model.

        Args:
            input_dim: Number of input features
            dropout_rate: Dropout probability (default: 0.5)
        """
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.output = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, input_dim)

        Returns:
            Predicted returns (batch_size, 1)
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.output(x)


def load_fnn_from_dir(load_dir: str, device: str = "cpu") -> Tuple[AssetPricingFNN, List[str], Dict[str, Any]]:
    """
    Load pre-trained FNN model with configuration and feature columns.

    Expected files in load_dir:
    - model_config.json: Model hyperparameters
    - feature_columns.json: Ordered list of feature names
    - state_dict.pt: PyTorch state dictionary

    Args:
        load_dir: Directory containing model files
        device: Device to load model onto ('cpu' or 'cuda')

    Returns:
        Tuple of (model, feature_cols, config)

    Raises:
        FileNotFoundError: If required files are missing
    """
    load_dir = Path(load_dir)
    cfg_path = load_dir / "model_config.json"
    cols_path = load_dir / "feature_columns.json"
    state_path = load_dir / "state_dict.pt"

    # Verify all required files exist
    for p in [cfg_path, cols_path, state_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    # Load configuration and feature columns
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    with open(cols_path, "r") as f:
        feature_cols = json.load(f)

    # Load model state
    state = torch.load(state_path, map_location="cpu")

    # Initialize and load model
    model = AssetPricingFNN(
        input_dim=int(cfg["input_dim"]),
        dropout_rate=float(cfg["dropout_rate"])
    )
    model.load_state_dict(state)
    model.eval()

    logger.info(f"Loaded FNN model from {load_dir}")
    logger.debug(f"Model config: {cfg}")
    logger.debug(f"Feature columns: {len(feature_cols)}")

    return model, feature_cols, cfg


@torch.no_grad()
def predict_in_batches(
    model: nn.Module,
    X,
    batch_size: int = 512
) -> np.ndarray:
    """
    Run FNN inference on DataFrame or numpy array in batches.

    Args:
        model: PyTorch model in eval mode
        X: DataFrame or numpy array with features (rows=samples, cols=features)
        batch_size: Batch size for inference (default: 512)

    Returns:
        Array of predictions (n_samples,)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.debug(f"Prediction device: {device}")

    model = model.to(device)
    model.eval()

    # Convert to numpy if DataFrame
    if isinstance(X, pd.DataFrame):
        X_np = X.values
    else:
        X_np = np.asarray(X, dtype=np.float32)

    # Convert to tensor
    Xt = torch.tensor(X_np, dtype=torch.float32)
    ds = torch.utils.data.TensorDataset(Xt)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)

    # Batch prediction
    preds = []
    for (bx,) in dl:
        bx = bx.to(device)
        out = model(bx).detach().cpu().numpy().reshape(-1)
        preds.append(out)

    pred = np.concatenate(preds, axis=0)
    return pred


def r2_oos_zero(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute out-of-sample R² relative to zero benchmark.

    R²_OOS = 1 - SSE_model / SSE_zero
    where SSE_zero = sum((y_true - 0)²)

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Out-of-sample R² (can be negative if model worse than zero)
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)

    sse_model = np.sum((y_true - y_pred) ** 2)
    sse_zero = np.sum(y_true ** 2)

    if sse_zero == 0:
        return float('nan')

    return 1.0 - (sse_model / sse_zero)
