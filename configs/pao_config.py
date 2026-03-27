"""
PAO Training Configuration.

Configuration for Predict-and-Optimize differentiable portfolio optimization including:
- Universe size sweeps
- Training hyperparameters
- Loss functions
- Model architecture
- Cache settings
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

@dataclass
class PAOConfig:
    """
    Configuration for PAO portfolio optimization training.

    Attributes:
        Paths:
            data_dir: Directory with processed data
            fnn_dir: Directory with pre-trained FNN model
            out_dir: Output directory for PAO results

        Data Splits:
            train_end: Training period end (YYYYMM)
            val_end: Validation period end (YYYYMM)

        Universe Configuration:
            universe_min: Minimum universe size
            universe_max: Maximum universe size
            universe_draws: Number of random universe sizes to sample
            universe_seed: Random seed for universe size sampling
            universe_sizes: Explicit list of universe sizes (overrides random)

        Portfolio Parameters:
            topk: Number of assets in portfolio
            preselect_factor: Pre-selection multiplier
            lookback: Lookback window for covariance (months)

        Covariance Parameters:
            lam: EWMA decay parameter
            shrink: Shrinkage intensity
            ridge: Ridge regularization

        Optimization Grids:
            kappa_grid: Robustness parameters to test
            omega_modes: Uncertainty structures
            lambda_grid: Risk aversion parameters to test

        Training Configuration:
            loss_types: Loss functions to train ("return", "utility", "sharpe")
            epochs: Maximum training epochs
            patience: Early stopping patience
            lr: Learning rate
            weight_decay: AdamW weight decay
            grad_clip: Gradient clipping threshold
            sharpe_batch: Batch size for Sharpe loss (months)
            month_batch: Batch size for return/utility loss (months)

        Model Architecture:
            hidden_dims: Hidden layer sizes for score network
            dropout: Dropout rate

        Mu Transform:
            mu_transform: Prediction transformation ("raw", "zscore", "tanh_cap")
            mu_ref_mode: Reference scale computation ("raw_std", "winsor_std", "abs_quantile")
            mu_winsor_p_low: Lower percentile for winsorization
            mu_winsor_p_high: Upper percentile for winsorization
            mu_abs_quantile: Quantile for absolute value scaling
            mu_scale_mult: Scale multiplier
            mu_scale_min: Minimum scale
            mu_scale_max: Maximum scale
            mu_cap_mult: Cap multiplier for tanh
            mu_cap_min: Minimum cap
            mu_cap_max: Maximum cap

        Other:
            clip_lower: Target clipping threshold
            seed: Random seed
            device: Device for training ("cpu" or "cuda")
            pred_batch_size: Batch size for FNN inference
            cache_version: Cache version tag
            force_retrain: Force retraining even if model exists
            auto_retrain_on_selection_mismatch: Retrain if checkpoint selection metric differs
    """

    # ============================================================
    # Paths
    # ============================================================
    data_dir: str = "./data/processed/ready_data"
    fnn_dir: str = "./models/fnn_v1"
    out_dir: str = "./outputs/pao/results"

    # ============================================================
    # Data Splits
    # ============================================================
    train_end: int = 200512  # December 2005
    val_end: int = 201512    # December 2015, test starts 201601

    # ============================================================
    # Universe Size Sweep
    # ============================================================
    universe_min: int = 20
    universe_max: int = 100
    universe_draws: int = 20
    universe_seed: int = 123
    universe_sizes: Optional[List[int]] = None  # If set, overrides random draws

    # ============================================================
    # Portfolio Parameters (PTO-compatible)
    # ============================================================
    topk: int = 200
    preselect_factor: int = 3
    lookback: int = 60

    # ============================================================
    # EWMA Covariance Parameters (PTO-compatible)
    # ============================================================
    lam: float = 0.94
    shrink: float = 0.10
    ridge: float = 1e-6

    # ============================================================
    # Optimization Grids
    # ============================================================
    kappa_grid: List[float] = field(default_factory=lambda: [0.0, 0.1, 1.0])
    omega_modes: List[str] = field(default_factory=lambda: ["diagSigma", "identity"])
    lambda_grid: List[float] = field(default_factory=lambda: [5.0, 10.0, 20.0])

    # ============================================================
    # Training Objectives
    # ============================================================
    loss_types: List[str] = field(default_factory=lambda: ["return", "utility", "sharpe"])

    # ============================================================
    # Training Hyperparameters
    # ============================================================
    epochs: int = 50
    patience: int = 10
    lr: float = 5e-5
    weight_decay: float = 1e-5
    grad_clip: float = 5.0
    sharpe_batch: int = 12
    month_batch: int = 12

    # ============================================================
    # Model Architecture
    # ============================================================
    hidden_dims: Tuple[int, ...] = (32, 16, 8)
    dropout: float = 0.5

    # ============================================================
    # Target Processing
    # ============================================================
    clip_lower: float = -0.99

    # ============================================================
    # Determinism and Device
    # ============================================================
    seed: int = 42
    device: str = "cpu"

    # ============================================================
    # FNN Inference
    # ============================================================
    pred_batch_size: int = 512

    # ============================================================
    # Mu Transform Configuration
    # ============================================================

    # Transformation mode: "raw" | "zscore" | "tanh_cap"
    # - "raw": Use predictor outputs directly
    # - "zscore": Cross-sectional z-score each month, then scale
    # - "tanh_cap": Bounded outputs with tanh
    mu_transform: str = "zscore"

    # Reference scale computation for mu_transform
    # - "raw_std": Standard deviation of y_train
    # - "winsor_std": Standard deviation after winsorization
    # - "abs_quantile": Quantile of absolute values
    mu_ref_mode: str = "winsor_std"
    mu_winsor_p_low: float = 0.01
    mu_winsor_p_high: float = 0.99
    mu_abs_quantile: float = 0.75

    # Scale parameters
    mu_scale_mult: float = 1.0
    mu_scale_min: float = 0.002   # Monthly scale lower bound
    mu_scale_max: float = 0.05    # Monthly scale upper bound

    # Cap parameters (for tanh_cap mode)
    mu_cap_mult: float = 2.0
    mu_cap_min: float = 0.005
    mu_cap_max: float = 0.10

    # ============================================================
    # Cache Configuration
    # ============================================================
    cache_version: str = "v2.0"  # Bump to invalidate cache

    # ============================================================
    # Training Control
    # ============================================================
    force_retrain: bool = False
    auto_retrain_on_selection_mismatch: bool = True


def get_default_config() -> PAOConfig:
    """
    Get default E2E configuration.

    Returns:
        PAOConfig with default parameters
    """
    return PAOConfig()


def get_config_with_overrides(**kwargs) -> PAOConfig:
    """
    Get E2E configuration with custom overrides.

    Args:
        **kwargs: Configuration parameters to override

    Returns:
        PAOConfig with specified overrides

    Example:
        config = get_config_with_overrides(
            topk=100,
            lambda_grid=[3.0, 5.0],
            loss_types=["utility"]
        )
    """
    config = PAOConfig()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown config parameter: {key}")
    return config
