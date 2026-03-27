"""
PTO Backtest Configuration.

Configuration for Predict-Then-Optimize backtesting including:
- Universe selection parameters
- Portfolio optimization parameters
- Robustness grids
- Evaluation settings
"""
from pathlib import Path
from typing import List

# ============================================================
# Paths
# ============================================================

# Base directory
BASE_DIR = Path(".")

# Data directory
DATA_DIR = BASE_DIR / "data" / "processed" / "ready_data"

# FNN model directory
FNN_DIR = BASE_DIR / "models" / "fnn_v1"

# Output directory
OUT_DIR = BASE_DIR / "outputs" / "pto" / "results"

# ============================================================
# Data Splits
# ============================================================

# Training period: up to and including December 2005
TRAIN_END = 200512

# Validation period: January 2006 to December 2015
VAL_END = 201512

# Test period: January 2016 onwards (>= 201601)

# ============================================================
# Universe Selection
# ============================================================

# Number of assets to hold in portfolio
TOPK = 200

# Pre-selection factor (select TOPK * PRESELECT_FACTOR before optimization)
# Reduces computational burden while maintaining diversification
PRESELECT_FACTOR = 3

# Lookback window for covariance estimation (months)
LOOKBACK = 60

# Minimum number of assets required (if fewer, use equal-weight fallback)
MIN_ASSETS = 10

# ============================================================
# Covariance Estimation (EWMA)
# ============================================================

# EWMA decay parameter (λ)
# Higher values (e.g., 0.94) give more weight to recent observations
LAM = 0.94

# Shrinkage intensity toward diagonal (Ledoit-Wolf style)
# Helps with numerical stability and estimation error
SHRINK = 0.10

# Ridge regularization added to diagonal
# Ensures positive definiteness
RIDGE = 1e-6

# Whether to project to PSD after shrinkage
PSD_PROJ = True

# ============================================================
# Portfolio Optimization
# ============================================================

# Risk aversion parameter (λ)
# Objective: max μ'w - (λ/2) w'Σw  s.t. sum(w)=1, w≥0
LAMBDA = 5.0

# ============================================================
# Robust Optimization Grids
# ============================================================

# Uncertainty structure modes
# - "diagSigma": Ω = diag(Σ), uncertainty proportional to variance
# - "identity": Ω = I, uniform uncertainty
OMEGA_MODES: List[str] = ["diagSigma", "identity"]

# Robustness parameter (κ)
# Higher values increase robustness to prediction errors
# κ=0 reduces to standard MVO
KAPPA_GRID: List[float] = [0.0, 0.1, 0.5, 1.0, 10.0]

# ============================================================
# Inference Settings
# ============================================================

# Batch size for FNN predictions
BATCH_SIZE_PRED = 512

# ============================================================
# Evaluation Settings
# ============================================================

# Rolling window for Sharpe ratio calculation (months)
ROLL = 36

# Whether to compute rolling metrics
COMPUTE_ROLLING = True

# ============================================================
# Execution Control
# ============================================================

# Whether to save detailed results
SAVE_DETAILED = True

# Whether to save monthly weights
SAVE_WEIGHTS = True

# Whether to print progress
VERBOSE = True

# Device for PyTorch inference
DEVICE = "cpu"
