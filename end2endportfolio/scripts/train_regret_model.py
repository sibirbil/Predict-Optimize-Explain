#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train FNN on panel triplets (works with new triplets or Chen-style files).
"""

import json
import sys
import numpy as np
import torch
from pathlib import Path

# ---------- project-relative paths ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import (  # noqa: E402
    load_panel,
    clean_missing_xy,
    scale_features,
    build_return_history,
)
from src.model import FNN  # noqa: E402
from src.train import train_regret_model  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"

TRAIN_NPZ = DATA_DIR / "panel_train.npz"
VALID_NPZ = DATA_DIR / "panel_valid.npz"
TEST_NPZ = DATA_DIR / "panel_test.npz"
MODEL_PATH = MODELS_DIR / "trained_model.pth"
FEATURE_MASK_PATH = ARTIFACTS_DIR / "feature_mask.json"
VARIANCE_THRESHOLD = 5e-4


def _apply_feature_mask(X_list, mask):
    return [x[:, mask] for x in X_list]

# ---------- 1) Load panels (handles both schemas) ----------
train_panel = load_panel(str(TRAIN_NPZ))
valid_panel = load_panel(str(VALID_NPZ))
test_panel = load_panel(str(TEST_NPZ))

# ---------- 2) Clean missing jointly on X and y ----------
X_train_list, y_train_list, asset_ids_train, _tickers_train = clean_missing_xy(
    train_panel.X, train_panel.y, train_panel.asset_ids, train_panel.tickers, name="train"
)
X_valid_list, y_valid_list, asset_ids_valid, _tickers_valid = clean_missing_xy(
    valid_panel.X, valid_panel.y, valid_panel.asset_ids, valid_panel.tickers, name="valid"
)
X_test_list, y_test_list, asset_ids_test, _tickers_test = clean_missing_xy(
    test_panel.X, test_panel.y, test_panel.asset_ids, test_panel.tickers, name="test"
)

if not X_train_list:
    raise RuntimeError("No usable training snapshots after cleaning.")

feature_names = train_panel.feature_names
if not isinstance(feature_names, np.ndarray) or feature_names.shape[0] != X_train_list[0].shape[1]:
    feature_names = np.array([f"f{i}" for i in range(X_train_list[0].shape[1])])

train_matrix = np.vstack(X_train_list)
variances = train_matrix.var(axis=0, ddof=1)
mask = variances >= VARIANCE_THRESHOLD
if not np.any(mask):
    raise RuntimeError("Variance threshold removed all features; lower VARIANCE_THRESHOLD.")

if not np.all(mask):
    dropped = feature_names[~mask]
    print(f"[info] Dropping low-variance features (<{VARIANCE_THRESHOLD}): {', '.join(dropped)}")
else:
    dropped = []

feature_names = feature_names[mask]
variances = variances[mask]
X_train_list = _apply_feature_mask(X_train_list, mask)
if X_valid_list:
    X_valid_list = _apply_feature_mask(X_valid_list, mask)
if X_test_list:
    X_test_list = _apply_feature_mask(X_test_list, mask)

# ---------- 3) Fit scaler on TRAIN X only and transform ----------
X_train_list, X_valid_list, X_test_list, scaler = scale_features(
    X_train_list, X_valid_list, X_test_list
)

# ---------- 4) Build model with correct input dim ----------
D = len(feature_names)

mask_payload = {
    "variance_threshold": VARIANCE_THRESHOLD,
    "kept_features": feature_names.tolist(),
    "dropped_features": [str(f) for f in dropped],
    "variances": {str(name): float(var) for name, var in zip(feature_names, variances)},
}
FEATURE_MASK_PATH.parent.mkdir(parents=True, exist_ok=True)
with FEATURE_MASK_PATH.open("w", encoding="utf-8") as f_mask:
    json.dump(mask_payload, f_mask, indent=2)
print(f"[info] Feature mask saved to {FEATURE_MASK_PATH}")

model = FNN(input_dim=D)

# ---------- 5) Training hyperparams ----------
lambda_ = 100.0
epochs = 5
learning_rate = 1e-3
batch_size = 30
batches_per_timestamp = 10

# ---------- 6) Build return history for covariance estimation ----------
return_history = build_return_history(train_panel.asset_ids, train_panel.dates, train_panel.y)

# ---------- 7) Train ----------
train_regret_model(
    model=model,
    lambda_=lambda_,
    X_train_list=X_train_list,
    y_train_list=y_train_list,
    X_valid_list=X_valid_list,
    y_valid_list=y_valid_list,
    X_test_list=X_test_list,
    y_test_list=y_test_list,
    asset_ids_train_list=asset_ids_train,
    asset_ids_valid_list=asset_ids_valid,
    asset_ids_test_list=asset_ids_test,
    return_history=return_history,
    epochs=epochs,
    lr=learning_rate,
    batch_size=batch_size,
    batches_per_timestamp=batches_per_timestamp,
)

# ---------- 7) Save + quick reload sanity ----------
MODELS_DIR.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model training complete. Parameters saved to {MODEL_PATH}.")

# Optional: verify load works with same input_dim
_reloaded = FNN(input_dim=D)
_reloaded.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
_reloaded.eval()
