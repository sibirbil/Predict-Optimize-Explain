"""
Universe selection for portfolio construction.
"""
import numpy as np
import pandas as pd
import torch.nn as nn
from typing import Optional, Tuple, List

try:
    from ..utils.dates import shift_yyyymm
    from ..utils.validation import align_features
    from ..models.fnn import predict_in_batches
except ImportError:
    # Fallback for direct imports
    from utils.dates import shift_yyyymm
    from utils.validation import align_features
    from models.fnn import predict_in_batches


def select_universe_pto_style(
    t: int,
    meta_split: pd.DataFrame,
    X_split: pd.DataFrame,
    returns_wide: pd.DataFrame,
    fnn_model: nn.Module,
    feature_cols: List[str],
    id_col: str,
    topk: int,
    preselect_factor: int,
    lookback: int,
    pred_batch_size: int = 512
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Select portfolio universe using PTO-style FNN pre-selection + full-history filter.

    Selection pipeline:
    1. Predict returns for all assets at month t using FNN
    2. Pre-select top (topk × preselect_factor) assets by prediction
    3. Filter to assets with complete lookback-month history
    4. Select final topk from eligible assets

    Args:
        t: Decision month (YYYYMM format)
        meta_split: Metadata for split (train/val/test)
        X_split: Features for split
        returns_wide: Wide-format return matrix (index=yyyymm, columns=asset_ids)
        fnn_model: Pre-trained FNN model for prediction
        feature_cols: Ordered list of feature names
        id_col: Asset identifier column name
        topk: Final number of assets to select
        preselect_factor: Pre-selection multiplier (e.g., 3 → select top 3×topk first)
        lookback: Number of months of history required
        pred_batch_size: Batch size for FNN inference (default: 512)

    Returns:
        Tuple of (sel_idx, sel_assets, sel_preds):
        - sel_idx: Row indices in X_split (topk,)
        - sel_assets: Asset IDs (topk,)
        - sel_preds: FNN predictions (topk,)
        Or None if cannot form valid universe

    Note:
        Assets must have NO missing returns in lookback window [t-lookback, t-1]
    """
    preK = int(topk * preselect_factor)

    # Filter to month t
    mask_t = (meta_split["yyyymm"].astype(int) == int(t))
    if mask_t.sum() == 0:
        return None

    # Get predictions for month t
    idx_t = np.where(mask_t.values)[0]
    X_t_aligned = align_features(X_split.iloc[idx_t], feature_cols)
    preds = predict_in_batches(fnn_model, X_t_aligned.values, batch_size=pred_batch_size)
    ids = meta_split.iloc[idx_t][id_col].values

    # Create prediction DataFrame
    df = pd.DataFrame({"row_idx": idx_t, "asset": ids, "pred": preds})

    # Handle duplicates: keep highest prediction per asset
    if df["asset"].duplicated().any():
        df = df.sort_values("pred", ascending=False)
        df = df.drop_duplicates(subset=["asset"], keep="first")

    # Check if enough assets for pre-selection
    if len(df) < preK:
        return None

    # Pre-select top (topk × preselect_factor)
    df_pre = df.nlargest(preK, "pred", keep="first")

    # Define lookback window [t-lookback, t-1]
    start = shift_yyyymm(int(t), -lookback)
    end = shift_yyyymm(int(t), -1)
    window = returns_wide.loc[(returns_wide.index >= start) & (returns_wide.index <= end)]

    # Check if sufficient historical data
    if window.shape[0] < lookback:
        return None

    # Filter to pre-selected assets with available history
    pre_assets = df_pre["asset"].tolist()
    valid_cols = [a for a in pre_assets if a in window.columns]
    if len(valid_cols) < topk:
        return None

    # Identify assets with complete history (no NaN in lookback window)
    hist_pre = window[valid_cols]
    eligible_assets = hist_pre.columns[hist_pre.notna().all(axis=0)].tolist()

    if len(eligible_assets) < topk:
        return None

    # Final selection: top topk from eligible assets
    df_final = df_pre[df_pre["asset"].isin(eligible_assets)].nlargest(topk, "pred", keep="first")

    if len(df_final) < topk:
        return None

    # Extract results
    sel_idx = df_final["row_idx"].values.astype(np.int64)
    sel_assets = df_final["asset"].values
    sel_preds = df_final["pred"].values.astype(np.float32)

    return sel_idx, sel_assets, sel_preds
