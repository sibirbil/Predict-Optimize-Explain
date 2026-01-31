"""
Data loading utilities for building return matrices and aligning data.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any

logger = logging.getLogger(__name__)


def build_returns_wide(
    meta_train: pd.DataFrame,
    meta_val: pd.DataFrame,
    meta_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    id_col: str
) -> pd.DataFrame:
    """
    Build wide-format return matrix for covariance lookback windows.

    Combines train/val/test returns into a single matrix indexed by yyyymm
    with columns for each asset.

    Args:
        meta_train: Training metadata with yyyymm and id_col
        meta_val: Validation metadata
        meta_test: Test metadata
        y_train: Training returns
        y_val: Validation returns
        y_test: Test returns
        id_col: Asset identifier column name

    Returns:
        DataFrame with:
        - Index: yyyymm (months)
        - Columns: asset IDs
        - Values: returns (float32)

    Warns:
        If duplicate (yyyymm, asset_id) combinations found
    """
    # Build train component
    df_tr = meta_train[[id_col, "yyyymm"]].copy()
    df_tr["ret"] = y_train.values.astype(float)

    # Build validation component
    df_va = meta_val[[id_col, "yyyymm"]].copy()
    df_va["ret"] = y_val.values.astype(float)

    # Build test component
    df_te = meta_test[[id_col, "yyyymm"]].copy()
    df_te["ret"] = y_test.values.astype(float)

    # Concatenate all splits
    df_all = pd.concat([df_tr, df_va, df_te], ignore_index=True)
    df_all["yyyymm"] = df_all["yyyymm"].astype(int)

    # Check for duplicates
    dup_rate = df_all.duplicated(subset=["yyyymm", id_col]).mean()
    if dup_rate > 0:
        logger.warning(f"Duplicate (yyyymm, asset_id) entries found: {dup_rate:.6f} rate")

    # Pivot to wide format
    wide = df_all.pivot(index="yyyymm", columns=id_col, values="ret").sort_index()
    wide = wide.astype(np.float32)

    logger.info(f"Built returns_wide: shape={wide.shape}, range={wide.index.min()}-{wide.index.max()}")
    return wide
