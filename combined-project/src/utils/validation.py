"""
Data validation and alignment utilities.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple


def patch_targets_inplace(data: Dict[str, Any], clip_lower: float = -0.99) -> None:
    """
    Detect and convert target scale (percentage vs decimal), then clip lower bounds.

    Modifies the data dict in place for keys: y_train, y_val, y_test.

    Args:
        data: Dictionary containing y_train, y_val, y_test as pandas Series
        clip_lower: Lower bound for clipping returns (default: -0.99 = -99% loss)

    Side Effects:
        - Converts percentage scale to decimal (divides by 100) if detected
        - Clips all targets at clip_lower threshold
    """
    print("\n--- [Patch] Targets ---")
    train_mean = float(data["y_train"].mean())

    # Detect scale: if mean magnitude > 0.1, assume percentage scale
    if abs(train_mean) > 0.1:
        print(f">> Detected Percentage Scale (Mean={train_mean:.2f}). Dividing all targets by 100.")
        for key in ["y_train", "y_val", "y_test"]:
            data[key] = data[key] / 100.0
    else:
        print(f">> Detected Decimal Scale (Mean={train_mean:.4f}). No scaling needed.")

    # Clip lower bounds (prevent impossible negative returns)
    for key in ["y_train", "y_val", "y_test"]:
        pre_min = float(data[key].min())
        data[key] = data[key].clip(lower=clip_lower)
        post_min = float(data[key].min())
        print(f"{key}: Min clipped from {pre_min:.4f} to {post_min:.4f}")

    print("\nTarget stats after patch:")
    print(data["y_train"].describe()[["mean", "min", "max", "std"]])


def detect_id_col(df: pd.DataFrame) -> str:
    """
    Auto-detect asset identifier column from common naming conventions.

    Args:
        df: DataFrame to search for ID column

    Returns:
        Name of the detected ID column

    Raises:
        ValueError: If no known ID column found
    """
    candidates = [
        "permno", "PERMNO",
        "asset_id", "id",
        "ticker", "TICKER",
        "gvkey", "GVKEY",
        "cusip", "CUSIP"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not detect an asset id column. Expected one of: {candidates}")


def strict_metadata_alignment(
    metadata: pd.DataFrame,
    train_end: int,
    val_end: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split metadata by YYYYMM cutoffs into train/val/test sets.

    Args:
        metadata: Full metadata DataFrame with 'yyyymm' column
        train_end: Training period end date (YYYYMM format, e.g., 200512)
        val_end: Validation period end date (YYYYMM format, e.g., 201512)

    Returns:
        Tuple of (meta_train, meta_val, meta_test)

    Raises:
        ValueError: If 'yyyymm' column not found in metadata
    """
    if "yyyymm" not in metadata.columns:
        raise ValueError("metadata must contain a 'yyyymm' column.")

    full_meta = metadata.copy()
    full_meta["yyyymm"] = full_meta["yyyymm"].astype(int)

    mask_train = full_meta["yyyymm"] <= int(train_end)
    mask_val = (full_meta["yyyymm"] > int(train_end)) & (full_meta["yyyymm"] <= int(val_end))
    mask_test = full_meta["yyyymm"] > int(val_end)

    meta_train = full_meta[mask_train].copy()
    meta_val = full_meta[mask_val].copy()
    meta_test = full_meta[mask_test].copy()

    return meta_train, meta_val, meta_test


def enforce_alignment_or_slice_fallback(
    data: Dict[str, Any],
    meta_train: pd.DataFrame,
    meta_val: pd.DataFrame,
    meta_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Validate metadata/features/targets alignment across splits, with fallback slicing.

    If row counts don't match, falls back to PTO-style sequential slicing.

    Args:
        data: Dictionary with keys X_train, X_val, X_test, y_train, y_val, y_test, metadata
        meta_train: Train metadata (from strict_metadata_alignment)
        meta_val: Validation metadata
        meta_test: Test metadata

    Returns:
        Tuple of validated (meta_train, meta_val, meta_test)

    Raises:
        ValueError: If metadata doesn't contain 'yyyymm' column
    """
    n_tr = len(data["X_train"])
    n_va = len(data["X_val"])
    n_te = len(data["X_test"])

    # Check if all splits are aligned
    ok = (
        len(meta_train) == n_tr and
        len(meta_val) == n_va and
        len(meta_test) == n_te and
        len(data["y_train"]) == n_tr and
        len(data["y_val"]) == n_va and
        len(data["y_test"]) == n_te
    )

    if ok:
        return meta_train, meta_val, meta_test

    # Fallback: sequential slicing (PTO-style)
    print(">> WARNING: meta/X/y mismatch; applying PTO-style slicing fallback for ALL splits.")
    full_meta = data["metadata"].copy().reset_index(drop=True)

    meta_train = full_meta.iloc[:n_tr].copy()
    meta_val = full_meta.iloc[n_tr:n_tr + n_va].copy()
    meta_test = full_meta.iloc[n_tr + n_va:n_tr + n_va + n_te].copy()

    # Validate yyyymm column exists
    for m in (meta_train, meta_val, meta_test):
        if "yyyymm" not in m.columns:
            raise ValueError("metadata must contain 'yyyymm'.")
        m["yyyymm"] = m["yyyymm"].astype(int)

    return meta_train, meta_val, meta_test


def align_features(X_df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Align DataFrame columns to match expected feature list.

    Args:
        X_df: DataFrame with feature columns
        feature_cols: Ordered list of expected feature column names

    Returns:
        DataFrame with columns reordered to match feature_cols

    Raises:
        ValueError: If any expected features are missing from X_df
    """
    missing = [c for c in feature_cols if c not in X_df.columns]
    if missing:
        raise ValueError(f"Missing {len(missing)} feature columns, e.g. {missing[:10]}")
    return X_df.loc[:, feature_cols]
