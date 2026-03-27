"""
E2E Cache Builder Module.

Builds month-level cache for PAO training with pre-computed:
- Universe selection (top-k assets per month)
- FNN predictions
- Covariance matrices (EWMA with shrinkage)
- Realized returns

Cache invalidation based on configuration hash.
"""
from pathlib import Path
from typing import List, Dict, Any
import time

import numpy as np
import pandas as pd
import torch.nn as nn

try:
    from ..utils.io import save_json, load_json, sha1_of_list, safe_rm_tree
    from ..utils.dates import shift_yyyymm
    from ..utils.validation import align_features
    from ..optimization.risk import (
        ewma_cov_full_history_matrix,
        sigma_vol_from_cov,
        compute_sigma_factor_for_risk
    )
    from ..portfolio.selection import select_universe_pto_style
except ImportError:
    from utils.io import save_json, load_json, sha1_of_list, safe_rm_tree
    from utils.dates import shift_yyyymm
    from utils.validation import align_features
    from optimization.risk import (
        ewma_cov_full_history_matrix,
        sigma_vol_from_cov,
        compute_sigma_factor_for_risk
    )
    from portfolio.selection import select_universe_pto_style


def expected_cache_manifest(
    split_name: str,
    id_col: str,
    feature_cols: List[str],
    fnn_dir: str,
    cache_version: str,
    topk: int,
    preselect_factor: int,
    lookback: int,
    lam: float,
    shrink: float,
    ridge: float,
    clip_lower: float,
) -> Dict[str, Any]:
    """
    Generate expected cache manifest for validation.

    Cache is invalidated if any of these parameters change.

    Args:
        split_name: Split name (train/val/test)
        id_col: Asset identifier column
        feature_cols: Feature column names
        fnn_dir: FNN model directory path
        cache_version: Cache version string (e.g. "v2.0")
        topk: Number of assets to select
        preselect_factor: Pre-selection factor for universe
        lookback: Lookback window for covariance
        lam: EWMA decay parameter
        shrink: Shrinkage intensity
        ridge: Ridge regularization
        clip_lower: Target clipping threshold

    Returns:
        Dictionary with cache configuration
    """
    return {
        "cache_version": str(cache_version),
        "split": split_name,
        "id_col": id_col,
        "fnn_dir": fnn_dir,
        "feature_cols_sha1": sha1_of_list(feature_cols),
        "topk": int(topk),
        "preselect_factor": int(preselect_factor),
        "lookback": int(lookback),
        "lam": float(lam),
        "shrink": float(shrink),
        "ridge": float(ridge),
        "clip_lower": float(clip_lower),
    }


def manifest_matches(existing: dict, expected: dict) -> bool:
    """
    Check if existing cache manifest matches expected configuration.

    Args:
        existing: Loaded manifest from cache
        expected: Expected manifest from current config

    Returns:
        True if all expected keys match existing values
    """
    for k, v in expected.items():
        if existing.get(k) != v:
            return False
    return True


def get_universe_sizes(
    universe_sizes: List[int] = None,
    universe_seed: int = 123,
    universe_min: int = 20,
    universe_max: int = 200,
    universe_draws: int = 10,
) -> List[int]:
    """
    Get list of universe sizes for PAO training.

    If universe_sizes is provided, use those values.
    Otherwise, randomly sample from [universe_min, universe_max].

    Args:
        universe_sizes: Explicit list of universe sizes (optional)
        universe_seed: Random seed for sampling
        universe_min: Minimum universe size
        universe_max: Maximum universe size
        universe_draws: Number of random draws

    Returns:
        Sorted list of unique universe sizes
    """
    if universe_sizes is not None and len(universe_sizes) > 0:
        sizes = [int(x) for x in universe_sizes]
    else:
        rng = np.random.default_rng(int(universe_seed))
        sizes = rng.integers(
            int(universe_min),
            int(universe_max) + 1,
            size=int(universe_draws),
        ).tolist()

    # Sanitize: keep only valid sizes, unique, sorted
    sizes = [s for s in sizes if int(universe_min) <= int(s) <= int(universe_max)]
    sizes = sorted(set(int(s) for s in sizes))
    return sizes


def build_month_cache(
    split_name: str,
    X_df: pd.DataFrame,
    y_ser: pd.Series,
    meta_split: pd.DataFrame,
    returns_wide: pd.DataFrame,
    fnn_model: nn.Module,
    feature_cols: List[str],
    id_col: str,
    fnn_dir: str,
    out_dir: Path,
    cache_version: str,
    topk: int,
    preselect_factor: int,
    lookback: int,
    lam: float,
    shrink: float,
    ridge: float,
    clip_lower: float,
    pred_batch_size: int = 2048,
) -> Path:
    """
    Build month-level cache for PAO training.

    For each month in the split:
    1. Select top-k universe using FNN predictions
    2. Compute EWMA covariance on lookback window
    3. Extract features, realized returns, risk factors
    4. Save to .npz file

    Cache is validated against manifest. If configuration changed, cache is rebuilt.

    Args:
        split_name: Split name (train/val/test)
        X_df: Feature DataFrame for this split
        y_ser: Target Series for this split
        meta_split: Metadata DataFrame for this split
        returns_wide: Wide-format returns (months × assets)
        fnn_model: Pre-trained FNN model for predictions
        feature_cols: Feature column names
        id_col: Asset identifier column
        fnn_dir: FNN model directory path
        out_dir: Output directory for cache
        cache_version: Cache version string (e.g. "v2.0")
        topk: Number of assets to select
        preselect_factor: Pre-selection factor
        lookback: Lookback window for covariance
        lam: EWMA decay parameter
        shrink: Shrinkage intensity
        ridge: Ridge regularization
        clip_lower: Target clipping threshold
        pred_batch_size: Batch size for FNN predictions

    Returns:
        Path to cache directory

    Side Effects:
        Creates cache directory with:
        - manifest.json: Configuration
        - YYYYMM.npz: Monthly data (X, y, Sigma_factor, sigma_vol, fnn_preds)
        - months.csv: Summary of prepared months
    """
    cache_dir = out_dir / "month_cache" / f"topk_{int(topk)}" / split_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"

    # Strict alignment check for this split
    if len(X_df) != len(y_ser) or len(X_df) != len(meta_split):
        raise ValueError(
            f"[{split_name}] X/y/meta mismatch: "
            f"X={len(X_df)} y={len(y_ser)} meta={len(meta_split)}"
        )

    expected = expected_cache_manifest(
        split_name=split_name,
        id_col=id_col,
        feature_cols=feature_cols,
        fnn_dir=fnn_dir,
        cache_version=cache_version,
        topk=topk,
        preselect_factor=preselect_factor,
        lookback=lookback,
        lam=lam,
        shrink=shrink,
        ridge=ridge,
        clip_lower=clip_lower,
    )

    # Check if cache is valid
    if manifest_path.exists():
        existing = load_json(manifest_path)
        if manifest_matches(existing, expected):
            print(f"[{split_name}] month_cache valid -> skipping rebuild")
            return cache_dir
        else:
            print(f"[{split_name}] month_cache invalid (config changed) -> rebuilding")
            safe_rm_tree(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)

    # Prepare metadata
    meta = meta_split.reset_index(drop=True).copy()
    meta["yyyymm"] = meta["yyyymm"].astype(int)

    months = sorted(meta["yyyymm"].unique().tolist())
    n_prepared, n_skipped = 0, 0
    rows = []
    t0 = time.time()

    for i, t in enumerate(months):
        # Select universe for this month
        sel = select_universe_pto_style(
            t=t,
            meta_split=meta,
            X_split=X_df,
            returns_wide=returns_wide,
            fnn_model=fnn_model,
            feature_cols=feature_cols,
            id_col=id_col,
            topk=topk,
            preselect_factor=preselect_factor,
            lookback=lookback,
            pred_batch_size=pred_batch_size,
        )
        if sel is None:
            n_skipped += 1
            continue

        idx_sel, assets, fnn_preds = sel

        # Extract features and targets
        X_t = align_features(X_df.iloc[idx_sel], feature_cols).values.astype(np.float32)
        y_t = y_ser.iloc[idx_sel].values.astype(np.float32)

        # Get historical returns for covariance
        start = shift_yyyymm(int(t), -lookback)
        end = shift_yyyymm(int(t), -1)
        window = returns_wide.loc[
            (returns_wide.index >= start) & (returns_wide.index <= end), assets
        ]

        if window.shape[0] < lookback:
            n_skipped += 1
            continue

        R = window.values.astype(np.float64)
        if np.isnan(R).any():
            n_skipped += 1
            continue

        # Compute covariance
        Sigma = ewma_cov_full_history_matrix(
            R, lam=lam, shrink=shrink, ridge=ridge, psd_proj=True
        )
        if Sigma is None:
            n_skipped += 1
            continue

        # Compute risk factors
        Sigma_factor = compute_sigma_factor_for_risk(Sigma, eps=1e-8)
        sigma_vol = sigma_vol_from_cov(Sigma)

        # Save month cache
        np.savez_compressed(
            cache_dir / f"{int(t)}.npz",
            yyyymm=np.int32(t),
            assets=assets,
            X=X_t,
            y=y_t,
            Sigma_factor=Sigma_factor,
            sigma_vol=sigma_vol,
            fnn_preds=fnn_preds,
        )

        n_prepared += 1
        rows.append({
            "yyyymm": int(t),
            "n_assets": int(topk),
            "window_rows": int(window.shape[0]),
        })

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            print(
                f"  [{split_name}] {i+1}/{len(months)} months | "
                f"prepared={n_prepared} | skipped={n_skipped} | "
                f"elapsed={elapsed:.1f}s"
            )

    # Save manifest and summary
    expected["n_prepared"] = int(n_prepared)
    expected["n_skipped"] = int(n_skipped)
    expected["feature_dim"] = int(len(feature_cols))
    save_json(manifest_path, expected)
    pd.DataFrame(rows).to_csv(cache_dir / "months.csv", index=False)

    elapsed = time.time() - t0
    print(
        f"[{split_name}] Prepared={n_prepared} | Skipped={n_skipped} | "
        f"Total time={elapsed:.1f}s"
    )
    return cache_dir
