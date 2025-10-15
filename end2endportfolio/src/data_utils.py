#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for loading panel-style asset data and preparing training snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

FILL_VALUE = -99.99
DROP_FEATURES = {"ME_mil"}  # redundant with ME_dollars


@dataclass
class PanelData:
    X: np.ndarray  # shape (T, N, D)
    y: np.ndarray  # shape (T, N)
    dates: np.ndarray  # shape (T,)
    feature_names: np.ndarray  # shape (D,)
    asset_ids: Optional[np.ndarray] = None  # shape (T, N), strings or ints
    tickers: Optional[np.ndarray] = None  # shape (T, N)


def load_panel(npz_path: str) -> PanelData:
    """
    Load an NPZ panel in either the new triplet format or the legacy Chen format.
    Returns a PanelData object preserving asset identifiers when available.
    """
    z = np.load(npz_path, allow_pickle=False)
    files = set(z.files)

    asset_ids = None
    tickers = None
    T = N = None

    if {"data", "y", "dates"}.issubset(files):
        X = z["data"].astype(np.float32)
        y = z["y"].astype(np.float32)
        dates = z["dates"]
        T, N = X.shape[:2]
        feature_names = (
            z["feature_names"].astype(str)
            if "feature_names" in files
            else np.array([f"f{i}" for i in range(X.shape[-1])], dtype="<U32")
        )
        for key in ("permno", "firm_id", "firmIds", "asset_id", "gvkey"):
            if key in files:
                asset_ids = z[key]
                break
        if "ticker" in files:
            tickers = z["ticker"]
    elif {"data", "date"}.issubset(files):
        arr = z["data"].astype(np.float32)
        y = arr[..., 0]
        X = arr[..., 1:]
        dates = z["date"]
        T, N = X.shape[:2]
        if "variable" in files:
            vars_ = z["variable"].astype(str).tolist()
            feature_names = np.array([v for v in vars_ if v.lower() != "ret"], dtype="<U32")
            if feature_names.size != X.shape[-1]:
                feature_names = np.array([f"f{i}" for i in range(X.shape[-1])], dtype="<U32")
        else:
            feature_names = np.array([f"f{i}" for i in range(X.shape[-1])], dtype="<U32")
        if "permno" in files:
            asset_ids = z["permno"]
        if "ticker" in files:
            tickers = z["ticker"]
    else:
        raise KeyError(f"Unrecognized NPZ schema for {npz_path}. Keys={sorted(z.files)}")

    def _align_panel_ids(arr: Optional[np.ndarray], target_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        if arr is None:
            return None
        arr_np = np.asarray(arr)
        if arr_np.ndim == 1:
            if arr_np.shape[0] == target_shape[1]:
                return np.tile(arr_np, (target_shape[0], 1))
            if arr_np.shape[0] == target_shape[0]:
                return np.repeat(arr_np[:, None], target_shape[1], axis=1)
            return None
        if arr_np.ndim == 2 and arr_np.shape == target_shape:
            return arr_np
        return None

    if isinstance(feature_names, np.ndarray):
        keep_mask = np.ones(feature_names.shape[0], dtype=bool)
        for feature in DROP_FEATURES:
            if feature in feature_names:
                keep_mask &= feature_names != feature
        if not np.all(keep_mask):
            X = X[..., keep_mask]
            feature_names = feature_names[keep_mask]

    if T is None or N is None:
        T, N = X.shape[:2]

    asset_ids = _align_panel_ids(asset_ids, (T, N))
    tickers = _align_panel_ids(tickers, (T, N))

    asset_ids = asset_ids.astype(str) if asset_ids is not None else None
    tickers = tickers.astype(str) if tickers is not None else None

    return PanelData(X=X, y=y, dates=dates, feature_names=feature_names, asset_ids=asset_ids, tickers=tickers)


def clean_missing_xy(
    X_3d: np.ndarray,
    y_2d: np.ndarray,
    asset_ids_2d: Optional[np.ndarray] = None,
    tickers_2d: Optional[np.ndarray] = None,
    *,
    name: str = "dataset",
    fill_value: float = FILL_VALUE,
) -> Tuple[
    List[np.ndarray],
    List[np.ndarray],
    Optional[List[np.ndarray]],
    Optional[List[np.ndarray]],
]:
    """
    Remove rows containing the fill sentinel or non-finite targets for each timestamp.
    Returns lists of feature matrices, return vectors, and optional asset IDs.
    """
    T, _, _ = X_3d.shape
    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    id_list: Optional[List[np.ndarray]] = [] if asset_ids_2d is not None else None
    ticker_list: Optional[List[np.ndarray]] = [] if tickers_2d is not None else None

    for t in range(T):
        X_t = X_3d[t]
        y_t = y_2d[t]
        mask = np.isfinite(y_t)
        mask &= np.all(np.isfinite(X_t), axis=1)
        if fill_value is not None:
            mask &= np.all(X_t != fill_value, axis=1)
            mask &= y_t != fill_value

        X_filtered = X_t[mask]
        y_filtered = y_t[mask]
        X_list.append(X_filtered)
        y_list.append(y_filtered)

        if id_list is not None:
            ids_t = asset_ids_2d[t][mask]
            id_list.append(ids_t.astype(str))

        if ticker_list is not None:
            tickers_t = tickers_2d[t][mask]
            ticker_list.append(tickers_t.astype(str))

    return X_list, y_list, id_list, ticker_list


def scale_features(
    X_train_list: List[np.ndarray],
    X_valid_list: List[np.ndarray],
    X_test_list: List[np.ndarray],
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], StandardScaler]:
    """
    Fit a StandardScaler on concatenated training features and transform all splits.
    """
    scaler = StandardScaler()
    all_train_X = np.vstack(X_train_list)
    scaler.fit(all_train_X)

    def _transform(items: List[np.ndarray]) -> List[np.ndarray]:
        return [scaler.transform(x) for x in items]

    return _transform(X_train_list), _transform(X_valid_list), _transform(X_test_list), scaler


ReturnHistory = Dict[str, Dict[str, float]]


def build_return_history(
    asset_ids_2d: Optional[np.ndarray],
    dates: np.ndarray,
    returns_2d: np.ndarray,
    *,
    fill_value: float = FILL_VALUE,
    history: Optional[ReturnHistory] = None,
) -> ReturnHistory:
    """Construct or update a mapping asset_id -> {date -> return}."""
    if history is None:
        history = {}
    if asset_ids_2d is None:
        return history

    date_values = dates.astype(str)
    T, N = returns_2d.shape
    for t in range(T):
        ids_row = asset_ids_2d[t]
        ret_row = returns_2d[t]
        date_str = str(date_values[t])
        for idx in range(N):
            asset_id = str(ids_row[idx])
            ret = ret_row[idx]
            if not np.isfinite(ret):
                continue
            if fill_value is not None and ret == fill_value:
                continue
            history.setdefault(asset_id, {})[date_str] = float(ret)
    return history


def covariance_from_history(
    asset_ids: Sequence[str],
    history: ReturnHistory,
    *,
    ridge: float = 1e-4,
    min_overlap: int = 3,
) -> np.ndarray:
    """
    Estimate a covariance matrix for the requested assets using overlapping monthly returns.
    """
    asset_ids = [str(a) for a in asset_ids]
    if not asset_ids:
        return np.zeros((0, 0), dtype=np.float64)

    intersect_dates: Optional[set] = None
    for asset in asset_ids:
        asset_hist = history.get(asset)
        if not asset_hist:
            continue
        dates = set(asset_hist.keys())
        intersect_dates = dates if intersect_dates is None else intersect_dates & dates

    used_dates = sorted(intersect_dates) if intersect_dates else []

    def _diagonal_covariance() -> np.ndarray:
        variances = []
        for asset in asset_ids:
            asset_hist = history.get(asset, {})
            values = np.array(list(asset_hist.values()), dtype=np.float64)
            if values.size < 2:
                variances.append(1.0)
            else:
                variances.append(float(np.var(values, ddof=1)))
        return np.diag(np.array(variances, dtype=np.float64))

    if len(used_dates) < min_overlap:
        # Fall back to diagonal variance using each asset's own history
        cov = _diagonal_covariance()
    else:
        rows: List[List[float]] = []
        fallback = False
        for asset in asset_ids:
            asset_hist = history.get(asset)
            if not asset_hist:
                fallback = True
                break
            try:
                rows.append([asset_hist[date] for date in used_dates])
            except KeyError:
                fallback = True
                break
        if fallback:
            cov = _diagonal_covariance()
        else:
            matrix = np.array(rows, dtype=np.float64)
            centered = matrix - matrix.mean(axis=1, keepdims=True)
            denom = max(len(used_dates) - 1, 1)
            cov = centered @ centered.T / denom

    cov += ridge * np.eye(len(asset_ids))
    return cov
