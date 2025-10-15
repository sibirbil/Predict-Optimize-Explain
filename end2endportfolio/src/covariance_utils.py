#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for constructing return panels and covariance matrices from ticker data.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.data_utils import FILL_VALUE


def build_ticker_return_panel(
    splits: Sequence[Tuple[np.ndarray, Sequence[np.ndarray], Optional[Sequence[Optional[np.ndarray]]]]],
) -> Optional[pd.DataFrame]:
    """
    Construct a date-indexed dataframe of ticker returns aggregated across dataset splits.

    Args:
        splits: Sequence of tuples (dates, returns_list, tickers_list), where each item corresponds
                to a panel (train/valid/test). The returns_list and tickers_list must be aligned with
                the dates array.

    Returns:
        pd.DataFrame indexed by date with columns per ticker. Missing combinations remain NaN.
        Returns None if no usable records are found.
    """
    records: List[Tuple[pd.Timestamp, str, float]] = []
    for dates, returns_list, tickers_list in splits:
        if dates is None or len(dates) == 0:
            continue
        if tickers_list is None:
            continue
        for date, ret_vec, tick_vec in zip(dates, returns_list, tickers_list):
            if tick_vec is None or len(tick_vec) == 0:
                continue
            tick_arr = np.asarray(tick_vec)
            ret_arr = np.asarray(ret_vec, dtype=float)
            if tick_arr.shape[0] != ret_arr.shape[0]:
                continue
            mask = np.isfinite(ret_arr)
            mask &= ret_arr != FILL_VALUE
            if not mask.any():
                continue
            date_ts = pd.to_datetime(date)
            for ticker, value in zip(tick_arr[mask], ret_arr[mask]):
                records.append((date_ts, str(ticker), float(value)))

    if not records:
        return None

    df = pd.DataFrame(records, columns=["date", "ticker", "return"])
    pivot = df.pivot_table(index="date", columns="ticker", values="return", aggfunc="first")
    pivot.sort_index(inplace=True)
    return pivot


def covariance_from_ticker_panel(
    tickers: Sequence[str],
    ticker_panel: Optional[pd.DataFrame],
    *,
    min_periods: int = 3,
    ridge: float = 1e-4,
) -> Optional[np.ndarray]:
    """
    Derive a covariance matrix for the supplied tickers using a historical ticker return panel.

    Args:
        tickers: Sequence of ticker symbols to include.
        ticker_panel: DataFrame returned by build_ticker_return_panel.
        min_periods: Minimum overlapping observations required for covariance estimation.
        ridge: Diagonal ridge added for numerical stability.

    Returns:
        Covariance matrix (np.ndarray) or None if insufficient data.
    """
    if ticker_panel is None or len(tickers) == 0:
        return None

    tickers = [str(t) for t in tickers]
    available = [t for t in tickers if t in ticker_panel.columns]
    if not available:
        return None

    subset = ticker_panel[available]
    subset = subset.dropna(how="all")
    if subset.empty:
        return None

    cov_df = subset.cov(min_periods=min_periods)
    n = len(tickers)
    cov_matrix = np.zeros((n, n), dtype=np.float64)

    for i, ti in enumerate(tickers):
        for j, tj in enumerate(tickers):
            if ti in cov_df.index and tj in cov_df.columns:
                val = cov_df.loc[ti, tj]
                if np.isfinite(val):
                    cov_matrix[i, j] = float(val)

    # Ensure reasonable diagonals when variance is missing/too small
    for idx, ticker in enumerate(tickers):
        if cov_matrix[idx, idx] == 0.0:
            if ticker in subset.columns:
                series = subset[ticker].dropna()
                var = float(series.var(ddof=1)) if series.size >= max(2, min_periods) else 1.0
            else:
                var = 1.0
            if not np.isfinite(var) or var <= 0.0:
                var = 1.0
            cov_matrix[idx, idx] = var

    cov_matrix = np.nan_to_num(cov_matrix, nan=0.0)
    cov_matrix += ridge * np.eye(n, dtype=np.float64)
    return cov_matrix
