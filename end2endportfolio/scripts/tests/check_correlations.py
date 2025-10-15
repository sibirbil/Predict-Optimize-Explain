#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Report the most (positively/negatively) correlated tickers across panels."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("KMP_CREATE_SHARED", "0")

import argparse
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys  # noqa: E402

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import clean_missing_xy, load_panel  # noqa: E402
from src.covariance_utils import build_ticker_return_panel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List top correlated ticker pairs.")
    parser.add_argument("--train", default="data/panel_train.npz", help="Training panel")
    parser.add_argument("--valid", default="data/panel_valid.npz", help="Validation panel")
    parser.add_argument("--test", default="data/panel_test.npz", help="Test panel")
    parser.add_argument("--min-periods", type=int, default=12, help="Minimum overlap for correlation.")
    parser.add_argument("--top", type=int, default=20, help="Number of top pairs to display.")
    return parser.parse_args()


def load_split(npz_path: Path):
    panel = load_panel(str(npz_path))
    X_list, y_list, _, tickers_list = clean_missing_xy(panel.X, panel.y, panel.asset_ids, panel.tickers)
    return panel.dates, y_list, tickers_list


def flatten_pairs(matrix: pd.DataFrame) -> List[Tuple[float, float, str, str]]:
    abs_corr = matrix.abs()
    mask = np.triu(np.ones_like(abs_corr, dtype=bool), k=1)
    vals: List[Tuple[float, float, str, str]] = []
    rows = abs_corr.index.to_list()
    cols = abs_corr.columns.to_list()
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            if mask[i, j]:
                value = abs_corr.iat[i, j]
                if not np.isnan(value):
                    vals.append((value, matrix.iat[i, j], r, c))
    return vals


def main() -> None:
    args = parse_args()

    splits = []
    for path in (args.train, args.valid, args.test):
        panel_path = PROJECT_ROOT / path
        if panel_path.exists():
            splits.append(load_split(panel_path))

    ticker_panel = build_ticker_return_panel(splits)
    if ticker_panel is None or ticker_panel.empty:
        raise RuntimeError("No ticker data available to compute correlations.")

    corr = ticker_panel.corr(min_periods=args.min_periods)
    corr = corr.dropna(how="all", axis=0).dropna(how="all", axis=1)

    pairs = flatten_pairs(corr)
    pairs.sort(reverse=True, key=lambda x: x[0])

    top_n = args.top
    print(f"Top {top_n} most positively correlated pairs:")
    for abs_val, corr_val, a, b in pairs[:top_n]:
        print(f"{a:6s} - {b:6s} => corr={corr_val:.3f}")

    neg_pairs = [p for p in pairs if p[1] < 0]
    neg_pairs.sort(key=lambda x: x[1])
    if neg_pairs:
        print(f"\nTop {min(top_n, len(neg_pairs))} most negatively correlated pairs:")
        for abs_val, corr_val, a, b in neg_pairs[:top_n]:
            print(f"{a:6s} - {b:6s} => corr={corr_val:.3f}")
    else:
        print("\nNo negative correlations found with the specified minimum overlap.")


if __name__ == "__main__":
    main()
