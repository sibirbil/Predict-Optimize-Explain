#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick diagnostic for covariance estimation using ticker history."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys  # noqa: E402

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import clean_missing_xy, load_panel  # noqa: E402
from src.covariance_utils import (  # noqa: E402
    build_ticker_return_panel,
    covariance_from_ticker_panel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect ticker-based covariance matrices.")
    parser.add_argument("--train", default="data/panel_train.npz", help="Training panel path")
    parser.add_argument("--valid", default="data/panel_valid.npz", help="Validation panel path")
    parser.add_argument("--test", default="data/panel_test.npz", help="Test panel path")
    parser.add_argument("--tickers", nargs="+", required=True, help="Ticker symbols to examine")
    parser.add_argument(
        "--date",
        default=None,
        help="Optional ISO date; covariance uses data up to this date (inclusive).",
    )
    parser.add_argument(
        "--min-periods",
        type=int,
        default=6,
        help="Minimum overlap when computing covariance (default=6).",
    )
    return parser.parse_args()


def load_split(panel_path: Path):
    panel = load_panel(str(panel_path))
    X_list, y_list, _, tickers_list = clean_missing_xy(
        panel.X, panel.y, panel.asset_ids, panel.tickers
    )
    return panel.dates, y_list, tickers_list


def main() -> None:
    args = parse_args()

    splits = []
    for path in (args.train, args.valid, args.test):
        panel_path = PROJECT_ROOT / path
        if not panel_path.exists():
            continue
        splits.append(load_split(panel_path))

    ticker_panel = build_ticker_return_panel(splits)
    if ticker_panel is None:
        raise RuntimeError("No ticker history could be assembled; check panel/ticker availability.")

    if args.date is not None:
        ticker_panel = ticker_panel.loc[:args.date]

    cov = covariance_from_ticker_panel(args.tickers, ticker_panel, min_periods=args.min_periods)
    if cov is None:
        raise RuntimeError("Insufficient overlap to compute covariance for supplied tickers.")

    eigvals = np.linalg.eigvalsh(cov)
    print("Tickers:", ", ".join(args.tickers))
    print("Covariance matrix:\n", cov)
    print("Eigenvalues:", eigvals)
    if eigvals.min() < 0:
        print("[warn] Covariance is not positive semi-definite (numerical jitter?).")


if __name__ == "__main__":
    main()
