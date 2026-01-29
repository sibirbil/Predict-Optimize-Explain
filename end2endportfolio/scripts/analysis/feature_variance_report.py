#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Report feature variance across NPZ panels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import load_panel, clean_missing_xy  # noqa: E402


def load_features(npz_path: Path) -> pd.DataFrame:
    panel = load_panel(str(npz_path))
    X_list, _, _, _ = clean_missing_xy(panel.X, panel.y, panel.asset_ids, panel.tickers)
    if not X_list:
        return pd.DataFrame()
    X = np.vstack(X_list)
    return pd.DataFrame(X, columns=panel.feature_names)


def aggregate_features(paths: List[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        if not p.exists():
            print(f"[warn] missing {p}, skipping.")
            continue
        df = load_features(p)
        if df.empty:
            print(f"[warn] no usable data in {p}, skipping.")
            continue
        df["source"] = p.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=0, ignore_index=True)


def summarize(features: pd.DataFrame) -> pd.DataFrame:
    numeric = features.drop(columns=["source"], errors="ignore")
    stats = pd.DataFrame({
        "variance": numeric.var(axis=0, ddof=1),
        "std": numeric.std(axis=0, ddof=1),
        "mean": numeric.mean(axis=0),
        "min": numeric.min(axis=0),
        "max": numeric.max(axis=0),
        "nonzero_pct": (numeric != 0).sum(axis=0) / numeric.shape[0] * 100.0,
    })
    stats.sort_values("variance", inplace=True)
    return stats


def parse_args():
    parser = argparse.ArgumentParser("Feature variance reporter")
    parser.add_argument(
        "--panels",
        nargs="+",
        default=["panel_train.npz", "panel_valid.npz", "panel_test.npz"],
        help="NPZ files under data/ to include",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="reports/distribution/feature_variance.csv",
        help="CSV output path",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    panel_paths = [PROJECT_ROOT / "data" / name for name in args.panels]
    feat_df = aggregate_features(panel_paths)
    if feat_df.empty:
        print("[error] no feature data gathered; aborting.")
        return
    stats = summarize(feat_df)
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(out_path, float_format="%.8f")
    print(f"[info] variance report written to {out_path}")
    print(stats.head(10))


if __name__ == "__main__":
    main()
