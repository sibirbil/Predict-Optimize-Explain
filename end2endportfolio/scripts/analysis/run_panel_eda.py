#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exploratory analysis for panel data (NPZ)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import load_panel, clean_missing_xy, FILL_VALUE  # noqa: E402


def describe_panel(panel_path: Path):
    panel = load_panel(str(panel_path))
    print(f"\n=== {panel_path.name} ===")
    T, N, D = panel.X.shape
    print(f"Timestamps: {T}, Assets per timestamp: {N}, Features: {D}")
    print(f"Dates span: {panel.dates[0]} -> {panel.dates[-1]}")
    if panel.asset_ids is not None:
        ids = panel.asset_ids.reshape(-1)
        unique_ids = np.unique(ids)
        print(f"Unique asset IDs: {len(unique_ids)}")
        if len(unique_ids) <= 10:
            print("Sample IDs:", unique_ids)
    if panel.tickers is not None:
        tickers = panel.tickers.reshape(-1)
        unique_tickers = np.unique(tickers)
        print(f"Unique tickers: {len(unique_tickers)}")
        if len(unique_tickers) <= 10:
            print("Sample tickers:", unique_tickers)

    X_list, y_list, id_list, tickers_list = clean_missing_xy(
        panel.X, panel.y, panel.asset_ids, panel.tickers
    )
    returns = np.concatenate(y_list) if y_list else np.array([])

    counts = [len(y) for y in y_list]
    date_segments = [np.repeat(panel.dates[i], counts[i]) for i in range(len(counts)) if counts[i] > 0]
    if date_segments:
        dates_clean = np.concatenate(date_segments)
        years = pd.to_datetime(dates_clean).year
    else:
        dates_clean = np.array([])
        years = np.array([])

    tickers_flat = None
    if tickers_list is not None and len(tickers_list):
        tickers_flat = np.concatenate(tickers_list)

    asset_ids_flat = None
    if id_list is not None and len(id_list):
        asset_ids_flat = np.concatenate(id_list)

    print("Returns summary:")
    print(pd.Series(returns).describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]))

    feature_matrix = np.vstack(X_list) if X_list else np.empty((0, panel.X.shape[-1]))
    feature_df = pd.DataFrame(feature_matrix, columns=panel.feature_names)
    if not feature_df.empty:
        print("\nFeature summary (first 10 columns):")
        print(feature_df.iloc[:, : min(10, feature_df.shape[1])].describe().T)
    else:
        print("\n[warn] No feature data after cleaning.")

    return {
        "returns": returns,
        "features": feature_matrix,
        "feature_names": panel.feature_names,
        "years": years,
        "dates": dates_clean,
        "asset_ids": asset_ids_flat,
        "tickers": tickers_flat,
    }


def plot_distributions(data, output_dir: Path, panel_name: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    returns = data["returns"]
    feature_matrix = data["features"]
    feature_names = data["feature_names"]
    years = data["years"]

    if returns.size == 0 or feature_matrix.size == 0:
        print(f"[warn] No data available for plotting {panel_name}.")
        return

    plt.figure(figsize=(8, 4))
    plt.hist(returns, bins=100, alpha=0.7, color="steelblue", edgecolor="none")
    plt.title(f"{panel_name} Returns Distribution")
    plt.xlabel("Return")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_dir / f"{panel_name}_returns_hist.png", dpi=150)
    plt.close()

    feature_var = feature_matrix.var(axis=0)
    top_idx = np.argsort(feature_var)[-10:]
    top_features = feature_names[top_idx]
    plt.figure(figsize=(10, 6))
    for idx in top_idx:
        col = feature_matrix[:, idx]
        plt.hist(col, bins=80, alpha=0.4, label=feature_names[idx], histtype="step")
    plt.title(f"{panel_name} Top-Variance Features")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / f"{panel_name}_top_features_hist.png", dpi=150)
    plt.close()

    returns_df = pd.DataFrame({"return": returns, "year": years})
    yearly_stats = returns_df.groupby("year")["return"].agg(["mean", "std", "count"]).reset_index()
    plt.figure(figsize=(10, 5))
    plt.plot(yearly_stats["year"], yearly_stats["mean"], color="navy", label="Mean return")
    plt.fill_between(
        yearly_stats["year"],
        yearly_stats["mean"] - yearly_stats["std"],
        yearly_stats["mean"] + yearly_stats["std"],
        color="skyblue",
        alpha=0.3,
        label="Mean ± std",
    )
    plt.title(f"{panel_name} Yearly Return Trend")
    plt.xlabel("Year")
    plt.ylabel("Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{panel_name}_yearly_return_trend.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(yearly_stats["year"], yearly_stats["count"], color="grey", alpha=0.7)
    plt.title(f"{panel_name} Observations per Year")
    plt.xlabel("Year")
    plt.ylabel("Observation count")
    plt.tight_layout()
    plt.savefig(output_dir / f"{panel_name}_yearly_counts.png", dpi=150)
    plt.close()

    feature_year_df = pd.DataFrame(feature_matrix, columns=feature_names)
    feature_year_df["year"] = years
    yearly_feature_means = feature_year_df.groupby("year")[top_features].mean()
    plt.figure(figsize=(10, 6))
    for feature in top_features:
        plt.plot(yearly_feature_means.index, yearly_feature_means[feature], label=feature)
    plt.title(f"{panel_name} Yearly Mean of Top Features")
    plt.xlabel("Year")
    plt.ylabel("Mean (feature units)")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / f"{panel_name}_yearly_feature_means.png", dpi=150)
    plt.close()

    recent_years = sorted(yearly_stats["year"].unique())[-20:]
    if recent_years:
        recent_data = returns_df[returns_df["year"].isin(recent_years)]
        plt.figure(figsize=(12, 6))
        box_data = [recent_data[recent_data["year"] == year]["return"] for year in recent_years]
        plt.boxplot(box_data, labels=recent_years, showfliers=False)
        plt.title(f"{panel_name} Return Distribution (last {len(recent_years)} years)")
        plt.xlabel("Year")
        plt.ylabel("Return")
        plt.xticks(rotation=45)
        plt.tight_layout()
    plt.savefig(output_dir / f"{panel_name}_recent_year_boxplot.png", dpi=150)
    plt.close()


def write_ticker_summary(tickers: Optional[np.ndarray], output_dir: Path, panel_name: str, top_k: int):
    if tickers is None or tickers.size == 0:
        return
    tickers_series = pd.Series(tickers).astype(str)
    counts = tickers_series.value_counts().head(top_k)
    counts.to_csv(output_dir / f"{panel_name}_top_tickers.csv", header=["count"])


def parse_args():
    parser = argparse.ArgumentParser("Panel EDA")
    parser.add_argument(
        "--panels",
        nargs="+",
        default=["panel_train.npz", "panel_valid.npz", "panel_test.npz"],
        help="List of NPZ files under data/ to profile",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="reports/distribution/panel_eda",
        help="Directory to store plots",
    )
    parser.add_argument(
        "--top-tickers",
        type=int,
        default=20,
        help="How many tickers to include in the frequency summary",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = PROJECT_ROOT / args.out
    top_k = args.top_tickers
    for panel_name in args.panels:
        panel_path = PROJECT_ROOT / "data" / panel_name
        if not panel_path.exists():
            print(f"[warn] Missing {panel_path}, skipping.")
            continue
        stats = describe_panel(panel_path)
        safe_name = panel_name.replace(".npz", "")
        plot_distributions(stats, out_dir, safe_name)
        write_ticker_summary(stats.get("tickers"), out_dir, safe_name, top_k)


if __name__ == "__main__":
    main()
