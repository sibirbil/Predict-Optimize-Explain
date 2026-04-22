from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .paths import REPO_ROOT


@dataclass(frozen=True)
class WealthRunSpec:
    label: str
    run_dir: Path
    split: str = "test"
    quantiles: int = 10


def load_manifest(path: Path) -> list[WealthRunSpec]:
    payload = json.loads(path.read_text())
    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("Manifest must contain a non-empty 'runs' list.")
    specs: list[WealthRunSpec] = []
    for item in runs:
        if not isinstance(item, dict):
            raise ValueError("Each manifest run entry must be an object.")
        raw_run_dir = Path(item["run_dir"])
        if raw_run_dir.is_absolute():
            run_dir = raw_run_dir
        else:
            run_dir = REPO_ROOT / raw_run_dir
            if not run_dir.exists() and raw_run_dir.parts and raw_run_dir.parts[0] == REPO_ROOT.name:
                run_dir = REPO_ROOT.joinpath(*raw_run_dir.parts[1:])
        specs.append(
            WealthRunSpec(
                label=str(item["label"]),
                run_dir=run_dir,
                split=str(item.get("split", "test")),
                quantiles=int(item.get("quantiles", 10)),
            )
        )
    return specs


def top_minus_bottom_spread_from_predictions(prediction_frame: pd.DataFrame, quantiles: int) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for yyyymm, group in prediction_frame.groupby("yyyymm", sort=True):
        if len(group) < quantiles:
            spread = float("nan")
        else:
            ranked = group[["target", "prediction"]].copy()
            try:
                ranked["bucket"] = pd.qcut(
                    ranked["prediction"],
                    q=quantiles,
                    labels=False,
                    duplicates="drop",
                )
            except ValueError:
                ranked["bucket"] = np.nan

            if ranked["bucket"].isna().all():
                spread = float("nan")
            else:
                bottom_bucket = ranked["bucket"].min()
                top_bucket = ranked["bucket"].max()
                if pd.isna(bottom_bucket) or pd.isna(top_bucket) or bottom_bucket == top_bucket:
                    spread = float("nan")
                else:
                    top_mean = ranked.loc[ranked["bucket"] == top_bucket, "target"].mean()
                    bottom_mean = ranked.loc[ranked["bucket"] == bottom_bucket, "target"].mean()
                    spread = float(top_mean - bottom_mean)
        rows.append({"yyyymm": int(yyyymm), "top_minus_bottom_spread": spread})
    return pd.DataFrame(rows)


def load_monthly_spread(run_dir: Path, split: str, quantiles: int) -> pd.DataFrame:
    monthly_path = run_dir / f"monthly_metrics_{split}.csv"
    if monthly_path.exists():
        frame = pd.read_csv(monthly_path)
        if {"yyyymm", "top_minus_bottom_spread"}.issubset(frame.columns):
            return frame[["yyyymm", "top_minus_bottom_spread"]].copy()

    prediction_path = run_dir / "predictions" / f"{split}_predictions.parquet"
    if not prediction_path.exists():
        raise FileNotFoundError(f"Could not find monthly metrics or predictions for {run_dir}")
    predictions = pd.read_parquet(prediction_path).copy()
    predictions["yyyymm"] = predictions["yyyymm"].astype(int)
    return top_minus_bottom_spread_from_predictions(predictions, quantiles=quantiles)


def build_return_and_wealth_frames(specs: list[WealthRunSpec]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, float | str | int]] = []

    for spec in specs:
        monthly = load_monthly_spread(spec.run_dir, split=spec.split, quantiles=spec.quantiles)
        monthly = monthly.rename(columns={"top_minus_bottom_spread": spec.label})
        return_frames.append(monthly)

    merged = return_frames[0]
    for frame in return_frames[1:]:
        merged = merged.merge(frame, on="yyyymm", how="outer")
    merged = merged.sort_values("yyyymm").reset_index(drop=True)
    merged["date"] = pd.to_datetime(merged["yyyymm"].astype(str), format="%Y%m")

    wealth = merged[["yyyymm", "date"]].copy()
    for spec in specs:
        series = merged[spec.label].fillna(0.0).astype(float)
        wealth[spec.label] = np.cumprod(1.0 + series.to_numpy())
        valid = merged[spec.label].dropna().astype(float)
        wealth_valid = np.cumprod(1.0 + valid.to_numpy()) if not valid.empty else np.array([1.0])
        peak = np.maximum.accumulate(wealth_valid)
        drawdown = wealth_valid / peak - 1.0
        summary_rows.append(
            {
                "label": spec.label,
                "n_months": int(valid.shape[0]),
                "mean_monthly_spread": float(valid.mean()) if not valid.empty else 0.0,
                "vol_monthly_spread": float(valid.std(ddof=1)) if valid.shape[0] > 1 else 0.0,
                "final_wealth": float(wealth_valid[-1]),
                "cum_simple_return": float(wealth_valid[-1] - 1.0),
                "max_drawdown": float(drawdown.min()) if drawdown.size else 0.0,
                "best_month": float(valid.max()) if not valid.empty else 0.0,
                "worst_month": float(valid.min()) if not valid.empty else 0.0,
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("final_wealth", ascending=False).reset_index(drop=True)
    return merged, wealth, summary


def plot_wealth_paths(wealth: pd.DataFrame, title: str, output_path: Path) -> None:
    plt.figure(figsize=(12, 6))
    for column in wealth.columns:
        if column in {"yyyymm", "date"}:
            continue
        plt.plot(wealth["date"], wealth[column], label=column, linewidth=2)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Wealth (start = 1)")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
