from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, pearsonr, spearmanr

from .data import TARGET_COLUMN

PREDICTION_COLUMN = "prediction"


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = y_true - y_pred
    mse = float(np.mean(np.square(residual)))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residual)))

    sse_model = float(np.sum(np.square(residual)))
    sse_zero = float(np.sum(np.square(y_true)))
    y_mean = float(np.mean(y_true))
    sse_mean = float(np.sum(np.square(y_true - y_mean)))

    r2 = 1.0 - sse_model / sse_mean if sse_mean > 0 else 0.0
    r2_zero = 1.0 - sse_model / sse_zero if sse_zero > 0 else 0.0
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else 0.0
    if np.isnan(corr):
        corr = 0.0

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": float(r2),
        "r2_zero": float(r2_zero),
        "correlation": corr,
    }


def save_predictions(path: Path, metadata: pd.DataFrame, y_true: np.ndarray, predictions: np.ndarray) -> None:
    frame = metadata.copy()
    frame[TARGET_COLUMN] = y_true
    frame[PREDICTION_COLUMN] = predictions
    frame["residual"] = y_true - predictions
    frame.to_parquet(path, index=False)


def summarize_prediction_dir(predictions_dir: Path, run_dir: Path, quantiles: int = 10) -> dict[str, object]:
    frames = {
        split: pd.read_parquet(predictions_dir / f"{split}_predictions.parquet")
        for split in ("train", "val", "test")
    }

    train_mean = float(frames["train"][TARGET_COLUMN].mean())
    split_metrics = pd.DataFrame(
        [
            {
                "split": split,
                **compute_split_metrics(frame, train_mean=train_mean),
            }
            for split, frame in frames.items()
        ]
    )
    split_metrics.to_csv(run_dir / "split_metrics.csv", index=False)

    monthly_summary: dict[str, dict[str, float]] = {}
    portfolio_rows: list[dict[str, float | int | str]] = []
    for split in ("val", "test"):
        monthly = compute_monthly_metrics(frames[split], quantiles=quantiles)
        monthly.to_csv(run_dir / f"monthly_metrics_{split}.csv", index=False)
        monthly_summary[split] = {
            "mean_monthly_ic": safe_mean(monthly["pearson_ic"]),
            "mean_monthly_rank_ic": safe_mean(monthly["spearman_ic"]),
            "mean_monthly_sign_accuracy": safe_mean(monthly["sign_accuracy"]),
        }
        portfolio_rows.append(
            {
                "split": split,
                "quantiles": quantiles,
                "mean_top_minus_bottom_spread": safe_mean(monthly["top_minus_bottom_spread"]),
                "n_months_with_spread": int(monthly["top_minus_bottom_spread"].notna().sum()),
            }
        )

    portfolio = pd.DataFrame(portfolio_rows)
    portfolio.to_csv(run_dir / "portfolio_sort_metrics.csv", index=False)

    summary = {
        "split_metrics": split_metrics.set_index("split").to_dict(orient="index"),
        "monthly_metrics": monthly_summary,
        "portfolio_sort_metrics": portfolio.to_dict(orient="records"),
    }
    with (run_dir / "evaluation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def compute_split_metrics(frame: pd.DataFrame, train_mean: float) -> dict[str, float]:
    target = frame[TARGET_COLUMN].to_numpy(dtype=float)
    prediction = frame[PREDICTION_COLUMN].to_numpy(dtype=float)
    residual = target - prediction

    mse = float(np.mean(np.square(residual)))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residual)))
    pearson_value = safe_statistic(lambda: pearsonr(target, prediction).statistic)
    spearman_value = safe_statistic(lambda: spearmanr(target, prediction).statistic)
    sign_accuracy = float(np.mean(np.sign(target) == np.sign(prediction)))

    sse_model = float(np.sum(np.square(residual)))
    sse_zero = float(np.sum(np.square(target)))
    sse_train_mean = float(np.sum(np.square(target - train_mean)))

    r2_zero = 1.0 - sse_model / sse_zero if sse_zero > 0 else 0.0
    r2_train_mean = 1.0 - sse_model / sse_train_mean if sse_train_mean > 0 else 0.0

    return {
        "n_obs": float(len(frame)),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "pearson_correlation": pearson_value,
        "spearman_correlation": spearman_value,
        "sign_accuracy": sign_accuracy,
        "r2_oos_zero_benchmark": float(r2_zero),
        "r2_oos_train_mean_benchmark": float(r2_train_mean),
    }


def compute_monthly_metrics(frame: pd.DataFrame, quantiles: int) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for yyyymm, group in frame.groupby("yyyymm", sort=True):
        target = group[TARGET_COLUMN].to_numpy(dtype=float)
        prediction = group[PREDICTION_COLUMN].to_numpy(dtype=float)
        rows.append(
            {
                "yyyymm": int(yyyymm),
                "n_assets": int(len(group)),
                "pearson_ic": safe_statistic(lambda: pearsonr(target, prediction).statistic),
                "spearman_ic": safe_statistic(lambda: spearmanr(target, prediction).statistic),
                "sign_accuracy": float(np.mean(np.sign(target) == np.sign(prediction))),
                "top_minus_bottom_spread": compute_top_minus_bottom_spread(group, quantiles),
            }
        )
    return pd.DataFrame(rows)


def compute_top_minus_bottom_spread(group: pd.DataFrame, quantiles: int) -> float:
    if len(group) < quantiles:
        return float("nan")
    ranked = group[[TARGET_COLUMN, PREDICTION_COLUMN]].copy()
    try:
        ranked["bucket"] = pd.qcut(
            ranked[PREDICTION_COLUMN],
            q=quantiles,
            labels=False,
            duplicates="drop",
        )
    except ValueError:
        return float("nan")

    if ranked["bucket"].isna().all():
        return float("nan")

    bottom_bucket = ranked["bucket"].min()
    top_bucket = ranked["bucket"].max()
    if pd.isna(bottom_bucket) or pd.isna(top_bucket) or bottom_bucket == top_bucket:
        return float("nan")

    top_mean = ranked.loc[ranked["bucket"] == top_bucket, TARGET_COLUMN].mean()
    bottom_mean = ranked.loc[ranked["bucket"] == bottom_bucket, TARGET_COLUMN].mean()
    return float(top_mean - bottom_mean)


def safe_statistic(compute) -> float:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConstantInputWarning)
            value = compute()
    except Exception:
        return 0.0
    if value is None or np.isnan(value):
        return 0.0
    return float(value)


def safe_mean(series: pd.Series) -> float:
    non_null = series.dropna()
    if non_null.empty:
        return 0.0
    return float(non_null.mean())
