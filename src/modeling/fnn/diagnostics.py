from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.data import TARGET_COLUMN
from src.modeling.benchmarks.runner import build_comparison_table
from src.modeling.fnn.evaluation import (
    PREDICTION_COLUMN,
    MonthlyEvaluationConfig,
    compute_monthly_metrics,
    compute_split_metrics,
    load_prediction_artifact,
    safe_mean,
    safe_statistic,
)
from src.utils.paths import resolve_repo_path


@dataclass(frozen=True)
class StabilityAssessment:
    split: str
    positive_year_share_ic: float
    positive_year_share_rank_ic: float
    positive_year_share_r2: float
    best_year_abs_ic_share: float
    major_instability: bool
    summary: str

    def to_dict(self) -> dict[str, object]:
        return {
            "split": self.split,
            "positive_year_share_ic": self.positive_year_share_ic,
            "positive_year_share_rank_ic": self.positive_year_share_rank_ic,
            "positive_year_share_r2": self.positive_year_share_r2,
            "best_year_abs_ic_share": self.best_year_abs_ic_share,
            "major_instability": self.major_instability,
            "summary": self.summary,
        }


def build_prediction_diagnostic_report(
    run_name: str,
    output_root: str | Path = "batuhan/artifacts/fnn_transfer",
    monthly_config: MonthlyEvaluationConfig | None = None,
) -> Path:
    monthly_config = monthly_config or MonthlyEvaluationConfig()
    run_dir = resolve_repo_path(output_root) / run_name
    predictions_dir = run_dir / "predictions"

    split_frames = {
        split: load_prediction_artifact(predictions_dir / f"{split}_predictions.parquet")
        for split in ("train", "val", "test")
    }
    train_mean = float(split_frames["train"][TARGET_COLUMN].mean())

    prediction_distribution = pd.DataFrame(
        [
            build_prediction_distribution_row(split=split, frame=frame)
            for split, frame in split_frames.items()
        ]
    )
    prediction_distribution.to_csv(run_dir / "prediction_distribution.csv", index=False)

    yearly_outputs: dict[str, pd.DataFrame] = {}
    stability_reports: dict[str, StabilityAssessment] = {}
    portfolio_rows: list[dict[str, object]] = []

    for split in ("val", "test"):
        frame = split_frames[split]
        monthly_frame = _load_or_compute_monthly_frame(
            run_dir=run_dir,
            split=split,
            frame=frame,
            monthly_config=monthly_config,
        )
        yearly_frame = compute_yearly_metrics(
            frame=frame,
            monthly_frame=monthly_frame,
            train_mean=train_mean,
        )
        yearly_frame.to_csv(run_dir / f"yearly_metrics_{split}.csv", index=False)
        yearly_outputs[split] = yearly_frame

        portfolio_summary = compute_portfolio_monotonicity_summary(
            frame=frame,
            quantiles=monthly_config.quantiles,
            split=split,
        )
        portfolio_rows.extend(portfolio_summary["bucket_rows"])
        pd.DataFrame(portfolio_summary["bucket_rows"]).to_csv(
            run_dir / f"portfolio_monotonicity_{split}.csv",
            index=False,
        )

        stability_reports[split] = assess_subperiod_stability(split=split, yearly_frame=yearly_frame)

    comparison = build_pto_comparison_report(run_dir=run_dir)

    summary = {
        "run_name": run_name,
        "output_root": str(output_root),
        "prediction_distribution": prediction_distribution.to_dict(orient="records"),
        "stability": {split: report.to_dict() for split, report in stability_reports.items()},
        "top_minus_bottom_summary": {
            split: summarize_top_minus_bottom(yearly_outputs[split]) for split in ("val", "test")
        },
        "comparison_models": comparison.to_dict(orient="records"),
    }
    with (run_dir / "diagnostic_report.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    with (run_dir / "diagnostic_report.md").open("w", encoding="utf-8") as handle:
        handle.write(render_diagnostic_markdown(run_name, yearly_outputs, stability_reports))

    return run_dir


def build_pto_comparison_report(run_dir: Path) -> pd.DataFrame:
    comparison = build_comparison_table(
        rows=[
            ("zero_shot_pretrained_backbone", "batuhan/artifacts/fnn_transfer", run_dir.name),
            ("current_clean_fnn_archived", "batuhan/archive/artifacts/fnn", "full_fnn_baseline_20260404"),
            ("legacy_style_fnn_archived", "batuhan/archive/artifacts/fnn", "full_fnn_legacy_style_20260404"),
            ("elastic_net_archived", "batuhan/archive/artifacts/benchmarks", "elastic_net_full_suite_20260404"),
        ]
    )
    comparison = comparison[
        [
            "model",
            "run_name",
            "val_r2_oos_zero",
            "test_r2_oos_zero",
            "val_sign_accuracy",
            "test_sign_accuracy",
            "val_mean_monthly_ic",
            "test_mean_monthly_ic",
            "val_mean_monthly_rank_ic",
            "test_mean_monthly_rank_ic",
        ]
    ].copy()

    spread_rows = []
    for _, row in comparison.iterrows():
        root = resolve_repo_path(
            {
                "zero_shot_pretrained_backbone": "batuhan/artifacts/fnn_transfer",
                "current_clean_fnn_archived": "batuhan/archive/artifacts/fnn",
                "legacy_style_fnn_archived": "batuhan/archive/artifacts/fnn",
                "elastic_net_archived": "batuhan/archive/artifacts/benchmarks",
            }[row["model"]]
        )
        portfolio = pd.read_csv(root / row["run_name"] / "portfolio_sort_metrics.csv")
        portfolio = portfolio.set_index("split")
        spread_rows.append(
            {
                "model": row["model"],
                "val_mean_top_minus_bottom_spread": float(
                    portfolio.loc["val", "mean_top_minus_bottom_spread"]
                ),
                "test_mean_top_minus_bottom_spread": float(
                    portfolio.loc["test", "mean_top_minus_bottom_spread"]
                ),
            }
        )
    comparison = comparison.merge(pd.DataFrame(spread_rows), on="model", how="left")
    comparison.to_csv(run_dir / "pto_backbone_comparison.csv", index=False)
    with (run_dir / "pto_backbone_comparison.md").open("w", encoding="utf-8") as handle:
        handle.write(comparison.to_markdown(index=False))
    return comparison


def build_prediction_distribution_row(split: str, frame: pd.DataFrame) -> dict[str, float | str]:
    prediction = frame[PREDICTION_COLUMN]
    return {
        "split": split,
        "n_obs": float(len(frame)),
        "prediction_mean": float(prediction.mean()),
        "prediction_std": float(prediction.std()),
        "prediction_min": float(prediction.min()),
        "prediction_p01": float(prediction.quantile(0.01)),
        "prediction_p05": float(prediction.quantile(0.05)),
        "prediction_p50": float(prediction.quantile(0.50)),
        "prediction_p95": float(prediction.quantile(0.95)),
        "prediction_p99": float(prediction.quantile(0.99)),
        "prediction_max": float(prediction.max()),
    }


def compute_yearly_metrics(
    frame: pd.DataFrame,
    monthly_frame: pd.DataFrame,
    train_mean: float,
) -> pd.DataFrame:
    yearly_rows: list[dict[str, float | int]] = []
    working = frame.copy()
    working["year"] = working["yyyymm"] // 100
    monthly = monthly_frame.copy()
    monthly["year"] = monthly["yyyymm"] // 100

    for year, group in working.groupby("year", sort=True):
        year_monthly = monthly.loc[monthly["year"] == year]
        metrics = compute_split_metrics(group, train_mean=train_mean)
        yearly_rows.append(
            {
                "year": int(year),
                "n_obs": float(len(group)),
                "r2_oos_zero_benchmark": metrics["r2_oos_zero_benchmark"],
                "sign_accuracy": metrics["sign_accuracy"],
                "mean_monthly_ic": safe_mean(year_monthly["pearson_ic"]),
                "mean_monthly_rank_ic": safe_mean(year_monthly["spearman_ic"]),
                "mean_top_minus_bottom_spread": safe_mean(year_monthly["top_minus_bottom_spread"]),
            }
        )
    return pd.DataFrame(yearly_rows)


def compute_portfolio_monotonicity_summary(
    frame: pd.DataFrame,
    quantiles: int,
    split: str,
) -> dict[str, object]:
    bucket_frames: list[pd.DataFrame] = []
    top_bottom_values: list[float] = []

    for yyyymm, group in frame.groupby("yyyymm", sort=True):
        if len(group) < quantiles:
            continue
        ranked = group[[TARGET_COLUMN, PREDICTION_COLUMN]].copy()
        try:
            ranked["bucket"] = pd.qcut(
                ranked[PREDICTION_COLUMN],
                q=quantiles,
                labels=False,
                duplicates="drop",
            )
        except ValueError:
            continue
        if ranked["bucket"].isna().all():
            continue
        ranked = ranked.dropna(subset=["bucket"]).copy()
        ranked["bucket"] = ranked["bucket"].astype(int) + 1
        grouped = ranked.groupby("bucket", as_index=False)[TARGET_COLUMN].mean()
        grouped["yyyymm"] = int(yyyymm)
        bucket_frames.append(grouped)
        if grouped["bucket"].nunique() >= 2:
            top_bottom_values.append(
                float(
                    grouped.loc[grouped["bucket"] == grouped["bucket"].max(), TARGET_COLUMN].iloc[0]
                    - grouped.loc[grouped["bucket"] == grouped["bucket"].min(), TARGET_COLUMN].iloc[0]
                )
            )

    if not bucket_frames:
        return {"bucket_rows": [], "summary": {}}

    bucket_frame = pd.concat(bucket_frames, ignore_index=True)
    bucket_summary = (
        bucket_frame.groupby("bucket", as_index=False)
        .agg(
            mean_target=(TARGET_COLUMN, "mean"),
            std_target=(TARGET_COLUMN, "std"),
            n_months=(TARGET_COLUMN, "count"),
        )
        .reset_index(drop=True)
    )
    bucket_summary["split"] = split
    bucket_summary["quantiles"] = quantiles

    monotonicity_ratio = float(
        np.mean(np.diff(bucket_summary["mean_target"].to_numpy(dtype=float)) > 0)
    ) if len(bucket_summary) > 1 else 0.0
    monotonicity_spearman = safe_statistic(
        lambda: spearmanr(bucket_summary["bucket"], bucket_summary["mean_target"]).statistic
    )
    positive_spread_share = float(np.mean(np.asarray(top_bottom_values) > 0)) if top_bottom_values else 0.0
    spread_std = float(np.std(top_bottom_values, ddof=1)) if len(top_bottom_values) > 1 else 0.0
    spread_mean = float(np.mean(top_bottom_values)) if top_bottom_values else 0.0
    spread_tstat = (
        float(spread_mean / (spread_std / np.sqrt(len(top_bottom_values))))
        if len(top_bottom_values) > 1 and spread_std > 0
        else 0.0
    )
    bucket_summary["monotonicity_ratio"] = monotonicity_ratio
    bucket_summary["monotonicity_spearman"] = monotonicity_spearman
    bucket_summary["positive_top_minus_bottom_share"] = positive_spread_share
    bucket_summary["mean_top_minus_bottom_spread"] = spread_mean
    bucket_summary["top_minus_bottom_tstat"] = spread_tstat

    return {
        "bucket_rows": bucket_summary.to_dict(orient="records"),
        "summary": {
            "monotonicity_ratio": monotonicity_ratio,
            "monotonicity_spearman": monotonicity_spearman,
            "positive_top_minus_bottom_share": positive_spread_share,
            "mean_top_minus_bottom_spread": spread_mean,
            "top_minus_bottom_tstat": spread_tstat,
        },
    }


def assess_subperiod_stability(split: str, yearly_frame: pd.DataFrame) -> StabilityAssessment:
    ic_values = yearly_frame["mean_monthly_ic"].to_numpy(dtype=float)
    rank_values = yearly_frame["mean_monthly_rank_ic"].to_numpy(dtype=float)
    r2_values = yearly_frame["r2_oos_zero_benchmark"].to_numpy(dtype=float)

    positive_year_share_ic = float(np.mean(ic_values > 0))
    positive_year_share_rank_ic = float(np.mean(rank_values > 0))
    positive_year_share_r2 = float(np.mean(r2_values > 0))
    abs_ic_sum = float(np.abs(ic_values).sum())
    best_year_abs_ic_share = float(np.abs(ic_values).max() / abs_ic_sum) if abs_ic_sum > 0 else 0.0

    major_instability = bool(
        positive_year_share_ic < 0.5
        or positive_year_share_rank_ic < 0.5
        or best_year_abs_ic_share > 0.5
    )
    if major_instability:
        summary = (
            f"{split} performance looks concentrated or unstable: "
            f"positive IC years={positive_year_share_ic:.2f}, "
            f"positive rank-IC years={positive_year_share_rank_ic:.2f}, "
            f"best-year IC share={best_year_abs_ic_share:.2f}."
        )
    else:
        summary = (
            f"{split} performance is reasonably distributed across years: "
            f"positive IC years={positive_year_share_ic:.2f}, "
            f"positive rank-IC years={positive_year_share_rank_ic:.2f}, "
            f"best-year IC share={best_year_abs_ic_share:.2f}."
        )
    return StabilityAssessment(
        split=split,
        positive_year_share_ic=positive_year_share_ic,
        positive_year_share_rank_ic=positive_year_share_rank_ic,
        positive_year_share_r2=positive_year_share_r2,
        best_year_abs_ic_share=best_year_abs_ic_share,
        major_instability=major_instability,
        summary=summary,
    )


def summarize_top_minus_bottom(yearly_frame: pd.DataFrame) -> dict[str, float]:
    spread = yearly_frame["mean_top_minus_bottom_spread"].to_numpy(dtype=float)
    return {
        "mean": float(np.mean(spread)),
        "min": float(np.min(spread)),
        "max": float(np.max(spread)),
        "positive_year_share": float(np.mean(spread > 0)),
    }


def render_diagnostic_markdown(
    run_name: str,
    yearly_outputs: dict[str, pd.DataFrame],
    stability_reports: dict[str, StabilityAssessment],
) -> str:
    lines = [f"# Diagnostic Report: {run_name}", ""]
    for split in ("val", "test"):
        lines.append(f"## {split.upper()}")
        lines.append(stability_reports[split].summary)
        lines.append("")
        lines.append(yearly_outputs[split].to_markdown(index=False))
        lines.append("")
    return "\n".join(lines)


def _load_or_compute_monthly_frame(
    run_dir: Path,
    split: str,
    frame: pd.DataFrame,
    monthly_config: MonthlyEvaluationConfig,
) -> pd.DataFrame:
    monthly_path = run_dir / f"monthly_metrics_{split}.csv"
    if monthly_path.exists():
        return pd.read_csv(monthly_path)
    monthly_frame = compute_monthly_metrics(frame=frame, quantiles=monthly_config.quantiles)
    monthly_frame.to_csv(monthly_path, index=False)
    return monthly_frame
