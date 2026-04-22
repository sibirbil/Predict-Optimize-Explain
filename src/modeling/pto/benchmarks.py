from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.configs.pto.robust_v1 import PTOV1Config
from src.modeling.pto.data import DecisionMonth, prepare_decision_months
from src.modeling.pto.metrics import compute_turnover, summarize_backtest, weights_diagnostics
from src.modeling.pto.solver import solve_robust_long_only


@dataclass(frozen=True)
class StrategySpec:
    name: str
    weight_builder: Callable[[DecisionMonth], np.ndarray | None]


def load_selected_candidate(run_dir: str | Path) -> dict[str, object]:
    summary_path = Path(run_dir) / "validation_selection_summary.json"
    with summary_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    selected = payload["selected_config"]
    return {
        "lambda": float(selected["lambda"]),
        "kappa": float(selected["kappa"]),
        "omega_mode": str(selected["omega_mode"]),
        "label": str(selected["label"]),
    }


def robust_selected_strategy(selected_config: dict[str, object]) -> StrategySpec:
    def build_weights(decision: DecisionMonth) -> np.ndarray | None:
        return solve_robust_long_only(
            mu=decision.frame["prediction"].to_numpy(dtype=float),
            sigma=decision.covariance,
            lambda_=float(selected_config["lambda"]),
            kappa=float(selected_config["kappa"]),
            omega_mode=str(selected_config["omega_mode"]),
        )

    return StrategySpec(name="robust_pto_baseline", weight_builder=build_weights)


def equal_weight_strategy() -> StrategySpec:
    def build_weights(decision: DecisionMonth) -> np.ndarray:
        n_assets = len(decision.frame)
        return np.full(n_assets, 1.0 / n_assets, dtype=float)

    return StrategySpec(name="equal_weight", weight_builder=build_weights)


def minimum_variance_strategy() -> StrategySpec:
    def build_weights(decision: DecisionMonth) -> np.ndarray | None:
        zeros = np.zeros(len(decision.frame), dtype=float)
        return solve_robust_long_only(
            mu=zeros,
            sigma=decision.covariance,
            lambda_=1.0,
            kappa=0.0,
            omega_mode="identity",
        )

    return StrategySpec(name="minimum_variance", weight_builder=build_weights)


def backtest_strategy(
    decision_months: list[DecisionMonth],
    strategy: StrategySpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []

    previous_weights: pd.Series | None = None
    previous_returns: pd.Series | None = None

    for decision in decision_months:
        frame = decision.frame.sort_values("permno").reset_index(drop=True)
        permnos = frame["permno"].astype(int).tolist()

        weights = strategy.weight_builder(decision)
        if weights is None:
            weights = np.full(len(frame), 1.0 / len(frame), dtype=float)
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()

        current_weights = pd.Series(weights, index=permnos, dtype=float)
        current_total_returns = pd.Series(frame["ret_tplus1"].to_numpy(dtype=float), index=permnos)
        current_excess_returns = pd.Series(frame["excess_ret"].to_numpy(dtype=float), index=permnos)
        current_rf = pd.Series(frame["Rfree"].to_numpy(dtype=float), index=permnos)

        turnover = compute_turnover(
            previous_weights=previous_weights,
            previous_returns=previous_returns,
            current_weights=current_weights,
        )
        diagnostics = weights_diagnostics(weights)
        monthly_rows.append(
            {
                "strategy": strategy.name,
                "split": decision.split,
                "yyyymm": decision.yyyymm,
                "n_assets": float(len(frame)),
                "portfolio_return": float(np.dot(weights, current_total_returns.to_numpy(dtype=float))),
                "portfolio_excess_return": float(np.dot(weights, current_excess_returns.to_numpy(dtype=float))),
                "portfolio_risk_free": float(np.dot(weights, current_rf.to_numpy(dtype=float))),
                "turnover": turnover,
                "herfindahl": diagnostics["herfindahl"],
                "effective_n": diagnostics["effective_n"],
                "max_weight": diagnostics["max_weight"],
                "active_positions": diagnostics["active_positions"],
            }
        )
        for permno, weight, pred, ret, excess_ret in zip(
            permnos,
            weights,
            frame["prediction"].to_numpy(dtype=float),
            current_total_returns.to_numpy(dtype=float),
            current_excess_returns.to_numpy(dtype=float),
        ):
            weight_rows.append(
                {
                    "strategy": strategy.name,
                    "split": decision.split,
                    "yyyymm": decision.yyyymm,
                    "permno": int(permno),
                    "weight": float(weight),
                    "prediction": float(pred),
                    "realized_return": float(ret),
                    "realized_excess_return": float(excess_ret),
                }
            )

        previous_weights = current_weights
        previous_returns = current_total_returns

    return pd.DataFrame(monthly_rows), pd.DataFrame(weight_rows)


def backtest_cash_benchmark(decision_months: list[DecisionMonth]) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly_rows: list[dict[str, object]] = []
    for decision in decision_months:
        rf = float(decision.frame["Rfree"].astype(float).mean())
        monthly_rows.append(
            {
                "strategy": "risk_free_cash",
                "split": decision.split,
                "yyyymm": decision.yyyymm,
                "n_assets": 1.0,
                "portfolio_return": rf,
                "portfolio_excess_return": 0.0,
                "portfolio_risk_free": rf,
                "turnover": 0.0,
                "herfindahl": 1.0,
                "effective_n": 1.0,
                "max_weight": 1.0,
                "active_positions": 1.0,
            }
        )
    return pd.DataFrame(monthly_rows), pd.DataFrame(
        columns=[
            "strategy",
            "split",
            "yyyymm",
            "permno",
            "weight",
            "prediction",
            "realized_return",
            "realized_excess_return",
        ]
    )


def build_pto_benchmark_package(config: PTOV1Config, run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    selected_config = load_selected_candidate(run_dir)
    backbone_config = config.load_backbone_config()
    decision_months = prepare_decision_months(
        backbone_config=backbone_config,
        lookback_months=config.lookback_months,
        ewma_lambda=config.ewma_lambda,
        shrinkage=config.covariance_shrinkage,
        ridge=config.covariance_ridge,
        min_assets=config.min_assets,
    )

    strategies = [
        robust_selected_strategy(selected_config),
        equal_weight_strategy(),
        minimum_variance_strategy(),
    ]

    summary_rows: list[dict[str, object]] = []
    monthly_frames: list[pd.DataFrame] = []
    weight_frames: list[pd.DataFrame] = []

    for split in ("val", "test"):
        for strategy in strategies:
            monthly_frame, weights_frame = backtest_strategy(decision_months[split], strategy)
            summary_rows.append(
                {
                    "strategy": strategy.name,
                    "split": split,
                    **summarize_backtest(monthly_frame),
                }
            )
            monthly_frames.append(monthly_frame)
            weight_frames.append(weights_frame)

        cash_monthly, cash_weights = backtest_cash_benchmark(decision_months[split])
        summary_rows.append(
            {
                "strategy": "risk_free_cash",
                "split": split,
                **summarize_backtest(cash_monthly),
            }
        )
        monthly_frames.append(cash_monthly)
        weight_frames.append(cash_weights)

    comparison = pd.DataFrame(summary_rows).sort_values(["split", "sharpe_ratio"], ascending=[True, False])
    monthly = pd.concat(monthly_frames, ignore_index=True)
    non_empty_weights = [frame for frame in weight_frames if not frame.empty]
    if non_empty_weights:
        weights = pd.concat(non_empty_weights, ignore_index=True)
    else:
        weights = pd.DataFrame(
            columns=[
                "strategy",
                "split",
                "yyyymm",
                "permno",
                "weight",
                "prediction",
                "realized_return",
                "realized_excess_return",
            ]
        )

    comparison.to_csv(run_dir / "pto_benchmark_comparison.csv", index=False)
    monthly.to_csv(run_dir / "pto_benchmark_monthly_returns.csv", index=False)
    weights.to_csv(run_dir / "pto_benchmark_weights.csv", index=False)
    (run_dir / "selected_robust_baseline.json").write_text(
        json.dumps(selected_config, indent=2),
        encoding="utf-8",
    )
    (run_dir / "pto_benchmark_summary.md").write_text(
        build_markdown_summary(comparison=comparison, selected_config=selected_config),
        encoding="utf-8",
    )
    return run_dir


def build_markdown_summary(comparison: pd.DataFrame, selected_config: dict[str, object]) -> str:
    lines = [
        "# PTO Benchmark Comparison",
        "",
        "## Frozen robust PTO baseline",
        f"- label: `{selected_config['label']}`",
        f"- lambda: `{selected_config['lambda']}`",
        f"- kappa: `{selected_config['kappa']}`",
        f"- omega: `{selected_config['omega_mode']}`",
        "",
    ]
    for split in ("val", "test"):
        split_frame = comparison.loc[comparison["split"] == split].copy()
        lines.append(f"## {split.upper()} summary")
        lines.append("")
        for _, row in split_frame.iterrows():
            lines.append(
                "- "
                + (
                    f"{row['strategy']}: ann_return={row['annualized_return']:.6f}, "
                    f"ann_excess={row['annualized_excess_return']:.6f}, "
                    f"ann_vol={row['annualized_volatility']:.6f}, "
                    f"sharpe={row['sharpe_ratio']:.6f}, "
                    f"max_dd={row['max_drawdown']:.6f}, "
                    f"avg_turnover={row['average_turnover']:.6f}, "
                    f"avg_herfindahl={row['average_herfindahl']:.6f}, "
                    f"avg_effective_n={row['average_effective_n']:.2f}, "
                    f"avg_max_weight={row['average_max_weight']:.6f}, "
                    f"avg_active_positions={row['average_active_positions']:.2f}"
                )
            )
        lines.append("")
    return "\n".join(lines)
