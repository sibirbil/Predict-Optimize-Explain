from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.configs.pto.robust_v1 import PTOV1Config
from src.modeling.pto.data import DecisionMonth, prepare_decision_months
from src.modeling.pto.metrics import compute_turnover, summarize_backtest, weights_diagnostics
from src.modeling.pto.solver import solve_robust_long_only


@dataclass(frozen=True)
class CandidateConfig:
    lambda_: float
    kappa: float
    omega_mode: str

    def label(self) -> str:
        return f"lambda={self.lambda_}__kappa={self.kappa}__omega={self.omega_mode}"

    def to_dict(self) -> dict[str, object]:
        return {
            "lambda": self.lambda_,
            "kappa": self.kappa,
            "omega_mode": self.omega_mode,
            "label": self.label(),
        }


def run_pto_pipeline(config: PTOV1Config) -> Path:
    backbone_config = config.load_backbone_config()
    decision_months = prepare_decision_months(
        backbone_config=backbone_config,
        lookback_months=config.lookback_months,
        ewma_lambda=config.ewma_lambda,
        shrinkage=config.covariance_shrinkage,
        ridge=config.covariance_ridge,
        min_assets=config.min_assets,
    )

    candidate_grid = [
        CandidateConfig(lambda_=lambda_, kappa=kappa, omega_mode=omega_mode)
        for lambda_, kappa, omega_mode in itertools.product(
            config.lambda_grid,
            config.kappa_grid,
            config.omega_grid,
        )
    ]

    stage1_months = select_stage1_months(decision_months["val"], config)
    stage1_table, _ = evaluate_candidate_grid(
        decision_months=stage1_months,
        candidates=candidate_grid,
        config=config,
        stage_name="stage1",
    )
    finalists = select_stage2_finalists(stage1_table, config)

    stage2_table, stage2_results = evaluate_candidate_grid(
        decision_months=decision_months["val"],
        candidates=finalists,
        config=config,
        stage_name="stage2",
    )

    selected_row = select_final_candidate(stage2_table)
    selected_label = str(selected_row["label"])
    selected_candidate, val_monthly, val_weights = stage2_results[selected_label]
    test_monthly, test_weights = backtest_split(decision_months["test"], selected_candidate)
    test_summary = summarize_backtest(test_monthly)

    run_dir = config.resolve_output_root() / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")

    stage1_table.to_csv(run_dir / "tuning_table_stage1.csv", index=False)
    stage2_table.to_csv(run_dir / "tuning_table_stage2.csv", index=False)
    val_monthly.to_csv(run_dir / "selected_validation_monthly_returns.csv", index=False)
    val_weights.to_csv(run_dir / "selected_validation_weights.csv", index=False)
    test_monthly.to_csv(run_dir / "test_monthly_portfolio.csv", index=False)
    test_weights.to_csv(run_dir / "test_monthly_weights.csv", index=False)

    selection_summary = {
        "selected_label": selected_label,
        "selected_config": selected_candidate.to_dict(),
        "stage1_mode": config.stage1_mode,
        "stage1_n_months": config.stage1_n_months,
        "stage1_stride": config.stage1_stride,
        "stage2_n_finalists": config.stage2_n_finalists,
        "stage1_candidate_count": len(candidate_grid),
        "stage2_candidate_count": len(finalists),
        "validation_summary": summarize_backtest(val_monthly),
        "test_summary": test_summary,
        "screening_rule": {
            "max_avg_turnover": config.max_avg_turnover,
            "max_avg_weight": config.max_avg_weight,
            "max_avg_herfindahl": config.max_avg_herfindahl,
            "min_avg_effective_n": config.min_avg_effective_n,
            "min_avg_active_positions": config.min_avg_active_positions,
        },
    }
    (run_dir / "validation_selection_summary.json").write_text(
        json.dumps(selection_summary, indent=2),
        encoding="utf-8",
    )
    (run_dir / "selected_robust_config.json").write_text(
        json.dumps(selected_candidate.to_dict(), indent=2),
        encoding="utf-8",
    )
    (run_dir / "final_test_summary.json").write_text(
        json.dumps(test_summary, indent=2),
        encoding="utf-8",
    )
    return run_dir


def evaluate_candidate_grid(
    decision_months: list[DecisionMonth],
    candidates: list[CandidateConfig],
    config: PTOV1Config,
    stage_name: str,
) -> tuple[pd.DataFrame, dict[str, tuple[CandidateConfig, pd.DataFrame, pd.DataFrame]]]:
    candidate_rows: list[dict[str, object]] = []
    candidate_results: dict[str, tuple[CandidateConfig, pd.DataFrame, pd.DataFrame]] = {}
    for candidate in candidates:
        monthly_frame, weights_frame = backtest_split(decision_months, candidate)
        metrics = summarize_backtest(monthly_frame)
        screens = diversification_screens(config=config, summary=metrics)
        candidate_rows.append(
            {
                **candidate.to_dict(),
                **metrics,
                **screens,
                "stage": stage_name,
            }
        )
        candidate_results[candidate.label()] = (candidate, monthly_frame, weights_frame)

    table = pd.DataFrame(candidate_rows).sort_values(
        ["passed_screens", "sharpe_ratio", "annualized_volatility", "average_herfindahl"],
        ascending=[False, False, True, True],
    )
    return table, candidate_results


def select_stage1_months(
    decision_months: list[DecisionMonth],
    config: PTOV1Config,
) -> list[DecisionMonth]:
    if config.stage1_mode == "first_n":
        return decision_months[: config.stage1_n_months]
    if config.stage1_mode == "stride":
        return decision_months[:: config.stage1_stride]
    raise ValueError(f"Unknown stage1_mode={config.stage1_mode}")


def select_stage2_finalists(stage1_table: pd.DataFrame, config: PTOV1Config) -> list[CandidateConfig]:
    screened = stage1_table.loc[stage1_table["passed_screens"] == True].copy()  # noqa: E712
    if screened.empty:
        screened = stage1_table.copy()
    finalists = screened.sort_values(
        [
            "sharpe_ratio",
            "annualized_volatility",
            "average_herfindahl",
            "average_turnover",
        ],
        ascending=[False, True, True, True],
    ).head(config.stage2_n_finalists)
    return [
        CandidateConfig(
            lambda_=float(row["lambda"]),
            kappa=float(row["kappa"]),
            omega_mode=str(row["omega_mode"]),
        )
        for _, row in finalists.iterrows()
    ]


def backtest_split(
    decision_months: list[DecisionMonth],
    candidate: CandidateConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []

    previous_weights: pd.Series | None = None
    previous_returns: pd.Series | None = None

    for decision in decision_months:
        frame = decision.frame.sort_values("permno").reset_index(drop=True)
        permnos = frame["permno"].astype(int).tolist()
        mu = frame["prediction"].to_numpy(dtype=float)
        weights = solve_robust_long_only(
            mu=mu,
            sigma=decision.covariance,
            lambda_=candidate.lambda_,
            kappa=candidate.kappa,
            omega_mode=candidate.omega_mode,
        )
        if weights is None:
            weights = np.full(len(frame), 1.0 / len(frame), dtype=float)

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


def diversification_screens(config: PTOV1Config, summary: dict[str, float]) -> dict[str, object]:
    passed = bool(
        summary
        and summary["average_turnover"] <= config.max_avg_turnover
        and summary["average_max_weight"] <= config.max_avg_weight
        and summary["average_herfindahl"] <= config.max_avg_herfindahl
        and summary["average_effective_n"] >= config.min_avg_effective_n
        and summary["average_active_positions"] >= config.min_avg_active_positions
    )
    return {"passed_screens": passed}


def select_final_candidate(tuning_table: pd.DataFrame) -> pd.Series:
    screened = tuning_table.loc[tuning_table["passed_screens"] == True].copy()  # noqa: E712
    if screened.empty:
        screened = tuning_table.copy()
    screened = screened.sort_values(
        [
            "sharpe_ratio",
            "annualized_volatility",
            "average_herfindahl",
            "average_turnover",
        ],
        ascending=[False, True, True, True],
    )
    return screened.iloc[0]
