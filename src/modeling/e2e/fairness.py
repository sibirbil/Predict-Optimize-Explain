from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.configs.e2e import E2EV1Config
from src.modeling.e2e.data import E2EDecisionMonth, prepare_e2e_months
from src.modeling.e2e.portfolio import robust_portfolio_weights
from src.utils.paths import resolve_repo_path


@dataclass(frozen=True)
class SolverAuditConfig:
    run_name: str = "e2e_v2_full_month_20260404"
    output_root: str = "batuhan/artifacts/e2e"
    val_months: int = 6
    test_months: int = 8
    tolerance_objective: float = 1e-4
    tolerance_weights_l1: float = 0.05
    reference_solver_mode: str = "mirror_descent"
    comparison_solver_mode: str = "exact_cvxpy"

    def resolve_run_dir(self) -> Path:
        return resolve_repo_path(self.output_root) / self.run_name


def run_solver_fairness_audit(config: SolverAuditConfig | None = None) -> Path:
    config = config or SolverAuditConfig()
    run_dir = config.resolve_run_dir()
    with (run_dir / "config.json").open("r", encoding="utf-8") as handle:
        e2e_config = E2EV1Config.from_json(run_dir / "config.json")
    with (run_dir / "validation_selection_summary.json").open("r", encoding="utf-8") as handle:
        selection_summary = json.load(handle)
    candidate = selection_summary["selected_candidate"]

    month_data = prepare_e2e_months(e2e_config)
    output_dir = run_dir / "solver_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    audit_rows: list[dict[str, object]] = []
    for split, limit in (("val", config.val_months), ("test", config.test_months)):
        predictions = pd.read_parquet(run_dir / f"{split}_asset_predictions.parquet")
        by_month = {int(month.yyyymm): month for month in month_data[split]}
        selected_months = sorted(by_month)[:limit]
        for yyyymm in selected_months:
            decision = by_month[yyyymm]
            month_predictions = predictions.loc[predictions["yyyymm"].astype(int) == yyyymm].copy()
            month_predictions = align_prediction_month(decision, month_predictions)

            mu = month_predictions["prediction"].to_numpy(dtype=np.float64)
            sigma = decision.covariance.astype(np.float64, copy=False)
            reference_weights = solve_solver(
                mu=mu,
                sigma=sigma,
                lambda_=float(candidate["lambda"]),
                kappa=float(candidate["kappa"]),
                omega_mode=str(candidate["omega_mode"]),
                solver_mode=config.reference_solver_mode,
                config=e2e_config,
            )
            comparison_weights = solve_solver(
                mu=mu,
                sigma=sigma,
                lambda_=float(candidate["lambda"]),
                kappa=float(candidate["kappa"]),
                omega_mode=str(candidate["omega_mode"]),
                solver_mode=config.comparison_solver_mode,
                config=e2e_config,
            )
            month_returns = month_predictions["ret_tplus1"].to_numpy(dtype=np.float64)
            month_excess = month_predictions["excess_ret"].to_numpy(dtype=np.float64)
            reference_objective = objective_value(
                weights=reference_weights,
                mu=mu,
                sigma=sigma,
                lambda_=float(candidate["lambda"]),
                kappa=float(candidate["kappa"]),
                omega_mode=str(candidate["omega_mode"]),
            )
            comparison_objective = objective_value(
                weights=comparison_weights,
                mu=mu,
                sigma=sigma,
                lambda_=float(candidate["lambda"]),
                kappa=float(candidate["kappa"]),
                omega_mode=str(candidate["omega_mode"]),
            )
            audit_rows.append(
                {
                    "split": split,
                    "yyyymm": yyyymm,
                    "n_assets": len(mu),
                    "reference_solver_mode": config.reference_solver_mode,
                    "comparison_solver_mode": config.comparison_solver_mode,
                    "reference_objective": reference_objective,
                    "comparison_objective": comparison_objective,
                    "objective_gap_comparison_minus_reference": comparison_objective - reference_objective,
                    "reference_total_return": float(reference_weights @ month_returns),
                    "comparison_total_return": float(comparison_weights @ month_returns),
                    "total_return_gap_comparison_minus_reference": float((comparison_weights - reference_weights) @ month_returns),
                    "reference_excess_return": float(reference_weights @ month_excess),
                    "comparison_excess_return": float(comparison_weights @ month_excess),
                    "excess_return_gap_comparison_minus_reference": float((comparison_weights - reference_weights) @ month_excess),
                    "weight_l1_gap": float(np.abs(comparison_weights - reference_weights).sum()),
                    "weight_max_abs_gap": float(np.abs(comparison_weights - reference_weights).max()),
                    "weight_cosine_similarity": cosine_similarity(reference_weights, comparison_weights),
                }
            )

    monthly_table = pd.DataFrame(audit_rows)
    monthly_table.to_csv(output_dir / "monthly_solver_comparison.csv", index=False)
    summary = build_audit_summary(monthly_table, config)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text(build_audit_markdown(summary), encoding="utf-8")
    return output_dir


def align_prediction_month(
    decision: E2EDecisionMonth,
    prediction_month: pd.DataFrame,
) -> pd.DataFrame:
    expected_permnos = decision.metadata["permno"].astype(int).tolist()
    observed_permnos = prediction_month["permno"].astype(int).tolist()
    if expected_permnos == observed_permnos:
        return prediction_month.reset_index(drop=True)
    aligned = decision.metadata[["permno"]].merge(
        prediction_month,
        on="permno",
        how="left",
        validate="one_to_one",
    )
    if aligned["prediction"].isna().any():
        raise ValueError(f"Missing predictions after alignment for yyyymm={decision.yyyymm}.")
    return aligned.reset_index(drop=True)


def solve_solver(
    mu: np.ndarray,
    sigma: np.ndarray,
    lambda_: float,
    kappa: float,
    omega_mode: str,
    solver_mode: str,
    config: E2EV1Config,
) -> np.ndarray:
    weights = robust_portfolio_weights(
        mu=torch.from_numpy(mu.astype(np.float32, copy=False)),
        sigma=torch.from_numpy(sigma.astype(np.float32, copy=False)),
        lambda_=lambda_,
        kappa=kappa,
        omega_mode=omega_mode,
        solver_mode=solver_mode,
        steps=int(config.mirror_descent_steps),
        step_size=float(config.mirror_descent_step_size),
        fixed_n_assets=config.exact_fixed_n_assets,
    )
    return weights.detach().cpu().numpy().astype(np.float64, copy=False)


def build_omega_matrix(sigma: np.ndarray, omega_mode: str) -> np.ndarray:
    normalized = omega_mode.lower()
    if normalized == "identity":
        return np.eye(sigma.shape[0], dtype=np.float64)
    if normalized == "diag_sigma":
        return np.diag(np.sqrt(np.clip(np.diag(sigma), 1e-12, None)))
    raise ValueError(f"Unknown omega_mode={omega_mode}")


def objective_value(
    weights: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    lambda_: float,
    kappa: float,
    omega_mode: str,
    omega: np.ndarray | None = None,
) -> float:
    omega = build_omega_matrix(sigma, omega_mode) if omega is None else omega
    robustness = float(kappa) * np.linalg.norm(omega @ weights, ord=2)
    risk = (float(lambda_) / 2.0) * float(weights @ sigma @ weights)
    return float(mu @ weights) - robustness - risk


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 0:
        return 0.0
    return float(np.dot(left, right) / denom)


def build_audit_summary(monthly_table: pd.DataFrame, config: SolverAuditConfig) -> dict[str, object]:
    split_summary: dict[str, dict[str, object]] = {}
    for split, frame in monthly_table.groupby("split"):
        split_summary[split] = {
            "months_audited": int(len(frame)),
            "mean_objective_gap_comparison_minus_reference": float(frame["objective_gap_comparison_minus_reference"].mean()),
            "max_objective_gap_comparison_minus_reference": float(frame["objective_gap_comparison_minus_reference"].max()),
            "mean_weight_l1_gap": float(frame["weight_l1_gap"].mean()),
            "max_weight_l1_gap": float(frame["weight_l1_gap"].max()),
            "mean_weight_max_abs_gap": float(frame["weight_max_abs_gap"].mean()),
            "max_weight_max_abs_gap": float(frame["weight_max_abs_gap"].max()),
            "mean_weight_cosine_similarity": float(frame["weight_cosine_similarity"].mean()),
            "mean_total_return_gap_comparison_minus_reference": float(frame["total_return_gap_comparison_minus_reference"].mean()),
            "max_abs_total_return_gap_comparison_minus_reference": float(frame["total_return_gap_comparison_minus_reference"].abs().max()),
        }
    overall = {
        "months_audited": int(len(monthly_table)),
        "mean_objective_gap_comparison_minus_reference": float(monthly_table["objective_gap_comparison_minus_reference"].mean()),
        "max_objective_gap_comparison_minus_reference": float(monthly_table["objective_gap_comparison_minus_reference"].max()),
        "mean_weight_l1_gap": float(monthly_table["weight_l1_gap"].mean()),
        "max_weight_l1_gap": float(monthly_table["weight_l1_gap"].max()),
        "mean_weight_cosine_similarity": float(monthly_table["weight_cosine_similarity"].mean()),
        "mean_total_return_gap_comparison_minus_reference": float(monthly_table["total_return_gap_comparison_minus_reference"].mean()),
        "max_abs_total_return_gap_comparison_minus_reference": float(monthly_table["total_return_gap_comparison_minus_reference"].abs().max()),
    }
    acceptable = bool(
        overall["max_objective_gap_comparison_minus_reference"] <= config.tolerance_objective
        and overall["max_weight_l1_gap"] <= config.tolerance_weights_l1
    )
    return {
        "audit_config": {
            "run_name": config.run_name,
            "output_root": config.output_root,
            "val_months": config.val_months,
            "test_months": config.test_months,
            "tolerance_objective": config.tolerance_objective,
            "tolerance_weights_l1": config.tolerance_weights_l1,
            "reference_solver_mode": config.reference_solver_mode,
            "comparison_solver_mode": config.comparison_solver_mode,
        },
        "overall": overall,
        "by_split": split_summary,
        "solver_approximation_acceptable": acceptable,
        "interpretation": (
            "Reference solver is close enough to the comparison solver on the audited sample."
            if acceptable
            else "Reference solver shows materially large deviations versus the comparison solver on the audited sample."
        ),
    }


def build_audit_markdown(summary: dict[str, object]) -> str:
    overall = summary["overall"]
    lines = [
        "# E2E Solver Comparison",
        "",
        f"- months audited: `{overall['months_audited']}`",
        f"- reference solver: `{summary['audit_config']['reference_solver_mode']}`",
        f"- comparison solver: `{summary['audit_config']['comparison_solver_mode']}`",
        f"- solver approximation acceptable: `{summary['solver_approximation_acceptable']}`",
        f"- mean objective gap (comparison - reference): `{overall['mean_objective_gap_comparison_minus_reference']:.8f}`",
        f"- max objective gap (comparison - reference): `{overall['max_objective_gap_comparison_minus_reference']:.8f}`",
        f"- mean weight L1 gap: `{overall['mean_weight_l1_gap']:.6f}`",
        f"- max weight L1 gap: `{overall['max_weight_l1_gap']:.6f}`",
        f"- mean cosine similarity: `{overall['mean_weight_cosine_similarity']:.6f}`",
        f"- max abs total-return gap: `{overall['max_abs_total_return_gap_comparison_minus_reference']:.8f}`",
        "",
        summary["interpretation"],
        "",
    ]
    for split, payload in summary["by_split"].items():
        lines.append(f"## {split.upper()}")
        lines.append(
            f"- mean objective gap: `{payload['mean_objective_gap_comparison_minus_reference']:.8f}`, "
            f"max objective gap: `{payload['max_objective_gap_comparison_minus_reference']:.8f}`, "
            f"mean weight L1 gap: `{payload['mean_weight_l1_gap']:.6f}`, "
            f"max abs total-return gap: `{payload['max_abs_total_return_gap_comparison_minus_reference']:.8f}`"
        )
        lines.append("")
    return "\n".join(lines)
