import argparse
import datetime
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.scenario5 import (
    DEFAULT_BASE_SEED,
    DECISION_METRICS,
    Scenario5Runtime,
    anchor_mahalanobis_squared,
    box_violation_squared,
    l2_squared,
    mahalanobis_chi2_summary,
    mahalanobis_squared,
    regime_classifier,
    summarize_ess,
)


def parse_float_grid(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_grid(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def available_months(start_date: int, end_date: int) -> list[int]:
    macro_path = ROOT / "runtime_universe500" / "data" / "macro_final.parquet"
    frame = pd.read_parquet(macro_path)
    months = frame["yyyymm"].astype(int)
    return sorted(month for month in months.tolist() if int(start_date) <= month <= int(end_date))


def make_runtime_args(
    *,
    date: int,
    model: str,
    objective: str,
    reg_mode: str,
    constraint_mode: str,
    l2reg: float,
    eta: float,
    beta: float,
    random_start_scale: float,
    n_seeds: int,
    n_steps: int,
    base_seed: int,
    probe: str = "decision_fragility",
    model_a: str = "locked_e2e",
    model_b: str = "standardized_pto",
) -> SimpleNamespace:
    return SimpleNamespace(
        date=int(date),
        probe=str(probe),
        model=str(model),
        model_a=str(model_a),
        model_b=str(model_b),
        objective=str(objective),
        reg_mode=str(reg_mode),
        constraint_mode=str(constraint_mode),
        l2reg=float(l2reg),
        eta=float(eta),
        beta=float(beta),
        random_start_scale=float(random_start_scale),
        n_seeds=int(n_seeds),
        n_steps=int(n_steps),
        base_seed=int(base_seed),
    )


def one_config_scan(runtime: Scenario5Runtime) -> dict[str, object]:
    rows = []
    start_time = time.time()
    for seed_idx in range(1, runtime.args.n_seeds + 1):
        chain_seed = None if runtime.args.base_seed < 0 else int(runtime.args.base_seed) + seed_idx - 1
        if chain_seed is not None:
            torch.manual_seed(chain_seed)
        generator = None
        if chain_seed is not None:
            generator = torch.Generator(device=runtime.m0.device)
            generator.manual_seed(chain_seed)
        random_draw = torch.rand((9,), dtype=runtime.m0.dtype, device=runtime.m0.device, generator=generator)
        m_start = runtime.m0 + (runtime.upper - runtime.lower) * runtime.args.random_start_scale * random_draw
        m_last, m_traj, info = runtime.run_mala_chain(m_start, runtime.args.n_steps)
        ess = summarize_ess(m_traj, info["accepted"])
        final_decision, _ = runtime.decision_record(m_last)
        start_regime = runtime.regime_summary(m_start)
        final_regime = runtime.regime_summary(m_last)
        mah2 = mahalanobis_squared(m_last, runtime.m0)
        mah = mahalanobis_chi2_summary(mah2)
        anchor_mah2 = anchor_mahalanobis_squared(m_last, runtime.m0)
        anchor_mah = mahalanobis_chi2_summary(anchor_mah2)
        diff = m_last.detach().float() - runtime.m0.detach().float()
        anchor_empirical_mah2 = float((diff @ (runtime.empirical_macro_cov_inv @ diff)).item())
        anchor_empirical_mah = mahalanobis_chi2_summary(anchor_empirical_mah2)
        reference_model = runtime.args.model if runtime.args.model in runtime.anchor_metrics else runtime.model_names[0]
        anchor_return = runtime.anchor_metrics[reference_model]["return"]
        final_return = final_decision[f"{reference_model}_return"]
        anchor_sharpe = runtime.anchor_metrics[reference_model]["sharpe"]
        final_sharpe = final_decision[f"{reference_model}_sharpe"]
        decision_values = {}
        for key, value in final_decision.items():
            if isinstance(value, (int, float, bool, str)):
                decision_values[key] = value
        rows.append(
            {
                "seed": seed_idx,
                "target_metric": float(final_decision["target_metric"]),
                "accept_rate": float(info["accept_rate"]),
                "mean_accept_prob": float(info["mean_accept_prob"]),
                "ess_mean": float(ess["ess_mean"]),
                "l2_dist": l2_squared(m_last, runtime.m0) ** 0.5,
                "mah_dist": mah2 ** 0.5,
                "mah_chi2_tail": mah["mah_chi2_tail"],
                "anchor_mah_dist": anchor_mah2 ** 0.5,
                "anchor_mah_chi2_tail": anchor_mah["mah_chi2_tail"],
                "anchor_empirical_mah_dist": anchor_empirical_mah2 ** 0.5,
                "anchor_empirical_mah_chi2_tail": anchor_empirical_mah["mah_chi2_tail"],
                "box_violation": box_violation_squared(m_last, runtime.lower, runtime.upper) ** 0.5,
                "start_regime": start_regime["label"],
                "final_regime": final_regime["label"],
                "final_prob_expansion": float(final_regime["probabilities"].get("expansion", 0.0)),
                "final_prob_contraction": float(final_regime["probabilities"].get("contraction", 0.0)),
                "final_prob_financial_stress": float(final_regime["probabilities"].get("financial_stress", 0.0)),
                "return_delta": float(final_return - anchor_return),
                "sharpe_delta": float(final_sharpe - anchor_sharpe),
                "final_return": float(final_return),
                "final_sharpe": float(final_sharpe),
                "entropy_delta": float(final_decision.get("delta_entropy", np.nan)),
                "hhi_delta": float(final_decision.get("delta_hhi", np.nan)),
                "effective_n_delta": float(final_decision.get("delta_effective_n", np.nan)),
                "max_weight_delta": float(final_decision.get("delta_max_weight", np.nan)),
                "top10_weight_delta": float(final_decision.get("delta_top10_weight", np.nan)),
                **decision_values,
            }
        )
    frame = pd.DataFrame(rows)
    anchor_regime = runtime.regime_summary(runtime.m0)
    anchor_probs = anchor_regime["probabilities"]
    out: dict[str, object] = {
        "date": runtime.date,
        "probe": runtime.args.probe,
        "model": runtime.args.model,
        "model_a": runtime.args.model_a,
        "model_b": runtime.args.model_b,
        "objective": runtime.args.objective,
        "score_target_direction": objective_direction(runtime.args.objective),
        "reg_mode": runtime.args.reg_mode,
        "constraint_mode": runtime.args.constraint_mode,
        "l2reg": runtime.args.l2reg,
        "eta": runtime.args.eta,
        "beta": runtime.args.beta,
        "random_start_scale": runtime.args.random_start_scale,
        "n_seeds": runtime.args.n_seeds,
        "n_steps": runtime.args.n_steps,
        "anchor_regime": anchor_regime["label"],
        "anchor_true_label": regime_classifier.historical_label(runtime.date) or "unknown",
        "anchor_return": runtime.anchor_metrics[reference_model]["return"],
        "anchor_sharpe": runtime.anchor_metrics[reference_model]["sharpe"],
        "anchor_entropy": runtime.anchor_metrics[reference_model]["entropy"],
        "anchor_hhi": runtime.anchor_metrics[reference_model]["hhi"],
        "anchor_effective_n": runtime.anchor_metrics[reference_model]["effective_n"],
        "anchor_max_weight": runtime.anchor_metrics[reference_model]["max_weight"],
        "anchor_top10_weight": runtime.anchor_metrics[reference_model]["top10_weight"],
        "anchor_prob_expansion": float(anchor_probs.get("expansion", 0.0)),
        "anchor_prob_contraction": float(anchor_probs.get("contraction", 0.0)),
        "anchor_prob_financial_stress": float(anchor_probs.get("financial_stress", 0.0)),
        "elapsed_seconds": time.time() - start_time,
    }
    if runtime.args.probe != "decision_fragility":
        a_name = runtime.args.model_a
        b_name = runtime.args.model_b
        out.update(
            {
                "anchor_model_a_return": runtime.anchor_metrics[a_name]["return"],
                "anchor_model_b_return": runtime.anchor_metrics[b_name]["return"],
                "anchor_model_a_sharpe": runtime.anchor_metrics[a_name]["sharpe"],
                "anchor_model_b_sharpe": runtime.anchor_metrics[b_name]["sharpe"],
                "anchor_return_gap_a_minus_b": runtime.anchor_metrics[a_name]["return"] - runtime.anchor_metrics[b_name]["return"],
                "anchor_sharpe_gap_a_minus_b": runtime.anchor_metrics[a_name]["sharpe"] - runtime.anchor_metrics[b_name]["sharpe"],
            }
        )
    for col in [
        "target_metric",
        "accept_rate",
        "mean_accept_prob",
        "ess_mean",
        "l2_dist",
        "mah_dist",
        "mah_chi2_tail",
        "anchor_mah_dist",
        "anchor_mah_chi2_tail",
        "anchor_empirical_mah_dist",
        "anchor_empirical_mah_chi2_tail",
        "box_violation",
        "return_delta",
        "sharpe_delta",
        "final_return",
        "final_sharpe",
        "entropy_delta",
        "delta_entropy",
        "hhi_delta",
        "delta_hhi",
        "effective_n_delta",
        "delta_effective_n",
        "max_weight_delta",
        "delta_max_weight",
        "top10_weight_delta",
        "delta_top10_weight",
        "diversification_score",
        "allocation_l1_from_anchor",
        "allocation_l1_gap",
        "return_gap_a_minus_b",
        "sharpe_gap_a_minus_b",
        "return_gap_improvement_for_b",
        "sharpe_gap_improvement_for_b",
        "final_prob_expansion",
        "final_prob_contraction",
        "final_prob_financial_stress",
    ]:
        if col in frame.columns:
            out[f"{col}_mean"] = float(frame[col].mean())
            out[f"{col}_median"] = float(frame[col].median())
            out[f"{col}_min"] = float(frame[col].min())
            out[f"{col}_max"] = float(frame[col].max())
    if "b_return_matches_or_beats_a" in frame.columns:
        out["b_return_match_or_beat_share"] = float(frame["b_return_matches_or_beats_a"].astype(bool).mean())
    for model_name in runtime.model_names:
        for metric_name in DECISION_METRICS:
            col = f"{model_name}_{metric_name}"
            if col in frame.columns:
                out[f"{col}_median"] = float(frame[col].median())
                out[f"{col}_mean"] = float(frame[col].mean())
    out["final_expansion_share"] = float((frame["final_regime"] == "expansion").mean())
    out["final_contraction_share"] = float((frame["final_regime"] == "contraction").mean())
    out["final_financial_stress_share"] = float((frame["final_regime"] == "financial_stress").mean())
    out["final_non_anchor_share"] = float((frame["final_regime"] != anchor_regime["label"]).mean())
    out["seed_rows"] = frame.to_dict(orient="records")
    return out


def objective_direction(objective: str) -> str:
    """Direction of the reported target metric for scan scoring."""
    if objective in {"allocation_l1_close"}:
        return "minimize"
    return "maximize"


def infer_story(args: argparse.Namespace) -> str:
    if args.story != "auto":
        return args.story
    if args.probe == "decision_fragility" and args.model == "locked_e2e" and args.objective in {"entropy_max", "hhi_min"}:
        return "diversify"
    if (
        args.probe == "training_gap"
        and args.model_a == "locked_e2e"
        and args.model_b == "standardized_pto"
        and args.objective == "b_beats_a"
    ):
        return "pto_catchup"
    return "generic"


def score_row(row: dict[str, object], args: argparse.Namespace) -> tuple[bool, float, list[str]]:
    reasons = []
    story = infer_story(args)
    direction = objective_direction(str(row.get("objective", args.objective)))
    target_median = float(row["target_metric_median"])
    if story == "diversify":
        if float(row.get("anchor_hhi", 0.0)) < args.min_anchor_hhi and float(row.get("anchor_max_weight", 0.0)) < args.min_anchor_max_weight:
            reasons.append("anchor_concentration")
        if target_median < args.min_diversification_improvement:
            reasons.append("diversification")
        if float(row.get("delta_hhi_median", row.get("hhi_delta_median", 0.0))) >= -args.min_hhi_reduction:
            reasons.append("hhi_reduction")
    elif story == "pto_catchup":
        anchor_gap = float(row.get("anchor_return_gap_a_minus_b", 0.0))
        if anchor_gap < args.min_anchor_return_gap:
            reasons.append("anchor_gap")
        if float(row.get("b_return_match_or_beat_share", 0.0)) < args.min_pto_match_share:
            reasons.append("pto_match")
        if float(row.get("return_gap_improvement_for_b_median", 0.0)) < args.min_gap_improvement:
            reasons.append("gap_improvement")
    else:
        if direction == "minimize":
            if target_median > args.max_target_median:
                reasons.append("target")
        elif target_median < args.min_target_median:
            reasons.append("target")
    if row["l2_dist_median"] > args.max_l2_median:
        reasons.append("l2")
    tail_for_gate = float(row.get("anchor_empirical_mah_chi2_tail_median", row["mah_chi2_tail_median"]))
    if tail_for_gate < args.min_mah_tail_median:
        reasons.append("mah_tail")
    if row["accept_rate_mean"] < args.min_accept_rate or row["accept_rate_mean"] > args.max_accept_rate:
        reasons.append("accept")
    if row["box_violation_mean"] > args.max_box_violation:
        reasons.append("box")
    if abs(row["return_delta_median"]) > args.max_abs_return_delta:
        reasons.append("return_delta")
    if args.require_regime_shift and row["final_non_anchor_share"] < args.min_non_anchor_share:
        reasons.append("regime_shift")
    passed = not reasons
    score = 0.0
    if story == "diversify":
        score += 4.0 * max(0.0, target_median)
        score += 2.0 * max(0.0, float(row.get("anchor_hhi", 0.0)) - args.min_anchor_hhi)
        score += 1.0 * max(0.0, float(row.get("anchor_max_weight", 0.0)) - args.min_anchor_max_weight)
        score += 1.0 * max(0.0, -float(row.get("delta_hhi_median", row.get("hhi_delta_median", 0.0))))
        score += 0.5 * max(0.0, float(row.get("delta_effective_n_median", row.get("effective_n_delta_median", 0.0))))
    elif story == "pto_catchup":
        score += 20.0 * max(0.0, float(row.get("anchor_return_gap_a_minus_b", 0.0)))
        score += 10.0 * max(0.0, float(row.get("return_gap_improvement_for_b_median", 0.0)))
        score += 3.0 * float(row.get("b_return_match_or_beat_share", 0.0))
        score += 1.0 * max(0.0, -float(row.get("return_gap_a_minus_b_median", 0.0)))
    elif direction == "minimize":
        score += 3.0 * max(0.0, args.max_target_median - target_median)
    else:
        score += 3.0 * target_median
    score -= 0.35 * max(0.0, float(row["l2_dist_median"]) - 1.0)
    score += 1.5 * min(tail_for_gate, 0.25)
    score -= 0.75 * max(0.0, args.min_mah_tail_median - tail_for_gate)
    if "return_delta_median" in row:
        score -= 0.50 * max(0.0, abs(float(row["return_delta_median"])) - 0.01)
    score -= 1.5 * abs(float(row["accept_rate_mean"]) - 0.50)
    score += 0.35 * float(row["final_non_anchor_share"])
    if not passed:
        score -= 0.5 * len(reasons)
    return passed, score, reasons


def write_scan_outputs(
    out_dir: Path,
    rows: list[dict[str, object]],
    seed_detail_rows: list[dict[str, object]],
    args: argparse.Namespace,
    dates: list[int],
    l2regs: list[float],
    etas: list[float],
    betas: list[float],
) -> pd.DataFrame:
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["passed", "score"], ascending=[False, False])
    result.to_csv(out_dir / "scan_summary.csv", index=False)
    pd.DataFrame(seed_detail_rows).to_csv(out_dir / "scan_seed_details.csv", index=False)
    config = vars(args).copy()
    config.update({"dates": dates, "l2regs": l2regs, "etas": etas, "betas": betas})
    (out_dir / "scan_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan anchor months and hyperparameters for paper-grade Scenario 5 locked_e2e fragility."
    )
    parser.add_argument("--candidate-dates", default=None, help="Comma-separated yyyymm list. Overrides start/end range.")
    parser.add_argument("--start-date", type=int, default=201801)
    parser.add_argument("--end-date", type=int, default=202412)
    parser.add_argument("--probe", choices=["decision_fragility", "training_gap", "pair_disagreement"], default="decision_fragility")
    parser.add_argument("--story", choices=["auto", "generic", "diversify", "pto_catchup"], default="auto")
    parser.add_argument("--model", default="locked_e2e")
    parser.add_argument("--model-a", default="locked_e2e")
    parser.add_argument("--model-b", default="standardized_pto")
    parser.add_argument("--objective", default="allocation_l1_from_anchor")
    parser.add_argument("--reg-mode", choices=["l2", "var1"], default="l2")
    parser.add_argument("--constraint-mode", choices=["clip", "none", "box_barrier", "strong_soft"], default="box_barrier")
    parser.add_argument("--l2regs", default="0.3,0.5,0.7,1.0")
    parser.add_argument("--etas", default=None, help="Comma-separated MALA step sizes. Defaults are story-specific.")
    parser.add_argument("--betas", default="10")
    parser.add_argument("--n-seeds", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--random-start-scale", type=float, default=0.0)
    parser.add_argument("--min-target-median", type=float, default=0.30)
    parser.add_argument("--max-target-median", type=float, default=0.30)
    parser.add_argument("--max-l2-median", type=float, default=1.60)
    parser.add_argument("--min-mah-tail-median", type=float, default=0.05)
    parser.add_argument("--min-accept-rate", type=float, default=0.40)
    parser.add_argument("--max-accept-rate", type=float, default=0.60)
    parser.add_argument("--max-box-violation", type=float, default=1e-8)
    parser.add_argument("--max-abs-return-delta", type=float, default=0.03)
    parser.add_argument("--min-anchor-hhi", type=float, default=0.08)
    parser.add_argument("--min-anchor-max-weight", type=float, default=0.15)
    parser.add_argument("--min-diversification-improvement", type=float, default=0.02)
    parser.add_argument("--min-hhi-reduction", type=float, default=0.005)
    parser.add_argument("--min-anchor-return-gap", type=float, default=0.005)
    parser.add_argument("--min-pto-match-share", type=float, default=0.50)
    parser.add_argument("--min-gap-improvement", type=float, default=0.005)
    parser.add_argument("--require-regime-shift", action="store_true")
    parser.add_argument("--min-non-anchor-share", type=float, default=0.25)
    parser.add_argument("--top-k", type=int, default=30)
    args = parser.parse_args()
    supplied = set(sys.argv[1:])
    if args.story == "diversify":
        args.probe = "decision_fragility"
        args.model = "locked_e2e"
        if "--objective" not in supplied:
            args.objective = "entropy_max"
    elif args.story == "pto_catchup":
        args.probe = "training_gap"
        args.model_a = "locked_e2e"
        args.model_b = "standardized_pto"
        if "--objective" not in supplied:
            args.objective = "b_beats_a"

    story = infer_story(args)
    if args.etas is None:
        if story == "diversify":
            args.etas = "0.05,0.09,0.12"
        elif story == "pto_catchup":
            args.etas = "0.005,0.01,0.015,0.02"
        else:
            args.etas = "0.02,0.05,0.09"
    dates = parse_int_grid(args.candidate_dates) if args.candidate_dates else available_months(args.start_date, args.end_date)
    l2regs = parse_float_grid(args.l2regs)
    etas = parse_float_grid(args.etas)
    betas = parse_float_grid(args.betas)
    total = len(dates) * len(l2regs) * len(etas) * len(betas)
    scan_key = {
        "diversify": "scenario_e2e_diversify_anchor_scan",
        "pto_catchup": "scenario_pto_catchup_anchor_scan",
    }.get(story, "scenario5_fragility_anchor_scan")
    out_dir = ROOT / "scenario_outputs" / scan_key / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    seed_detail_rows = []
    run_idx = 0
    for date in dates:
        for l2reg in l2regs:
            for eta in etas:
                for beta in betas:
                    run_idx += 1
                    print(f"[{run_idx}/{total}] date={date} l2reg={l2reg:g} eta={eta:g} beta={beta:g}", flush=True)
                    rt_args = make_runtime_args(
                        date=date,
                        model=args.model,
                        objective=args.objective,
                        reg_mode=args.reg_mode,
                        constraint_mode=args.constraint_mode,
                        l2reg=l2reg,
                        eta=eta,
                        beta=beta,
                        random_start_scale=args.random_start_scale,
                        n_seeds=args.n_seeds,
                        n_steps=args.n_steps,
                        base_seed=args.base_seed,
                        probe=args.probe,
                        model_a=args.model_a,
                        model_b=args.model_b,
                    )
                    try:
                        runtime = Scenario5Runtime(rt_args)
                        row = one_config_scan(runtime)
                        seed_rows = row.pop("seed_rows")
                        passed, score, reasons = score_row(row, args)
                        row["passed"] = bool(passed)
                        row["score"] = float(score)
                        row["fail_reasons"] = ",".join(reasons)
                        rows.append(row)
                        for seed_row in seed_rows:
                            seed_detail_rows.append({**{k: row[k] for k in ["date", "l2reg", "eta", "beta"]}, **seed_row})
                        print(
                            f"  target_med={row['target_metric_median']:.3f}, "
                            f"L2_med={row['l2_dist_median']:.2f}, "
                            f"emp_tail_med={row.get('anchor_empirical_mah_chi2_tail_median', row['mah_chi2_tail_median']):.3g}, "
                            f"accept={row['accept_rate_mean']:.1%}, "
                            f"regime_shift={row['final_non_anchor_share']:.1%}, "
                            f"passed={passed}",
                            flush=True,
                        )
                    except Exception as exc:
                        rows.append(
                            {
                                "date": date,
                                "l2reg": l2reg,
                                "eta": eta,
                                "beta": beta,
                                "passed": False,
                                "score": -999.0,
                                "fail_reasons": f"error:{type(exc).__name__}",
                                "error": str(exc),
                            }
                        )
                        print(f"  failed: {type(exc).__name__}: {exc}", flush=True)
                    write_scan_outputs(out_dir, rows, seed_detail_rows, args, dates, l2regs, etas, betas)

    result = write_scan_outputs(out_dir, rows, seed_detail_rows, args, dates, l2regs, etas, betas)

    print(f"\nResults saved -> {out_dir.relative_to(ROOT)}")
    if not result.empty:
        cols = [
            "date",
            "l2reg",
            "eta",
            "anchor_hhi",
            "anchor_max_weight",
            "anchor_effective_n",
            "anchor_return_gap_a_minus_b",
            "target_metric_median",
            "delta_entropy_median",
            "delta_hhi_median",
            "delta_effective_n_median",
            "return_gap_a_minus_b_median",
            "return_gap_improvement_for_b_median",
            "b_return_match_or_beat_share",
            "l2_dist_median",
            "mah_chi2_tail_median",
            "anchor_empirical_mah_chi2_tail_median",
            "accept_rate_mean",
            "final_non_anchor_share",
            "return_delta_median",
            "anchor_regime",
            "passed",
            "score",
            "fail_reasons",
        ]
        available = [col for col in cols if col in result.columns]
        print("\nTop candidates:")
        print(result[available].head(args.top_k).to_string(index=False))


if __name__ == "__main__":
    main()
