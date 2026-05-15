from __future__ import annotations

import argparse
import datetime
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RUNTIME = ROOT / "runtime_universe500"
MACRO_COLUMNS = ["dp", "ep", "bm", "ntis", "tbl", "tms", "dfy", "svar", "infl"]


def parse_dates(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def candidate_dates(start_date: int | None, end_date: int | None, explicit: list[int] | None) -> list[int]:
    if explicit is not None:
        return sorted(set(explicit))
    macro = pd.read_parquet(RUNTIME / "data" / "macro_final.parquet")
    meta_test = pd.read_parquet(RUNTIME / "data" / "metadata_test.parquet")
    macro["yyyymm"] = macro["yyyymm"].astype(int)
    meta_test["yyyymm"] = meta_test["yyyymm"].astype(int)
    dates = sorted(set(macro["yyyymm"]).intersection(set(meta_test["yyyymm"])))
    if start_date is not None:
        dates = [date for date in dates if date >= start_date]
    if end_date is not None:
        dates = [date for date in dates if date <= end_date]
    return dates


def latest_run_stamp(stdout: str) -> str:
    match = re.search(r"Results saved .+ \((\d{8}_\d{6})\)", stdout)
    if not match:
        raise RuntimeError("Could not parse scenario4 timestamp from stdout.")
    return match.group(1)


def run_scenario4(date: int, args: argparse.Namespace) -> dict[str, object]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "scenario4.py"),
        "--date",
        str(date),
        "--n-seeds",
        str(args.n_seeds),
        "--n-steps",
        str(args.n_steps),
        "--reg-mode",
        "var1",
        "--constraint-mode",
        args.constraint_mode,
        "--contrast-function",
        args.target,
        "--l2reg",
        str(args.l2reg),
        "--eta",
        str(args.eta),
        "--beta",
        str(args.beta),
        "--random-start-scale",
        str(args.random_start_scale),
        "--base-seed",
        str(args.base_seed),
    ]
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if proc.returncode != 0:
        return {
            "date": date,
            "status": "failed",
            "error": proc.stdout[-4000:],
        }

    stamp = latest_run_stamp(proc.stdout)
    out_dir = ROOT / "scenario_outputs" / f"scenario4_{date}"
    config = json.loads((out_dir / f"config_{stamp}.json").read_text())
    final = pd.read_csv(out_dir / f"final_state_diagnostics_{stamp}.csv")
    sample = pd.read_csv(out_dir / f"generated_macro_sample_postburnin_{stamp}.csv")

    anchor_probs = config["ANCHOR_REGIME"]["probabilities"]
    anchor_regime = config["ANCHOR_REGIME"]["label"]
    anchor_expansion_prob = float(anchor_probs.get("expansion", 0.0))
    anchor_contraction_prob = float(anchor_probs.get("contraction", 0.0))
    anchor_financial_stress_prob = float(anchor_probs.get("financial_stress", 0.0))

    final_fs_prob = float(final.get("prob_financial_stress", pd.Series(dtype=float)).mean())
    final_expansion_prob = float(final.get("prob_expansion", pd.Series(dtype=float)).mean())
    final_contraction_prob = float(final.get("prob_contraction", pd.Series(dtype=float)).mean())
    sample_fs_prob = float(sample.get("prob_financial_stress", pd.Series(dtype=float)).mean())
    sample_expansion_prob = float(sample.get("prob_expansion", pd.Series(dtype=float)).mean())
    final_fs_share = float((final["regime"] == "financial_stress").mean()) if "regime" in final else np.nan
    final_expansion_share = float((final["regime"] == "expansion").mean()) if "regime" in final else np.nan
    final_gap_mean = float(final["return_gap"].mean())
    final_gap_median = float(final["return_gap"].median())
    sc_win_rate = float((final["winner"] == "summer_child").mean())
    ww_win_rate = float((final["winner"] == "winter_wolf").mean())
    median_tail = float(final["mah_chi2_tail"].median()) if "mah_chi2_tail" in final else np.nan
    inside_95 = float((final["mah_chi2_percentile"] <= 0.95).mean()) if "mah_chi2_percentile" in final else np.nan

    shifts = {}
    for col in MACRO_COLUMNS:
        std_col = f"{col}_std"
        if std_col in sample:
            anchor_std = float((sample[col].iloc[0] - sample[col].iloc[0]) * 0.0)
            # Prefer raw shift columns in final state summaries when present.
            shifts[f"median_{col}_std"] = float(sample[std_col].median())
    raw_shift_cols = [col for col in final.columns if col in MACRO_COLUMNS]
    top_shift_text = ""
    if raw_shift_cols:
        historical = pd.read_csv(out_dir / f"historical_macro_panel_{stamp}.csv")
        anchor = historical.loc[historical["yyyymm"].astype(int).eq(date), MACRO_COLUMNS].iloc[0]
        med_shift = final[MACRO_COLUMNS].median() - anchor
        top = med_shift.abs().sort_values(ascending=False).head(3).index.tolist()
        top_shift_text = ", ".join(f"{col} {med_shift[col]:+.4f}" for col in top)

    anchor_gap = float(config["ANCHOR_RETURN_GAP"])
    if args.target == "sc_beats_ww":
        pass_anchor = anchor_gap < 0
        pass_flip = final_gap_mean > 0 and sc_win_rate >= args.min_win_rate
        target_margin = final_gap_mean
        target_win_rate = sc_win_rate
    else:
        pass_anchor = anchor_gap > 0
        pass_flip = final_gap_mean < 0 and ww_win_rate >= args.min_win_rate
        target_margin = -final_gap_mean
        target_win_rate = ww_win_rate
    pass_tail = np.isfinite(median_tail) and median_tail >= args.min_tail
    final_regime_mode = str(final["regime"].mode().iloc[0]) if "regime" in final and not final.empty else ""
    expansion_prob_delta = final_expansion_prob - anchor_expansion_prob
    contraction_prob_delta = final_contraction_prob - anchor_contraction_prob
    hard_regime_change = final_regime_mode != anchor_regime
    pass_not_stress = (
        np.isfinite(final_fs_prob)
        and final_fs_prob <= args.max_financial_stress_prob
        and (not np.isfinite(final_fs_share) or final_fs_share <= args.max_financial_stress_share)
    )
    pass_regime_story = (
        hard_regime_change
        or expansion_prob_delta >= args.min_expansion_prob_delta
    )
    passed = pass_anchor and pass_flip and pass_tail and pass_not_stress and pass_regime_story

    score = (
        100.0 * target_margin
        + 2.0 * target_win_rate
        + 1.0 * min(median_tail if np.isfinite(median_tail) else 0.0, 0.5)
        - 1.0 * max(final_fs_prob if np.isfinite(final_fs_prob) else 1.0, 0.0)
        + 2.0 * max(expansion_prob_delta, 0.0)
        + 0.75 * max(final_expansion_share, 0.0)
        + (1.0 if hard_regime_change else 0.0)
    )

    return {
        "date": date,
        "status": "ok",
        "stamp": stamp,
        "passed": bool(passed),
        "target": args.target,
        "pass_anchor_opposite_model_wins": bool(pass_anchor),
        "pass_counterfactual_target_model_wins": bool(pass_flip),
        "pass_var1_tail": bool(pass_tail),
        "pass_not_financial_stress": bool(pass_not_stress),
        "pass_regime_story": bool(pass_regime_story),
        "score": float(score),
        "anchor_gap": anchor_gap,
        "anchor_summer_return": float(config["ANCHOR_SUMMER_RETURN"]),
        "anchor_winter_return": float(config["ANCHOR_WINTER_RETURN"]),
        "anchor_regime": anchor_regime,
        "anchor_true_label": config.get("ANCHOR_TRUE_LABEL"),
        "anchor_contraction_prob": anchor_contraction_prob,
        "anchor_expansion_prob": anchor_expansion_prob,
        "anchor_financial_stress_prob": anchor_financial_stress_prob,
        "final_gap_mean": final_gap_mean,
        "final_gap_median": final_gap_median,
        "sc_win_rate": sc_win_rate,
        "ww_win_rate": ww_win_rate,
        "target_margin_mean": target_margin,
        "target_win_rate": target_win_rate,
        "median_mah_dist": float(final["mah_dist"].median()),
        "median_mah_tail": median_tail,
        "share_inside_95": inside_95,
        "final_financial_stress_prob": final_fs_prob,
        "final_contraction_prob": final_contraction_prob,
        "final_expansion_prob": final_expansion_prob,
        "expansion_prob_delta": expansion_prob_delta,
        "contraction_prob_delta": contraction_prob_delta,
        "sample_financial_stress_prob": sample_fs_prob,
        "sample_expansion_prob": sample_expansion_prob,
        "final_financial_stress_share": final_fs_share,
        "final_expansion_share": final_expansion_share,
        "final_regime_mode": final_regime_mode,
        "hard_regime_change": bool(hard_regime_change),
        "top_raw_macro_shifts": top_shift_text,
        **shifts,
    }


def write_markdown(frame: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> None:
    ranked = frame.sort_values(["passed", "score"], ascending=[False, False])
    lines = [
        "# Scenario 4 VAR(1) Anchor Scan",
        "",
        "Selection rules:",
        "",
        f"- anchor has the opposite model winning for target `{args.target}`",
        "- generated final states have the target model winning",
        f"- median VAR(1) chi-square tail probability >= `{args.min_tail}`",
        f"- final financial-stress probability <= `{args.max_financial_stress_prob}`",
        f"- final financial-stress regime share <= `{args.max_financial_stress_share}`",
        f"- hard regime changes, or expansion probability rises by at least `{args.min_expansion_prob_delta:.0%}`",
        "",
        "Top candidates:",
        "",
    ]
    cols = [
        "date",
        "passed",
        "score",
        "anchor_gap",
        "final_gap_mean",
        "sc_win_rate",
        "ww_win_rate",
        "target_margin_mean",
        "target_win_rate",
        "median_mah_tail",
        "final_financial_stress_prob",
        "anchor_expansion_prob",
        "final_expansion_prob",
        "expansion_prob_delta",
        "final_regime_mode",
        "hard_regime_change",
        "top_raw_macro_shifts",
        "stamp",
    ]
    lines.append(ranked[cols].head(20).to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    (output_dir / "anchor_scan_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Systematically scan Scenario 4 VAR(1) anchors.")
    parser.add_argument("--candidate-dates", type=str, default=None, help="Comma-separated yyyymm list.")
    parser.add_argument("--start-date", type=int, default=None)
    parser.add_argument("--end-date", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--n-seeds", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=250)
    parser.add_argument("--l2reg", type=float, default=0.05)
    parser.add_argument("--eta", type=float, default=0.001)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--constraint-mode", choices=["box_barrier", "none", "clip", "strong_soft"], default="box_barrier")
    parser.add_argument("--random-start-scale", type=float, default=0.0)
    parser.add_argument("--base-seed", type=int, default=20260424)
    parser.add_argument("--target", choices=["sc_beats_ww", "ww_beats_sc"], default="sc_beats_ww")
    parser.add_argument("--min-tail", type=float, default=0.05)
    parser.add_argument("--min-win-rate", type=float, default=0.75)
    parser.add_argument("--min-sc-win-rate", type=float, default=None, help="Deprecated alias for --min-win-rate.")
    parser.add_argument("--max-financial-stress-prob", type=float, default=0.50)
    parser.add_argument("--max-financial-stress-share", type=float, default=0.50)
    parser.add_argument("--min-expansion-prob-delta", type=float, default=0.25)
    args = parser.parse_args()
    if args.min_sc_win_rate is not None:
        args.min_win_rate = args.min_sc_win_rate

    dates = candidate_dates(args.start_date, args.end_date, parse_dates(args.candidate_dates))
    if args.max_candidates is not None:
        dates = dates[: args.max_candidates]
    if not dates:
        raise SystemExit("No candidate dates selected.")

    output_dir = ROOT / "scenario_outputs" / "scenario4_var1_anchor_scan" / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, date in enumerate(dates, start=1):
        print(f"[{idx}/{len(dates)}] scanning {date}")
        row = run_scenario4(date, args)
        rows.append(row)
        pd.DataFrame(rows).to_csv(output_dir / "anchor_scan_partial.csv", index=False)
        if row.get("status") == "ok":
            print(
                f"  gap_anchor={row['anchor_gap']:+.2%}, gap_final={row['final_gap_mean']:+.2%}, "
                f"tail={row['median_mah_tail']:.3g}, fs_prob={row['final_financial_stress_prob']:.1%}, "
                f"passed={row['passed']}"
            )
        else:
            print(f"  failed: {str(row.get('error', ''))[:240]}")

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "anchor_scan_results.csv", index=False)
    write_markdown(frame[frame["status"].eq("ok")].copy(), output_dir, args)
    print(f"\nResults saved -> {output_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
