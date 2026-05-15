"""Build a submission-readiness table for scenario runs.

The report is intentionally conservative: it separates VAR(1)-forecast /
VAR-innovation diagnostics from empirical-anchor diagnostics so manuscript text
does not overclaim plausibility when only the empirical metric passes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _timestamp_from_config(path: Path) -> str | None:
    match = re.search(r"config_(.+)\.json$", path.name)
    return match.group(1) if match else None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _first_existing(run_dir: Path, names: list[str]) -> Path | None:
    for name in names:
        path = run_dir / name
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def _mode_value(frame: pd.DataFrame, col: str) -> str:
    if col not in frame.columns or frame.empty:
        return ""
    values = frame[col].dropna()
    if values.empty:
        return ""
    mode = values.astype(int).mode()
    return str(int(mode.iloc[0])) if not mode.empty else ""


def _median(frame: pd.DataFrame, col: str) -> float:
    if col not in frame.columns or frame.empty:
        return float("nan")
    return float(pd.to_numeric(frame[col], errors="coerce").median())


def _mean(frame: pd.DataFrame, col: str) -> float:
    if col not in frame.columns or frame.empty:
        return float("nan")
    return float(pd.to_numeric(frame[col], errors="coerce").mean())


def _regime_mix(frame: pd.DataFrame) -> str:
    if "regime" not in frame.columns or frame.empty:
        return ""
    shares = frame["regime"].value_counts(normalize=True).mul(100.0)
    return "; ".join(f"{name}:{value:.1f}%" for name, value in shares.items())


def _target_info(final_frame: pd.DataFrame, scenario: str) -> tuple[str, float]:
    if "target_metric" in final_frame.columns:
        name = str(final_frame["target_metric_name"].dropna().iloc[0]) if "target_metric_name" in final_frame.columns and final_frame["target_metric_name"].notna().any() else "target_metric"
        return name, _median(final_frame, "target_metric")
    if scenario == "scenario4" and "return_gap" in final_frame.columns:
        return "SC_minus_WW_return_gap", _median(final_frame, "return_gap")
    return "", float("nan")


def _verdict(row: dict[str, Any]) -> tuple[str, str]:
    path = str(row["run_path"])
    target = float(row.get("target_median", float("nan")))
    n = int(row.get("n_seed_rows", 0) or 0)
    accept = float(row.get("accept_rate_mean", float("nan")))
    box = float(row.get("box_violation_mean", float("nan")))
    emp_tail = float(row.get("empirical_anchor_tail_median", float("nan")))
    probe = str(row.get("probe", ""))
    objective = str(row.get("objective", ""))

    if "scenario4_202004/runs/20260424_102040" in path:
        return "main", "Locked Q1 flagship; use NN analog narrative and avoid unsupported VAR-plausibility claim."
    if n <= 0:
        return "reject", "Missing or empty seed/final diagnostics."
    if n < 5:
        return "reject", "Too few completed seeds for manuscript use."

    production_ready = (
        n >= 20
        and 0.20 <= accept <= 0.85
        and (np.isnan(box) or box <= 1e-8)
        and (not np.isnan(emp_tail) and emp_tail >= 0.05)
    )
    if production_ready and probe == "training_gap" and objective == "allocation_l1_gap" and target >= 0.50:
        return "main", "Production-sized Q2 candidate; frame as empirically local if VAR tail is small."
    if production_ready and probe == "decision_fragility" and target >= 0.30:
        return "appendix", "Production-sized fragility evidence, but Q3 should stay optional unless story is cleaner."
    if n >= 5:
        return "appendix", "Diagnostic or sensitivity evidence; useful for meeting/appendix, not main claim."
    return "reject", "Does not meet minimum manuscript criteria."


def summarize_run(config_path: Path) -> dict[str, Any]:
    run_dir = config_path.parent
    ts = _timestamp_from_config(config_path)
    cfg = _read_json(config_path)
    scenario = run_dir.parents[1].name.split("_", 1)[0] if len(run_dir.parents) > 1 else ""
    date = cfg.get("DATE", "")

    final_path = _first_existing(
        run_dir,
        [
            f"final_state_diagnostics_{ts}_enriched.csv",
            f"final_state_diagnostics_{ts}.csv",
        ],
    )
    seed_path = _first_existing(
        run_dir,
        [
            f"seed_summary_{ts}_enriched.csv",
            f"seed_summary_{ts}.csv",
        ],
    )
    sample_path = _first_existing(
        run_dir,
        [
            f"generated_macro_sample_postburnin_{ts}_enriched.csv",
            f"generated_macro_sample_postburnin_{ts}.csv",
            f"generated_macro_sample_{ts}.csv",
        ],
    )

    final_frame = _read_csv(final_path) if final_path else pd.DataFrame()
    seed_frame = _read_csv(seed_path) if seed_path else pd.DataFrame()
    sample_frame = _read_csv(sample_path) if sample_path else pd.DataFrame()
    target_name, target_median = _target_info(final_frame, scenario)

    row: dict[str, Any] = {
        "run_path": str(run_dir.relative_to(ROOT)),
        "scenario": scenario,
        "date": date,
        "probe": cfg.get("PROBE", "sc_beats_ww" if scenario == "scenario4" else ""),
        "objective": cfg.get("OBJECTIVE", cfg.get("CONTRAST_FUNCTION", "sc_beats_ww" if scenario == "scenario4" else "")),
        "n_seeds_config": cfg.get("N_SEEDS", ""),
        "n_seed_rows": len(seed_frame) if not seed_frame.empty else len(final_frame),
        "target_name": target_name,
        "target_median": target_median,
        "accept_rate_mean": _mean(final_frame if "accept_rate" in final_frame.columns else seed_frame, "accept_rate"),
        "box_violation_mean": _mean(final_frame if "box_violation" in final_frame.columns else seed_frame, "box_violation"),
        "final_regime_mix": _regime_mix(final_frame),
        "sample_regime_mix": _regime_mix(sample_frame),
        "var_forecast_tail_median": _median(final_frame, "mah_chi2_tail"),
        "anchor_var_tail_median": _median(final_frame, "anchor_mah_chi2_tail"),
        "empirical_anchor_tail_median": _median(final_frame, "anchor_empirical_mah_chi2_tail"),
        "nn_var1_modal_yyyymm": _mode_value(final_frame, "nn_var1_yyyymm_1"),
        "nn_hist_modal_yyyymm": _mode_value(final_frame, "nn_hist_yyyymm_1"),
        "nn_eucl_modal_yyyymm": _mode_value(final_frame, "nn_eucl_yyyymm_1"),
        "config_path": str(config_path.relative_to(ROOT)),
    }
    row["manuscript_verdict"], row["verdict_reason"] = _verdict(row)
    return row


def build_report(output_prefix: Path) -> tuple[Path, Path, pd.DataFrame]:
    config_paths = sorted((ROOT / "scenario_outputs").glob("scenario*_*/runs/*/config_*.json"))
    rows = [summarize_run(path) for path in config_paths]
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(
            ["manuscript_verdict", "scenario", "date", "run_path"],
            key=lambda s: s.map({"main": 0, "appendix": 1, "reject": 2}).fillna(s) if s.name == "manuscript_verdict" else s,
        )

    csv_path = output_prefix.with_suffix(".csv")
    md_path = output_prefix.with_suffix(".md")
    frame.to_csv(csv_path, index=False)

    display_cols = [
        "manuscript_verdict",
        "date",
        "probe",
        "objective",
        "n_seed_rows",
        "target_name",
        "target_median",
        "accept_rate_mean",
        "empirical_anchor_tail_median",
        "nn_var1_modal_yyyymm",
        "nn_hist_modal_yyyymm",
        "nn_eucl_modal_yyyymm",
        "run_path",
        "verdict_reason",
    ]
    available = [col for col in display_cols if col in frame.columns]
    lines = [
        "# Submission Readiness Report",
        "",
        "Generated from local `scenario_outputs/*/runs/*/config_*.json` artifacts.",
        "",
        frame[available].to_markdown(index=False) if not frame.empty else "_No scenario runs found._",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path, frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate scenario submission-readiness CSV and Markdown reports.")
    parser.add_argument("--output-prefix", type=Path, default=ROOT / "SUBMISSION_READINESS_REPORT")
    args = parser.parse_args()
    output_prefix = args.output_prefix
    if not output_prefix.is_absolute():
        output_prefix = ROOT / output_prefix
    csv_path, md_path, frame = build_report(output_prefix)
    print(f"Wrote {csv_path.relative_to(ROOT)}")
    print(f"Wrote {md_path.relative_to(ROOT)}")
    if not frame.empty:
        print(frame["manuscript_verdict"].value_counts().to_string())


if __name__ == "__main__":
    main()
