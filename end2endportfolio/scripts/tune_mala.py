#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grid search driver for MALA hyperparameters.

For each combination of (eta, beta, risk) the script:
  1. Runs `generate_scenarios.py`
  2. Runs `run_scenario_diagnostics.py`
  3. Extracts summary metrics and aggregates them in a CSV table
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
SCENARIO_DIR = ARTIFACTS_DIR / "scenarios" / "mala_grid"
REPORTS_DIR = PROJECT_ROOT / "reports" / "diagnostics" / "mala_grid"
GEN_SCRIPT = PROJECT_ROOT / "scripts" / "generate_scenarios.py"
DIAG_SCRIPT = PROJECT_ROOT / "scripts" / "run_scenario_diagnostics.py"

DEFAULT_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "KMP_DUPLICATE_LIB_OK": "TRUE",
    "KMP_INIT_AT_FORK": "FALSE",
    "OMP_WAIT_POLICY": "PASSIVE",
    "PYTORCH_ENABLE_MPS_FALLBACK": "1",
}


def parse_floats(csv_str: str) -> List[float]:
    return [float(x.strip()) for x in csv_str.split(",") if x.strip()]


def build_cmd(base: List[str], extra_env: Dict[str, str]) -> Dict[str, object]:
    env = os.environ.copy()
    env.update(DEFAULT_ENV)
    env.update(extra_env)
    return {"args": base, "env": env, "check": True}


def run_generation(args, eta: float, beta: float, risk: float, run_id: str, scenario_path: Path) -> None:
    scenario_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(GEN_SCRIPT),
        "--timestamp",
        str(args.timestamp),
        "--steps",
        str(args.steps),
        "--eta",
        f"{eta}",
        "--beta",
        f"{beta}",
        "--risk",
        f"{risk}",
        "--clusters",
        str(args.clusters),
        "--centroid-weight",
        f"{args.centroid_weight}",
        "--output",
        str(scenario_path),
    ]
    if args.benchmark is not None:
        cmd.extend(["--benchmark", str(args.benchmark)])
    print(f"[run] generating scenarios for {run_id}")
    subprocess.run(**build_cmd(cmd, {}))


def run_diagnostics(args, scenario_path: Path, diag_dir: Path) -> None:
    diag_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(DIAG_SCRIPT),
        "--in",
        str(scenario_path),
        "--out",
        str(diag_dir),
        "--alpha",
        str(args.alpha),
        "--topk",
        str(args.topk),
        "--load-key",
        args.load_key,
    ]
    print(f"[run] diagnostics -> {diag_dir}")
    subprocess.run(**build_cmd(cmd, {}))


def read_csv_row(path: Path) -> Dict[str, float]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            return {k: float(v) for k, v in row.items()}
    raise RuntimeError(f"No rows found in {path}")


def summarize_run(diag_dir: Path) -> Dict[str, float]:
    global_row = read_csv_row(diag_dir / "global_diagnostics.csv")
    cov_row = read_csv_row(diag_dir / "cov_corr_diagnostics.csv")
    ew_row = read_csv_row(diag_dir / "equal_weight_portfolio_diagnostics.csv")
    summary = {
        "global_mean": global_row["global_mean"],
        "global_std": global_row["global_std"],
        "global_skew": global_row["global_skew"],
        "global_kurtosis": global_row["global_excess_kurtosis"],
        "avg_pairwise_corr": cov_row["avg_pairwise_corr"],
        "effective_rank": cov_row["effective_rank"],
        "top_eigenvalue": cov_row["top_eigenvalue"],
        "EW_mean": ew_row["mean"],
        "EW_std": ew_row["std"],
        "EW_VaR": ew_row["VaR_alpha"],
        "EW_CVaR": ew_row["CVaR_alpha"],
    }
    return summary


def format_run_id(eta: float, beta: float, risk: float) -> str:
    return f"eta{eta:.4f}_beta{beta:.2f}_risk{risk:.3f}".replace(".", "p")


def main():
    parser = argparse.ArgumentParser(description="Grid search for MALA hyperparameters.")
    parser.add_argument("--etas", type=str, required=True, help="Comma-separated list of eta values.")
    parser.add_argument("--betas", type=str, required=True, help="Comma-separated list of beta values.")
    parser.add_argument("--risks", type=str, required=True, help="Comma-separated list of lambda (risk) values.")
    parser.add_argument("--timestamp", type=int, default=0, help="Validation timestamp index to seed from.")
    parser.add_argument("--steps", type=int, default=800, help="Number of MALA steps per run.")
    parser.add_argument("--clusters", type=int, default=4, help="Centroid count for regularisation.")
    parser.add_argument("--centroid-weight", type=float, default=0.1, help="Centroid penalty weight.")
    parser.add_argument("--benchmark", type=float, default=None, help="Override benchmark return.")
    parser.add_argument("--alpha", type=float, default=0.95, help="Diagnostics VaR/CVaR alpha.")
    parser.add_argument("--topk", type=int, default=50, help="Diagnostics eigenvalue summary count.")
    parser.add_argument("--load-key", default="trajectory", help="Key inside scenario files containing the matrix.")
    parser.add_argument("--summary", default=str(REPORTS_DIR / "summary.csv"), help="Path to write summary CSV.")
    args = parser.parse_args()

    etas = parse_floats(args.etas)
    betas = parse_floats(args.betas)
    risks = parse_floats(args.risks)

    SCENARIO_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, float]] = []

    for eta, beta, risk in product(etas, betas, risks):
        run_id = format_run_id(eta, beta, risk)
        scenario_path = SCENARIO_DIR / f"{run_id}.pt"
        diag_dir = REPORTS_DIR / run_id
        try:
            run_generation(args, eta, beta, risk, run_id, scenario_path)
            run_diagnostics(args, scenario_path, diag_dir)
            summary = summarize_run(diag_dir)
            summary.update({"eta": eta, "beta": beta, "risk": risk, "run_id": run_id})
            records.append(summary)
        except subprocess.CalledProcessError as exc:
            print(f"[warn] run {run_id} failed with returncode {exc.returncode}")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[warn] run {run_id} failed: {exc}")

    if records:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "run_id",
            "eta",
            "beta",
            "risk",
            "global_mean",
            "global_std",
            "global_skew",
            "global_kurtosis",
            "avg_pairwise_corr",
            "effective_rank",
            "top_eigenvalue",
            "EW_mean",
            "EW_std",
            "EW_VaR",
            "EW_CVaR",
        ]
        with summary_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in records:
                writer.writerow(rec)
        print(f"[info] summary written to {summary_path}")
    else:
        print("[warn] no successful runs recorded.")


if __name__ == "__main__":
    main()
