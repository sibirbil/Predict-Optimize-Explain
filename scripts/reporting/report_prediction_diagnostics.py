from __future__ import annotations

import argparse
from pathlib import Path

from src.modeling.fnn.diagnostics import build_prediction_diagnostic_report
from src.modeling.fnn.evaluation import MonthlyEvaluationConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build compact diagnostics for a saved prediction run.")
    parser.add_argument("--run-name", required=True, type=str, help="Saved run folder name.")
    parser.add_argument(
        "--output-root",
        default="batuhan/artifacts/fnn_transfer",
        type=Path,
        help="Root directory that contains the saved run folder.",
    )
    parser.add_argument(
        "--quantiles",
        default=10,
        type=int,
        help="Number of within-month quantiles used for monotonicity diagnostics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = build_prediction_diagnostic_report(
        run_name=args.run_name,
        output_root=args.output_root,
        monthly_config=MonthlyEvaluationConfig(quantiles=args.quantiles),
    )
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
