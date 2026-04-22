from __future__ import annotations

import argparse

from src.modeling.portfolio_diagnostics import (
    build_difference_portfolio_diagnostics_package,
    build_monthly_portfolio_diagnostics_package,
    build_presentation_portfolio_diagnostics_package,
    build_portfolio_diagnostics_package,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build portfolio diagnostics packages.")
    parser.add_argument("--pto-run-name", type=str, default="pto_robust_v1_staged_20260404")
    parser.add_argument("--e2e-run-name", type=str, default="e2e_v2_full_month_20260404")
    parser.add_argument(
        "--yearly-output-run-name",
        type=str,
        default="test_period_wealth_and_industry_diagnostics_20260404",
    )
    parser.add_argument(
        "--monthly-output-run-name",
        type=str,
        default="test_period_monthly_diagnostics_20260404",
    )
    parser.add_argument(
        "--difference-output-run-name",
        type=str,
        default="test_period_difference_diagnostics_20260404",
    )
    parser.add_argument(
        "--presentation-output-run-name",
        type=str,
        default="test_period_presentation_figures_20260404",
    )
    parser.add_argument(
        "--package",
        choices=("yearly", "monthly", "difference", "presentation", "both", "all"),
        default="both",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.package in {"yearly", "both", "all"}:
        yearly_output_dir = build_portfolio_diagnostics_package(
            pto_run_name=args.pto_run_name,
            e2e_run_name=args.e2e_run_name,
            output_run_name=args.yearly_output_run_name,
        )
        print(f"yearly_output_dir={yearly_output_dir}")
    if args.package in {"monthly", "both", "all"}:
        monthly_output_dir = build_monthly_portfolio_diagnostics_package(
            pto_run_name=args.pto_run_name,
            e2e_run_name=args.e2e_run_name,
            output_run_name=args.monthly_output_run_name,
        )
        print(f"monthly_output_dir={monthly_output_dir}")
    if args.package in {"difference", "all"}:
        difference_output_dir = build_difference_portfolio_diagnostics_package(
            pto_run_name=args.pto_run_name,
            e2e_run_name=args.e2e_run_name,
            output_run_name=args.difference_output_run_name,
        )
        print(f"difference_output_dir={difference_output_dir}")
    if args.package in {"presentation", "all"}:
        presentation_output_dir = build_presentation_portfolio_diagnostics_package(
            pto_run_name=args.pto_run_name,
            e2e_run_name=args.e2e_run_name,
            output_run_name=args.presentation_output_run_name,
        )
        print(f"presentation_output_dir={presentation_output_dir}")


if __name__ == "__main__":
    main()
