from __future__ import annotations

import argparse

from src.modeling.portfolio_diagnostics import build_contrastive_event_study_package


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the broad SummerChild vs WinterWolf event-study diagnostics package.")
    parser.add_argument(
        "--contrastive-run-name",
        type=str,
        default="summerchild_winterwolf_exact_layer_20260409",
    )
    parser.add_argument(
        "--output-run-name",
        type=str,
        default="summerchild_winterwolf_broad_event_study_20260409",
    )
    parser.add_argument(
        "--contrastive-output-root",
        type=str,
        default="batuhan/artifacts/e2e_contrastive",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="batuhan/artifacts/portfolio_diagnostics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = build_contrastive_event_study_package(
        contrastive_run_name=args.contrastive_run_name,
        output_run_name=args.output_run_name,
        contrastive_output_root=args.contrastive_output_root,
        output_root=args.output_root,
    )
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
