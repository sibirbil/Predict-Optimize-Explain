from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from src.configs.e2e import ContrastiveE2EConfig, DEFAULT_CONTRASTIVE_E2E_CONFIG_PATH
from src.modeling.e2e import run_contrastive_e2e_training, run_kappa_sensitivity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SummerChild / WinterWolf contrastive E2E training.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONTRASTIVE_E2E_CONFIG_PATH,
        help="Path to the contrastive E2E config JSON.",
    )
    parser.add_argument("--run-name", type=str, default=None, help="Optional run-name override.")
    parser.add_argument("--kappa-sensitivity", action="store_true", help="Run kappa sensitivity grid after main pair.")
    parser.add_argument("--kappa-sensitivity-only", action="store_true", help="Run only the kappa sensitivity grid (skip main pair).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ContrastiveE2EConfig.from_json(args.config)
    if args.run_name is not None:
        config = replace(config, run_name=args.run_name)

    if not args.kappa_sensitivity_only:
        run_dir = run_contrastive_e2e_training(config)
        print(f"main_run_dir={run_dir}")

    if args.kappa_sensitivity or args.kappa_sensitivity_only:
        sensitivity_config = replace(
            config,
            kappa_sensitivity_grid=(0.25, 0.5, 1.0) if config.kappa_sensitivity_grid is None else config.kappa_sensitivity_grid,
        )
        sensitivity_dir = run_kappa_sensitivity(sensitivity_config)
        print(f"sensitivity_dir={sensitivity_dir}")


if __name__ == "__main__":
    main()
