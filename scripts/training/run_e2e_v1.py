from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from src.configs.e2e import DEFAULT_E2E_CONFIG_PATH, E2EV1Config
from src.modeling.e2e import run_e2e_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run clean batuhan E2E / PAO v1.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_E2E_CONFIG_PATH,
        help="Path to the E2E v1 config JSON.",
    )
    parser.add_argument("--run-name", type=str, default=None, help="Optional run-name override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = E2EV1Config.from_json(args.config)
    if args.run_name is not None:
        config = replace(config, run_name=args.run_name)
    run_dir = run_e2e_training(config)
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
