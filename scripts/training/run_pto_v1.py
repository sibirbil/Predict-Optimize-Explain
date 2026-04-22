from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from src.configs.pto.robust_v1 import DEFAULT_PTO_CONFIG_PATH, PTOV1Config
from src.modeling.pto.tuner import run_pto_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run clean batuhan PTO v1 robust optimization.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_PTO_CONFIG_PATH,
        help="Path to the PTO v1 config JSON.",
    )
    parser.add_argument("--run-name", type=str, default=None, help="Optional run-name override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PTOV1Config.from_json(args.config)
    if args.run_name is not None:
        config = replace(config, run_name=args.run_name)
    run_dir = run_pto_pipeline(config)
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
