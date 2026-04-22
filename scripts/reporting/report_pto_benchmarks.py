from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from src.configs.pto.robust_v1 import DEFAULT_PTO_CONFIG_PATH, PTOV1Config
from src.modeling.pto.benchmarks import build_pto_benchmark_package


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PTO benchmark comparison package for a saved PTO run.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_PTO_CONFIG_PATH,
        help="Path to the PTO v1 config JSON.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        required=True,
        help="Saved PTO run name under the configured PTO artifact root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PTOV1Config.from_json(args.config)
    config = replace(config, run_name=args.run_name)
    run_dir = config.resolve_output_root() / config.run_name
    build_pto_benchmark_package(config=config, run_dir=run_dir)
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
