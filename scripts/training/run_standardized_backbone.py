from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from src.configs.model import FNNTrainConfig
from src.modeling.fnn.trainer import run_fnn_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FNN backbone retraining.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/configs/fnn/standardized_backbone.json"),
        help="Path to the FNN config JSON.",
    )
    parser.add_argument("--run-name", type=str, default=None, help="Optional run-name override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = FNNTrainConfig.from_json(args.config)
    if args.run_name is not None:
        config = replace(config, run_name=args.run_name)
    run_dir = run_fnn_training(config=config)
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
