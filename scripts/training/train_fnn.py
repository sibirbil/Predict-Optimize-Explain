from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
WORKSPACE_ROOT = THIS_FILE.parents[1]
SRC_ROOT = WORKSPACE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.fnn21.config import FNNTrainConfig
from src.fnn21.trainer import run_fnn_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a local 21April FNN on universe_500_v21April_stdz_macro.")
    parser.add_argument("--config", type=Path, required=True, help="Path to FNN config JSON.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional run-name override.")
    parser.add_argument("--epochs", type=int, default=None, help="Optional epochs override.")
    parser.add_argument("--max-train-rows", type=int, default=None, help="Optional train row cap.")
    parser.add_argument("--max-val-rows", type=int, default=None, help="Optional val row cap.")
    parser.add_argument("--max-test-rows", type=int, default=None, help="Optional test row cap.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = FNNTrainConfig.from_json(args.config)
    if args.run_name is not None:
        config = replace(config, run_name=args.run_name)
    if args.epochs is not None:
        config = replace(config, epochs=args.epochs)
    if args.max_train_rows is not None:
        config = replace(config, max_train_rows=args.max_train_rows)
    if args.max_val_rows is not None:
        config = replace(config, max_val_rows=args.max_val_rows)
    if args.max_test_rows is not None:
        config = replace(config, max_test_rows=args.max_test_rows)
    run_dir = run_fnn_training(config)
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
