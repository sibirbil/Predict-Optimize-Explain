from __future__ import annotations

import argparse
import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
WORKSPACE_ROOT = THIS_FILE.parents[1]
SRC_ROOT = WORKSPACE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.fnn21.tuner import run_fnn_grid_search


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local 21April FNN grid search.")
    parser.add_argument("--config", type=Path, required=True, help="Path to FNN tuning config JSON.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional run-name override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = run_fnn_grid_search(args.config, run_name=args.run_name)
    print(f"grid_search_dir={run_dir}")


if __name__ == "__main__":
    main()
