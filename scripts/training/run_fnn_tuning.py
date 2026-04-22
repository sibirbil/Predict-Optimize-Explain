from __future__ import annotations

import argparse
from pathlib import Path
from src.modeling.fnn.tuner import run_fnn_grid_search

def main() -> None:
    parser = argparse.ArgumentParser(description="Run FNN grid search.")
    parser.add_argument("--config", type=Path, required=True, help="Path to FNN config with grids.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional run-name override.")
    args = parser.parse_args()
    
    run_dir = run_fnn_grid_search(config_path=args.config, run_name=args.run_name)
    print(f"grid_search_dir={run_dir}")

if __name__ == "__main__":
    main()
