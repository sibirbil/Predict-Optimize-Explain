from __future__ import annotations

import argparse
import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
WORKSPACE_ROOT = THIS_FILE.parents[1]
SRC_ROOT = WORKSPACE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.fnn21.evaluation import summarize_prediction_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute local 21April FNN evaluation artifacts from saved predictions.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory under 21April/artifacts/fnn/...")
    parser.add_argument("--quantiles", type=int, default=10, help="Number of prediction quantiles for spread diagnostics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    predictions_dir = run_dir / "predictions"
    summary = summarize_prediction_dir(predictions_dir, run_dir, quantiles=args.quantiles)
    print(f"evaluation_updated={run_dir}")
    print(f"val_mean_monthly_ic={summary['monthly_metrics']['val']['mean_monthly_ic']:.6f}")
    print(f"test_mean_monthly_ic={summary['monthly_metrics']['test']['mean_monthly_ic']:.6f}")


if __name__ == "__main__":
    main()
