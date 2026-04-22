from __future__ import annotations

import argparse
import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
WORKSPACE_ROOT = THIS_FILE.parents[1]
SRC_ROOT = WORKSPACE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.eda21.paths import ARTIFACTS_ROOT, DATASET_ROOT
from src.eda21.summary import write_eda_outputs
from src.eda21.universe_500 import load_universe_500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run first-pass EDA for 21April/universe_500_v21April_stdz_macro."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT,
        help="Dataset root to inspect.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARTIFACTS_ROOT / "universe_500_v21April_stdz_macro_eda",
        help="Directory where EDA artifacts will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_universe_500(args.dataset_root)
    write_eda_outputs(args.output_dir, bundle)
    print(args.output_dir)


if __name__ == "__main__":
    main()
