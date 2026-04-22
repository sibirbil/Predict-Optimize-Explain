from __future__ import annotations

import argparse
import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
WORKSPACE_ROOT = THIS_FILE.parents[1]
SRC_ROOT = WORKSPACE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.eda21.build_stdz_macro_dataset import build_standardized_macro_dataset
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build 21April/universe_500_v21April_stdz_macro from the base v21April dataset."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Source dataset root. Pass the external/base dataset explicitly.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=WORKSPACE_ROOT / "universe_500_v21April_stdz_macro",
        help="Output dataset root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = build_standardized_macro_dataset(
        source_root=args.source_root,
        output_root=args.output_root,
    )
    print(out)


if __name__ == "__main__":
    main()
