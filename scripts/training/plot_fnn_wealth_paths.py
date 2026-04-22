from __future__ import annotations

import argparse
import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
WORKSPACE_ROOT = THIS_FILE.parents[1]
SRC_ROOT = WORKSPACE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.fnn21.wealth import build_return_and_wealth_frames, load_manifest, plot_wealth_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build FNN long-short wealth path comparison artifacts from saved run outputs."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to JSON manifest listing run directories and labels.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("21April/artifacts/fnn/wealth_compare"),
        help="Directory for generated CSV and PNG outputs.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="FNN Test Long-Short Wealth Paths",
        help="Chart title.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = load_manifest(args.manifest.resolve())
    returns, wealth, summary = build_return_and_wealth_frames(specs)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    returns.to_csv(output_dir / "returns_monthly.csv", index=False)
    wealth.to_csv(output_dir / "wealth_paths.csv", index=False)
    summary.to_csv(output_dir / "wealth_summary.csv", index=False)
    plot_wealth_paths(wealth, title=args.title, output_path=output_dir / "wealth_paths.png")

    print(f"output_dir={output_dir}")
    print(f"wealth_plot={output_dir / 'wealth_paths.png'}")
    print(f"wealth_summary={output_dir / 'wealth_summary.csv'}")


if __name__ == "__main__":
    main()
