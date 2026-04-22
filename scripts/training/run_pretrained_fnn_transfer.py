from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from src.modeling.fnn.transfer import PretrainedTransferConfig, run_pretrained_transfer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run compatibility audit plus zero-shot transfer evaluation for an old FNN bundle."
    )
    parser.add_argument(
        "--pretrained-dir",
        type=str,
        default=None,
        help="Directory containing feature_columns.json, model_config.json, and state_dict.pt.",
    )
    parser.add_argument("--run-name", type=str, default=None, help="Optional transfer run-name override.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional root directory for saved transfer artifacts.",
    )
    parser.add_argument("--device", type=str, default=None, help="Optional device override.")
    parser.add_argument("--quantiles", type=int, default=None, help="Optional quantile override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PretrainedTransferConfig()

    overrides = {}
    if args.pretrained_dir is not None:
        overrides["pretrained_dir"] = args.pretrained_dir
    if args.run_name is not None:
        overrides["run_name"] = args.run_name
    if args.output_root is not None:
        overrides["output_root"] = str(args.output_root)
    if args.device is not None:
        overrides["device"] = args.device
    if args.quantiles is not None:
        overrides["quantiles"] = args.quantiles

    config = replace(config, **overrides)
    run_dir = run_pretrained_transfer(config=config)
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
