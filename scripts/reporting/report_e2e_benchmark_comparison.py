from __future__ import annotations

import argparse
from pathlib import Path

from src.modeling.e2e import (
    E2EBenchmarkSpec,
    build_e2e_benchmark_comparison,
    build_e2e_benchmark_comparison_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the clean E2E benchmark comparison report.")
    parser.add_argument(
        "--pto-config",
        type=Path,
        default=Path("src/configs/pto/robust_v1_selected_baseline.json"),
        help="Path to the frozen PTO baseline config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Run directory where the comparison report should be saved.",
    )
    parser.add_argument(
        "--include-run",
        nargs=2,
        action="append",
        metavar=("MODEL_NAME", "RUN_DIR"),
        required=True,
        help="Model label and E2E run directory to include.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = [
        E2EBenchmarkSpec(model=model_name, run_dir=Path(run_dir))
        for model_name, run_dir in args.include_run
    ]
    comparison = build_e2e_benchmark_comparison(
        pto_baseline_config_path=args.pto_config,
        benchmark_specs=specs,
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "e2e_benchmark_comparison.csv", index=False)
    (output_dir / "e2e_benchmark_comparison.md").write_text(
        build_e2e_benchmark_comparison_markdown(comparison),
        encoding="utf-8",
    )
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
