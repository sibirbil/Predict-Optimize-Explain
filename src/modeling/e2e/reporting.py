from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.utils.paths import resolve_repo_path


@dataclass(frozen=True)
class E2EBenchmarkSpec:
    model: str
    run_dir: Path


def build_e2e_benchmark_comparison(
    pto_baseline_config_path: str | Path,
    benchmark_specs: list[E2EBenchmarkSpec],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    with resolve_repo_path(pto_baseline_config_path).open("r", encoding="utf-8") as handle:
        pto_config = json.load(handle)
    pto_run_dir = resolve_repo_path(pto_config["output_root"]) / str(pto_config["run_name"])
    with (pto_run_dir / "validation_selection_summary.json").open("r", encoding="utf-8") as handle:
        pto_selection = json.load(handle)
    with (pto_run_dir / "final_test_summary.json").open("r", encoding="utf-8") as handle:
        pto_test = json.load(handle)
    rows.append({"model": "frozen_robust_pto_baseline", "split": "val", **pto_selection["validation_summary"]})
    rows.append({"model": "frozen_robust_pto_baseline", "split": "test", **pto_test})

    for spec in benchmark_specs:
        with spec.run_dir.joinpath("validation_selection_summary.json").open("r", encoding="utf-8") as handle:
            selection = json.load(handle)
        with spec.run_dir.joinpath("final_test_summary.json").open("r", encoding="utf-8") as handle:
            test_summary = json.load(handle)
        rows.append(
            {
                "model": spec.model,
                "split": "val",
                **selection["validation_summary"],
                "label": selection["selected_candidate"]["label"],
                "solver_mode": selection.get("solver_mode"),
            }
        )
        rows.append(
            {
                "model": spec.model,
                "split": "test",
                **test_summary,
                "label": selection["selected_candidate"]["label"],
                "solver_mode": selection.get("solver_mode"),
            }
        )

    return pd.DataFrame(rows)


def build_e2e_benchmark_comparison_markdown(comparison: pd.DataFrame) -> str:
    lines = [
        "# E2E Benchmark Comparison",
        "",
        "Compares the frozen PTO baseline against the preserved E2E benchmark runs.",
        "",
    ]
    for split in ("val", "test"):
        frame = comparison.loc[comparison["split"] == split].copy()
        frame = frame.sort_values("sharpe_ratio", ascending=False)
        lines.append(f"## {split.upper()}")
        for _, row in frame.iterrows():
            lines.append(
                "- "
                + f"{row['model']}: Sharpe `{row['sharpe_ratio']:.6f}`, "
                + f"ann. excess `{row['annualized_excess_return']:.6f}`, "
                + f"vol `{row['annualized_volatility']:.6f}`, "
                + f"max DD `{row['max_drawdown']:.6f}`"
            )
        lines.append("")
    return "\n".join(lines).strip() + "\n"
