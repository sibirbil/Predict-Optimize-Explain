"""All-macro MALA trace plots for Mehmet delivery.

This script plots full-chain macro traces from the saved 3D tensor bundles:
[seed, step, macro_variable]. It creates standardized and unstandardized figures
for Scenario 4 and E2E diversification.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.macro_scaler import MacroScaler  # noqa: E402
from src.modules.runtime_regime import MACRO_ORDER  # noqa: E402


DEFAULT_S4_RUN = ROOT / "scenario_outputs" / "scenario4_202004" / "runs" / "20260509_072049"
DEFAULT_E2E_RUN = ROOT / "scenario_outputs" / "scenario_e2e_diversify_202004" / "runs" / "20260509_113131_213629"
DEFAULT_OUT = ROOT / "scenario_outputs" / "mehmet_delivery" / "trace_plots_202004_20ksteps_4seeds"

SCENARIO_COLORS = {
    "scenario4": ["#2667a0", "#5e9f6e", "#c45b4f", "#7a5195"],
    "e2e": ["#1f6f78", "#5a9f4f", "#b6554d", "#8264a8"],
}


def newest(run_dir: Path, pattern: str) -> Path:
    matches = sorted(run_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file in {run_dir} matching {pattern}")
    return matches[-1]


def load_tensor(path: Path) -> tuple[torch.Tensor, list[str], dict]:
    payload = torch.load(path, map_location="cpu")
    tensor = payload["tensor"].detach().cpu().to(dtype=torch.float32)
    macro_columns = [str(x) for x in payload.get("macro_columns", list(MACRO_ORDER))]
    return tensor, macro_columns, payload


def anchor_values(run_dir: Path, macro_columns: list[str], standardized: bool) -> np.ndarray:
    hist = pd.read_csv(newest(run_dir, "historical_macro_panel_*.csv"))
    scenario_name = run_dir.parent.parent.name
    match = re.search(r"_(\d{6})$", scenario_name)
    anchor = int(match.group(1)) if match else 202004
    row = hist[hist["yyyymm"].astype(int).eq(anchor)]
    if row.empty:
        row = hist[hist["yyyymm"].astype(int).eq(202004)]
    if row.empty:
        raise ValueError(f"Could not find {anchor} or 202004 anchor in {run_dir}.")
    raw = row[macro_columns].iloc[0].to_numpy(dtype=np.float32)
    if not standardized:
        return raw
    scaler = MacroScaler.load(ROOT / "runtime_universe500" / "scaler")
    mean = scaler.mean.detach().cpu().numpy().astype(np.float32)
    std = scaler.std.detach().cpu().numpy().astype(np.float32)
    order = list(MACRO_ORDER)
    idx = [order.index(col) for col in macro_columns]
    return (raw - mean[idx]) / std[idx]


def plot_all_macro_traces(
    tensor: torch.Tensor,
    macro_columns: list[str],
    anchor: np.ndarray,
    out_path_stem: Path,
    title: str,
    unit_label: str,
    colors: list[str],
    plot_step: int,
    burn_in_step: int,
) -> tuple[Path, Path]:
    data = tensor.numpy()
    n_seed, n_step, n_macro = data.shape
    if n_macro != len(macro_columns):
        raise ValueError(f"Tensor macro dimension {n_macro} != columns {len(macro_columns)}")
    step_idx = np.arange(1, n_step + 1, max(1, plot_step))

    fig, axes = plt.subplots(3, 3, figsize=(13.5, 9.2), sharex=True)
    axes = axes.ravel()
    for macro_idx, (ax, macro) in enumerate(zip(axes, macro_columns)):
        for seed_idx in range(n_seed):
            ax.plot(
                step_idx,
                data[seed_idx, :: max(1, plot_step), macro_idx],
                color=colors[seed_idx % len(colors)],
                linewidth=0.75,
                alpha=0.80,
                label=f"seed {seed_idx + 1}" if macro_idx == 0 else None,
            )
        ax.axhline(float(anchor[macro_idx]), color="#111111", linewidth=0.9, linestyle="--", alpha=0.85)
        if burn_in_step > 0:
            ax.axvline(burn_in_step, color="#888888", linewidth=0.75, linestyle=":", alpha=0.65)
        ax.set_title(macro, fontsize=10.5, fontweight="bold", loc="left")
        ax.grid(alpha=0.18, linewidth=0.6)
        ax.tick_params(labelsize=8)
    for ax in axes[6:]:
        ax.set_xlabel("MALA step", fontsize=9)
    for ax in axes[::3]:
        ax.set_ylabel(unit_label, fontsize=9)
    axes[0].legend(frameon=False, ncol=4, fontsize=8, loc="upper right")
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    fig.text(
        0.5,
        0.012,
        f"Dashed horizontal line: April 2020 anchor. Dotted vertical line: 50% burn-in marker. Visual plotting step: every {max(1, plot_step)} MALA iterations.",
        ha="center",
        fontsize=8.8,
        color="#333333",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    pdf = out_path_stem.with_suffix(".pdf")
    png = out_path_stem.with_suffix(".png")
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(png, dpi=260, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return pdf, png


def write_readme(out_dir: Path, records: list[dict[str, str]], plot_step: int) -> None:
    lines = [
        "All-macro MALA trace plots for Mehmet",
        "",
        "These figures visualize the full 20,000-step MALA trajectories saved as 3D tensors.",
        "Each panel is one macro variable; each colored line is one seed.",
        "",
        "Shape of underlying tensors:",
        "[4, 20000, 9] = [seed, step, macro_variable]",
        "",
        "Macro variables:",
        ", ".join(MACRO_ORDER),
        "",
        "Plotting note:",
        f"For readability and smaller PDF size, lines are drawn every {plot_step} MALA steps.",
        "The underlying tensor files still contain every step.",
        "",
        "Files:",
    ]
    for record in records:
        lines.append(f"- {Path(record['pdf']).name}")
        lines.append(f"- {Path(record['png']).name}")
    (out_dir / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create all-macro trace plots for Mehmet delivery.")
    parser.add_argument("--scenario4-run-dir", type=Path, default=DEFAULT_S4_RUN)
    parser.add_argument("--e2e-run-dir", type=Path, default=DEFAULT_E2E_RUN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--plot-step", type=int, default=10)
    parser.add_argument("--burn-in-step", type=int, default=10000)
    args = parser.parse_args()

    args.scenario4_run_dir = args.scenario4_run_dir.resolve()
    args.e2e_run_dir = args.e2e_run_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str]] = []

    specs = [
        ("scenario4", "Scenario 4", args.scenario4_run_dir),
        ("e2e", "E2E diversification", args.e2e_run_dir),
    ]
    for key, label, run_dir in specs:
        standardized_path = newest(run_dir, "trajectories_standardized_3d_*.pt")
        raw_path = newest(run_dir, "trajectories_unstandardized_3d_*.pt") if list(run_dir.glob("trajectories_unstandardized_3d_*.pt")) else newest(run_dir, "trajectories_raw_3d_*.pt")

        for units_key, tensor_path, unit_label, title_suffix in [
            ("standardized", standardized_path, "z-score", "standardized macro space"),
            ("unstandardized", raw_path, "raw units", "unstandardized macro units"),
        ]:
            tensor, macro_columns, payload = load_tensor(tensor_path)
            anchor = anchor_values(run_dir, macro_columns, standardized=(units_key == "standardized"))
            stem = out_dir / f"{key}_{units_key}_all_macro_traces"
            pdf, png = plot_all_macro_traces(
                tensor=tensor,
                macro_columns=macro_columns,
                anchor=anchor,
                out_path_stem=stem,
                title=f"{label}: all macro MALA traces ({title_suffix})",
                unit_label=unit_label,
                colors=SCENARIO_COLORS[key],
                plot_step=args.plot_step,
                burn_in_step=args.burn_in_step,
            )
            records.append(
                {
                    "scenario": key,
                    "units": units_key,
                    "source_tensor": str(tensor_path.relative_to(ROOT)),
                    "tensor_shape": json.dumps(list(tensor.shape)),
                    "pdf": str(pdf.relative_to(ROOT)),
                    "png": str(png.relative_to(ROOT)),
                    "payload_units": str(payload.get("units", "")),
                }
            )

    with (out_dir / "trace_plot_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    write_readme(out_dir, records, plot_step=max(1, args.plot_step))
    print(f"Wrote {len(records)} trace figures -> {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
