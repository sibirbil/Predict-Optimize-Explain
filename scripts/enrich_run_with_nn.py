"""Backfill nearest-neighbor (NN) historical-month diagnostics onto an existing scenario run.

Reads the timestamped CSVs in a run directory, computes top-3 historical
analogs under three metrics (VAR(1) Mahalanobis, historical Mahalanobis, and
Euclidean in z-score space), and writes ``*_enriched.csv`` siblings without
touching the original artifacts. Also emits one paper-grade figure showing each
seed's top-3 analogs per metric, color-coded by historical regime label.

Usage::

    python scripts/enrich_run_with_nn.py \\
        scenario_outputs/scenario4_202004/runs/20260424_102040
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modules.nn_historical import HistoricalNNIndex, build_index_from_runtime
from src.modules.runtime_regime import MACRO_ORDER, regime_classifier

MACRO_COLS = list(MACRO_ORDER)
STD_COLS = [f"{c}_std" for c in MACRO_COLS]
METRICS = ("var1", "hist", "eucl")
TS_RE = re.compile(r"_(\d{8}_\d{6})\.csv$")


def discover_timestamp(run_dir: Path) -> str:
    """Find the timestamp suffix used by this run's CSVs."""
    candidates: set[str] = set()
    for path in run_dir.glob("*.csv"):
        m = TS_RE.search(path.name)
        if m:
            candidates.add(m.group(1))
    if not candidates:
        raise FileNotFoundError(f"No timestamped CSVs found in {run_dir}")
    if len(candidates) > 1:
        raise RuntimeError(f"Multiple timestamps in {run_dir}: {sorted(candidates)}; pass --timestamp")
    return candidates.pop()


def attach_nn_to_frame(
    frame: pd.DataFrame,
    index: HistoricalNNIndex,
    std_cols: list[str],
    k: int = 3,
) -> pd.DataFrame:
    missing = [c for c in std_cols if c not in frame.columns]
    if missing:
        raise KeyError(f"Frame missing standardized macro columns: {missing}")
    states_std = frame[std_cols].to_numpy(dtype=np.float64)
    nn = index.attach(states_std, k=k)
    nn.index = frame.index
    return pd.concat([frame, nn], axis=1)


def enrich_seed_summary(
    seed_summary: pd.DataFrame,
    final_enriched: pd.DataFrame,
) -> pd.DataFrame:
    """Pull top-1 final-state NN per metric onto the per-seed summary."""
    cols = ["seed"]
    for m in METRICS:
        cols += [f"nn_{m}_yyyymm_1", f"nn_{m}_dist_1"]
    pull = final_enriched[cols].copy()
    rename = {}
    for m in METRICS:
        rename[f"nn_{m}_yyyymm_1"] = f"final_nn_{m}_yyyymm_1"
        rename[f"nn_{m}_dist_1"] = f"final_nn_{m}_dist_1"
    pull = pull.rename(columns=rename)
    return seed_summary.merge(pull, on="seed", how="left")


def historical_regime_lookup(yyyymm_array: np.ndarray) -> np.ndarray:
    """Map yyyymm to historical regime label using the runtime classifier."""
    out = np.empty(yyyymm_array.shape, dtype=object)
    for idx, yy in enumerate(yyyymm_array.astype(int)):
        try:
            out[idx] = regime_classifier.historical_label(int(yy))
        except Exception:
            out[idx] = "unknown"
    return out


def plot_nn_storyline(
    final_enriched: pd.DataFrame,
    out_path: Path,
    run_label: str,
) -> None:
    import matplotlib.pyplot as plt

    n_seeds = len(final_enriched)
    fig, axes = plt.subplots(3, 1, figsize=(11, 1.0 + 0.3 * n_seeds), sharex=False)
    regime_colors = {
        "expansion": "#2ca02c",
        "contraction": "#d62728",
        "financial_stress": "#1f1f1f",
        "unknown": "#9e9e9e",
    }
    for ax, metric in zip(axes, METRICS):
        for rank in range(1, 4):
            yy = final_enriched[f"nn_{metric}_yyyymm_{rank}"].to_numpy()
            dd = final_enriched[f"nn_{metric}_dist_{rank}"].to_numpy()
            regs = historical_regime_lookup(yy)
            colors = [regime_colors.get(r, regime_colors["unknown"]) for r in regs]
            seeds = final_enriched["seed"].to_numpy()
            xs = np.full_like(seeds, fill_value=rank, dtype=float) + np.linspace(-0.15, 0.15, n_seeds)
            ax.scatter(seeds, np.full(n_seeds, rank) + 0.0, c=colors, s=40, edgecolor="black", linewidth=0.3)
            for s, r, label, dist in zip(seeds, [rank] * n_seeds, yy, dd):
                ax.annotate(
                    f"{int(label)}",
                    xy=(s, r),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    fontsize=6,
                )
        ax.set_title(f"Top-3 historical analogs ({metric}) per seed", fontsize=10)
        ax.set_ylabel("rank (1=closest)")
        ax.set_ylim(0.5, 3.5)
        ax.set_yticks([1, 2, 3])
        ax.set_xlabel("seed")
        ax.grid(axis="y", alpha=0.3)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=name)
        for name, c in regime_colors.items()
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"NN historical analogs — {run_label}", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill NN-to-historical-yyyymm onto an existing scenario run.")
    parser.add_argument("run_dir", type=Path, help="Path to the run directory (e.g., scenario_outputs/scenario4_202004/runs/20260424_102040).")
    parser.add_argument("--timestamp", type=str, default=None, help="Override timestamp suffix discovery.")
    parser.add_argument("-k", type=int, default=3, help="Top-k neighbors to save per metric (default 3).")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"{run_dir} is not a directory")
    ts = args.timestamp or discover_timestamp(run_dir)

    final_path = run_dir / f"final_state_diagnostics_{ts}.csv"
    chain_path = run_dir / f"generated_macro_sample_postburnin_{ts}.csv"
    seed_path = run_dir / f"seed_summary_{ts}.csv"
    if not final_path.exists():
        raise FileNotFoundError(final_path)
    if not chain_path.exists():
        raise FileNotFoundError(chain_path)

    print(f"[enrich] run_dir = {run_dir}")
    print(f"[enrich] timestamp = {ts}")
    print(f"[enrich] building NN index from runtime ...")
    index = build_index_from_runtime(macro_columns=MACRO_COLS)
    print(f"[enrich] index covers {len(index.yyyymm)} historical months ({index.yyyymm.min()}-{index.yyyymm.max()})")

    final_df = pd.read_csv(final_path)
    chain_df = pd.read_csv(chain_path)
    print(f"[enrich] final-state rows = {len(final_df)}, chain rows = {len(chain_df)}")

    final_enriched = attach_nn_to_frame(final_df, index, STD_COLS, k=args.k)
    chain_enriched = attach_nn_to_frame(chain_df, index, STD_COLS, k=args.k)

    final_out = run_dir / f"final_state_diagnostics_{ts}_enriched.csv"
    chain_out = run_dir / f"generated_macro_sample_postburnin_{ts}_enriched.csv"
    final_enriched.to_csv(final_out, index=False)
    chain_enriched.to_csv(chain_out, index=False)
    print(f"[enrich] wrote {final_out.name}")
    print(f"[enrich] wrote {chain_out.name}")

    if seed_path.exists():
        seed_df = pd.read_csv(seed_path)
        seed_enriched = enrich_seed_summary(seed_df, final_enriched)
        seed_out = run_dir / f"seed_summary_{ts}_enriched.csv"
        seed_enriched.to_csv(seed_out, index=False)
        print(f"[enrich] wrote {seed_out.name}")

    fig_out = run_dir / f"nn_storyline_{ts}.png"
    plot_nn_storyline(final_enriched, fig_out, run_label=run_dir.name)
    print(f"[enrich] wrote {fig_out.name} (+ .pdf)")

    headline = []
    for m in METRICS:
        col = f"nn_{m}_yyyymm_1"
        modes = final_enriched[col].mode()
        median_dist = float(final_enriched[f"nn_{m}_dist_1"].median())
        top_yy = int(modes.iloc[0]) if not modes.empty else -1
        headline.append(f"{m}: modal yyyymm={top_yy}, median dist={median_dist:.3f}")
    print("[enrich] headline NN summary:")
    for line in headline:
        print(f"   - {line}")


if __name__ == "__main__":
    main()
