"""Lightweight E2E anchor screen for a more visual diversification story.

This script does not run MALA chains. It scores economically selected candidate
anchor months by:

- locked E2E anchor concentration
- regime-boundary position from classifier probabilities
- PCA location relative to historical macro regimes

After this screen, run short pilot scenario5 chains for the top few candidates
and inspect the enhanced geography plots before committing to a long run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.scenario5 import (  # noqa: E402
    DEFAULT_BASE_SEED,
    DEFAULT_BETA,
    DEFAULT_ETA,
    DEFAULT_L2REG,
    Scenario5Runtime,
)
from src.data.macro_scaler import MacroScaler  # noqa: E402
from src.modules.runtime_regime import MACRO_ORDER  # noqa: E402


DEFAULT_DATES = [201812, 202003, 202004, 202005, 202009, 202010, 202211, 202212]
DEFAULT_OUT = ROOT / "scenario_outputs" / "e2e_anchor_story_screen"


def runtime_args(date: int) -> SimpleNamespace:
    return SimpleNamespace(
        date=int(date),
        probe="decision_fragility",
        model="locked_e2e",
        model_a="locked_e2e",
        model_b="standardized_pto",
        objective="entropy_max",
        n_seeds=1,
        n_steps=1,
        base_seed=DEFAULT_BASE_SEED,
        reg_mode="l2",
        constraint_mode="box_barrier",
        l2reg=DEFAULT_L2REG,
        eta=DEFAULT_ETA,
        beta=DEFAULT_BETA,
        random_start_scale=0.0,
        save_trajectory_tensors=False,
        trajectory_format="pt",
        trajectory_burn_in_frac=0.5,
        trajectory_thin=1,
    )


def zscore(values: pd.Series) -> pd.Series:
    std = float(values.std(ddof=0))
    if not np.isfinite(std) or std <= 1e-12:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - float(values.mean())) / std


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank E2E diversification anchor candidates for visual story quality.")
    parser.add_argument("--dates", nargs="*", type=int, default=DEFAULT_DATES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--pilot-steps", type=int, default=1000)
    parser.add_argument("--pilot-seeds", type=int, default=4)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    macro = list(MACRO_ORDER)
    macro_panel = pd.read_parquet(ROOT / "runtime_universe500" / "data" / "macro_final.parquet")
    macro_panel["yyyymm"] = macro_panel["yyyymm"].astype(int)
    regime_panel = pd.read_csv(ROOT / "runtime_universe500" / "regime" / "regime_probability_panel.csv")
    regime_panel["yyyymm"] = regime_panel["yyyymm"].astype(int)
    scaler = MacroScaler.load(ROOT / "runtime_universe500" / "scaler")
    mean = scaler.mean.detach().cpu().numpy().astype(float)
    std = scaler.std.detach().cpu().numpy().astype(float)
    hist_std = (macro_panel[macro].to_numpy(dtype=float) - mean.reshape(1, -1)) / std.reshape(1, -1)
    pca = PCA(n_components=2, random_state=0).fit(hist_std)
    pca_coords = pca.transform(hist_std)
    pca_lookup = dict(zip(macro_panel["yyyymm"].astype(int), pca_coords))

    rows = []
    for date in args.dates:
        print(f"Evaluating anchor {date}...")
        runtime = Scenario5Runtime(runtime_args(date))
        metrics = runtime.anchor_metrics["locked_e2e"]
        probs = regime_panel.loc[regime_panel["yyyymm"].eq(int(date))]
        if probs.empty:
            raise ValueError(f"Date {date} not found in regime probability panel.")
        prob_row = probs.iloc[0]
        regime_probs = {
            "financial_stress": float(prob_row["financial_stress"]),
            "contraction": float(prob_row["contraction"]),
            "expansion": float(prob_row["expansion"]),
        }
        ordered = sorted(regime_probs.items(), key=lambda item: item[1], reverse=True)
        boundary_margin = ordered[0][1] - ordered[1][1]
        boundary_score = 1.0 - boundary_margin
        pca_xy = pca_lookup[int(date)]
        rows.append(
            {
                "date": int(date),
                "anchor_entropy": float(metrics["entropy"]),
                "anchor_hhi": float(metrics["hhi"]),
                "anchor_effective_n": float(metrics["effective_n"]),
                "anchor_max_weight": float(metrics["max_weight"]),
                "anchor_top10_weight": float(metrics["top10_weight"]),
                "top_regime": ordered[0][0],
                "top_regime_prob": ordered[0][1],
                "second_regime": ordered[1][0],
                "second_regime_prob": ordered[1][1],
                "boundary_margin": boundary_margin,
                "boundary_score": boundary_score,
                "pca1": float(pca_xy[0]),
                "pca2": float(pca_xy[1]),
            }
        )

    frame = pd.DataFrame(rows)
    frame["visual_story_score"] = (
        0.35 * zscore(frame["anchor_max_weight"])
        + 0.25 * zscore(frame["anchor_hhi"])
        + 0.30 * zscore(frame["boundary_score"])
        + 0.10 * zscore(frame["anchor_top10_weight"])
    )
    frame = frame.sort_values("visual_story_score", ascending=False)
    out_csv = args.out_dir / "e2e_anchor_visual_story_candidates.csv"
    frame.to_csv(out_csv, index=False)

    commands = []
    for row in frame.head(5).itertuples(index=False):
        commands.append(
            "python scripts/scenario5.py "
            f"--date {int(row.date)} "
            "--probe decision_fragility --model locked_e2e --objective entropy_max "
            "--reg-mode l2 --constraint-mode box_barrier "
            "--l2reg 0.3 --eta 0.09 --beta 10 --random-start-scale 0.0 "
            f"--n-seeds {args.pilot_seeds} --n-steps {args.pilot_steps}"
        )
    (args.out_dir / "pilot_commands.sh").write_text("\n".join(commands) + "\n", encoding="utf-8")
    print(frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nWrote {out_csv.relative_to(ROOT)}")
    print(f"Wrote {(args.out_dir / 'pilot_commands.sh').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
