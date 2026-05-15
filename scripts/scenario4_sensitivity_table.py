"""Cross-anchor identifiability table for the locked sc_beats_ww scenario.

Reruns ``scripts/scenario4.py`` at a small set of alternative anchors that share
the same regime context as the locked anchor 202004 (financial stress), then
aggregates the median z-shift per macro variable into one table. Stable signs
and magnitudes across anchors are the AE-rebuttal evidence that the
framework's identified macro shifts are not anchor-specific artifacts.

Usage::

    python scripts/scenario4_sensitivity_table.py \\
        --anchors 200810,200903,202003 \\
        --n-seeds 10 --n-steps 500 \\
        --include-locked-run scenario_outputs/scenario4_202004/runs/20260424_102040
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MACRO_COLS = ["dp", "ep", "bm", "ntis", "tbl", "tms", "dfy", "svar", "infl"]
TS_RE = re.compile(r"_(\d{8}_\d{6})\.csv$")
RUN_STAMP_RE = re.compile(r"Results saved .+ \((\d{8}_\d{6})\)")


def discover_timestamp(run_dir: Path) -> str:
    candidates: set[str] = set()
    for path in run_dir.glob("*.csv"):
        m = TS_RE.search(path.name)
        if m:
            candidates.add(m.group(1))
    if len(candidates) != 1:
        raise RuntimeError(f"Could not pin timestamp in {run_dir} (found {sorted(candidates)})")
    return candidates.pop()


def median_shifts(run_dir: Path, ts: str) -> dict[str, float]:
    final = pd.read_csv(run_dir / f"final_state_diagnostics_{ts}.csv")
    shifts: dict[str, float] = {}
    for col in MACRO_COLS:
        std_col = f"{col}_std"
        if std_col not in final.columns:
            shifts[f"median_shift_{col}_z"] = float("nan")
            continue
        values = final[std_col].to_numpy(dtype=float)
        shifts[f"median_shift_{col}_z"] = float(np.median(values))
    if "mah_dist" in final.columns:
        shifts["median_mah_dist"] = float(final["mah_dist"].median())
    if "mah_chi2_tail" in final.columns:
        shifts["median_mah_chi2_tail"] = float(final["mah_chi2_tail"].median())
    if "winner" in final.columns:
        shifts["sc_win_rate"] = float((final["winner"] == "summer_child").mean())
    if "return_gap" in final.columns:
        shifts["mean_return_gap"] = float(final["return_gap"].mean())
    return shifts


def regime_label_for(date: int, ts: str, run_dir: Path) -> str:
    cfg = run_dir / f"config_{ts}.json"
    if not cfg.exists():
        return "unknown"
    payload = json.loads(cfg.read_text())
    anchor_regime = payload.get("ANCHOR_REGIME") or {}
    if isinstance(anchor_regime, dict) and "label" in anchor_regime:
        return str(anchor_regime["label"])
    return str(payload.get("ANCHOR_TRUE_LABEL", "unknown"))


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def run_scenario4(args: argparse.Namespace, anchor: int) -> Path:
    cmd = [
        sys.executable, str(ROOT / "scripts" / "scenario4.py"),
        "--date", str(anchor),
        "--n-seeds", str(args.n_seeds),
        "--n-steps", str(args.n_steps),
        "--reg-mode", args.reg_mode,
        "--constraint-mode", args.constraint_mode,
        "--contrast-function", "sc_beats_ww",
        "--l2reg", str(args.l2reg),
        "--eta", str(args.eta),
        "--beta", str(args.beta),
        "--random-start-scale", str(args.random_start_scale),
        "--base-seed", str(args.base_seed),
    ]
    print(f"[sensitivity] launching scenario4 for anchor={anchor}: {' '.join(cmd[-12:])}", flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"scenario4 failed for anchor {anchor}:\n{proc.stdout[-2000:]}")
    match = RUN_STAMP_RE.search(proc.stdout)
    if not match:
        raise RuntimeError(f"could not parse run stamp for anchor {anchor}")
    stamp = match.group(1)
    run_dir = ROOT / "scenario_outputs" / f"scenario4_{anchor}" / "runs" / stamp
    if not run_dir.is_dir():
        flat = ROOT / "scenario_outputs" / f"scenario4_{anchor}"
        if (flat / f"config_{stamp}.json").exists():
            run_dir = flat
    print(f"[sensitivity]   wrote {run_dir.relative_to(ROOT)}", flush=True)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="sc_beats_ww cross-anchor sensitivity / identifiability table.")
    parser.add_argument("--anchors", default="200810,200903,202003", help="Comma-separated yyyymm list of alternative anchors.")
    parser.add_argument("--include-locked-run", type=Path, default=None, help="Path to an already-run anchor (e.g., the locked 202004 run) to include in the table without re-running.")
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--l2reg", type=float, default=0.3)
    parser.add_argument("--eta", type=float, default=0.001)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--reg-mode", choices=["l2", "var1"], default="l2")
    parser.add_argument("--constraint-mode", choices=["box_barrier", "none", "clip", "strong_soft"], default="box_barrier")
    parser.add_argument("--random-start-scale", type=float, default=0.005)
    parser.add_argument("--base-seed", type=int, default=20260428)
    parser.add_argument("--out-tag", default=None, help="Subfolder name under scenario_outputs/scenario4_sensitivity/.")
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    if args.include_locked_run is not None:
        run_dir = args.include_locked_run.resolve()
        ts = discover_timestamp(run_dir)
        m = re.search(r"scenario4_(\d{6})", str(run_dir))
        if m is None:
            raise RuntimeError(f"could not infer anchor yyyymm from {run_dir}")
        anchor = int(m.group(1))
        row = {"anchor": anchor, "regime": regime_label_for(anchor, ts, run_dir), "source": "locked", "run_dir": str(run_dir.relative_to(ROOT)), "timestamp": ts}
        row.update(median_shifts(run_dir, ts))
        rows.append(row)

    for anchor in parse_int_list(args.anchors):
        try:
            run_dir = run_scenario4(args, anchor)
        except RuntimeError as exc:
            head = str(exc).splitlines()[0]
            tail = str(exc).splitlines()[-1] if "\n" in str(exc) else ""
            print(f"[sensitivity] SKIP anchor {anchor}: {head} | {tail}", flush=True)
            rows.append({"anchor": anchor, "regime": "unknown", "source": "failed", "run_dir": "", "timestamp": "", "error": head})
            continue
        ts = discover_timestamp(run_dir)
        row = {"anchor": anchor, "regime": regime_label_for(anchor, ts, run_dir), "source": "rerun", "run_dir": str(run_dir.relative_to(ROOT)), "timestamp": ts}
        row.update(median_shifts(run_dir, ts))
        rows.append(row)

    table = pd.DataFrame(rows)
    out_root = ROOT / "scenario_outputs" / "scenario4_sensitivity"
    tag = args.out_tag or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "sc_beats_ww_sensitivity_table.csv"
    table.to_csv(out_csv, index=False)

    md_lines = [
        "# sc_beats_ww identifiability sensitivity",
        "",
        f"_Hyperparameters_: L2REG={args.l2reg}, ETA={args.eta}, BETA={args.beta}, N_SEEDS={args.n_seeds}, N_STEPS={args.n_steps}",
        "",
        "Median z-shift per macro variable across anchors:",
        "",
    ]
    md_cols = ["anchor", "regime", "source"] + [f"median_shift_{c}_z" for c in MACRO_COLS] + ["median_mah_dist", "median_mah_chi2_tail", "sc_win_rate", "mean_return_gap"]
    available = [c for c in md_cols if c in table.columns]
    md_lines.append(table[available].to_markdown(index=False, floatfmt=".3f"))
    (out_dir / "sc_beats_ww_sensitivity_table.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"\n[sensitivity] wrote {out_csv.relative_to(ROOT)}")
    print(f"[sensitivity] wrote {(out_dir / 'sc_beats_ww_sensitivity_table.md').relative_to(ROOT)}")
    print()
    print(table[available].to_string(index=False))


if __name__ == "__main__":
    main()
