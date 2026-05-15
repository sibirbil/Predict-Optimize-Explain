from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.scenario4 import (  # noqa: E402
    ASSET_SIZE,
    MACRO_ORDER,
    ROOT as SCENARIO_ROOT,
    AllocationPipeline,
    construct_C2_21a,
    evaluate,
    load_exact_e2e_model_from_run,
    macro_df,
    macro_mean,
    macro_std,
    regime_classifier,
)


def parse_dates(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def candidate_dates(start_date: int | None, end_date: int | None, explicit: list[int] | None) -> list[int]:
    if explicit is not None:
        return sorted(set(explicit))
    data = macro_df.copy()
    data["yyyymm"] = data["yyyymm"].astype(int)
    dates = sorted(data["yyyymm"].unique().astype(int).tolist())
    if start_date is not None:
        dates = [date for date in dates if date >= start_date]
    if end_date is not None:
        dates = [date for date in dates if date <= end_date]
    return dates


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast anchor-level prefilter for Scenario 4.")
    parser.add_argument("--candidate-dates", type=str, default=None)
    parser.add_argument("--start-date", type=int, default=None)
    parser.add_argument("--end-date", type=int, default=None)
    parser.add_argument(
        "--target",
        choices=["sc_beats_ww", "ww_beats_sc"],
        default="sc_beats_ww",
        help="Desired generated winner; prefilter keeps anchors where the opposite model initially wins.",
    )
    parser.add_argument("--min-ww-edge", type=float, default=0.0025, help="For sc_beats_ww target, require WW-SC anchor edge at least this much.")
    parser.add_argument("--min-sc-edge", type=float, default=0.0025, help="For ww_beats_sc target, require SC-WW anchor edge at least this much.")
    parser.add_argument("--max-financial-stress-prob", type=float, default=0.50)
    parser.add_argument(
        "--true-labels",
        type=str,
        default=None,
        help="Optional comma-separated true historical labels to keep, e.g. financial_stress,contraction.",
    )
    parser.add_argument("--min-contraction-prob", type=float, default=0.0)
    parser.add_argument("--min-stress-or-contraction-prob", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()
    true_labels = set(parse_dates(None) or [])
    if args.true_labels:
        true_labels = {label.strip() for label in args.true_labels.split(",") if label.strip()}

    runtime = SCENARIO_ROOT / "runtime_universe500"
    summer_model, _ = load_exact_e2e_model_from_run(runtime / "models/summer_child")
    winter_model, _ = load_exact_e2e_model_from_run(runtime / "models/winter_wolf")

    rows = []
    dates = candidate_dates(args.start_date, args.end_date, parse_dates(args.candidate_dates))
    for idx, date in enumerate(dates, start=1):
        try:
            raw_values = macro_df.loc[macro_df["yyyymm"].astype(int).eq(date), list(MACRO_ORDER)]
            if raw_values.empty:
                continue
            m0_raw = torch.tensor(raw_values.to_numpy(dtype=np.float32, copy=True)[0], dtype=torch.float32)
            m0 = (m0_raw - macro_mean) / macro_std
            sigma, builder, returns, permnos = construct_C2_21a(date, ASSET_SIZE)
            summer_pi = AllocationPipeline(summer_model, sigma)
            winter_pi = AllocationPipeline(winter_model, sigma)
            summer_eval, _ = evaluate(m0, builder, returns, sigma, summer_pi)
            winter_eval, _ = evaluate(m0, builder, returns, sigma, winter_pi)
            summer_return = float(summer_eval[0].item())
            winter_return = float(winter_eval[0].item())
            summer_sharpe = float(summer_eval[2].item())
            winter_sharpe = float(winter_eval[2].item())
            regime = regime_classifier.classify(m0_raw.numpy())
            probs = regime["probabilities"]
            rows.append(
                {
                    "date": date,
                    "n_assets": len(permnos),
                    "summer_return": summer_return,
                    "winter_return": winter_return,
                    "anchor_gap_sc_minus_ww": summer_return - winter_return,
                    "ww_edge": winter_return - summer_return,
                    "summer_sharpe": summer_sharpe,
                    "winter_sharpe": winter_sharpe,
                    "anchor_regime": regime["label"],
                    "anchor_true_label": regime_classifier.historical_label(date),
                    "prob_contraction": float(probs.get("contraction", 0.0)),
                    "prob_expansion": float(probs.get("expansion", 0.0)),
                    "prob_financial_stress": float(probs.get("financial_stress", 0.0)),
                }
            )
            print(f"[{idx}/{len(dates)}] {date}: anchor_gap={summer_return - winter_return:+.2%}, regime={regime['label']}")
        except Exception as exc:
            rows.append({"date": date, "error": repr(exc)})
            print(f"[{idx}/{len(dates)}] {date}: failed {exc}")

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise SystemExit("No rows produced.")

    out_dir = ROOT / "scenario_outputs" / "scenario4_anchor_prefilter"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    frame.to_csv(out_dir / f"anchor_prefilter_{stamp}.csv", index=False)

    ok = frame[frame.get("error").isna() if "error" in frame else np.ones(len(frame), dtype=bool)].copy()
    edge_mask = ok["ww_edge"] >= args.min_ww_edge
    sort_edge = "ww_edge"
    if args.target == "ww_beats_sc":
        edge_mask = ok["anchor_gap_sc_minus_ww"] >= args.min_sc_edge
        sort_edge = "anchor_gap_sc_minus_ww"

    selected = ok[
        edge_mask
        & (ok["prob_financial_stress"] <= args.max_financial_stress_prob)
        & (ok["prob_contraction"] >= args.min_contraction_prob)
        & ((ok["prob_contraction"] + ok["prob_financial_stress"]) >= args.min_stress_or_contraction_prob)
    ].sort_values([sort_edge, "prob_expansion"], ascending=[False, False])
    if true_labels:
        selected = selected[selected["anchor_true_label"].isin(true_labels)]
    selected.head(args.top_k).to_csv(out_dir / f"anchor_prefilter_selected_{stamp}.csv", index=False)
    print(f"\nResults saved -> {out_dir.relative_to(ROOT)}")
    print("Selected dates:")
    print(",".join(str(int(date)) for date in selected["date"].head(args.top_k)))


if __name__ == "__main__":
    main()
