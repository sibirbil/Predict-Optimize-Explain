#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inspect portfolio weight concentration in evaluation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check weight distributions for concentration issues.")
    parser.add_argument("weights", help="Path to weights JSON emitted by evaluate_regret_model.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Alert if any weight exceeds this value (default 0.5).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of top allocations to print per date (default 5).",
    )
    return parser.parse_args()


def load_weights(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Weights JSON must be a list of records.")
    return data


def main() -> None:
    args = parse_args()
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(weights_path)

    records = load_weights(weights_path)
    by_date = {}
    for rec in records:
        date = rec.get("date", "unknown")
        by_date.setdefault(date, []).append(rec)

    for date, entries in sorted(by_date.items()):
        entries.sort(key=lambda r: r.get("weight", 0.0), reverse=True)
        top_entries = entries[: args.top]
        max_weight = top_entries[0]["weight"] if top_entries else 0.0
        alert = " !!!" if max_weight > args.threshold else ""
        print(f"{date}: max_weight={max_weight:.4f}{alert}")
        for rec in top_entries:
            ticker = rec.get("ticker", "-")
            asset_id = rec.get("asset_id", "-")
            print(
                f"  {ticker:>8} (asset_id={asset_id:<8}) weight={rec['weight']:.4f} "
                f"pred={rec['prediction']:.4f} ret={rec['return']:.4f}"
            )


if __name__ == "__main__":
    main()
