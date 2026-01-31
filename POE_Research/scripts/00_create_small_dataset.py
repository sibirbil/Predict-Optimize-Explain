#!/usr/bin/env python3
"""
Create a small subset of the data for fast testing.

Selects assets with complete history from 198001 to 202412 and creates
a reduced dataset that can be used to quickly test all functionality.

Usage:
    python scripts/00_create_small_dataset.py --n-assets 500
    python scripts/00_create_small_dataset.py --n-assets 100 --output-dir data/processed/small_data
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(
        description='Create small subset of data for fast testing'
    )
    parser.add_argument(
        '--n-assets',
        type=int,
        default=500,
        help='Number of assets to keep (default: 500)'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/processed/ready_data',
        help='Input data directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed/small_data',
        help='Output directory for small dataset'
    )
    parser.add_argument(
        '--start-date',
        type=int,
        default=198001,
        help='Start date (YYYYMM)'
    )
    parser.add_argument(
        '--end-date',
        type=int,
        default=202412,
        help='End date (YYYYMM)'
    )
    parser.add_argument(
        '--min-coverage',
        type=float,
        default=0.95,
        help='Minimum coverage ratio (e.g., 0.95 = 95%% of months)'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("CREATE SMALL DATASET FOR FAST TESTING")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Input dir: {args.input_dir}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Target assets: {args.n_assets}")
    print(f"  Date range: {args.start_date} - {args.end_date}")
    print(f"  Min coverage: {args.min_coverage * 100:.1f}%")
    print()

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    input_path = Path(args.input_dir)

    # ========================================
    # Step 1: Load metadata only
    # ========================================
    print("[1/6] Loading metadata to find assets...")
    metadata = pd.read_parquet(input_path / "metadata.parquet")

    # Detect ID column
    id_col = None
    for col in ['permno', 'PERMNO', 'asset_id', 'id']:
        if col in metadata.columns:
            id_col = col
            break

    if id_col is None:
        raise ValueError("Could not detect ID column in metadata")

    print(f"  ID column: {id_col}")
    print(f"  Total observations: {len(metadata):,}")

    # ========================================
    # Step 2: Find assets with good coverage
    # ========================================
    print("\n[2/6] Finding assets with complete history...")

    # Filter by date range
    metadata['yyyymm'] = metadata['yyyymm'].astype(int)
    metadata_filtered = metadata[
        (metadata['yyyymm'] >= args.start_date) &
        (metadata['yyyymm'] <= args.end_date)
    ].copy()

    # Count observations per asset
    total_months = len(metadata_filtered['yyyymm'].unique())
    asset_counts = metadata_filtered[id_col].value_counts()

    print(f"  Total months in range: {total_months}")
    print(f"  Total unique assets: {len(asset_counts)}")

    # Find assets with sufficient coverage
    min_obs = int(total_months * args.min_coverage)
    good_assets = asset_counts[asset_counts >= min_obs].index.tolist()

    print(f"  Assets with >={args.min_coverage*100:.0f}% coverage: {len(good_assets)}")

    if len(good_assets) < args.n_assets:
        print(f"\n  WARNING: Only {len(good_assets)} assets have {args.min_coverage*100:.0f}% coverage")
        print(f"  Requested {args.n_assets} assets. Adjusting...")
        args.n_assets = len(good_assets)

    # Select top N assets by coverage
    selected_assets = set(good_assets[:args.n_assets])

    print(f"\n  Selected {len(selected_assets)} assets")
    print(f"  Asset ID range: {min(selected_assets)} - {max(selected_assets)}")

    # ========================================
    # Step 3: Filter metadata
    # ========================================
    print("\n[3/6] Filtering metadata...")

    metadata_small = metadata[metadata[id_col].isin(selected_assets)].copy()

    # Split metadata back into train/val/test
    train_end = 200512
    val_end = 201512

    mask_train = metadata_small['yyyymm'] <= train_end
    mask_val = (metadata_small['yyyymm'] > train_end) & (metadata_small['yyyymm'] <= val_end)
    mask_test = metadata_small['yyyymm'] > val_end

    print(f"  Full metadata: {len(metadata_small)} rows")
    print(f"  Train: {mask_train.sum()} rows")
    print(f"  Val: {mask_val.sum()} rows")
    print(f"  Test: {mask_test.sum()} rows")

    # ========================================
    # Step 4: Filter features and targets (memory-efficient)
    # ========================================
    print("\n[4/6] Filtering features and targets...")

    # Load and filter each split separately to save memory
    print("  Loading train split...")
    meta_train_full = metadata[metadata['yyyymm'] <= train_end].reset_index(drop=True)
    mask_train_full = meta_train_full[id_col].isin(selected_assets)

    X_train = pd.read_parquet(input_path / "X_train.parquet")
    y_train = pd.read_parquet(input_path / "y_train.parquet")

    X_train_small = X_train[mask_train_full].reset_index(drop=True)
    y_train_small = y_train[mask_train_full].reset_index(drop=True)
    meta_train_small = meta_train_full[mask_train_full].reset_index(drop=True)

    print(f"    X_train: {X_train.shape} → {X_train_small.shape}")
    del X_train, y_train, meta_train_full  # Free memory

    print("  Loading val split...")
    meta_val_full = metadata[(metadata['yyyymm'] > train_end) & (metadata['yyyymm'] <= val_end)].reset_index(drop=True)
    mask_val_full = meta_val_full[id_col].isin(selected_assets)

    X_val = pd.read_parquet(input_path / "X_val.parquet")
    y_val = pd.read_parquet(input_path / "y_val.parquet")

    X_val_small = X_val[mask_val_full].reset_index(drop=True)
    y_val_small = y_val[mask_val_full].reset_index(drop=True)
    meta_val_small = meta_val_full[mask_val_full].reset_index(drop=True)

    print(f"    X_val: {X_val.shape} → {X_val_small.shape}")
    del X_val, y_val, meta_val_full  # Free memory

    print("  Loading test split...")
    meta_test_full = metadata[metadata['yyyymm'] > val_end].reset_index(drop=True)
    mask_test_full = meta_test_full[id_col].isin(selected_assets)

    X_test = pd.read_parquet(input_path / "X_test.parquet")
    y_test = pd.read_parquet(input_path / "y_test.parquet")

    X_test_small = X_test[mask_test_full].reset_index(drop=True)
    y_test_small = y_test[mask_test_full].reset_index(drop=True)
    meta_test_small = meta_test_full[mask_test_full].reset_index(drop=True)

    print(f"    X_test: {X_test.shape} → {X_test_small.shape}")
    del X_test, y_test, meta_test_full  # Free memory

    # Combine metadata
    metadata_small = pd.concat([meta_train_small, meta_val_small, meta_test_small], ignore_index=True)

    # ========================================
    # Step 5: Filter auxiliary data
    # ========================================
    print("\n[5/6] Filtering auxiliary data...")

    # Filter firm_clean if it exists
    firm_small = None
    if (input_path / "firm_clean.parquet").exists():
        firm_clean = pd.read_parquet(input_path / "firm_clean.parquet")
        firm_small = firm_clean[firm_clean[id_col].isin(selected_assets)].copy()
        print(f"  firm_clean: {firm_clean.shape} → {firm_small.shape}")
        del firm_clean

    # Keep all macro data
    macro_small = None
    if (input_path / "macro_clean.parquet").exists():
        macro_small = pd.read_parquet(input_path / "macro_clean.parquet")
        print(f"  macro_clean: {macro_small.shape} (kept all)")

    macro_final_small = None
    if (input_path / "macro_final.parquet").exists():
        macro_final_small = pd.read_parquet(input_path / "macro_final.parquet")
        print(f"  macro_final: {macro_final_small.shape} (kept all)")

    # ========================================
    # Step 6: Save small dataset
    # ========================================
    print("\n[6/6] Saving small dataset...")

    # Save each file separately
    X_train_small.to_parquet(output_path / "X_train.parquet", index=False)
    X_val_small.to_parquet(output_path / "X_val.parquet", index=False)
    X_test_small.to_parquet(output_path / "X_test.parquet", index=False)

    y_train_small.to_parquet(output_path / "y_train.parquet", index=False)
    y_val_small.to_parquet(output_path / "y_val.parquet", index=False)
    y_test_small.to_parquet(output_path / "y_test.parquet", index=False)

    metadata_small.to_parquet(output_path / "metadata.parquet", index=False)

    if firm_small is not None:
        firm_small.to_parquet(output_path / "firm_clean.parquet", index=False)
    if macro_small is not None:
        macro_small.to_parquet(output_path / "macro_clean.parquet", index=False)
    if macro_final_small is not None:
        macro_final_small.to_parquet(output_path / "macro_final.parquet", index=False)

    print(f"\n✅ Small dataset saved to: {output_path}")

    # ========================================
    # Summary statistics
    # ========================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\nOriginal dataset:")
    print(f"  Total observations: {len(metadata):,}")
    print(f"  Unique assets: {metadata[id_col].nunique():,}")
    print(f"  Features: {X_train_small.shape[1]}")

    print(f"\nSmall dataset:")
    print(f"  Total observations: {len(metadata_small):,}")
    print(f"  Unique assets: {len(selected_assets)}")
    print(f"  Features: {X_train_small.shape[1]}")
    print(f"  Size reduction: {len(metadata_small) / len(metadata) * 100:.1f}%")

    print(f"\nTrain/Val/Test split:")
    print(f"  Train: {len(X_train_small):,} obs")
    print(f"  Val: {len(X_val_small):,} obs")
    print(f"  Test: {len(X_test_small):,} obs")

    # Asset coverage stats
    coverage_stats = metadata_small.groupby(id_col)['yyyymm'].count()
    print(f"\nAsset coverage:")
    print(f"  Mean: {coverage_stats.mean():.0f} months")
    print(f"  Min: {coverage_stats.min():.0f} months")
    print(f"  Max: {coverage_stats.max():.0f} months")

    print("\n" + "=" * 80)
    print("USAGE")
    print("=" * 80)
    print(f"\nTo use the small dataset, run:")
    print(f"  python scripts/02_run_pto.py --data-dir {args.output_dir} --topk 50")
    print(f"  python scripts/03_run_e2e.py --data-dir {args.output_dir} --topk 50")
    print("\nThis will run much faster for testing!")
    print("=" * 80)


if __name__ == "__main__":
    main()
