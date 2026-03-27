#!/usr/bin/env python3
"""
Data Preparation Script.

Processes raw firm and macro data into train/val/test splits.

Workflow:
1. Load raw firm characteristics and macro predictors
2. Process firm data (drop missing, clean features)
3. Process target variable (scale detection, clipping)
4. Process macro data (interpolation, alignment)
5. Build interactions and final feature matrix
6. Split into train/val/test
7. Save processed data

Usage:
    python scripts/01_prepare_data.py
    python scripts/01_prepare_data.py --firm-data data/raw/signed_predictors_wide.parquet
    python scripts/01_prepare_data.py --train-end 200512 --val-end 201512
"""
import argparse
import sys
from pathlib import Path

# Add project root and src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd

from data.processors import FirmOnlyProcessor, TargetProcessor, MacroDataProcessor
from data.builders import HighRAMDatasetBuilder
from data.storage import DataStorageEngine
from utils.io import save_json
import configs.data_config as cfg


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Prepare POE research dataset from raw data'
    )

    # Input paths
    parser.add_argument(
        '--firm-data',
        type=str,
        default=str(cfg.RAW_DATA_DIR / cfg.FIRM_DATA_FILE),
        help='Path to firm characteristics parquet file'
    )
    parser.add_argument(
        '--macro-data',
        type=str,
        default=str(cfg.RAW_DATA_DIR / cfg.MACRO_DATA_FILE),
        help='Path to macro predictors CSV file'
    )

    # Output path
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(cfg.PROCESSED_DATA_DIR),
        help='Output directory for processed data'
    )

    # Date parameters
    parser.add_argument(
        '--start-date',
        type=int,
        default=cfg.FIRM_START_DATE,
        help='Start date (YYYYMM format)'
    )
    parser.add_argument(
        '--end-date',
        type=int,
        default=cfg.FIRM_END_DATE,
        help='End date (YYYYMM format)'
    )
    parser.add_argument(
        '--train-end',
        type=int,
        default=cfg.TRAIN_END,
        help='Training period end (YYYYMM)'
    )
    parser.add_argument(
        '--val-end',
        type=int,
        default=cfg.VAL_END,
        help='Validation period end (YYYYMM)'
    )

    # Processing parameters
    parser.add_argument(
        '--missing-thresh',
        type=float,
        default=cfg.MISSING_THRESH,
        help='Missing data threshold for dropping columns'
    )
    parser.add_argument(
        '--target-lower',
        type=float,
        default=cfg.TARGET_LOWER_LIMIT,
        help='Physical lower limit for returns'
    )
    parser.add_argument(
        '--target-upper-q',
        type=float,
        default=cfg.TARGET_UPPER_QUANTILE,
        help='Upper quantile for clipping extreme returns'
    )

    # Control flags
    parser.add_argument(
        '--no-interactions',
        action='store_true',
        help='Skip creating firm × macro interactions'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=100000,
        help='Chunk size for memory-efficient processing (default: 100000)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=cfg.RANDOM_STATE,
        help='Random seed'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=cfg.VERBOSE,
        help='Print detailed progress'
    )

    return parser.parse_args()


def main():
    """Main data preparation pipeline."""
    args = parse_args()

    # Set random seed
    np.random.seed(args.seed)

    if args.verbose:
        print("=" * 70)
        print("POE RESEARCH - DATA PREPARATION")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  Firm data: {args.firm_data}")
        print(f"  Macro data: {args.macro_data}")
        print(f"  Output dir: {args.output_dir}")
        print(f"  Date range: {args.start_date} to {args.end_date}")
        print(f"  Train end: {args.train_end}")
        print(f"  Val end: {args.val_end}")
        print(f"  Missing threshold: {args.missing_thresh}")
        print(f"  Random seed: {args.seed}")
        print()

    # ========================================
    # Step 1: Load raw data
    # ========================================
    if args.verbose:
        print("Step 1: Loading raw data...")

    firm_path = Path(args.firm_data)
    macro_path = Path(args.macro_data)

    if not firm_path.exists():
        raise FileNotFoundError(f"Firm data not found: {firm_path}")
    if not macro_path.exists():
        raise FileNotFoundError(f"Macro data not found: {macro_path}")

    firm_df = pd.read_parquet(firm_path)
    macro_df = pd.read_csv(macro_path)

    if args.verbose:
        print(f"  Firm data: {firm_df.shape}")
        print(f"  Macro data: {macro_df.shape}")

    # ========================================
    # Step 2: Process firm data
    # ========================================
    if args.verbose:
        print("\nStep 2: Processing firm characteristics...")

    processor = FirmOnlyProcessor(
        firm_path=firm_path,
        start_date=args.start_date,
        end_date=args.end_date,
        missing_thresh=args.missing_thresh
    )
    df_firm = processor.load_data()
    df_firm_clean = processor.process(df_firm)

    if args.verbose:
        print(f"  Clean firm data: {df_firm_clean.shape}")

    # ========================================
    # Step 3: Process target variable
    # ========================================
    if args.verbose:
        print("\nStep 3: Processing target variable...")

    target_proc = TargetProcessor(
        target_col=cfg.TARGET_COL,
        lower_limit=args.target_lower,
        upper_quantile=args.target_upper_q
    )
    df_firm_final = target_proc.process(df_firm_clean)

    if args.verbose:
        print(f"  Final firm data: {df_firm_final.shape}")
        print(f"  Target stats: mean={df_firm_final[cfg.TARGET_COL].mean():.4f}, "
              f"std={df_firm_final[cfg.TARGET_COL].std():.4f}")

    # ========================================
    # Step 4: Process macro data
    # ========================================
    if args.verbose:
        print("\nStep 4: Processing macro predictors...")

    macro_proc = MacroDataProcessor(
        file_path=macro_path,
        start_date=args.start_date,
        end_date=args.end_date
    )
    macro_final = macro_proc.load_and_process()

    if args.verbose:
        print(f"  Macro data: {macro_final.shape}")

    # ========================================
    # Step 5-7: Build interactions, split, and save (memory-efficient)
    # ========================================
    builder = HighRAMDatasetBuilder(df_firm_final, macro_final)
    builder.merge_and_calculate_target()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.no_interactions:
        # Fast path: No interactions
        if args.verbose:
            print("\nStep 5: Building feature matrix (no interactions)...")
            print("  Skipping interactions (--no-interactions flag set)")

        meta_cols = ['permno', 'yyyymm', 'ret_tplus1', 'excess_ret', 'Rfree', 'Price', 'Size']
        firm_cols = [
            c for c in builder.full_df.columns
            if c not in meta_cols and c not in macro_final.columns
        ]
        X_matrix = builder.full_df[firm_cols].copy()
        metadata = builder.full_df[['yyyymm', 'permno', 'excess_ret']].copy()

        if args.verbose:
            print(f"  Feature matrix: {X_matrix.shape}")
            print(f"  Metadata: {metadata.shape}")

        # Split data
        if args.verbose:
            print("\nStep 6: Splitting into train/val/test...")

        data_dict = builder.split_data(
            X_matrix,
            metadata,
            train_end=args.train_end,
            val_end=args.val_end
        )

        if args.verbose:
            print(f"  Train: X={data_dict['X_train'].shape}, y={len(data_dict['y_train'])}")
            print(f"  Val:   X={data_dict['X_val'].shape}, y={len(data_dict['y_val'])}")
            print(f"  Test:  X={data_dict['X_test'].shape}, y={len(data_dict['y_test'])}")

        # Save
        if args.verbose:
            print("\nStep 7: Saving processed data...")

        engine = DataStorageEngine(storage_dir=str(output_dir))
        engine.save_dataset(data_dict)

    else:
        # Memory-efficient chunked processing with interactions
        if args.verbose:
            print("\nStep 5-7: Memory-efficient chunked processing with interactions...")
            print(f"  Using chunk size: {args.chunk_size:,} rows")

        paths = builder.create_interactions_chunked(
            output_dir=str(output_dir),
            train_end=args.train_end,
            val_end=args.val_end,
            chunk_size=args.chunk_size
        )

        if args.verbose:
            print(f"\n  Files saved to: {output_dir}")

        # Load metadata for feature columns (we need to reconstruct data_dict structure)
        # The chunked method already saved the files, so we just need feature columns
        meta_cols = ['permno', 'yyyymm', 'ret_tplus1', 'excess_ret', 'Rfree', 'Price', 'Size']
        macro_predictors = [c for c in macro_final.columns if c not in ['yyyymm', 'Rfree']]
        firm_cols = [
            c for c in builder.full_df.columns
            if c not in meta_cols and c not in macro_final.columns
        ]

        # Create feature column list
        feature_cols = firm_cols.copy()
        for macro in macro_predictors:
            feature_cols.extend([f"{col}_x_{macro}" for col in firm_cols])

        # Save feature columns
        save_json(output_dir / "feature_columns.json", feature_cols)

        if args.verbose:
            print(f"  Saved {len(feature_cols)} feature columns")

        # Load one file to get sample counts
        y_train = pd.read_parquet(output_dir / "y_train.parquet")
        y_val = pd.read_parquet(output_dir / "y_val.parquet")
        y_test = pd.read_parquet(output_dir / "y_test.parquet")

        # Save dataset info
        dataset_info = {
            "n_features": len(feature_cols),
            "train_end": args.train_end,
            "val_end": args.val_end,
            "train_samples": len(y_train),
            "val_samples": len(y_val),
            "test_samples": len(y_test),
            "random_seed": args.seed,
            "missing_thresh": args.missing_thresh,
            "target_col": cfg.TARGET_COL,
            "chunked_processing": True,
            "chunk_size": args.chunk_size,
        }
        save_json(output_dir / "dataset_info.json", dataset_info)

    # Save additional metadata (for both paths)
    if args.no_interactions:
        # Save feature columns for no-interactions path
        feature_cols = X_matrix.columns.tolist()
        save_json(output_dir / "feature_columns.json", feature_cols)

        if args.verbose:
            print(f"  Saved {len(feature_cols)} feature columns")

        # Save dataset info
        dataset_info = {
            "n_features": len(feature_cols),
            "train_end": args.train_end,
            "val_end": args.val_end,
            "train_samples": len(data_dict['y_train']),
            "val_samples": len(data_dict['y_val']),
            "test_samples": len(data_dict['y_test']),
            "random_seed": args.seed,
            "missing_thresh": args.missing_thresh,
            "target_col": cfg.TARGET_COL,
            "chunked_processing": False,
        }
        save_json(output_dir / "dataset_info.json", dataset_info)

    # Save clean panels for scenario generation
    df_firm_final.to_parquet(
        output_dir / "firm_clean.parquet",
        engine="pyarrow",
        compression="snappy"
    )
    macro_final.to_parquet(
        output_dir / "macro_clean.parquet",
        engine="pyarrow",
        compression="snappy"
    )

    if args.verbose:
        print("\nData preparation complete!")
        print(f"  Output directory: {output_dir}")
        print("=" * 70)


if __name__ == "__main__":
    main()
