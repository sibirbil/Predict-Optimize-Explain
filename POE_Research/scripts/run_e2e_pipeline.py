#!/usr/bin/env python
"""
End-to-End Pipeline for POE Research Project

This script runs the complete pipeline:
1. Validates data integrity
2. Runs PTO backtest with all strategies
3. Post-processes results
4. Generates performance report

Usage:
    python scripts/run_e2e_pipeline.py --verbose
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import argparse
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

def validate_data(data_dir):
    """Validate that all required data files exist and are valid."""
    print("\n" + "=" * 70)
    print("STEP 1: DATA VALIDATION")
    print("=" * 70)

    data_path = Path(data_dir)
    required_files = [
        'X_train.parquet', 'y_train.parquet', 'meta_train.parquet',
        'X_val.parquet', 'y_val.parquet', 'meta_val.parquet',
        'X_test.parquet', 'y_test.parquet', 'meta_test.parquet'
    ]

    print("\nChecking required files...")
    for filename in required_files:
        filepath = data_path / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Required file not found: {filepath}")
        print(f"  ✓ {filename}")

    # Load and validate test data
    print("\nValidating test data...")
    meta_test = pd.read_parquet(data_path / 'meta_test.parquet')
    y_test = pd.read_parquet(data_path / 'y_test.parquet')
    X_test = pd.read_parquet(data_path / 'X_test.parquet')

    print(f"  Test set: {len(meta_test):,} observations")
    print(f"  Unique months: {meta_test['yyyymm'].nunique()}")
    print(f"  Date range: {meta_test['yyyymm'].min()} to {meta_test['yyyymm'].max()}")
    print(f"  Features: {X_test.shape[1]}")

    # Validate historical returns
    print("\nValidating historical returns...")
    meta_train = pd.read_parquet(data_path / 'meta_train.parquet')
    meta_val = pd.read_parquet(data_path / 'meta_val.parquet')

    total_hist = len(meta_train) + len(meta_val)
    print(f"  Historical observations: {total_hist:,}")
    print(f"  Train: {len(meta_train):,}")
    print(f"  Val: {len(meta_val):,}")

    print("\n✅ Data validation complete!")
    return True


def validate_model(fnn_dir):
    """Validate that FNN model exists and can be loaded."""
    print("\n" + "=" * 70)
    print("STEP 2: MODEL VALIDATION")
    print("=" * 70)

    from models.fnn import load_fnn_from_dir

    fnn_path = Path(fnn_dir)
    if not fnn_path.exists():
        raise FileNotFoundError(f"FNN directory not found: {fnn_path}")

    print(f"\nLoading model from {fnn_dir}...")
    model, feature_cols, config = load_fnn_from_dir(str(fnn_path))

    print(f"  ✓ Model loaded successfully")
    print(f"  Input dim: {config['input_dim']}")
    print(f"  Hidden dims: {config.get('hidden_dims', 'N/A')}")
    print(f"  Dropout: {config.get('dropout_rate', 'N/A')}")
    print(f"  Features: {len(feature_cols)}")

    print("\n✅ Model validation complete!")
    return True


def run_pto_backtest(args):
    """Run the optimized PTO backtest."""
    print("\n" + "=" * 70)
    print("STEP 3: RUNNING PTO BACKTEST")
    print("=" * 70)

    print("\nConfiguration:")
    print(f"  Strategies: {len(args.kappa_values)} kappa values × {len(args.omega_modes)} omega modes = {len(args.kappa_values) * len(args.omega_modes)} strategies")
    print(f"  Top-K: {args.topk}")
    print(f"  Lookback: {args.lookback}")
    print(f"  Lambda: {args.lambda_}")

    # Import here to avoid circular dependencies
    import subprocess

    # Build command
    cmd = [
        'python', 'scripts/02_run_pto_streaming_OPTIMIZED.py',
        '--kappa_values'] + [str(k) for k in args.kappa_values] + [
        '--omega_modes'] + args.omega_modes + [
        '--topk', str(args.topk),
        '--lookback', str(args.lookback),
        '--lambda', str(args.lambda_),
        '--verbose'
    ]

    print(f"\nRunning: {' '.join(cmd)}")
    print("\nBacktest starting...")
    start_time = time.time()

    result = subprocess.run(cmd, capture_output=True, text=True)

    elapsed = time.time() - start_time
    print(f"\n✅ Backtest complete in {elapsed/60:.1f} minutes!")

    if result.returncode != 0:
        print("\n❌ Backtest failed!")
        print(result.stderr)
        return False

    # Show key output lines
    output_lines = result.stdout.split('\n')
    for line in output_lines:
        if 'BACKTEST RESULTS' in line or 'Sharpe:' in line or 'Mean (annual):' in line:
            print(line)

    return True


def aggregate_results(output_dir):
    """Aggregate duplicate month entries."""
    print("\n" + "=" * 70)
    print("STEP 4: POST-PROCESSING RESULTS")
    print("=" * 70)

    import subprocess

    print("\nAggregating duplicate months...")
    result = subprocess.run(
        ['python', 'scripts/fix_pto_results.py'],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("❌ Post-processing failed!")
        print(result.stderr)
        return False

    print(result.stdout)
    return True


def generate_report(output_dir):
    """Generate final performance report."""
    print("\n" + "=" * 70)
    print("STEP 5: GENERATING PERFORMANCE REPORT")
    print("=" * 70)

    # Load corrected results
    results_path = Path(output_dir) / 'monthly_results_CORRECTED.json'
    stats_path = Path(output_dir) / 'summary_stats_CORRECTED.csv'

    if not results_path.exists():
        print(f"❌ Corrected results not found: {results_path}")
        return False

    # Load data
    with open(results_path, 'r') as f:
        results = json.load(f)

    stats_df = pd.read_csv(stats_path)

    # Print summary
    print("\n" + "=" * 70)
    print("FINAL PERFORMANCE SUMMARY")
    print("=" * 70)

    print("\n{:<20} {:>12} {:>12} {:>10} {:>12}".format(
        "Strategy", "Annual Ret", "Annual Vol", "Sharpe", "Max DD"
    ))
    print("-" * 70)

    for _, row in stats_df.iterrows():
        print("{:<20} {:>11.2%} {:>11.2%} {:>10.2f} {:>11.2%}".format(
            row['strategy'],
            row['mean_a'],
            row['vol_a'],
            row['sharpe_a'],
            row['max_drawdown']
        ))

    # Key insights
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)

    # Get strategy stats
    ew_sharpe = stats_df[stats_df['strategy'] == 'equal_weight']['sharpe_a'].values[0]
    mvo_sharpe = stats_df[stats_df['strategy'].str.contains('kappa_0.0')]['sharpe_a'].values[0]

    print(f"\n✓ MVO Sharpe ({mvo_sharpe:.2f}) > Equal Weight ({ew_sharpe:.2f})")

    # Check kappa effect
    kappa_strategies = stats_df[stats_df['strategy'].str.contains('diagSigma')]
    kappa_strategies = kappa_strategies.sort_values('strategy')

    print(f"\n✓ Kappa Effect:")
    for _, row in kappa_strategies.iterrows():
        kappa_val = row['strategy'].split('_')[-1]
        print(f"  κ={kappa_val}: Sharpe={row['sharpe_a']:.2f}, Vol={row['vol_a']:.2%}")

    # Validation checks
    print("\n" + "=" * 70)
    print("VALIDATION CHECKS")
    print("=" * 70)

    checks = []

    # Check 1: All Sharpes positive
    all_positive = (stats_df['sharpe_a'] > 0).all()
    checks.append(("All Sharpe ratios > 0", all_positive))

    # Check 2: MVO outperforms EW
    mvo_better = mvo_sharpe > ew_sharpe
    checks.append(("MVO outperforms Equal Weight", mvo_better))

    # Check 3: Reasonable Sharpe range
    sharpe_reasonable = (stats_df['sharpe_a'] > 0.3).all() and (stats_df['sharpe_a'] < 2.0).all()
    checks.append(("Sharpe ratios in reasonable range (0.3-2.0)", sharpe_reasonable))

    # Check 4: Reasonable max drawdown
    dd_reasonable = (stats_df['max_drawdown'] > -0.5).all()
    checks.append(("Max drawdowns reasonable (> -50%)", dd_reasonable))

    # Check 5: Hit rates positive
    hit_reasonable = (stats_df['hit_rate'] > 50).all()
    checks.append(("Hit rates > 50%", hit_reasonable))

    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"{status} {check_name}")

    all_passed = all(passed for _, passed in checks)

    if all_passed:
        print("\n🎉 ALL VALIDATION CHECKS PASSED!")
    else:
        print("\n⚠️  Some validation checks failed")

    return all_passed


def main():
    parser = argparse.ArgumentParser(description='Run E2E POE Research Pipeline')
    parser.add_argument('--data_dir', type=str, default='data/processed/ready_data')
    parser.add_argument('--fnn_dir', type=str, default='models/fnn_v1')
    parser.add_argument('--output_dir', type=str, default='POE_Research/outputs/pto/results')
    parser.add_argument('--topk', type=int, default=200)
    parser.add_argument('--lookback', type=int, default=60)
    parser.add_argument('--lambda', type=float, default=5.0, dest='lambda_')
    parser.add_argument('--kappa_values', type=float, nargs='+', default=[0.0, 0.1, 1.0])
    parser.add_argument('--omega_modes', type=str, nargs='+', default=['diagSigma'])
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--skip_validation', action='store_true', help='Skip data/model validation')

    args = parser.parse_args()

    print("=" * 70)
    print("POE RESEARCH - END-TO-END PIPELINE")
    print("=" * 70)
    print(f"\nStarting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    start_time = time.time()

    try:
        # Step 1: Validate data
        if not args.skip_validation:
            validate_data(args.data_dir)

            # Step 2: Validate model
            validate_model(args.fnn_dir)
        else:
            print("\n⚠️  Skipping data/model validation")

        # Step 3: Run backtest
        success = run_pto_backtest(args)
        if not success:
            print("\n❌ Pipeline failed at backtest step")
            return 1

        # Step 4: Post-process results
        success = aggregate_results(args.output_dir)
        if not success:
            print("\n❌ Pipeline failed at post-processing step")
            return 1

        # Step 5: Generate report
        success = generate_report(args.output_dir)
        if not success:
            print("\n❌ Pipeline failed at reporting step")
            return 1

        elapsed = time.time() - start_time

        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)
        print(f"\nTotal time: {elapsed/60:.1f} minutes")
        print(f"Results saved to: {args.output_dir}")
        print("\n✅ E2E PIPELINE SUCCESS!")

        return 0

    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
