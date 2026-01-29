#!/usr/bin/env python3
"""
Generate Summary CSVs for E2E Experiment Results

Creates summary CSV files for each category of E2E experiments:
1. Hyperparameter grid search
2. Universe sweep (standard training)
3. Summer child crisis scenario
4. Winter wolf crisis scenario
5. Master comparison across all categories

Each summary includes:
- Configuration parameters (topk, loss, gamma/lambda, kappa, omega)
- Test performance metrics (Sharpe, return, volatility, hit rate)
- Training metrics (best epoch, validation performance)
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import re

def parse_config_from_dirname(dirname: str) -> Dict[str, str]:
    """
    Parse configuration parameters from directory name.

    Example:
        'loss=utility__gamma=10.0__kappa=1.0__omega=diagSigma__mu=zscore'
        -> {'loss': 'utility', 'gamma': '10.0', 'kappa': '1.0', 'omega': 'diagSigma', 'mu': 'zscore'}
    """
    config = {}

    # Extract scenario prefix if present
    if dirname.startswith('scenario='):
        parts = dirname.split('__', 1)
        config['scenario'] = parts[0].replace('scenario=', '')
        dirname = parts[1] if len(parts) > 1 else ''

    # Parse key=value pairs
    pattern = r'(\w+)=([^_]+(?:_[^_]+)?)'
    matches = re.findall(pattern, dirname)

    for key, value in matches:
        # Handle multi-word values (e.g., diagSigma)
        config[key] = value

    return config

def load_model_metrics(run_dir: Path) -> Optional[Dict]:
    """
    Load metrics from a single trained model directory.

    Returns dict with:
    - config: model configuration
    - test_metrics: test set performance
    - train_metrics: training history
    """
    try:
        # Load config
        config_file = run_dir / 'config.json'
        if not config_file.exists():
            return None
        with open(config_file, 'r') as f:
            config = json.load(f)

        # Load test summary (optional)
        test_file = run_dir / 'test_summary.json'
        test_metrics = {}
        if test_file.exists():
            with open(test_file, 'r') as f:
                test_metrics = json.load(f)

        # Load train summary
        train_file = run_dir / 'train_summary.json'
        train_metrics = {}
        if train_file.exists():
            with open(train_file, 'r') as f:
                train_metrics = json.load(f)

        # Must have at least config
        return {
            'config': config,
            'test_metrics': test_metrics,
            'train_metrics': train_metrics
        }
    except Exception as e:
        print(f"Warning: Failed to load metrics from {run_dir}: {e}")
        return None

def create_summary_row(run_dir: Path, metrics: Dict) -> Dict:
    """Create a single row for the summary CSV."""
    config = metrics['config']
    test = metrics['test_metrics']
    train = metrics['train_metrics']

    # Parse config from directory name
    dir_config = parse_config_from_dirname(run_dir.name)

    row = {
        'run_name': run_dir.name,
        'scenario': dir_config.get('scenario', 'standard'),
        'topk': config.get('topk', dir_config.get('topk', 'N/A')),
        'loss': config.get('loss_type', dir_config.get('loss', 'N/A')),
        'lambda': config.get('lambda_', config.get('gamma', 'N/A')),  # Handle both names
        'kappa': config.get('kappa', dir_config.get('kappa', 'N/A')),
        'omega': config.get('omega_mode', dir_config.get('omega', 'N/A')),
        'mu_transform': config.get('mu_transform', dir_config.get('mu', 'N/A')),

        # Test metrics
        'test_sharpe': test.get('test_sharpe', 'N/A'),
        'test_return_ann': test.get('test_return_ann', 'N/A'),
        'test_vol_ann': test.get('test_vol_ann', 'N/A'),
        'test_hit_rate': test.get('test_hit_rate', 'N/A'),
        'test_turnover': test.get('test_turnover', 'N/A'),

        # Train metrics
        'best_epoch': train.get('best_epoch', 'N/A'),
        'best_val_metric': train.get('best_val_metric', 'N/A'),
        'final_train_loss': train.get('final_train_loss', 'N/A'),
    }

    return row

def generate_category_summary(category_path: Path, output_file: Path, category_name: str):
    """Generate summary CSV for a specific category."""
    print(f"\nProcessing {category_name}...")
    print(f"  Path: {category_path}")

    if not category_path.exists():
        print(f"  Warning: Path does not exist, skipping")
        return

    # Find all run directories
    run_dirs = [d for d in category_path.iterdir() if d.is_dir() and (d / 'config.json').exists()]

    if not run_dirs:
        print(f"  Warning: No valid run directories found")
        return

    print(f"  Found {len(run_dirs)} runs")

    # Collect metrics from all runs
    rows = []
    for run_dir in sorted(run_dirs):
        metrics = load_model_metrics(run_dir)
        if metrics:
            row = create_summary_row(run_dir, metrics)
            rows.append(row)

    if not rows:
        print(f"  Warning: No metrics loaded")
        return

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Remove columns where all values are 'N/A'
    cols_to_drop = []
    for col in df.columns:
        if (df[col] == 'N/A').all():
            cols_to_drop.append(col)

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"  Dropped {len(cols_to_drop)} all-N/A columns: {', '.join(cols_to_drop)}")

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"  ✓ Saved {len(rows)} runs to {output_file}")
    print(f"  Columns: {', '.join(df.columns)}")

    return df

def main():
    """Generate all summary CSVs."""
    base_dir = Path(__file__).parent.parent

    print("=" * 80)
    print("E2E Experiment Summary Generator")
    print("=" * 80)

    summaries = []

    # 1. Hyperparameter Grid (Standard Training)
    hyperparam_path = base_dir / 'outputs' / 'e2e' / 'standard_training' / 'hyperparameter_grid' / 'runs'
    hyperparam_csv = base_dir / 'outputs' / 'e2e' / 'standard_training' / 'hyperparameter_grid' / 'summary.csv'
    df1 = generate_category_summary(hyperparam_path, hyperparam_csv, "Hyperparameter Grid (Standard Training)")
    if df1 is not None:
        summaries.append(('hyperparameter_grid', df1))

    # 2. Universe Sweep (Standard Training)
    universe_path = base_dir / 'outputs' / 'e2e' / 'standard_training' / 'universe_sweep' / 'runs'
    universe_csv = base_dir / 'outputs' / 'e2e' / 'standard_training' / 'universe_sweep' / 'summary.csv'
    df2 = generate_category_summary(universe_path, universe_csv, "Universe Sweep (Standard Training)")
    if df2 is not None:
        summaries.append(('universe_sweep', df2))

    # 3. Summer Child - One Config
    summer_oneconfig_path = base_dir / 'outputs' / 'e2e' / 'crisis_scenarios' / 'summer_child_no_crisis' / 'runs_oneconfig'
    summer_oneconfig_csv = base_dir / 'outputs' / 'e2e' / 'crisis_scenarios' / 'summer_child_no_crisis' / 'summary_oneconfig.csv'
    df3 = generate_category_summary(summer_oneconfig_path, summer_oneconfig_csv, "Summer Child - One Config")
    if df3 is not None:
        summaries.append(('summer_child_oneconfig', df3))

    # 4. Summer Child - Universe Sweep
    summer_sweep_path = base_dir / 'outputs' / 'e2e' / 'crisis_scenarios' / 'summer_child_no_crisis' / 'runs_universe_sweep'
    summer_sweep_csv = base_dir / 'outputs' / 'e2e' / 'crisis_scenarios' / 'summer_child_no_crisis' / 'summary_universe_sweep.csv'
    df4 = generate_category_summary(summer_sweep_path, summer_sweep_csv, "Summer Child - Universe Sweep")
    if df4 is not None:
        summaries.append(('summer_child_sweep', df4))

    # 5. Winter Wolf - One Config
    winter_oneconfig_path = base_dir / 'outputs' / 'e2e' / 'crisis_scenarios' / 'winter_wolf_with_crisis' / 'runs_oneconfig'
    winter_oneconfig_csv = base_dir / 'outputs' / 'e2e' / 'crisis_scenarios' / 'winter_wolf_with_crisis' / 'summary_oneconfig.csv'
    df5 = generate_category_summary(winter_oneconfig_path, winter_oneconfig_csv, "Winter Wolf - One Config")
    if df5 is not None:
        summaries.append(('winter_wolf_oneconfig', df5))

    # 6. Winter Wolf - Universe Sweep
    winter_sweep_path = base_dir / 'outputs' / 'e2e' / 'crisis_scenarios' / 'winter_wolf_with_crisis' / 'runs_universe_sweep'
    winter_sweep_csv = base_dir / 'outputs' / 'e2e' / 'crisis_scenarios' / 'winter_wolf_with_crisis' / 'summary_universe_sweep.csv'
    df6 = generate_category_summary(winter_sweep_path, winter_sweep_csv, "Winter Wolf - Universe Sweep")
    if df6 is not None:
        summaries.append(('winter_wolf_sweep', df6))

    # 7. Master Summary (All Categories Combined)
    if summaries:
        print("\n" + "=" * 80)
        print("Creating Master Summary")
        print("=" * 80)

        # Add category column to each dataframe
        for category_name, df in summaries:
            df['category'] = category_name

        # Combine all summaries
        master_df = pd.concat([df for _, df in summaries], ignore_index=True)

        # Reorder columns to put category first
        cols = ['category'] + [col for col in master_df.columns if col != 'category']
        master_df = master_df[cols]

        # Remove columns where all values are 'N/A' in master summary
        cols_to_drop = []
        for col in master_df.columns:
            if col != 'category' and (master_df[col] == 'N/A').all():
                cols_to_drop.append(col)

        if cols_to_drop:
            master_df = master_df.drop(columns=cols_to_drop)
            print(f"  Dropped {len(cols_to_drop)} all-N/A columns from master: {', '.join(cols_to_drop)}")

        # Save master summary
        master_csv = base_dir / 'outputs' / 'e2e' / 'master_summary.csv'
        master_df.to_csv(master_csv, index=False)
        print(f"  ✓ Saved master summary with {len(master_df)} total runs to {master_csv}")

        # Print summary statistics
        print("\n" + "=" * 80)
        print("Summary Statistics by Category")
        print("=" * 80)

        # Only aggregate columns that exist
        agg_dict = {'run_name': 'count'}
        if 'test_sharpe' in master_df.columns:
            agg_dict['test_sharpe'] = lambda x: f"{pd.to_numeric(x, errors='coerce').mean():.3f}"
        if 'best_val_metric' in master_df.columns:
            agg_dict['best_val_metric'] = lambda x: f"{pd.to_numeric(x, errors='coerce').mean():.3f}"

        rename_dict = {'run_name': 'count'}
        if 'test_sharpe' in agg_dict:
            rename_dict['test_sharpe'] = 'avg_sharpe'
        if 'best_val_metric' in agg_dict:
            rename_dict['best_val_metric'] = 'avg_val_metric'

        print(master_df.groupby('category').agg(agg_dict).rename(columns=rename_dict))

    print("\n" + "=" * 80)
    print("Summary generation complete!")
    print("=" * 80)

if __name__ == '__main__':
    main()
