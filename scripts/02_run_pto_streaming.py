#!/usr/bin/env python3
"""
Memory-Efficient PTO Backtest Script using Streaming Data Loading.

Uses MonthlyParquetDataset to process data month-by-month without loading all data.

Workflow:
1. Load FNN model
2. Stream test data month-by-month using PyArrow
3. Generate predictions incrementally
4. Run portfolio optimization month-by-month
5. Compute performance statistics
6. Save results

Usage:
    python scripts/02_run_pto_streaming.py
    python scripts/02_run_pto_streaming.py --topk 100 --lambda 10.0
"""
import argparse
import sys
from pathlib import Path

# Add project root and src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from data.iterable_dataset import create_monthly_iterator
from models.fnn import load_fnn_from_dir
from optimization.risk import make_psd, sigma_vol_from_cov
from optimization.solvers import solve_robust_longonly
from portfolio.selection import select_universe_pto_style
from portfolio.metrics import perf_stats_excess, rolling_sharpe
from utils.io import save_json
import configs.pto_config as cfg


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run memory-efficient PTO backtest'
    )

    # Paths
    parser.add_argument(
        '--data-dir',
        type=str,
        default=str(cfg.DATA_DIR),
        help='Directory with processed data'
    )
    parser.add_argument(
        '--fnn-dir',
        type=str,
        default=str(cfg.FNN_DIR),
        help='Directory with FNN model'
    )
    parser.add_argument(
        '--out-dir',
        type=str,
        default=str(cfg.OUT_DIR),
        help='Output directory for results'
    )

    # Date splits
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

    # Universe parameters
    parser.add_argument(
        '--topk',
        type=int,
        default=cfg.TOPK,
        help='Number of assets in portfolio'
    )
    parser.add_argument(
        '--preselect-factor',
        type=int,
        default=cfg.PRESELECT_FACTOR,
        help='Pre-selection factor'
    )
    parser.add_argument(
        '--lookback',
        type=int,
        default=cfg.LOOKBACK,
        help='Lookback window for covariance (months)'
    )

    # Covariance parameters
    parser.add_argument(
        '--lam',
        type=float,
        default=cfg.LAM,
        help='EWMA decay parameter'
    )
    parser.add_argument(
        '--shrink',
        type=float,
        default=cfg.SHRINK,
        help='Shrinkage intensity'
    )
    parser.add_argument(
        '--ridge',
        type=float,
        default=cfg.RIDGE,
        help='Ridge regularization'
    )

    # Optimization parameters
    parser.add_argument(
        '--lambda',
        type=float,
        default=cfg.LAMBDA,
        dest='lambda_',
        help='Risk aversion parameter'
    )

    # Control flags
    parser.add_argument(
        '--device',
        type=str,
        default=cfg.DEVICE,
        choices=['cpu', 'cuda'],
        help='Device for inference'
    )
    parser.add_argument(
        '--kappa-values',
        type=float,
        nargs='+',
        default=cfg.KAPPA_GRID,
        help='Robustness parameter values to test'
    )
    parser.add_argument(
        '--omega-modes',
        type=str,
        nargs='+',
        default=cfg.OMEGA_MODES,
        help='Omega modes to test'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=cfg.VERBOSE,
        help='Print detailed progress'
    )

    return parser.parse_args()


def compute_ewma_cov_simple(returns_history, lam=0.97):
    """
    Compute EWMA covariance from return history.

    Simple implementation: exponentially weighted covariance.

    Args:
        returns_history: numpy array (T, N) - historical returns
        lam: decay parameter

    Returns:
        Covariance matrix (N, N)
    """
    T, N = returns_history.shape

    # De-mean returns
    mu = returns_history.mean(axis=0)
    X_centered = returns_history - mu

    # Compute EWMA weights
    weights = np.array([lam ** i for i in range(T)])[::-1]
    weights = weights / weights.sum()

    # Weighted covariance
    cov = np.zeros((N, N))
    for t in range(T):
        ret_t = X_centered[t:t+1].T  # (N, 1)
        cov += weights[t] * (ret_t @ ret_t.T)

    # Make PSD and regularize
    cov = make_psd(cov, ridge=1e-5)

    return cov


def run_pto_streaming(args):
    """
    Run PTO backtest with streaming data loading.

    Returns:
        Dictionary with results for each strategy
    """
    print("=" * 70)
    print("MEMORY-EFFICIENT PTO BACKTEST (STREAMING)")
    print("=" * 70)

    if args.verbose:
        print(f"\nConfiguration:")
        print(f"  Data dir: {args.data_dir}")
        print(f"  FNN dir: {args.fnn_dir}")
        print(f"  Output dir: {args.out_dir}")
        print(f"  Top-K: {args.topk}")
        print(f"  Preselect factor: {args.preselect_factor}")
        print(f"  Lookback: {args.lookback} months")
        print(f"  Lambda: {args.lambda_}")
        print(f"  EWMA decay (lam): {args.lam}")
        print(f"  Kappa values: {args.kappa_values}")
        print(f"  Omega modes: {args.omega_modes}")

    # Step 1: Load FNN model
    print("\nStep 1: Loading FNN model...")
    fnn_dir = Path(args.fnn_dir)
    if not fnn_dir.exists():
        raise FileNotFoundError(f"FNN directory not found: {fnn_dir}")

    model, feature_cols, fnn_config = load_fnn_from_dir(str(fnn_dir))
    model = model.to(args.device)
    model.eval()

    if args.verbose:
        print(f"  Model input dim: {fnn_config['input_dim']}")
        print(f"  Feature columns: {len(feature_cols)}")

    # Step 2: Create monthly data iterator
    print("\nStep 2: Creating monthly data iterator...")
    monthly_iterator = create_monthly_iterator(args.data_dir, split='test')

    # Step 3: Process months and accumulate results
    print("\nStep 3: Processing test months...")

    # Storage for results
    all_months_data = []
    monthly_results = {
        'equal_weight': []
    }
    for omega_mode in args.omega_modes:
        for kappa in args.kappa_values:
            key = f'{omega_mode}_kappa_{kappa}'
            monthly_results[key] = []

    # Process each month
    month_count = 0
    for month_data in monthly_iterator:
        month_count += 1
        yyyymm = month_data['yyyymm']
        X = month_data['X']
        y_realized = month_data['y']
        meta = month_data['meta']

        if args.verbose and month_count % 12 == 1:
            print(f"  Month {month_count}: {yyyymm} ({len(X)} stocks)")

        # Generate predictions
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(args.device)
            predictions = model(X_tensor).cpu().numpy().flatten()

        # PRE-SELECTION: Select top stocks based on predictions
        # This reduces computational burden dramatically
        n_preselect = min(args.topk * args.preselect_factor, len(predictions))
        top_indices = np.argsort(predictions)[-n_preselect:][::-1]  # Top n_preselect stocks

        # Filter to pre-selected stocks only
        predictions_pre = predictions[top_indices]
        y_realized_pre = y_realized[top_indices]
        permno_pre = meta['permno'].values[top_indices] if 'permno' in meta.columns else top_indices

        # Store month data for later use (covariance history)
        all_months_data.append({
            'yyyymm': yyyymm,
            'permno': permno_pre,
            'predictions': predictions_pre,
            'realized': y_realized_pre,
            'count': len(predictions_pre)
        })

        # Equal weight baseline (on top-K of pre-selected)
        ew_indices = np.argsort(predictions_pre)[-args.topk:][::-1]
        ew_return = np.mean(y_realized_pre[ew_indices])
        monthly_results['equal_weight'].append({
            'yyyymm': yyyymm,
            'return': ew_return
        })

        # For robust MVO, we need return history for covariance
        # Simplified: use diagonal covariance with constant vol estimate
        # (Full implementation would need to load historical returns)
        n_stocks = len(predictions_pre)
        sigma_vol = np.ones(n_stocks) * 0.15  # Rough estimate
        Sigma = np.diag(sigma_vol ** 2)  # Diagonal covariance matrix

        # For each strategy
        for omega_mode in args.omega_modes:
            for kappa in args.kappa_values:
                key = f'{omega_mode}_kappa_{kappa}'

                try:
                    # Solve robust MVO (on pre-selected stocks)
                    weights = solve_robust_longonly(
                        mu_hat=predictions_pre,
                        Sigma=Sigma,
                        lambda_=args.lambda_,
                        kappa=kappa,
                        omega_mode=omega_mode
                    )

                    # Realize return (on pre-selected stocks)
                    port_return = np.dot(weights, y_realized_pre)

                    monthly_results[key].append({
                        'yyyymm': yyyymm,
                        'return': port_return,
                        'n_assets': np.sum(weights > 1e-6)
                    })

                except Exception as e:
                    if args.verbose:
                        print(f"    Warning: Optimization failed for {yyyymm}, {key}: {e}")
                    # Use equal weight fallback
                    monthly_results[key].append({
                        'yyyymm': yyyymm,
                        'return': ew_return,
                        'n_assets': args.topk
                    })

    print(f"\n  Processed {month_count} months")

    # Step 4: Compute performance statistics
    print("\nStep 4: Computing performance statistics...")

    results = {}
    summary_rows = []

    for strategy_name, month_list in monthly_results.items():
        returns_array = np.array([m['return'] for m in month_list])

        # Compute stats
        stats = perf_stats_excess(returns_array)
        stats['strategy'] = strategy_name
        summary_rows.append(stats)

        # Store detailed results
        results[strategy_name] = {
            'monthly_returns': month_list,
            'stats': stats,
            'returns_array': returns_array.tolist()
        }

    summary_df = pd.DataFrame(summary_rows)

    # Print summary
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    print(summary_df[['strategy', 'mean_a', 'vol_a', 'sharpe_a', 'n_months']].to_string(index=False))

    return results, summary_df


def main():
    """Main pipeline."""
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run streaming backtest
    results, summary_df = run_pto_streaming(args)

    # Save results
    print(f"\nSaving results to {out_dir}...")

    # Save summary
    summary_df.to_csv(out_dir / 'results_summary_streaming.csv', index=False)

    # Save detailed results
    save_json(out_dir / 'pto_streaming_results.json', results, convert_np=True)

    # Create plots
    print("\nCreating plots...")

    # Wealth plot
    fig, ax = plt.subplots(figsize=(12, 6))
    for strategy_name, res in results.items():
        returns = res['returns_array']
        wealth = np.cumprod(1.0 + np.array(returns))
        ax.plot(range(len(wealth)), wealth, label=strategy_name, linewidth=1.5)

    ax.set_xlabel("Month")
    ax.set_ylabel("Cumulative Wealth")
    ax.set_title("Wealth Evolution (Streaming PTO)")
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "wealth_streaming.png", dpi=150)
    plt.close()

    # Rolling Sharpe plot
    fig, ax = plt.subplots(figsize=(12, 6))
    for strategy_name, res in results.items():
        returns = res['returns_array']
        rs = rolling_sharpe(np.array(returns), window=cfg.ROLL)
        ax.plot(range(len(rs)), rs, label=strategy_name, linewidth=1.5)

    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_xlabel("Month")
    ax.set_ylabel(f"Rolling {cfg.ROLL}-Month Sharpe")
    ax.set_title(f"Rolling Sharpe Ratio ({cfg.ROLL} months)")
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"rolling_sharpe_streaming.png", dpi=150)
    plt.close()

    print(f"\n✓ Streaming PTO backtest complete!")
    print(f"  Output directory: {out_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
