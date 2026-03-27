#!/usr/bin/env python3
"""
Memory-Efficient PAO Training Script using Streaming Data Loading.

Trains Predict-and-Optimize portfolio models without loading all data at once.

Workflow:
1. Load FNN model
2. Stream data to build month-level cache incrementally
3. Train PAO models using cached month data
4. Evaluate on test set
5. Save results

Usage:
    python scripts/03_run_pao_streaming.py
    python scripts/03_run_pao_streaming.py --topk 50 --loss-type utility --lambda 10.0
"""
import argparse
import sys
import copy
from pathlib import Path

# Add project root and src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
import torch

from data.iterable_dataset import create_monthly_iterator
from models.fnn import load_fnn_from_dir
from models.pao_portfolio import PAOPortfolioModel
from models.score_network import compute_mu_reference
from optimization.risk import make_psd, sigma_vol_from_cov
from cache.dataset import MonthCacheDataset
from training.trainer import train_one_run
from training.evaluation import eval_dataset, backtest_and_save
from utils.io import save_json, load_json
from utils.dates import shift_yyyymm
from configs.pao_config import PAOConfig, get_config_with_overrides


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train PAO with streaming data loading'
    )

    # Config file
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to JSON config file'
    )

    # Key parameters
    parser.add_argument(
        '--topk',
        type=int,
        default=None,
        help='Portfolio size'
    )
    parser.add_argument(
        '--lambda',
        type=float,
        default=None,
        dest='lambda_',
        help='Risk aversion parameter'
    )
    parser.add_argument(
        '--kappa',
        type=float,
        default=None,
        help='Robustness parameter'
    )
    parser.add_argument(
        '--loss-type',
        type=str,
        default=None,
        choices=['return', 'utility', 'sharpe'],
        help='Loss function'
    )

    # Paths
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Data directory'
    )
    parser.add_argument(
        '--fnn-dir',
        type=str,
        default=None,
        help='FNN model directory'
    )
    parser.add_argument(
        '--out-dir',
        type=str,
        default=None,
        help='Output directory'
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default=None,
        help='Cache directory (if None, uses out-dir/cache)'
    )

    # Control
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cpu', 'cuda'],
        help='Device for training'
    )
    parser.add_argument(
        '--rebuild-cache',
        action='store_true',
        help='Force rebuild cache even if exists'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Print detailed progress'
    )

    return parser.parse_args()


def build_month_cache_streaming(
    data_dir,
    fnn_model,
    cache_dir,
    split='train',
    topk=200,
    lookback=60,
    lam=0.97,
    shrink=0.1,
    ridge=1e-5,
    device='cpu',
    verbose=True
):
    """
    Build month-level cache by streaming data from parquet files.

    Args:
        data_dir: Directory with parquet files
        fnn_model: Trained FNN model
        cache_dir: Directory to save cache files
        split: Which split to process ('train', 'val', 'test')
        topk: Portfolio size
        lookback: Lookback window for covariance
        lam: EWMA decay
        shrink: Shrinkage intensity
        ridge: Ridge regularization
        device: Torch device
        verbose: Print progress

    Returns:
        List of cache file paths
    """
    if verbose:
        print(f"\nBuilding {split} cache with streaming...")

    cache_path = Path(cache_dir) / split
    cache_path.mkdir(parents=True, exist_ok=True)

    # Create monthly iterator
    monthly_iterator = create_monthly_iterator(data_dir, split=split)

    # Storage for historical returns (for covariance)
    returns_history = []  # List of (yyyymm, permno_array, returns_array)

    cache_files = []
    month_count = 0

    for month_data in monthly_iterator:
        month_count += 1
        yyyymm = month_data['yyyymm']
        X = month_data['X']
        y = month_data['y']
        meta = month_data['meta']

        if verbose and month_count % 12 == 1:
            print(f"  Processing month {month_count}: {yyyymm} ({len(X)} stocks)")

        # Generate predictions
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            predictions = fnn_model(X_tensor).cpu().numpy().flatten()

        # Get permnos (or use indices if not available)
        permnos = meta['permno'].values if 'permno' in meta.columns else np.arange(len(X))

        # Add to history
        returns_history.append({
            'yyyymm': yyyymm,
            'permno': permnos,
            'returns': y
        })

        # Compute covariance from lookback window
        if len(returns_history) >= lookback:
            history_window = returns_history[-lookback:]
        else:
            history_window = returns_history

        # Simplified covariance: use diagonal (volatility only)
        # Full implementation would align stocks across months
        hist_returns = np.array([h['returns'] for h in history_window])
        sigma_vol = np.std(hist_returns, axis=0)
        sigma_vol = np.maximum(sigma_vol, 0.01)  # Floor at 1%

        # For now, simple diagonal covariance
        Sigma = np.diag(sigma_vol ** 2)
        Sigma_psd = make_psd(Sigma, ridge=ridge)

        # Factor representation (Cholesky)
        try:
            L = np.linalg.cholesky(Sigma_psd)
            Sigma_factor = L  # (N, N) lower triangular
        except np.linalg.LinAlgError:
            # Fallback to diagonal
            Sigma_factor = np.diag(sigma_vol)

        # Save month cache
        month_file = cache_path / f"{yyyymm}.npz"
        np.savez_compressed(
            month_file,
            yyyymm=yyyymm,
            permno=permnos,
            mu_pred=predictions,
            Sigma_factor=Sigma_factor,
            sigma_vol=sigma_vol,
            actual_ret=y
        )
        cache_files.append(str(month_file))

    if verbose:
        print(f"  Built cache for {month_count} months")

    return cache_files


def main():
    """Main E2E training pipeline with streaming."""
    args = parse_args()

    # Load configuration
    if args.config:
        config_dict = load_json(Path(args.config))
        config = PAOConfig(**config_dict)
    else:
        config = PAOConfig()

    # Override with command-line arguments
    overrides = {}
    if args.topk is not None:
        overrides['topk'] = args.topk
    if args.lambda_ is not None:
        overrides['lambda_grid'] = [args.lambda_]
    if args.kappa is not None:
        overrides['kappa_grid'] = [args.kappa]
    if args.loss_type is not None:
        overrides['loss_types'] = [args.loss_type]
    if args.data_dir is not None:
        overrides['data_dir'] = args.data_dir
    if args.fnn_dir is not None:
        overrides['fnn_dir'] = args.fnn_dir
    if args.out_dir is not None:
        overrides['out_dir'] = args.out_dir
    if args.device is not None:
        overrides['device'] = args.device

    if overrides:
        config = get_config_with_overrides(**overrides)

    # Set cache directory
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
    else:
        cache_dir = Path(config.out_dir) / 'cache'

    # Set random seeds
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    if args.verbose:
        print("=" * 78)
        print("PAO TRAINING - STREAMING DATA LOADING")
        print("=" * 78)
        print(f"\nConfiguration:")
        print(f"  Device: {config.device}")
        print(f"  Data dir: {config.data_dir}")
        print(f"  Cache dir: {cache_dir}")
        print(f"  TOPK: {config.topk}")
        print(f"  Lambda grid: {config.lambda_grid}")
        print(f"  Kappa grid: {config.kappa_grid}")
        print(f"  Loss types: {config.loss_types}")
        print()

    # ========================================
    # Step 1: Load FNN model
    # ========================================
    if args.verbose:
        print("Step 1: Loading FNN model...")

    fnn_model, feature_cols, fnn_config = load_fnn_from_dir(
        config.fnn_dir
    )
    fnn_model = fnn_model.to(config.device)
    fnn_model.eval()

    if args.verbose:
        print(f"  Model input dim: {fnn_config['input_dim']}")

    # ========================================
    # Step 2: Build month-level cache
    # ========================================
    train_cache_exists = (cache_dir / 'train').exists()
    val_cache_exists = (cache_dir / 'val').exists()
    test_cache_exists = (cache_dir / 'test').exists()

    if args.rebuild_cache or not (train_cache_exists and val_cache_exists):
        if args.verbose:
            print("\nStep 2: Building month-level cache...")

        # Build train cache
        train_files = build_month_cache_streaming(
            data_dir=config.data_dir,
            fnn_model=fnn_model,
            cache_dir=cache_dir,
            split='train',
            topk=config.topk,
            lookback=config.lookback,
            lam=config.lam,
            shrink=config.shrink,
            ridge=config.ridge,
            device=config.device,
            verbose=args.verbose
        )

        # Build val cache
        val_files = build_month_cache_streaming(
            data_dir=config.data_dir,
            fnn_model=fnn_model,
            cache_dir=cache_dir,
            split='val',
            topk=config.topk,
            lookback=config.lookback,
            lam=config.lam,
            shrink=config.shrink,
            ridge=config.ridge,
            device=config.device,
            verbose=args.verbose
        )

        # Build test cache
        test_files = build_month_cache_streaming(
            data_dir=config.data_dir,
            fnn_model=fnn_model,
            cache_dir=cache_dir,
            split='test',
            topk=config.topk,
            lookback=config.lookback,
            lam=config.lam,
            shrink=config.shrink,
            ridge=config.ridge,
            device=config.device,
            verbose=args.verbose
        )
    else:
        if args.verbose:
            print("\nStep 2: Using existing cache...")
            print(f"  Cache directory: {cache_dir}")

    # ========================================
    # Step 3: Create datasets from cache
    # ========================================
    if args.verbose:
        print("\nStep 3: Creating datasets from cache...")

    train_ds = MonthCacheDataset(cache_dir / 'train')
    val_ds = MonthCacheDataset(cache_dir / 'val')
    test_ds = MonthCacheDataset(cache_dir / 'test')

    if args.verbose:
        print(f"  Train months: {len(train_ds)}")
        print(f"  Val months: {len(val_ds)}")
        print(f"  Test months: {len(test_ds)}")

    # ========================================
    # Step 4: Train PAO models
    # ========================================
    if args.verbose:
        print("\nStep 4: Training PAO models...")

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    # Grid search over configurations
    for loss_type in config.loss_types:
        for lambda_ in config.lambda_grid:
            for kappa in config.kappa_grid:
                for omega_mode in config.omega_modes:
                    for mu_transform in config.mu_transform:
                        run_name = f"{loss_type}_l{lambda_}_k{kappa}_{omega_mode}_{mu_transform}"

                        if args.verbose:
                            print(f"\n  Training: {run_name}")

                        # Create run directory
                        run_dir = out_dir / 'runs' / run_name
                        run_dir.mkdir(parents=True, exist_ok=True)

                        # Create model
                        input_dim = train_ds[0]['mu_pred'].shape[0]
                        model = PAOPortfolioModel(
                            input_dim=input_dim,
                            hidden_dims=config.hidden_dims,
                            lambda_=lambda_,
                            kappa=kappa,
                            omega_mode=omega_mode,
                            mu_transform=mu_transform
                        ).to(config.device)

                        # Train
                        train_config = copy.deepcopy(config)
                        train_config.lambda_ = lambda_
                        train_config.kappa = kappa
                        train_config.omega_mode = omega_mode
                        train_config.mu_transform = mu_transform

                        history = train_one_run(
                            run_dir=run_dir,
                            train_ds=train_ds,
                            val_ds=val_ds,
                            model=model,
                            config=train_config,
                            loss_type=loss_type,
                            eval_fn=None
                        )

                        # Evaluate on test
                        if args.verbose:
                            print(f"    Evaluating on test set...")

                        # Load best model
                        best_ckpt = run_dir / 'best_model.pt'
                        if best_ckpt.exists():
                            model.load_state_dict(torch.load(best_ckpt, map_location=config.device))

                        test_results = eval_dataset(model, test_ds, config.device)

                        # Save results
                        result = {
                            'run_name': run_name,
                            'loss_type': loss_type,
                            'lambda_': lambda_,
                            'kappa': kappa,
                            'omega_mode': omega_mode,
                            'mu_transform': mu_transform,
                            'test_stats': test_results
                        }
                        all_results.append(result)

                        if args.verbose:
                            print(f"    Test mean return: {test_results['mean_excess_ret']:.4f}")
                            print(f"    Test Sharpe: {test_results['sharpe_a']:.4f}")

    # ========================================
    # Step 5: Save summary
    # ========================================
    if args.verbose:
        print("\nStep 5: Saving results...")

    save_json(all_results, out_dir / 'pao_streaming_results.json', convert_np=True)

    # Create summary dataframe
    summary_rows = []
    for res in all_results:
        row = {
            'run_name': res['run_name'],
            'loss_type': res['loss_type'],
            'lambda_': res['lambda_'],
            'kappa': res['kappa'],
            'test_mean_ret': res['test_stats']['mean_excess_ret'],
            'test_sharpe': res['test_stats']['sharpe_a']
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / 'pao_streaming_summary.csv', index=False)

    if args.verbose:
        print("\n" + "=" * 78)
        print("SUMMARY")
        print("=" * 78)
        print(summary_df.to_string(index=False))
        print(f"\n✓ PAO streaming training complete!")
        print(f"  Output directory: {out_dir}")
        print("=" * 78)


if __name__ == '__main__':
    main()
