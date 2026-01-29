"""
Fix PTO results by aggregating duplicate months.

The optimized script had a bug where it processed each month multiple times.
This script aggregates the duplicates to get the correct statistics.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

def aggregate_monthly_results(results_dict):
    """Aggregate duplicate month entries by taking the mean."""
    aggregated = {}

    for strategy_name, month_list in results_dict.items():
        month_dict = defaultdict(list)

        for entry in month_list:
            month = entry['yyyymm']
            month_dict[month].append(entry['return'])

        # Create aggregated list with unique months
        aggregated[strategy_name] = [
            {'yyyymm': month, 'return': np.mean(returns)}
            for month, returns in sorted(month_dict.items())
        ]

    return aggregated

def compute_performance_stats(returns):
    """Compute annual performance statistics."""
    returns = np.array(returns)
    n_months = len(returns)

    mean_m = np.mean(returns)
    vol_m = np.std(returns, ddof=1)

    mean_a = mean_m * 12
    vol_a = vol_m * np.sqrt(12)
    sharpe_a = mean_a / vol_a if vol_a > 0 else 0

    cum_simple = np.prod(1 + returns) - 1

    # Drawdown
    cum_rets = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cum_rets)
    drawdown = (cum_rets - running_max) / running_max
    max_drawdown = np.min(drawdown)

    hit_rate = np.mean(returns > 0) * 100

    return {
        'n_months': n_months,
        'mean_m': mean_m,
        'vol_m': vol_m,
        'mean_a': mean_a,
        'vol_a': vol_a,
        'sharpe_a': sharpe_a,
        'cum_simple': cum_simple,
        'max_drawdown': max_drawdown,
        'hit_rate': hit_rate,
        'worst_month': np.min(returns),
        'best_month': np.max(returns)
    }

def main():
    # Load results
    results_path = Path('POE_Research/outputs/pto/results/monthly_results.json')

    print("Loading results...")
    with open(results_path, 'r') as f:
        results = json.load(f)

    print(f"Original data: {len(results['equal_weight'])} entries")

    # Aggregate duplicates
    print("\nAggregating duplicate months...")
    aggregated = aggregate_monthly_results(results)

    print(f"Aggregated data: {len(aggregated['equal_weight'])} unique months")

    # Compute corrected statistics
    print("\n" + "=" * 70)
    print("CORRECTED BACKTEST RESULTS")
    print("=" * 70)

    summary_data = []
    for strategy_name in aggregated.keys():
        returns = [entry['return'] for entry in aggregated[strategy_name]]
        stats = compute_performance_stats(returns)
        stats['strategy'] = strategy_name
        summary_data.append(stats)

        print(f"\n{strategy_name}:")
        print(f"  Mean (annual): {stats['mean_a']:.4f}")
        print(f"  Vol (annual): {stats['vol_a']:.4f}")
        print(f"  Sharpe: {stats['sharpe_a']:.4f}")
        print(f"  Cumulative: {stats['cum_simple']:.2%}")
        print(f"  Max Drawdown: {stats['max_drawdown']:.2%}")
        print(f"  Hit Rate: {stats['hit_rate']:.1f}%")
        print(f"  Months: {stats['n_months']}")

    # Save corrected results
    output_dir = Path('POE_Research/outputs/pto/results')

    print(f"\nSaving corrected results to {output_dir}...")

    with open(output_dir / 'monthly_results_CORRECTED.json', 'w') as f:
        json.dump(aggregated, f, indent=2)

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / 'summary_stats_CORRECTED.csv', index=False)

    print("✅ Done!")
    print("\nCorrected files created:")
    print("  - monthly_results_CORRECTED.json")
    print("  - summary_stats_CORRECTED.csv")

if __name__ == '__main__':
    main()
