# Portfolio Optimization with End-to-End Learning

## Overview

This project implements a complete pipeline for portfolio optimization with machine learning:
1. **Data Processing**: Firm characteristics, macro predictors, and forward returns
2. **Predict-Then-Optimize (PTO)**: Traditional two-stage approach with FNN prediction + Robust MVO
3. **End-to-End Learning (E2E)**: Differentiable portfolio optimization with joint training

**Key Features:**
- Memory-efficient streaming architecture (handles millions of observations with 3-4GB RAM)
- Robust mean-variance optimization with uncertainty quantification
- Crisis scenario analysis (training with/without crisis data)
- Comprehensive backtesting with 107 months of out-of-sample data

---

## Project Structure

```
POE_Research/
├── configs/                    # Configuration files
│   ├── data_config.py          # Data processing parameters
│   ├── pto_config.py           # PTO backtest parameters
│   └── e2e_config.py           # E2E training parameters
│
├── scripts/                    # Pipeline scripts
│   ├── 01_prepare_data.py      # Data preparation
│   ├── 02_run_pto_streaming.py # PTO backtesting (memory-efficient)
│   ├── 03_run_e2e_streaming.py # E2E training (memory-efficient)
│   ├── run_e2e_pipeline.py     # Complete E2E pipeline
│   └── generate_e2e_summaries.py # Summary CSV generation
│
├── src/                        # Source code
│   ├── cache/                  # E2E caching system
│   ├── data/                   # Data processing & loading
│   ├── models/                 # Neural network models
│   ├── optimization/           # Portfolio optimization
│   ├── portfolio/              # Backtesting & metrics
│   ├── training/               # E2E training loop
│   └── utils/                  # Utility functions
│
├── data/                       # Data directories
│   ├── raw/                    # Raw input data
│   └── processed/              # Processed datasets
│
├── models/                     # Pre-trained models
│   └── fnn_v1/                 # Shared FNN (used by PTO & E2E)
│
├── outputs/                    # Results
│   ├── pto/results/            # PTO backtest results (107 months)
│   └── e2e/                    # E2E training results
│       ├── standard_training/  # Hyperparameter grid & universe sweep
│       └── crisis_scenarios/   # Summer child vs Winter wolf
│
├── requirements.txt            # Python dependencies
├── setup.py                    # Package installation
├── README.md                   # This file
└── STREAMING_DATA_GUIDE.md     # Memory-efficient architecture details
```

---

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.10+
- CVXPY with CvxpyLayers
- PyArrow (for streaming)

### Setup

```bash
# Navigate to project directory
cd POE_Research

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

---

## Quick Start

### 1. Data Preparation

```bash
python scripts/01_prepare_data.py \
    --firm-data data/raw/signed_predictors_wide.parquet \
    --macro-data data/raw/PredictorData.csv \
    --output-dir data/processed/ready_data \
    --train-end 200512 \
    --val-end 201512
```

### 2. PTO Backtesting (Memory-Efficient)

```bash
python scripts/02_run_pto_streaming.py \
    --data-dir data/processed/ready_data \
    --fnn-dir models/fnn_v1 \
    --topk 200 \
    --lambda 5.0 \
    --kappa-values 0.0 0.1 0.5 1.0 \
    --verbose
```

**Outputs:**
- `outputs/pto/results/results_summary_excess.csv` - Performance metrics (11 strategies)
- `outputs/pto/results/returns_total_all_specs.csv` - Monthly returns (107 months)
- `outputs/pto/results/wealth_paths_total_all_specs.csv` - Cumulative wealth evolution

### 3. E2E Training (Memory-Efficient)

```bash
python scripts/03_run_e2e_streaming.py \
    --data-dir data/processed/ready_data \
    --fnn-dir models/fnn_v1 \
    --topk 50 \
    --loss-type utility \
    --lambda 10.0 \
    --kappa 0.5 \
    --verbose
```

---

## Key Results

### PTO Performance (107 Months: 2016-01 to 2024-11)

| Strategy | Sharpe Ratio | Annual Return | Annual Vol | Max Drawdown |
|----------|--------------|---------------|------------|--------------|
| **Robust MVO (κ=0.1)** | **1.197** | 18.35% | 15.33% | -26.03% |
| **Robust MVO (κ=0.5)** | **1.158** | 17.63% | 15.23% | -28.52% |
| **Robust MVO (κ=1.0)** | **1.081** | 17.37% | 16.06% | -29.81% |
| Standard MVO (κ=0.0) | 0.967 | 16.91% | 17.49% | -32.28% |
| Equal-Weight Benchmark | 0.801 | 17.49% | 21.85% | -34.60% |

**Key Findings:**
- **50% Sharpe improvement** over benchmark with robust optimization
- **Optimal robustness:** κ=0.1 achieves best risk-adjusted returns
- **Reduced drawdowns:** 8 percentage points vs equal-weight benchmark

### E2E Results (172 Trained Models)

**Categories:**
1. **Hyperparameter Grid** - 45 configs (loss × λ × κ × Ω)
2. **Universe Sweep** - 54 configs (portfolio sizes: 21-94 assets)
3. **Summer Child (Crisis-Naive)** - 56 configs (trained without crisis data)
4. **Winter Wolf (Crisis-Aware)** - 17 configs (trained with 2008 crisis)

**Crisis Scenario Findings:**
- Winter Wolf (crisis-aware) outperforms Summer Child (crisis-naive) by **9.5%** Sharpe
- Crisis training improves robustness without sacrificing normal-period performance

---

## Parameters

### Portfolio Construction

| Parameter | Default | Description |
|-----------|---------|-------------|
| `topk` | 200 | Number of assets in portfolio |
| `preselect_factor` | 3 | Pre-selection multiplier (universe = topk × 3) |
| `lookback` | 60 | Months for covariance estimation |

### EWMA Covariance

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lam` | 0.94 | Decay parameter (0.94 ≈ 16-month half-life) |
| `shrink` | 0.10 | Shrinkage toward diagonal matrix |
| `ridge` | 1e-6 | Ridge regularization for PSD |

### Optimization

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lambda_` | 5.0 | Risk aversion coefficient (λ) |
| `kappa` | 0.0 | Robustness parameter (0 = standard MVO) |
| `omega_mode` | diagSigma | Uncertainty structure (diagSigma or identity) |

### E2E Training

| Parameter | Default | Description |
|-----------|---------|-------------|
| `loss_type` | utility | Loss function (return, utility, sharpe) |
| `epochs` | 50 | Maximum training epochs |
| `patience` | 10 | Early stopping patience |
| `lr` | 5e-5 | Learning rate |

---

## Methodology

### Predict-Then-Optimize (PTO)

1. **Prediction:** Train FNN to predict next-month returns
   ```
   r̂_{t+1} = f_θ(X_t)
   ```

2. **Covariance Estimation:** EWMA with shrinkage
   ```
   Σ_t = λΣ_{t-1} + (1-λ)r_t r_t'
   ```

3. **Portfolio Optimization:** Standard MVO
   ```
   max_w  μ'w - (λ/2)w'Σw
   s.t.   Σw = 1, w ≥ 0
   ```

### Robust Mean-Variance Optimization

Adds uncertainty penalty to handle estimation risk:

```
max_w  μ'w - (λ/2)w'Σw - κ√(w'Ωw)
s.t.   Σw = 1, w ≥ 0
```

Where:
- **λ** controls risk aversion
- **κ** controls robustness (higher = more conservative)
- **Ω** defines uncertainty structure

### End-to-End Learning (E2E)

Jointly optimizes prediction and allocation:

1. **Score Network:** Learns asset scores
   ```
   s_t = g_φ(X_t)
   ```

2. **Differentiable Optimization:** Uses CvxpyLayers
   ```
   w_t = MVO(s_t, Σ_t)  [differentiable!]
   ```

3. **Backpropagation:** Gradients flow through optimization to prediction

**Loss Functions:**
- **Return:** `L = -r_p` (maximize return)
- **Utility:** `L = -(r_p - λ/2·w'Σw)` (risk-adjusted)
- **Sharpe:** `L = -mean(r_p)/std(r_p)` (Sharpe ratio)

---

## Memory Efficiency

**Dataset:** 2.2M observations × 1,400 features
**RAM Usage:** Only 3-4GB (fits on laptop!)

### Key Techniques

1. **Chunked Processing** - Process 100K rows at a time during data creation
2. **PyArrow Streaming** - Incremental batch iteration
3. **Month-Level Caching** - Store E2E inputs separately by month
4. **FastReturnsLookup** - Pre-computed pivot tables (55.8× speedup)

### Memory Comparison

| Approach | Memory | Notes |
|----------|--------|-------|
| Naive (load all) | 120GB+ | Won't fit in memory |
| This implementation | 3-4GB | Streaming architecture |

See [STREAMING_DATA_GUIDE.md](STREAMING_DATA_GUIDE.md) for implementation details.

---

## Results Files

### PTO Results (`outputs/pto/results/`)

- **results_summary_excess.csv** - Aggregate metrics for 11 strategies
- **returns_total_all_specs.csv** - 107 months of returns
- **wealth_paths_total_all_specs.csv** - Cumulative wealth evolution
- **PTO_RESULTS_SUMMARY.md** - Detailed documentation

### E2E Results (`outputs/e2e/`)

- **master_summary.csv** - All 172 models combined
- **standard_training/hyperparameter_grid/summary.csv** - 45 configs
- **standard_training/universe_sweep/summary.csv** - 54 configs
- **crisis_scenarios/summer_child_no_crisis/summary_*.csv** - 56 configs
- **crisis_scenarios/winter_wolf_with_crisis/summary_*.csv** - 17 configs

**Generate Summaries:**
```bash
python scripts/generate_e2e_summaries.py
```

---

## Documentation

- **README.md** (this file) - Project overview and usage
- **STREAMING_DATA_GUIDE.md** - Memory-efficient architecture details
- **outputs/pto/PTO_RESULTS_SUMMARY.md** - PTO results documentation
- **E2E_REORGANIZATION_COMPLETE.md** - E2E results structure
- **E2E_SUMMARY_CSVS.md** - Summary CSV documentation

---

## Citation

If using this code or results in research:

```
POE Research Project - Portfolio Optimization with End-to-End Learning
PTO Backtesting: 2016-01 to 2024-11 (107 months)
E2E Models: 172 trained configurations across multiple scenarios
POE_Research/
2026
```

---

## License

This project is provided for research purposes.

---

## Contact

For questions or issues, please open an issue in the repository.
