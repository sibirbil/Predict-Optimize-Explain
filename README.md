# Predict-Optimize-Explain

## Overview

This repository implements traditional **Predict-Then-Optimize (PTO)** with **predict-and-optimize (PAO)** decision pipelines. We also provide an approach based on **sample generation** for explaining different decision pipelines.

## Usage

### Step 1: Data Acquisition

```bash
# Download CRSP data (requires WRDS credentials)
cd scripts
Rscript 00_download_OAP.R

# Move to data directory
mkdir -p ../data/raw
mv temp/signed_predictors_all_wide.parquet ../data/raw/signed_predictors_wide.parquet

# Download Goyal-Welch macro data manually:
# https://docs.google.com/spreadsheets/d/1OIZg6htTK60wtnCVXvxAujvG1aKEOVYv
# Save as: data/raw/PredictorData.csv
```

### Step 2: Data Preparation

```bash
python scripts/01_prepare_data.py
```

**Output:** `data/processed/ready_data/` containing:
- `X_train.parquet`, `y_train.parquet`, `metadata_train.parquet`
- `X_val.parquet`, `y_val.parquet`, `metadata_val.parquet`
- `X_test.parquet`, `y_test.parquet`, `metadata_test.parquet`
- `feature_metadata.parquet` (column names)

### Step 3: Run PTO Pipeline

```bash
# Default configuration
python scripts/02_run_pto_streaming.py

# Custom parameters
python scripts/02_run_pto_streaming.py \
    --topk 100 \
    --lambda 10.0 \
    --kappa 1.0 \
    --omega-mode diagSigma
```

**Outputs:**
- `outputs/pto_results_*.json`: Performance metrics
- `models/fnn_v1/`: Trained prediction model

### Step 4: Train PAO Models

```bash
# Default configuration
python scripts/03_run_pao_streaming.py

# Specify loss function and risk aversion
python scripts/03_run_pao_streaming.py \
    --loss-type utility \
    --lambda 10.0 \
    --kappa 1.0 \
    --topk 60
```

**Outputs:**
- `outputs/pao_results_*.json`: Training logs
- `models/pao_*/`: Checkpoints for each configuration

### Step 5: Scenario Experiments

```bash
# Experiment 1: Counterfactual analysis with Sharpe-based portfolios
python scripts/04_scenario1.py

# Experiment 2: Counterfactual analysis with utility-based portfolios
python scripts/05_scenario2.py

# Experiment 3: Model contrast (summer child vs. winter wolf)
python scripts/06_scenario3.py
```
