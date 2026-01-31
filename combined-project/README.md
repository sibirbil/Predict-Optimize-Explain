# Predict-Optimize-Explain

## Overview

This repository implements traditional **Predict-Then-Optimize (PTO)** with **end-to-end differentiable optimization (E2E)**, and providing **scenario generation**.

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

### Step 4: Train E2E Models

```bash
# Default configuration
python scripts/03_run_e2e_streaming.py

# Specify loss function and risk aversion
python scripts/03_run_e2e_streaming.py \
    --loss-type utility \
    --lambda 10.0 \
    --kappa 1.0 \
    --topk 60
```

**Outputs:**
- `outputs/e2e_results_*.json`: Training logs
- `models/e2e_*/`: Checkpoints for each configuration

### Step 5: Scenario Experiments

```bash
# Experiment 1: Benchmark-based
python scripts/04_script1.py

# Experiment 2: Utility-based
python scripts/05_script2.py

# Experiment 3: Model contrast (summer child vs. winter wolf)
python scripts/06_script3.py
```
