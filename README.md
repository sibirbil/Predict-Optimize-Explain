## Overview

This repository implements **Predict-Then-Optimize (PTO)**, **Predict-And-Optimize (PAO)** portfolio optimization, and a **scenario-based explanation framework** for probing decision pipelines.


## Data Acquisition

```bash
# Download CRSP firm data (requires WRDS credentials)
Rscript scripts/00_download_OAP.R

# Download Goyal-Welch macro data manually:
# https://docs.google.com/spreadsheets/d/1OIZg6htTK60wtnCVXvxAujvG1aKEOVYv
# Save as: data/raw/PredictorData.csv
```

## Universe 500 Dataset

Builds the standardized-macro Universe500 dataset used by all PTO and PAO models.

```bash
# Prepare base data
python scripts/01_prepare_data.py

# Build Universe500 with standardized macro interactions
python scripts/data/build_universe_500_stdz_macro.py
```

## FNN Backbone

Train and tune the FNN.

```bash
python scripts/training/train_fnn.py
python scripts/training/tune_fnn.py
```

## PTO Pipeline

```bash
python scripts/training/run_pto_v1.py
```

## PAO Training

```bash
# Standard E2E grid search
python scripts/training/run_e2e_v1.py

# Contrastive training (SummerChild / WinterWolf)
python scripts/training/run_e2e_contrastive.py
```

Configs for each run live in `src/configs/e2e/`.

## Scenario Experiment

Runs MALA chains from a chosen anchor month to find macro states where SummerChild and WinterWolf produce maximally distinct realized returns.

```bash
python scripts/scenario4.py
```

Anchor month, regularizer mode (`var1` / `l2`), and chain hyperparameters are set at the top of `scripts/scenario4.py`.

## Reporting

```bash
python scripts/reporting/report_e2e_benchmark_comparison.py
python scripts/reporting/report_pto_benchmarks.py
python scripts/reporting/report_contrastive_event_study.py
# ... other reports in scripts/reporting/
```
