# Data and Models

Due to file size limitations, the following directories are **not included** in this GitHub repository:

## Data Files
- `data/raw/` - Raw input data
- `data/processed/` - Processed datasets

## Pre-trained Models
- `models/fnn_v1/` - Shared FNN model used by PTO and E2E

## Results
- `outputs/pto/results/` - PTO backtest results (5 CSV files)
- `outputs/e2e/` - E2E trained models and summaries (172 models + 7 summary CSVs)

---

## Download from Google Drive

**Google Drive Link:** [TO BE ADDED]

The Google Drive folder contains:
```
POE_Research_Data_and_Models/
├── data/
│   ├── raw/
│   │   ├── signed_predictors_wide.parquet
│   │   └── PredictorData.csv
│   └── processed/
│       └── ready_data/
│           ├── train.parquet
│           ├── validation.parquet
│           └── test.parquet
│
├── models/
│   └── fnn_v1/
│       ├── model.pth
│       └── config.json
│
└── outputs/
    ├── pto/
    │   └── results/
    │       ├── results_summary_excess.csv
    │       ├── returns_total_all_specs.csv
    │       ├── returns_excess_all_specs.csv
    │       ├── wealth_paths_total_all_specs.csv
    │       └── wealth_paths_excess_all_specs.csv
    │
    └── e2e/
        ├── master_summary.csv
        ├── standard_training/
        │   ├── hyperparameter_grid/    (45 models)
        │   └── universe_sweep/          (54 models)
        └── crisis_scenarios/
            ├── summer_child_no_crisis/  (56 models)
            └── winter_wolf_with_crisis/ (17 models)
```

---

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone [YOUR_REPO_URL]
   cd POE_Research
   ```

2. **Download data and models from Google Drive** (link above)

3. **Extract to project directory:**
   ```bash
   # Extract so the directory structure matches:
   POE_Research/
   ├── data/
   ├── models/
   ├── outputs/
   └── ... (other files from git)
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

5. **Verify setup:**
   ```bash
   # Check data is present
   ls data/raw/
   ls data/processed/ready_data/

   # Check models are present
   ls models/fnn_v1/

   # Check results are present
   ls outputs/pto/results/
   ls outputs/e2e/
   ```

---

## File Sizes (Approximate)

- **data/**: ~2-5 GB (raw parquet files)
- **models/**: ~10-50 MB (FNN checkpoint)
- **outputs/**: ~500 MB - 2 GB (172 E2E models + results)

**Total:** ~3-7 GB (too large for GitHub)

---

## Reproducing Results

Once data and models are in place, you can:

### Run PTO Backtesting
```bash
python scripts/02_run_pto_streaming.py \
    --data-dir data/processed/ready_data \
    --fnn-dir models/fnn_v1 \
    --topk 200 \
    --lambda 5.0 \
    --kappa-values 0.0 0.1 0.5 1.0
```

### Run E2E Training
```bash
python scripts/03_run_e2e_streaming.py \
    --data-dir data/processed/ready_data \
    --fnn-dir models/fnn_v1 \
    --topk 50 \
    --loss-type utility \
    --lambda 10.0 \
    --kappa 0.5
```

### Generate Summary CSVs
```bash
python scripts/generate_e2e_summaries.py
```
