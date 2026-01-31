# Data

## Overview

Two-stage pipeline: (1) download and merge raw data, (2) process into train/validation/test splits.

---

## Stage 1: Data Acquisition

### Prerequisites
- WRDS account with CRSP access

### Run Data Download

```bash
cd scripts
Rscript 00_download_OAP.R
```

**What it does:**
1. Downloads OpenSourceAssetPricing firm characteristics from Google Drive
2. Pulls CRSP monthly returns from WRDS (with delisting adjustments)
3. Merges and creates `ret_tplus1` (t+1 return target)

**Move output:**
```bash
mkdir -p data/raw
mv temp/signed_predictors_all_wide.parquet data/raw/signed_predictors_wide.parquet
```

**Download macro data:** Save [Goyal-Welch Dataset](https://docs.google.com/spreadsheets/d/1OIZg6htTK60wtnCVXvxAujvG1aKEOVYv/edit?gid=1660564386#gid=1660564386) as `data/raw/PredictorData.csv`

---

## Stage 2: Data Processing

### Prerequisites
- Python 3.8+: `pip install -r requirements.txt`
- Files in `data/raw/`: `signed_predictors_wide.parquet`, `PredictorData.csv`

### Run Processing

```bash
python scripts/01_prepare_data.py
```

**Pipeline:**
1. Load firm characteristics and macro predictors
2. Clean data (drop >70% missing, clip outliers)
3. Create firm × macro interactions, rank-normalize to [-1, 1]
4. Split: Train (≤2005), Validation (2006-2015), Test (≥2016)
5. Save to `data/processed/ready_data/`

**Output:** X/y/meta files for train/val/test + feature metadata + clean panels
---

## Data Sources

### Firm Characteristics
- **OpenSourceAssetPricing:** [Google Drive Folder](https://drive.google.com/drive/folders/1qQDuTsnyvWfEJR6nPBQZ8xxlq6bkLG_y)
- **CRSP Returns Code:** [GitHub Repository](https://github.com/OpenSourceAP/CrossSectionDemos/blob/main/dl_signals_add_crsp.R)
- **Our Modified Version:** [00_download_OAP.R](scripts/00_download_OAP.R)

**Citation:**
```bibtex
@article{ChenZimmermann2022,
  title={Open Source Cross-Sectional Asset Pricing},
  author={Chen, Andrew Y. and Tom Zimmermann},
  journal={Critical Finance Review},
  year={2022},
  volume={27},
  number={2},
  pages={207--264}
}
```

### Macroeconomic Data
- **Goyal-Welch Dataset:** [Google Sheets Link](https://docs.google.com/spreadsheets/d/1OIZg6htTK60wtnCVXvxAujvG1aKEOVYv/edit?gid=1660564386#gid=1660564386)

