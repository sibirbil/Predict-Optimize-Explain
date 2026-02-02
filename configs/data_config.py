"""
Data Processing Configuration.

Configuration for data preprocessing pipeline including:
- Random seeds
- Date splits (train/val/test)
- Missing data thresholds
- Target processing parameters
- Default paths
"""
from pathlib import Path

# ============================================================
# Random Seed
# ============================================================
RANDOM_STATE = 42

# ============================================================
# Date Splits
# ============================================================
# Training period: up to and including December 2005
TRAIN_END = 200512

# Validation period: January 2006 to December 2015
VAL_END = 201512

# Test period: January 2016 onwards (>= 201601)

# ============================================================
# Data Processing Parameters
# ============================================================

# Missing data threshold
# Drop columns with more than this fraction of missing values
MISSING_THRESH = 0.70

# Firm data processing
# Date range for firm data (YYYYMM format)
FIRM_START_DATE = 198001  # January 1980
FIRM_END_DATE = 202412    # December 2024

# Target processing
# Physical lower limit for returns (any return < -100% is impossible)
TARGET_LOWER_LIMIT = -1.0

# Upper quantile for clipping extreme positive returns
TARGET_UPPER_QUANTILE = 0.995  # 99.5th percentile

# Target column name
TARGET_COL = 'ret_tplus1'

# Percentage detection threshold
# If mean absolute return > 0.2, data is assumed to be in percentage scale
PERCENTAGE_THRESHOLD = 0.2

# ============================================================
# Default Paths (Relative)
# ============================================================

# Base directory for the project
BASE_DIR = Path(".")

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed" / "ready_data"

# Model directories
MODELS_DIR = BASE_DIR / "models"
FNN_MODEL_DIR = MODELS_DIR / "fnn_v1"

# Output directories
OUTPUTS_DIR = BASE_DIR / "outputs"
PTO_OUTPUTS_DIR = OUTPUTS_DIR / "pto"
PAO_OUTPUTS_DIR = OUTPUTS_DIR / "pao"

# ============================================================
# Default Input File Names
# ============================================================

# Firm characteristics (wide format with signed predictors)
FIRM_DATA_FILE = "signed_predictors_wide.parquet"

# Macro predictors (Goyal-Welch)
MACRO_DATA_FILE = "PredictorData.csv"

# ============================================================
# Interaction Terms
# ============================================================

# Whether to create firm × macro interaction terms
CREATE_INTERACTIONS = True

# ============================================================
# Rank Normalization
# ============================================================

# Whether to rank-normalize features
RANK_NORMALIZE = True

# Range for rank normalization
RANK_RANGE = (-1, 1)

# ============================================================
# Output Settings
# ============================================================

# Output file format
OUTPUT_FORMAT = "parquet"

# Files to save in processed data directory:
# - X_train.parquet, y_train.parquet, meta_train.parquet
# - X_val.parquet, y_val.parquet, meta_val.parquet
# - X_test.parquet, y_test.parquet, meta_test.parquet
# - feature_columns.json (list of feature names)
# - dataset_info.json (metadata about splits)

# ============================================================
# Validation Settings
# ============================================================

# Whether to perform strict validation checks
STRICT_VALIDATION = True

# Whether to display intermediate outputs
VERBOSE = True
