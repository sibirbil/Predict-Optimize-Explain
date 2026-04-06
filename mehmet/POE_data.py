# =========================================
# PREPROCESSING NOTEBOOK
# =========================================


import os
import math
import gc
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    HAVE_STATSMODELS = True
except Exception:
    HAVE_STATSMODELS = False

pd.set_option("display.max_columns", 200)
pd.set_option("display.max_rows", 50)
pd.set_option("display.width", 160)

RANDOM_STATE = 42
print("Imports complete.")

# ---------------------------------------------------------------------
# Load raw firm and macro data
# ---------------------------------------------------------------------
firm_data_path = Path("/content/drive/MyDrive/POE/signed_predictors_wide.parquet")
firm_df = pd.read_parquet(firm_data_path)

macro_data_path = Path('/content/drive/MyDrive/POE/PredictorData.csv')
macro_df = pd.read_csv(macro_data_path)

# ---------------------------------------------------------------------
# FirmOnlyProcessor
# ---------------------------------------------------------------------
class FirmOnlyProcessor:
    """
    Robust processor for firm-level data ONLY.
    - Diagnosis (Missingness, Outliers)
    - Dynamic Dropping of sparse columns
    - Winsorization (1%/99%)
    - Rank Normalization
    """
    def __init__(self, firm_path, start_date=198001, end_date=202412, missing_thresh=0.70):
        self.firm_path = firm_path
        self.start_date = start_date
        self.end_date = end_date
        self.missing_thresh = missing_thresh

    def load_data(self):
        print(f"Loading Firm Data from {self.firm_path}...")
        df = pd.read_parquet(self.firm_path)

        df = df[(df['yyyymm'] >= self.start_date) & (df['yyyymm'] <= self.end_date)].copy()

        self.meta_cols = ['permno', 'yyyymm', 'ret_tplus1', 'Price', 'Size']
        self.meta_cols = [c for c in self.meta_cols if c in df.columns]
        self.char_cols = [c for c in df.columns if c not in self.meta_cols]

        print(f"Initial Shape: {df.shape}")
        print(f"Initial Characteristics: {len(self.char_cols)}")
        return df

    def process(self, df):
        # 1) Drop very sparse characteristics
        missing_rates = df[self.char_cols].isnull().mean()
        drop_cols = missing_rates[missing_rates > self.missing_thresh].index.tolist()

        if drop_cols:
            print(f"\n[Dropping] Eliminating {len(drop_cols)} columns with >{self.missing_thresh:.0%} missing data.")
            df = df.drop(columns=drop_cols)
            self.char_cols = [c for c in self.char_cols if c not in drop_cols]

        print(f"Remaining Characteristics: {len(self.char_cols)}")

        print("\n[Processing] Winsorization (1/99%), median imputation, rank-normalization...")

        def clean_group(group):
            filled = group.fillna(group.median())
            filled = filled.fillna(0)

            lower = filled.quantile(0.01)
            upper = filled.quantile(0.99)
            clipped = filled.clip(lower, upper)

            ranks = clipped.rank(method='dense')
            min_r = 1
            max_r = ranks.max()

            if max_r > 1:
                return ((ranks - min_r) / (max_r - min_r)) * 2 - 1
            else:
                return filled * 0

        df[self.char_cols] = df.groupby('yyyymm')[self.char_cols].transform(clean_group)
        df[self.char_cols] = df[self.char_cols].fillna(0)

        return df

# --- firm pipeline ---
firm_path = "/content/drive/MyDrive/POE/signed_predictors_wide.parquet"
processor = FirmOnlyProcessor(firm_path)

df_firm = processor.load_data()
df_firm_clean = processor.process(df_firm)

print("\nFirm Data Processing Complete.")
print("Final Shape:", df_firm_clean.shape)

# ---------------------------------------------------------------------
# TargetProcessor
# ---------------------------------------------------------------------
class TargetProcessor:
    """
    Handles inspection and processing of the Raw Return (ret_tplus1).
    Automatically handles percentage vs decimal scaling issues.
    """
    def __init__(self, target_col='ret_tplus1', lower_limit=-1.0, upper_quantile=0.995):
        self.target_col = target_col
        self.physical_limit = lower_limit
        self.upper_q = upper_quantile

    def convert_to_decimal(self, df):
        target = df[self.target_col]
        mean_abs = target.abs().mean()
        if mean_abs > 0.2:
            print(f">> DETECTED PERCENTAGE DATA (Mean Abs: {mean_abs:.2f}). Dividing by 100.")
            df[self.target_col] = df[self.target_col] / 100.0
        else:
            print(f">> DETECTED DECIMAL DATA (Mean Abs: {mean_abs:.4f}). Keeping as is.")
        return df

    def process(self, df):
        df = self.convert_to_decimal(df)

        print(f"\n[Processing Target] Cleaning {self.target_col}...")

        lower_mask = df[self.target_col] < self.physical_limit
        if lower_mask.sum() > 0:
            print(f"   Clipping {lower_mask.sum()} impossible negative returns to {self.physical_limit}")
            df.loc[lower_mask, self.target_col] = self.physical_limit

        upper_limit = df[self.target_col].quantile(self.upper_q)
        print(f"   Clipping upside outliers > {upper_limit:.4f} (99.5\% quantile)")
        df[self.target_col] = df[self.target_col].clip(lower=None, upper=upper_limit)

        return df

target_proc = TargetProcessor(target_col='ret_tplus1')
df_firm_final = target_proc.process(df_firm_clean)

print("\nFinal Data Logic Check:")
desc = df_firm_final[['ret_tplus1']].describe().T
print(desc)

# ---------------------------------------------------------------------
# MacroDataProcessor
# ---------------------------------------------------------------------
class MacroDataProcessor:
    """
    Constructs the 8 canonical Goyal–Welch macro predictors.
    """
    def __init__(self, file_path, start_date=198001, end_date=202412):
        self.file_path = file_path
        self.start_date = start_date
        self.end_date = end_date
        self.predictors = ['dp', 'ep', 'bm', 'ntis', 'tbl', 'tms', 'dfy', 'svar']

    def _clean_data(self, df):
        numeric_cols = ['Index', 'D12', 'E12', 'b/m', 'tbl', 'AAA', 'BAA', 'lty', 'ntis', 'Rfree', 'svar']
        for col in numeric_cols:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(',', '', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    def load_and_process(self):
        df = pd.read_csv(self.file_path)
        df = self._clean_data(df)
        df = df[(df['yyyymm'] >= self.start_date) & (df['yyyymm'] <= self.end_date)].reset_index(drop=True)

        df['dp'] = np.log(df['D12']) - np.log(df['Index'])
        df['ep'] = np.log(df['E12']) - np.log(df['Index'])
        df.rename(columns={'b/m': 'bm'}, inplace=True)
        df['tms'] = df['lty'] - df['tbl']
        df['dfy'] = df['BAA'] - df['AAA']

        cols = ['yyyymm', 'Rfree'] + self.predictors
        if 'infl' in df.columns:
            cols.append('infl')

        self.df = df[cols].copy()
        return self.df

    def get_final_data(self):
        return self.df

macro_path = "/content/drive/MyDrive/POE/PredictorData.csv"
macro_proc = MacroDataProcessor(macro_path, start_date=198001, end_date=202412)
macro_proc.load_and_process()
macro_final = macro_proc.get_final_data()

print(f"\nFinal Macro Shape: {macro_final.shape}")

# ---------------------------------------------------------------------
# HighRAMDatasetBuilder 
# ---------------------------------------------------------------------
class HighRAMDatasetBuilder:
    def __init__(self, firm_df, macro_df):
        self.firm_df = firm_df.copy()
        self.macro_df = macro_df.copy()

    def merge_and_calculate_target(self):
        print("--- Step 1: Merging Firm & Macro Data ---")
        self.full_df = pd.merge(self.firm_df, self.macro_df, on='yyyymm', how='inner')

        print("Calculating Excess Returns (ret_tplus1 - Rfree)...")
        self.full_df['excess_ret'] = self.full_df['ret_tplus1'] - self.full_df['Rfree']
        self.full_df = self.full_df.dropna(subset=['excess_ret']).reset_index(drop=True)

        print(f"Total Rows: {len(self.full_df):,}")
        return self.full_df

    def create_interactions(self):
        print("\n--- Step 2: Generating Interaction Terms (In-Memory) ---")
        meta_cols = ['permno', 'yyyymm', 'ret_tplus1', 'excess_ret', 'Rfree', 'Price', 'Size']
        macro_predictors = [c for c in self.macro_df.columns if c not in ['yyyymm', 'Rfree']]

        firm_cols = [c for c in self.full_df.columns if c not in meta_cols and c not in self.macro_df.columns]

        print(f"Base Firm Features: {len(firm_cols)}")
        print(f"Macro Predictors: {len(macro_predictors)} ({macro_predictors})")

        X_parts = [self.full_df[firm_cols]]

        for macro in macro_predictors:
            print(f"   Interacting {len(firm_cols)} features with '{macro}'...")
            interaction_block = self.full_df[firm_cols].multiply(self.full_df[macro], axis=0)
            interaction_block.columns = [f"{col}_x_{macro}" for col in firm_cols]
            X_parts.append(interaction_block)

        print("Concatenating full feature matrix...")
        X_final = pd.concat(X_parts, axis=1)

        print(f"Final Feature Matrix Shape: {X_final.shape}")
        return X_final, self.full_df[['yyyymm', 'permno', 'excess_ret']]

    def split_data(self, X, metadata, train_end=200512, val_end=201512):
        print("\n--- Step 3: Train/Val/Test Split ---")

        train_mask = metadata['yyyymm'] <= train_end
        val_mask = (metadata['yyyymm'] > train_end) & (metadata['yyyymm'] <= val_end)
        test_mask = metadata['yyyymm'] > val_end

        X_train = X[train_mask]
        X_val = X[val_mask]
        X_test = X[test_mask]

        y_train = metadata.loc[train_mask, 'excess_ret']
        y_val   = metadata.loc[val_mask,  'excess_ret']
        y_test  = metadata.loc[test_mask, 'excess_ret']

        print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val':   X_val,   'y_val':   y_val,
            'X_test':  X_test,  'y_test':  y_test,
            'metadata': metadata
        }

builder = HighRAMDatasetBuilder(df_firm_final, macro_final)
_ = builder.merge_and_calculate_target()
X_matrix, metadata = builder.create_interactions()
data_dict = builder.split_data(X_matrix, metadata)

# ---------------------------------------------------------------------
# DataStorageEngine
# ---------------------------------------------------------------------
class DataStorageEngine:
    def __init__(self, storage_dir="/content/drive/MyDrive/POE/ready_data"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        print(f"Storage Engine initialized at: {self.storage_dir}")

    def save_dataset(self, data_dict):
        print("\n--- Saving X/y/metadata to Disk (Parquet) ---")
        for key, data in data_dict.items():
            file_path = self.storage_dir / f"{key}.parquet"
            if isinstance(data, pd.Series):
                data = data.to_frame(name='target')
            if isinstance(data, pd.DataFrame):
                print(f"Saving {key} ({data.shape})...")
                data.to_parquet(file_path, engine='pyarrow', compression='snappy')
        print("Save Complete.")

    def load_dataset(self):
        print("\n--- Loading Data from Disk ---")
        loaded_dict = {}
        files = list(self.storage_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError("No parquet files found in storage directory.")
        for file_path in files:
            key = file_path.stem
            print(f"Loading {key}...")
            df = pd.read_parquet(file_path)
            if key.startswith('y_'):
                loaded_dict[key] = df.iloc[:, 0]
            else:
                loaded_dict[key] = df
        return loaded_dict

storage = DataStorageEngine(storage_dir="/content/drive/MyDrive/POE/ready_data")
storage.save_dataset(data_dict)

# --- NEW: Save cleaned firm & macro panels for scenario generation ---
df_firm_final.to_parquet(
    "/content/drive/MyDrive/POE/ready_data/firm_clean.parquet",
    engine="pyarrow", compression="snappy"
)
macro_final.to_parquet(
    "/content/drive/MyDrive/POE/ready_data/macro_final.parquet",
    engine="pyarrow", compression="snappy"
)
print("\nSaved firm_clean.parquet and macro_final.parquet for scenario generation.")
