"""
Data processors for firm characteristics, targets, and macroeconomic variables.
"""
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


class FirmOnlyProcessor:
    """
    Robust processor for firm-level characteristics data.

    Processing pipeline:
    1. Filter by date range
    2. Drop sparse columns (> missing_thresh)
    3. Month-by-month processing:
       - Median imputation
       - Winsorization (1%/99%)
       - Dense rank transformation
       - Rescale to [-1, 1]
    """

    def __init__(
        self,
        firm_path: str,
        start_date: int = 198001,
        end_date: int = 202412,
        missing_thresh: float = 0.70
    ):
        """
        Initialize firm processor.

        Args:
            firm_path: Path to firm characteristics parquet file
            start_date: Start date in YYYYMM format (default: 198001)
            end_date: End date in YYYYMM format (default: 202412)
            missing_thresh: Threshold for dropping sparse columns (default: 0.70)
        """
        self.firm_path = Path(firm_path)
        self.start_date = start_date
        self.end_date = end_date
        self.missing_thresh = missing_thresh
        self.meta_cols = []
        self.char_cols = []

    def load_data(self) -> pd.DataFrame:
        """
        Load firm data and filter by date range.

        Returns:
            DataFrame with firm characteristics and metadata

        Side Effects:
            Sets self.meta_cols and self.char_cols
        """
        logger.info(f"Loading firm data from {self.firm_path}")
        df = pd.read_parquet(self.firm_path)

        # Filter by date range
        df = df[(df['yyyymm'] >= self.start_date) & (df['yyyymm'] <= self.end_date)].copy()

        # Identify metadata vs characteristic columns
        self.meta_cols = ['permno', 'yyyymm', 'ret_tplus1', 'Price', 'Size']
        self.meta_cols = [c for c in self.meta_cols if c in df.columns]
        self.char_cols = [c for c in df.columns if c not in self.meta_cols]

        logger.info(f"Initial shape: {df.shape}, characteristics: {len(self.char_cols)}")
        return df

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process firm characteristics with robust normalization.

        Pipeline:
        1. Drop columns with > missing_thresh missing data
        2. Group-wise (by month) processing:
           - Median imputation (then 0 if still missing)
           - Winsorize at 1%/99%
           - Dense rank
           - Rescale to [-1, 1]

        Args:
            df: DataFrame from load_data()

        Returns:
            Processed DataFrame with normalized characteristics
        """
        # Drop very sparse characteristics
        missing_rates = df[self.char_cols].isnull().mean()
        drop_cols = missing_rates[missing_rates > self.missing_thresh].index.tolist()

        if drop_cols:
            logger.info(f"Dropping {len(drop_cols)} sparse columns (>{self.missing_thresh:.0%} missing)")
            df = df.drop(columns=drop_cols)
            self.char_cols = [c for c in self.char_cols if c not in drop_cols]

        logger.info(f"Processing {len(self.char_cols)} characteristics: winsorization, imputation, rank-normalization")

        def clean_group(group):
            """Clean and normalize a single month's cross-section."""
            # Median imputation
            filled = group.fillna(group.median())
            filled = filled.fillna(0)

            # Winsorization at 1%/99%
            lower = filled.quantile(0.01)
            upper = filled.quantile(0.99)
            clipped = filled.clip(lower, upper)

            # Dense rank transformation
            ranks = clipped.rank(method='dense')
            min_r = 1
            max_r = ranks.max()

            # Rescale to [-1, 1]
            if max_r > 1:
                return ((ranks - min_r) / (max_r - min_r)) * 2 - 1
            else:
                # All values constant - return zeros (neutral signal)
                return filled * 0

        # Apply month-by-month transformation
        df[self.char_cols] = df.groupby('yyyymm')[self.char_cols].transform(clean_group)
        df[self.char_cols] = df[self.char_cols].fillna(0)

        return df


class TargetProcessor:
    """
    Processor for target variable (ret_tplus1).

    Handles:
    - Automatic scale detection (percentage vs decimal)
    - Physical limit clipping (impossible negative returns)
    - Upside outlier winsorization
    """

    def __init__(
        self,
        target_col: str = 'ret_tplus1',
        lower_limit: float = -1.0,
        upper_quantile: float = 0.995
    ):
        """
        Initialize target processor.

        Args:
            target_col: Name of target column (default: 'ret_tplus1')
            lower_limit: Physical lower bound (default: -1.0 = -100% loss)
            upper_quantile: Quantile for upside clipping (default: 0.995)
        """
        self.target_col = target_col
        self.physical_limit = lower_limit
        self.upper_q = upper_quantile

    def convert_to_decimal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and convert percentage scale to decimal.

        Detection logic: If mean(abs(target)) > 0.2, assume percentage scale.

        Args:
            df: DataFrame with target column

        Returns:
            DataFrame with target in decimal scale
        """
        target = df[self.target_col]
        mean_abs = target.abs().mean()

        if mean_abs > 0.2:
            logger.info(f"Detected percentage scale data (mean abs: {mean_abs:.2f}), converting to decimal")
            df[self.target_col] = df[self.target_col] / 100.0
        else:
            logger.debug(f"Detected decimal scale data (mean abs: {mean_abs:.4f})")

        return df

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process target variable with scale conversion and clipping.

        Args:
            df: DataFrame with target column

        Returns:
            DataFrame with cleaned target
        """
        df = self.convert_to_decimal(df)

        logger.info(f"Processing target variable: {self.target_col}")

        # Clip impossible negative returns
        lower_mask = df[self.target_col] < self.physical_limit
        if lower_mask.sum() > 0:
            logger.info(f"Clipping {lower_mask.sum()} impossible negative returns to {self.physical_limit}")
            df.loc[lower_mask, self.target_col] = self.physical_limit

        # Clip upside outliers
        upper_limit = df[self.target_col].quantile(self.upper_q)
        logger.info(f"Clipping upside outliers at {self.upper_q:.1%} quantile ({upper_limit:.4f})")
        df[self.target_col] = df[self.target_col].clip(lower=None, upper=upper_limit)

        return df


class MacroDataProcessor:
    """
    Processor for macroeconomic variables.

    Constructs 8 canonical Goyal-Welch predictors:
    - dp: Dividend-price ratio
    - ep: Earnings-price ratio
    - bm: Book-to-market
    - ntis: Net equity issuance
    - tbl: Treasury bill rate
    - tms: Term spread
    - dfy: Default spread
    - svar: Stock variance
    """

    def __init__(
        self,
        file_path: str,
        start_date: int = 198001,
        end_date: int = 202412
    ):
        """
        Initialize macro processor.

        Args:
            file_path: Path to macro data CSV file
            start_date: Start date in YYYYMM format (default: 198001)
            end_date: End date in YYYYMM format (default: 202412)
        """
        self.file_path = Path(file_path)
        self.start_date = start_date
        self.end_date = end_date
        self.predictors = ['dp', 'ep', 'bm', 'ntis', 'tbl', 'tms', 'dfy', 'svar']
        self.df = None

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean numeric columns (remove commas, convert to float).

        Args:
            df: Raw DataFrame from CSV

        Returns:
            DataFrame with cleaned numeric columns
        """
        numeric_cols = ['Index', 'D12', 'E12', 'b/m', 'tbl', 'AAA', 'BAA', 'lty', 'ntis', 'Rfree', 'svar']
        for col in numeric_cols:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(',', '', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    def load_and_process(self) -> pd.DataFrame:
        """
        Load macro data and construct Goyal-Welch predictors.

        Returns:
            DataFrame with yyyymm, Rfree, and 8 macro predictors
        """
        df = pd.read_csv(self.file_path)
        df = self._clean_data(df)

        # Filter by date range
        df = df[(df['yyyymm'] >= self.start_date) & (df['yyyymm'] <= self.end_date)].reset_index(drop=True)

        # Construct predictors
        df['dp'] = np.log(df['D12']) - np.log(df['Index'])
        df['ep'] = np.log(df['E12']) - np.log(df['Index'])
        df.rename(columns={'b/m': 'bm'}, inplace=True)
        df['tms'] = df['lty'] - df['tbl']
        df['dfy'] = df['BAA'] - df['AAA']

        # Select final columns
        cols = ['yyyymm', 'Rfree'] + self.predictors
        if 'infl' in df.columns:
            cols.append('infl')

        self.df = df[cols].copy()
        return self.df

    def get_final_data(self) -> pd.DataFrame:
        """
        Get processed macro data.

        Returns:
            DataFrame with macro predictors
        """
        return self.df
