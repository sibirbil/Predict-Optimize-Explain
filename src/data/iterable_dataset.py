"""
Iterable Dataset for memory-efficient data loading.

Uses PyArrow to stream data from parquet files without loading everything into memory.
"""
import torch
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
from torch.utils.data import IterableDataset
from pathlib import Path
from typing import Optional, Tuple


class ParquetIterableDataset(IterableDataset):
    """
    PyTorch IterableDataset that streams data from parquet files.

    Loads data in batches using PyArrow, avoiding memory issues with large files.

    Args:
        features_path: Path to X parquet file
        labels_path: Path to y parquet file
        meta_path: Optional path to metadata parquet file
        batch_size: Number of rows to load at once from parquet
        shuffle: Whether to shuffle within each batch
    """

    def __init__(
        self,
        features_path: str,
        labels_path: str,
        meta_path: Optional[str] = None,
        batch_size: int = 16384,
        shuffle: bool = True
    ):
        super().__init__()
        self.features_path = Path(features_path)
        self.labels_path = Path(labels_path)
        self.meta_path = Path(meta_path) if meta_path else None
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Validate files exist
        if not self.features_path.exists():
            raise FileNotFoundError(f"Features file not found: {self.features_path}")
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {self.labels_path}")
        if self.meta_path and not self.meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.meta_path}")

    def __iter__(self):
        """Iterate over data in batches."""
        # Open parquet files (doesn't load data yet)
        f_file = pq.ParquetFile(self.features_path)
        l_file = pq.ParquetFile(self.labels_path)

        # Verify same number of rows
        if f_file.metadata.num_rows != l_file.metadata.num_rows:
            raise ValueError(
                f"Row count mismatch: features={f_file.metadata.num_rows}, "
                f"labels={l_file.metadata.num_rows}"
            )

        # Create batch iterators
        f_batches = f_file.iter_batches(batch_size=self.batch_size)
        l_batches = l_file.iter_batches(batch_size=self.batch_size)

        # If metadata requested, also iterate over it
        if self.meta_path:
            m_file = pq.ParquetFile(self.meta_path)
            m_batches = m_file.iter_batches(batch_size=self.batch_size)
            iterator = zip(f_batches, l_batches, m_batches)
        else:
            iterator = zip(f_batches, l_batches)

        # Iterate over batches
        for batch_data in iterator:
            if self.meta_path:
                f_batch, l_batch, m_batch = batch_data
            else:
                f_batch, l_batch = batch_data

            # Convert to numpy then torch
            X = torch.from_numpy(f_batch.to_pandas().to_numpy()).float()
            y = torch.from_numpy(l_batch.to_pandas().to_numpy()).float().squeeze()

            # Shuffle within batch if requested
            if self.shuffle:
                perm = torch.randperm(X.size(0))
                X = X[perm]
                y = y[perm]

            # If metadata requested, include it
            if self.meta_path:
                meta = m_batch.to_pandas()
                if self.shuffle:
                    meta = meta.iloc[perm.numpy()]

                # Yield samples with metadata
                for i in range(len(X)):
                    yield X[i], y[i], meta.iloc[i]
            else:
                # Yield individual samples
                for xi, yi in zip(X, y):
                    yield xi, yi


class MonthlyParquetDataset(IterableDataset):
    """
    Dataset that yields data month-by-month for portfolio backtesting.

    Instead of individual samples, yields entire months of data.
    Useful for PTO/E2E where we need all stocks in a month together.

    Args:
        features_path: Path to X parquet file
        labels_path: Path to y parquet file
        meta_path: Path to metadata parquet file (required - need yyyymm)
        batch_size: Number of rows to load at once from parquet
    """

    def __init__(
        self,
        features_path: str,
        labels_path: str,
        meta_path: str,
        batch_size: int = 50000
    ):
        super().__init__()
        self.features_path = Path(features_path)
        self.labels_path = Path(labels_path)
        self.meta_path = Path(meta_path)
        self.batch_size = batch_size

        # Validate files exist
        for path in [self.features_path, self.labels_path, self.meta_path]:
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")

    def __iter__(self):
        """Iterate over months of data."""
        # Read entire dataset at once (simpler and faster for monthly iteration)
        # This is acceptable because we're only reading metadata to group by month

        # Read all data (PyArrow is efficient with parquet)
        X_df = pq.read_table(self.features_path).to_pandas()
        y_df = pq.read_table(self.labels_path).to_pandas()
        meta_df = pq.read_table(self.meta_path).to_pandas()

        # Convert features and labels to numpy once
        X_all = X_df.to_numpy()
        y_all = y_df.to_numpy().flatten()

        # Group by month and yield in sorted order
        for month in sorted(meta_df['yyyymm'].unique()):
            # Get indices for this month
            month_mask = meta_df['yyyymm'] == month
            month_indices = month_mask.values

            yield {
                'yyyymm': month,
                'X': X_all[month_indices],
                'y': y_all[month_indices],
                'meta': meta_df[month_mask].reset_index(drop=True)
            }


def create_data_loaders(
    data_dir: str,
    batch_size: int = 256,
    parquet_batch_size: int = 16384,
    shuffle_train: bool = True,
    load_train: bool = True,
    load_val: bool = True,
    load_test: bool = True
) -> dict:
    """
    Create PyTorch DataLoaders for train/val/test splits.

    Args:
        data_dir: Directory containing parquet files
        batch_size: Batch size for DataLoader
        parquet_batch_size: Number of rows to load from parquet at once
        shuffle_train: Whether to shuffle training data
        load_train: Whether to load training data
        load_val: Whether to load validation data
        load_test: Whether to load test data

    Returns:
        Dictionary with DataLoaders for requested splits
    """
    from torch.utils.data import DataLoader

    data_path = Path(data_dir)
    loaders = {}

    if load_train:
        train_dataset = ParquetIterableDataset(
            features_path=data_path / 'X_train.parquet',
            labels_path=data_path / 'y_train.parquet',
            batch_size=parquet_batch_size,
            shuffle=shuffle_train
        )
        loaders['train'] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False  # Shuffling done in dataset
        )

    if load_val:
        val_dataset = ParquetIterableDataset(
            features_path=data_path / 'X_val.parquet',
            labels_path=data_path / 'y_val.parquet',
            batch_size=parquet_batch_size,
            shuffle=False
        )
        loaders['val'] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )

    if load_test:
        test_dataset = ParquetIterableDataset(
            features_path=data_path / 'X_test.parquet',
            labels_path=data_path / 'y_test.parquet',
            batch_size=parquet_batch_size,
            shuffle=False
        )
        loaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )

    return loaders


def create_monthly_iterator(data_dir: str, split: str = 'test') -> MonthlyParquetDataset:
    """
    Create iterator that yields data month-by-month.

    Useful for portfolio backtesting where we need all stocks in each month.

    Args:
        data_dir: Directory containing parquet files
        split: Which split to load ('train', 'val', or 'test')

    Returns:
        MonthlyParquetDataset iterator
    """
    data_path = Path(data_dir)

    return MonthlyParquetDataset(
        features_path=data_path / f'X_{split}.parquet',
        labels_path=data_path / f'y_{split}.parquet',
        meta_path=data_path / f'meta_{split}.parquet'
    )
