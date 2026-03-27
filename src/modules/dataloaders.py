import torch
import pyarrow.parquet as pq
from torch.utils.data import IterableDataset, DataLoader
from pathlib import Path
import pandas as pd
import numpy as np
from src.modules.pao_model_defs import PAOPortfolioModel


TRAIN_END = 200512
VAL_END   = 201512

class ParquetIterableDataset(IterableDataset):
    def __init__(self, features_path, labels_path, batch_size=16384, shuffle = True):
        super().__init__()
        self.features_path = features_path
        self.labels_path = labels_path
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        f_file = pq.ParquetFile(self.features_path)
        l_file = pq.ParquetFile(self.labels_path)

        assert f_file.metadata.num_rows == l_file.metadata.num_rows

        f_batches = f_file.iter_batches(batch_size=self.batch_size)
        l_batches = l_file.iter_batches(batch_size=self.batch_size)

        for f_batch, l_batch in zip(f_batches, l_batches):

            X = torch.from_numpy(f_batch.to_pandas().to_numpy()).float()
            y = torch.from_numpy(l_batch.to_pandas().to_numpy()).float()

            if self.shuffle:
                perm = torch.randperm(X.size(0))
                X = X[perm]
                y = y[perm]
            
            for xi, yi in zip(X, y):
                yield xi, yi


def getIterableDatasets():
    dataset = ParquetIterableDataset('./Data/final_data/X_train.parquet',
                                 './Data/final_data/y_train.parquet', shuffle = True)
    trainloader = DataLoader(dataset, batch_size=256, shuffle= False)
    val_dataset = ParquetIterableDataset('./Data/final_data/X_val.parquet',
                                      './Data/final_data/y_val.parquet', shuffle = False)
    val_loader = DataLoader(val_dataset, batch_size= 1024, shuffle = False)

    test_dataset = ParquetIterableDataset('./Data/final_data/X_test.parquet',
                                      './Data/final_data/y_test.parquet', shuffle = False)
    test_loader = DataLoader(test_dataset, batch_size= 1024, shuffle = False)
    return {'dataset':dataset, 'trainloader':trainloader, 
            'val_dataset':val_dataset, 'val_loader':val_loader,
            'test_dataset':test_dataset, 'test_loader':test_loader}


class DataStorageEngine:
    def __init__(self, storage_dir, load_train = False):
        self.storage_dir = Path(storage_dir)
        self.load_train = load_train # train||test||val||None
        print(f"Initializing Loader from: {self.storage_dir}")

    def load_dataset(self):
        print("\n--- Loading Data ---")
        loaded_dict = {}
        files = list(self.storage_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found in {self.storage_dir}")

        for file_path in files:
            key = file_path.stem
            if (self.load_train == False) and key.endswith("train"):
                continue
            print(f"Loading {key}...")
            df = pd.read_parquet(file_path, engine = 'pyarrow').astype(np.float32)
            # Handle Series vs DataFrame
            if key.startswith('y_'):
                loaded_dict[key] = df.iloc[:, 0]
            else:
                loaded_dict[key] = df
        return loaded_dict
    



def strict_metadata_alignment(metadata: pd.DataFrame, train_end=TRAIN_END, val_end=VAL_END):

    if "yyyymm" not in metadata.columns:
        raise ValueError("metadata must contain a 'yyyymm' column.")

    full_meta = metadata.copy()
    full_meta["yyyymm"] = full_meta["yyyymm"].astype(int)

    mask_train = full_meta["yyyymm"] <= int(train_end)
    mask_val   = (full_meta["yyyymm"] > int(train_end)) & (full_meta["yyyymm"] <= int(val_end))
    mask_test  = full_meta["yyyymm"] > int(val_end)

    meta_train = full_meta[mask_train].copy()
    meta_val   = full_meta[mask_val].copy()
    meta_test  = full_meta[mask_test].copy()

    return meta_train, meta_val, meta_test

