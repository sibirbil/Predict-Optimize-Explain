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

from src.modules.dataloaders import DataStorageEngine
# --- 1. Data Loader ---
    
# --- 2. Execution & Inspection ---

# A. Load
# Change Path
storage = DataStorageEngine(storage_dir="./Data/final_data")
data = storage.load_dataset()


# NOTE: I DONT LIKE THE WAY DATA PATCHING IS DONE.
# We should know the format of the data. Not have to deal with guessing whether
# it is given in a percentage form or not.


# 2. Clip impossible returns to -1.0 (You cannot lose more than 100%)
for key in ['y_train', 'y_val', 'y_test']:
    pre_min = data[key].min()
    # Clip at -0.99 to prevent division by zero or log errors later if needed
    data[key] = data[key].clip(lower=-0.99)
    post_min = data[key].min()
    print(f"{key}: Min clipped from {pre_min:.4f} to {post_min:.4f}")

# 3. Verify Final Stats
print("\nTarget Stats after Patch:")
print(data['y_train'].describe()[['mean', 'min', 'max', 'std']])

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time

import torch
import torch.nn as nn

class AssetPricingFNN(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.5):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.output = nn.Linear(8, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.output(x)


## LOAD THE MODEL 

LOAD_DIR = "./models/fnn_v1"

with open(os.path.join(LOAD_DIR, "model_config.json"), "r") as f:
    cfg = json.load(f)

with open(os.path.join(LOAD_DIR, "feature_columns.json"), "r") as f:
    feature_cols = json.load(f)

state = torch.load(os.path.join(LOAD_DIR, "state_dict.pt"), map_location="cpu")

model = AssetPricingFNN(input_dim=cfg["input_dim"], dropout_rate=cfg["dropout_rate"])
model.load_state_dict(state)
model.eval()

print("Loaded model.")
print("Config:", cfg)
print("Feature cols:", len(feature_cols))


def align_features(X_df, feature_cols):
    missing = [c for c in feature_cols if c not in X_df.columns]
    extra = [c for c in X_df.columns if c not in feature_cols]

    if missing:
        raise ValueError(f"Missing {len(missing)} columns, e.g. {missing[:10]}")
    # drop extras and reorder
    X_aligned = X_df.loc[:, feature_cols]
    return X_aligned, extra

X_test_aligned, extra_cols = align_features(data["X_test"], feature_cols)
print("Aligned X_test shape:", X_test_aligned.shape)
print("Extra columns dropped:", len(extra_cols))


import numpy as np

X_small = X_test_aligned.iloc[:4096].to_numpy(dtype=np.float32, copy=False)
with torch.no_grad():
    yhat = model(torch.from_numpy(X_small)).squeeze(-1).numpy()

print("Pred stats:")
print("  mean/std:", float(yhat.mean()), float(yhat.std()))
print("  min/max :", float(yhat.min()), float(yhat.max()))


with open(os.path.join(LOAD_DIR, "train_history.json"), "r") as f:
    hist = json.load(f)

print("Last val_r2 (%):", 100 * hist["val_r2"][-1])
print("Best  val_r2 (%):", 100 * max(hist["val_r2"]))


import numpy as np
import pandas as pd
import torch

def predict_cpu(model, X_df, batch_size=65536):
    model.eval()
    X_np = X_df.to_numpy(dtype=np.float32, copy=False)
    preds = []

    with torch.no_grad():
        for i in range(0, X_np.shape[0], batch_size):
            xb = torch.from_numpy(X_np[i:i+batch_size])
            yb = model(xb).squeeze(-1).numpy()
            preds.append(yb)

    return np.concatenate(preds)

# Align first (you already did)
pred_test = predict_cpu(model, X_test_aligned, batch_size=65536)

# Build index-aligned results with RAW excess returns
idx = X_test_aligned.index
df_results = data["metadata"].loc[idx, ["yyyymm","permno","excess_ret"]].copy()
df_results = df_results.rename(columns={"excess_ret":"excess_ret_raw"})

df_results["pred_return"] = pred_test
df_results["excess_ret_clip1"] = df_results["excess_ret_raw"].clip(lower=-1.0)

# Save
save_path = "./outs/oos_predictions_TEST_from_loaded_model.parquet"
df_results.to_parquet(save_path)

print("Saved:", save_path)
print(df_results[["pred_return","excess_ret_raw"]].describe())



