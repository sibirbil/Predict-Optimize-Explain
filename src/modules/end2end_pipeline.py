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

# --- 1. Data Loader ---
class DataStorageEngine:
    def __init__(self, storage_dir="/content/drive/MyDrive/POE/ready_data"):
        self.storage_dir = Path(storage_dir)
        print(f"Initializing Loader from: {self.storage_dir}")

    def load_dataset(self):
        print("\n--- Loading Data ---")
        loaded_dict = {}
        files = list(self.storage_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found in {self.storage_dir}")

        for file_path in files:
            key = file_path.stem
            print(f"Loading {key}...")
            df = pd.read_parquet(file_path)
            # Handle Series vs DataFrame
            if key.startswith('y_'):
                loaded_dict[key] = df.iloc[:, 0]
            else:
                loaded_dict[key] = df
        return loaded_dict
    
# --- 2. Execution & Inspection ---

# A. Load
# Change Path
storage = DataStorageEngine(storage_dir="./Data/final_data_sub")
data = storage.load_dataset()

# B. Shape Verification
print("\n--- [1] Shape Verification ---")
print(f"X_train: {data['X_train'].shape} | y_train: {data['y_train'].shape}")
print(f"X_val:   {data['X_val'].shape}   | y_val:   {data['y_val'].shape}")
print(f"X_test:  {data['X_test'].shape}   | y_test:  {data['y_test'].shape}")

3# --- 3. Data Patching ---
print("\n--- [3] Patching Data Targets ---")

# 1. Check for Percentage vs Decimal Scale
# If mean is > 0.1 (10%), it's likely percentage. We need decimals for optimization stability.
train_mean = data['y_train'].mean()
if abs(train_mean) > 0.1:
    print(f">> Detected Percentage Scale (Mean={train_mean:.2f}). Dividing all targets by 100.")
    for key in ['y_train', 'y_val', 'y_test']:
        data[key] = data[key] / 100.0
else:
    print(f">> Detected Decimal Scale (Mean={train_mean:.4f}). No scaling needed.")

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


import os, json

LOAD_DIR = "/content/drive/MyDrive/POE/models/fnn_v1"

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
save_path = "/content/drive/MyDrive/POE/oos_predictions_TEST_from_loaded_model.parquet"
df_results.to_parquet(save_path)

print("Saved:", save_path)
print(df_results[["pred_return","excess_ret_raw"]].describe())


import numpy as np

X_val_aligned, _ = align_features(data["X_val"], feature_cols)
pred_val = predict_cpu(model, X_val_aligned, batch_size=65536)

y_val = data["y_val"].clip(lower=-1.0).to_numpy()

# R2 vs mean
sse = np.sum((y_val - pred_val)**2)
tss = np.sum((y_val - y_val.mean())**2)
r2 = 1.0 - sse / (tss + 1e-12)

print("VAL R2 vs mean (%):", 100*r2)


import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def evaluate_oos_standard(model, data_dict, batch_size=512):
    # 1. Auto-detect Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation on Device: {device}")

    model.to(device)
    model.eval()

    print("\n--- Generating Out-of-Sample Predictions (2016-2024) ---")

    # 2. Prepare Test Data (Keep on CPU initially)
    test_x = torch.tensor(data_dict['X_test'].values, dtype=torch.float32)
    test_y_true = data_dict['y_test'].clip(lower=-1.0).values

    # Standard Loader
    test_ds = torch.utils.data.TensorDataset(test_x)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    predictions = []

    # 3. Inference Loop
    with torch.no_grad():
        for batch in test_loader:
            bx = batch[0].to(device) # Move batch to GPU/CPU
            output = model(bx)
            predictions.append(output.cpu().numpy())

    # Flatten
    pred_y = np.concatenate(predictions).flatten()

    # 4. Calculate R2 OOS (Benchmark: Zero)
    sse_model = np.sum((test_y_true - pred_y)**2)
    sse_zero = np.sum(test_y_true**2)
    r2_oos = 1 - (sse_model / sse_zero)

    print(f"Test Set Rows: {len(test_y_true):,}")
    print(f">> R2_OOS (Zero Benchmark): {r2_oos:.4%}")

    return pred_y, r2_oos

# --- Execute ---
# Using 'model_std' from the CPU/GPU training step
pred_oos, r2_score = evaluate_oos_standard(model, data)

# --- Save Results for Optimization ---
print("\nConstructing Portfolio Dataset...")
df_results = data['metadata'][data['metadata']['yyyymm'] > 201512].copy().reset_index(drop=True)
df_results['pred_return'] = pred_oos
df_results['actual_return'] = data['y_test'].clip(lower=-1.0).values

# Check Stats
print(df_results[['pred_return', 'actual_return']].describe())

# Save
df_results.to_parquet("oos_predictions_2016_2024.parquet")
print("Saved to 'oos_predictions_2016_2024.parquet'. Ready for Portfolio Optimization.")




import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

from tqdm.auto import tqdm


# Cell 2: Rolling covariance engine
class RollingCovariance:
    """
    Builds a (yyyymm x permno) wide matrix of realized returns and provides
    a rolling covariance estimate for a requested month and a requested permno set.

    IMPORTANT indexing convention in your pipeline:
      - The row key 'yyyymm' corresponds to the *information/decision month t*.
      - The stored return column (excess_ret / y) corresponds to realized return over t+1.
      - Therefore, when we request Sigma at decision month t, we use returns up to t (inclusive),
        which correspond to rows <= t-1 in this decision-month indexing.
    """
    def __init__(self, metadata_df: pd.DataFrame, ret_col: str = "excess_ret",
                 lookback: int = 60, min_periods: int = 12):
        req = {"yyyymm", "permno", ret_col}
        missing = req - set(metadata_df.columns)
        if missing:
            raise KeyError(f"metadata_df missing columns: {missing}")

        df = metadata_df[["yyyymm", "permno", ret_col]].copy()

        # enforce clean types (prevents silent 'no matches' in covariance)
        df["yyyymm"] = df["yyyymm"].astype(int)
        df["permno"] = df["permno"].astype(int)

        self.lookback = int(lookback)
        self.min_periods = int(min_periods)

        self.returns_wide = (
            df.pivot_table(index="yyyymm", columns="permno", values=ret_col)
              .sort_index()
        )

        self.date_to_idx = {d: i for i, d in enumerate(self.returns_wide.index)}
        print(f"[RollingCovariance] returns_wide shape = {self.returns_wide.shape}")

    def get_covariance_batch(self, date_yyyymm: int, permnos: np.ndarray) -> np.ndarray:
        permnos = np.asarray(permnos).astype(int)
        K = len(permnos)

        # If date not in history, return a small diagonal matrix
        if int(date_yyyymm) not in self.date_to_idx:
            return (np.eye(K) * 0.01).astype(np.float32)

        end_idx = self.date_to_idx[int(date_yyyymm)]
        start_idx = max(0, end_idx - self.lookback)

        # only those permnos that exist in the wide return matrix
        valid_cols = [p for p in permnos if p in self.returns_wide.columns]
        if len(valid_cols) < 2:
            return (np.eye(K) * 0.01).astype(np.float32)

        hist = self.returns_wide.iloc[start_idx:end_idx][valid_cols]  # excludes current row by iloc slicing
        cov_sub = hist.cov(min_periods=self.min_periods).values

        # cleanup
        if np.isnan(cov_sub).any() or np.isinf(cov_sub).any():
            np.nan_to_num(cov_sub, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            np.fill_diagonal(cov_sub, np.diag(cov_sub) + 0.01)

        # map back to requested permno order
        final_cov = np.eye(K, dtype=np.float32) * 0.01
        col_to_idx = {c: i for i, c in enumerate(valid_cols)}
        idx_map = [col_to_idx.get(p, -1) for p in permnos]

        for i, ii in enumerate(idx_map):
            if ii == -1:
                continue
            for j, jj in enumerate(idx_map):
                if jj == -1:
                    continue
                final_cov[i, j] = cov_sub[ii, jj]

        # small ridge for numerical stability
        final_cov += np.eye(K, dtype=np.float32) * 1e-4
        return final_cov.astype(np.float32)



# Cell 3: E2E dataset (one sample = one month)
class E2E_Portfolio_Loader(Dataset):
    """
    One sample = one month t with K assets (mini-market).
    Returns tensors:
      X:     (K, F)
      y:     (K,)
      Sigma: (K, K)
      date:  scalar int yyyymm
    """
    def __init__(self, X_tensor: torch.Tensor, y_tensor: torch.Tensor,
                 metadata_df: pd.DataFrame, cov_engine: RollingCovariance,
                 n_assets: int = 50):
        assert len(X_tensor) == len(y_tensor) == len(metadata_df), \
            "X, y, and metadata must have the same number of rows (same ordering)."

        self.X = X_tensor
        self.y = y_tensor
        self.meta = metadata_df.copy()

        # enforce int keys
        self.meta["yyyymm"] = self.meta["yyyymm"].astype(int)
        self.meta["permno"] = self.meta["permno"].astype(int)

        # reset to positional index so tensor row i aligns with meta row i
        self.meta = self.meta.reset_index(drop=True)

        self.n_assets = int(n_assets)
        self.cov_engine = cov_engine

        # month -> list of row positions
        self.date_to_indices = self.meta.groupby("yyyymm").groups
        self.unique_dates = sorted(self.date_to_indices.keys())

    def __len__(self):
        return len(self.unique_dates)

    def __getitem__(self, idx: int):
        date = int(self.unique_dates[idx])
        valid_indices = np.asarray(list(self.date_to_indices[date]), dtype=int)
        n_available = len(valid_indices)

        # sample K assets from *this same month*
        replace = (n_available < self.n_assets)
        chosen = np.random.choice(valid_indices, self.n_assets, replace=replace)
        chosen = np.sort(chosen)

        # strict check: all chosen rows must have same yyyymm
        month_vals = self.meta.iloc[chosen]["yyyymm"].values
        if not np.all(month_vals == date):
            raise RuntimeError("Batch construction error: mixed months inside a single portfolio sample.")

        X_k = self.X[chosen]                  # (K,F)
        y_k = self.y[chosen]                  # (K,)
        permnos = self.meta.iloc[chosen]["permno"].values

        Sigma_np = self.cov_engine.get_covariance_batch(date, permnos)  # (K,K)
        Sigma = torch.tensor(Sigma_np, dtype=torch.float32)

        return {"X": X_k, "y": y_k, "Sigma": Sigma, "date": torch.tensor(date, dtype=torch.int64)}



# Cell 4: Predict-and-Optimize model (differentiable QP layer)
class PredictAndOptimize(nn.Module):
    def __init__(self, n_features: int, n_assets: int, gamma: float = 5.0):
        super().__init__()
        self.gamma = float(gamma)
        self.n_assets = int(n_assets)

        self.fnn = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

        # CVXPY problem: maximize mu^T w - (gamma/2) * ||L^T w||^2  s.t. w>=0, sum w = 1
        w = cp.Variable(self.n_assets)
        mu_param = cp.Parameter(self.n_assets)
        LT_param = cp.Parameter((self.n_assets, self.n_assets))

        risk_term = cp.sum_squares(LT_param @ w)  # = ||L^T w||^2
        ret_term = mu_param @ w

        objective = cp.Maximize(ret_term - (self.gamma / 2.0) * risk_term)
        constraints = [cp.sum(w) == 1, w >= 0]
        problem = cp.Problem(objective, constraints)

        self.opt_layer = CvxpyLayer(problem, parameters=[mu_param, LT_param], variables=[w])

    @staticmethod
    def robust_factorize(Sigma: torch.Tensor) -> torch.Tensor:
        """
        Returns L such that Sigma =_ L L^T.
        """
        # symmetrize
        Sigma = 0.5 * (Sigma + Sigma.transpose(-1, -2))
        # ridge
        eye = torch.eye(Sigma.shape[-1], device=Sigma.device, dtype=Sigma.dtype)
        Sigma = Sigma + 1e-6 * eye

        try:
            L = torch.linalg.cholesky(Sigma)  # (B,K,K) lower-triangular
            return L
        except RuntimeError:
            eigvals, eigvecs = torch.linalg.eigh(Sigma)
            eigvals = torch.clamp(eigvals, min=1e-8)
            # symmetric sqrt: V sqrt(D)
            L = eigvecs @ torch.diag_embed(torch.sqrt(eigvals))
            return L

    def forward(self, X: torch.Tensor, Sigma: torch.Tensor):
        """
        X:     (B,K,F)
        Sigma: (B,K,K)
        Returns:
          w_star: (B,K)
          mu_hat: (B,K)
        """
        B, K, F = X.shape
        assert K == self.n_assets, f"K={K} must match n_assets={self.n_assets}"

        mu_hat = self.fnn(X.reshape(-1, F)).reshape(B, K)  # (B,K)

        # factorize Sigma -> L, pass L^T
        L = self.robust_factorize(Sigma)                  # (B,K,K)
        LT = L.transpose(-1, -2)

        w_star, = self.opt_layer(mu_hat.double(), LT.double())
        return w_star.float(), mu_hat



# Cell 5: Regret loss + training loop
class RegretLoss(nn.Module):
    def __init__(self, gamma: float):
        super().__init__()
        self.gamma = float(gamma)

    def forward(self, w_pred, w_opt, y_realized, Sigma):
        # realized portfolio return
        ret_pred = (w_pred * y_realized).sum(dim=1)
        risk_pred = torch.einsum("bi,bij,bj->b", w_pred, Sigma, w_pred)
        util_pred = ret_pred - (self.gamma / 2.0) * risk_pred

        with torch.no_grad():
            ret_opt = (w_opt * y_realized).sum(dim=1)
            risk_opt = torch.einsum("bi,bij,bj->b", w_opt, Sigma, w_opt)
            util_opt = ret_opt - (self.gamma / 2.0) * risk_opt

        return (util_opt - util_pred).mean()


def train_regret_model(model, train_loader, val_loader, n_epochs=10, lr=1e-3, device="cpu"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    loss_fn = RegretLoss(gamma=model.gamma)

    train_hist, val_hist = [], []
    print(f"--- Regret training on {device} ---")

    for epoch in range(n_epochs):
        # ---------- train ----------
        model.train()
        train_losses = []

        for batch in train_loader:
            X = batch["X"].to(device)            # (B,K,F)
            y = batch["y"].to(device)            # (B,K)
            Sigma = batch["Sigma"].to(device)    # (B,K,K)

            w_pred, _ = model(X, Sigma)

            # oracle weights: solve same QP but with realized y as "mu"
            L = model.robust_factorize(Sigma)
            LT = L.transpose(-1, -2)
            with torch.no_grad():
                w_opt, = model.opt_layer(y.double(), LT.double())
                w_opt = w_opt.float()

            loss = loss_fn(w_pred, w_opt, y, Sigma)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

        avg_train = float(np.mean(train_losses))

        # ---------- val ----------
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                Xv = batch["X"].to(device)
                yv = batch["y"].to(device)
                Sigv = batch["Sigma"].to(device)

                w_pred_v, _ = model(Xv, Sigv)

                Lv = model.robust_factorize(Sigv)
                LTv = Lv.transpose(-1, -2)
                w_opt_v, = model.opt_layer(yv.double(), LTv.double())
                w_opt_v = w_opt_v.float()

                vloss = loss_fn(w_pred_v, w_opt_v, yv, Sigv)
                val_losses.append(vloss.item())

        avg_val = float(np.mean(val_losses))
        train_hist.append(avg_train)
        val_hist.append(avg_val)

        scheduler.step(avg_val)
        print(f"Epoch {epoch+1:02d} | Train regret: {avg_train:.6f} | Val regret: {avg_val:.6f}")

    return train_hist, val_hist


# Cell 6: Load data, align indices, build datasets/loaders (with checks)

READY_DIR = Path("/content/drive/MyDrive/POE/ready_data")

def load_ready(name: str):
    df = pd.read_parquet(READY_DIR / f"{name}.parquet")
    if name.startswith("y_"):
        return df.iloc[:, 0]   # Series
    return df

X_train = load_ready("X_train")
y_train = load_ready("y_train")
X_val   = load_ready("X_val")
y_val   = load_ready("y_val")
metadata_full = load_ready("metadata")

print("Loaded:")
print("X_train", X_train.shape, "| y_train", y_train.shape)
print("X_val  ", X_val.shape,   "| y_val  ", y_val.shape)
print("metadata", metadata_full.shape)

# Required columns
assert {"yyyymm", "permno"}.issubset(metadata_full.columns), "metadata must contain yyyymm and permno."
assert "excess_ret" in metadata_full.columns, "metadata must contain excess_ret for covariance."

# Clean types (prevents silent mismatches in groupby/cov)
metadata_full = metadata_full.copy()
metadata_full["yyyymm"] = metadata_full["yyyymm"].astype(int)
metadata_full["permno"] = metadata_full["permno"].astype(int)

# Alignment via index (your diagnostics already showed 100% feasibility)
meta_train = metadata_full.loc[X_train.index].copy()
meta_val   = metadata_full.loc[X_val.index].copy()

# Strong ordering check: ensure meta rows are in same order as X rows
# (loc preserves the order of the indexer; this should pass)
assert (meta_train.index.values == X_train.index.values).all(), "meta_train order mismatch vs X_train index order."
assert (meta_val.index.values == X_val.index.values).all(), "meta_val order mismatch vs X_val index order."

# Convert to tensors in that exact row order
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_val_tensor   = torch.tensor(X_val.values,   dtype=torch.float32)
y_val_tensor   = torch.tensor(y_val.values,   dtype=torch.float32)

# Covariance engine uses the FULL metadata history
cov_engine = RollingCovariance(metadata_full, ret_col="excess_ret", lookback=60, min_periods=12)

N_ASSETS = 50
train_dataset = E2E_Portfolio_Loader(X_train_tensor, y_train_tensor, meta_train, cov_engine, n_assets=N_ASSETS)
val_dataset   = E2E_Portfolio_Loader(X_val_tensor,   y_val_tensor,   meta_val,   cov_engine, n_assets=N_ASSETS)

print(f"Train months: {len(train_dataset)} | Val months: {len(val_dataset)}")

BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, drop_last=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=False)

# Quick batch sanity (no plotting, just shapes)
batch = next(iter(train_loader))
print("Batch shapes:")
print("X:", batch["X"].shape, "  (B,K,F)")
print("y:", batch["y"].shape, "  (B,K)")
print("Sigma:", batch["Sigma"].shape, "  (B,K,K)")
print("date:", batch["date"][:5])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Cell 7: Build model, train, save

device = "cpu"  # strongly recommended with CVXPYLayers
n_features = X_train.shape[1]
GAMMA = 5.0

model = PredictAndOptimize(n_features=n_features, n_assets=N_ASSETS, gamma=GAMMA)

N_EPOCHS = 20
LR = 1e-3

train_hist, val_hist = train_regret_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    n_epochs=N_EPOCHS,
    lr=LR,
    device=device,
)

MODEL_PATH = READY_DIR / "predict_optimize_e2e.pt"
torch.save(model.state_dict(), MODEL_PATH)
print(f"Saved E2E model to {MODEL_PATH}")


# Cell 8: Plot regret curves

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.plot(train_hist, label="Train regret")
plt.plot(val_hist, label="Val regret", linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("Average regret")
plt.title("PAO Predict-and-Optimize Training (Regret minimization)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


macro_final = data["macro_final"].copy()
print("\n--- Macro Final Loaded ---")
print(macro_final.shape)
print(macro_final.head())

assert "yyyymm" in macro_final.columns, "macro_final must have a yyyymm column!"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


N_MACRO = 9


class DifferentiableInteractionReconstructor(nn.Module):
    """
    Rebuilds the full feature tensor X from:
      - Z_firm: base firm characteristics (first block of features)
      - M_macro: macro variables in the model's own feature scale

    It assumes the original training feature construction was:
      X = [Z_firm, Z_firm * M1, Z_firm * M2, ..., Z_firm * M_N_macro]
    """
    def __init__(self, n_firm_features, n_macro_features):
        super().__init__()
        self.n_firm = n_firm_features
        self.n_macro = n_macro_features

    def forward(self, Z_firm, M_macro):
        """
        Z_firm:  (B, K, N_firm)   fixed firm features
        M_macro: (B, N_macro)     optimizable macro variables

        Returns:
          X_reconstructed: (B, K, N_firm * (1 + N_macro))
        """
        # Base block
        features_list = [Z_firm]

        # Interaction blocks: Z_firm * M_j
        for m in range(self.n_macro):
            # M_macro[:, m]: (B,) -> (B,1,1) for broadcasting
            macro_val = M_macro[:, m].view(-1, 1, 1)
            interaction_block = Z_firm * macro_val   # (B, K, N_firm)
            features_list.append(interaction_block)

        # Concatenate along feature dimension
        X_reconstructed = torch.cat(features_list, dim=2)
        return X_reconstructed



def compute_portfolio_utility_batch(model, X, Sigma):
    """
    Computes batch Markowitz utility using the trained predict-and-optimize model.

    Args:
      model:  PredictAndOptimize
      X:      (B, K, F_total)
      Sigma:  (B, K, K)

    Returns:
      util:   (B,) tensor of utilities
    """
    with torch.no_grad():
        w_star, mu_hat = model(X, Sigma)          # w_star: (B,K), mu_hat: (B,K)
        port_ret = torch.sum(w_star * mu_hat, dim=1)     # (B,)
        port_risk = torch.einsum("bi,bij,bj->b", w_star, Sigma, w_star)
        util = port_ret - (model.gamma / 2.0) * port_risk
    return util


def run_gradient_scenario(
    model,
    reconstructor,
    Z_firm_fixed,      # (B, K, N_firm)
    M_init,            # (B, N_macro)
    Sigma_fixed,       # (B, K, K)
    target_utility=-0.15,
    n_steps=500,
    step_size=0.1,
    noise_scale=0.01,
    prior_lambda=0.1,
):
    """
    Gradient-based inverse design of macro variables:

    We treat M_macro as parameters and adjust them to make the
    model's portfolio utility approach the target_utility.

    Energy per batch element:
      E_i(M) = (U_i(M) - target)^2 + prior_lambda * ||M_i||^2.
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad = False  # freeze model during scenario search

    M = M_init.clone().detach().requires_grad_(True)
    history = []

    print(f"--- Scenario Search (Target Utility: {target_utility:.4f}) ---")
    print(f"Steps: {n_steps}, step_size: {step_size}, noise_scale: {noise_scale}")

    for step in range(n_steps):
        if M.grad is not None:
            M.grad.zero_()

        # 1) Rebuild feature tensor X(M)
        X_gen = reconstructor(Z_firm_fixed, M)  # (B, K, F_total)

        # 2) Forward through the model (no_grad for weights, but M requires grad)
        w_star, mu_hat = model(X_gen, Sigma_fixed)  # w_star: (B,K), mu_hat: (B,K)

        # 3) Compute portfolio utility per batch element
        port_ret = torch.sum(w_star * mu_hat, dim=1)                     # (B,)
        port_risk = torch.einsum("bi,bij,bj->b", w_star, Sigma_fixed, w_star)
        current_util = port_ret - (model.gamma / 2.0) * port_risk        # (B,)

        # 4) Energy: mismatch to target + quadratic prior on M
        mse_loss = (current_util - target_utility) ** 2                  # (B,)
        prior_loss = prior_lambda * torch.sum(M ** 2, dim=1)             # (B,)
        energy = mse_loss + prior_loss                                   # (B,)

        # 5) Backprop: need scalar, so sum across batch
        energy.sum().backward()

        with torch.no_grad():
            grad = M.grad
            noise = torch.randn_like(M) * noise_scale

            # Langevin-style update: M_new = M - step * grad + noise
            M_new = M - step_size * grad + noise
            M.copy_(M_new)

        # Track mean utility for logging / plotting
        history.append(current_util.mean().item())

        if (step + 1) % 50 == 0:
            print(f"Step {step+1:4d}/{n_steps} | mean U = {history[-1]:.4f}")

    return M.detach(), history


# Grab a seed batch from validation loader
batch = next(iter(val_loader))
X_seed = batch["X"].to(DEVICE)          # (B, K, F_total)
Sigma_seed = batch["Sigma"].to(DEVICE)  # (B, K, K)

B, K, F_total = X_seed.shape
print(f"Seed batch shapes: X={X_seed.shape}, Sigma={Sigma_seed.shape}")

# Infer N_FIRM from total feature dimension and N_MACRO
#     We assume: F_total = N_FIRM * (1 + N_MACRO)
N_FIRM = F_total // (1 + N_MACRO)
assert N_FIRM * (1 + N_MACRO) == F_total, "Feature dimension is not divisible by (1 + N_MACRO). Check N_MACRO."

print(f"Inferred N_FIRM = {N_FIRM}, N_MACRO = {N_MACRO}")

# Build reconstructor
reconstructor = DifferentiableInteractionReconstructor(
    n_firm_features=N_FIRM,
    n_macro_features=N_MACRO,
).to(DEVICE)

# Extract base firm block directly from model inputs
# This ensures we are perfectly aligned with how the model was trained.
Z_firm_seed = X_seed[:, :, :N_FIRM]     # (B, K, N_FIRM)

# === Step 1: Identify month of seed batch ===
# Convert to clean integer YYYYMM
seed_date = 201905
print("Using seed month:", seed_date)

# === Step 2: Extract TRUE macro row for that month ===
real_macro_row = macro_final[macro_final["yyyymm"] == seed_date]

# If exact month doesn't exist → fallback to nearest available macro month
if real_macro_row.empty:
    print(f"No exact macro row for {seed_date}, searching nearest month...")
    # find index of closest month in macro_final
    nearest_idx = (macro_final["yyyymm"] - seed_date).abs().argmin()
    real_macro_row = macro_final.iloc[[nearest_idx]]
    print(f"   → Using nearest available month:", int(real_macro_row["yyyymm"].iloc[0]))

# Extract macro values as numpy vector
macro_cols = ["dp", "ep", "bm", "ntis", "tbl", "tms", "dfy", "svar", "infl"]
m_real_np = real_macro_row[macro_cols].values.astype(np.float32)[0]

# === Step 3: Build initial M tensor by repeating across batch ===
M_init = torch.tensor(
    np.tile(m_real_np, (B, 1)),      # (B, n_macro)
    dtype=torch.float32,
    device=DEVICE,
    requires_grad=True
)

print("Initial macro conditions:", M_init[0])


# Set crash target utility and run scenario search
TARGET_UTILITY = -0.15   # severe crash level

crash_M, crash_hist = run_gradient_scenario(
    model=model,
    reconstructor=reconstructor,
    Z_firm_fixed=Z_firm_seed,
    M_init=M_init,
    Sigma_fixed=Sigma_seed,
    target_utility=TARGET_UTILITY,
    n_steps=500,
    step_size=0.1,
    noise_scale=0.01,
    prior_lambda=0.1,
)



plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(crash_hist, label="Mean utility along trajectory")
plt.axhline(TARGET_UTILITY, color="red", linestyle="--", label="Target utility")
plt.xlabel("Step")
plt.ylabel("Portfolio utility")
plt.title("Scenario Generation Trajectory")
plt.grid(alpha=0.3)
plt.legend()

# Average macro shift across batch
macro_shifts = crash_M.mean(dim=0).cpu().numpy()
macro_names = ['dp', 'ep', 'bm', 'ntis', 'tbl', 'tms', 'dfy', 'svar', 'infl'][:N_MACRO]

plt.subplot(1, 2, 2)
colors = ["green" if x > 0 else "red" for x in macro_shifts]
plt.bar(macro_names, macro_shifts, color=colors)
plt.axhline(0.0, color="black", linewidth=0.8)
plt.title("Average Macro Shifts in Crash Scenario")
plt.ylabel("Macro shift (model units)")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("Mean Macro Shift Vector (per macro dim):", np.round(macro_shifts, 2))

# Sanity check: utilities before vs after scenario
with torch.no_grad():
    U_before = compute_portfolio_utility_batch(model, X_seed, Sigma_seed).mean().item()

    # Rebuild X under crash_M
    X_crash = reconstructor(Z_firm_seed, crash_M)
    U_after = compute_portfolio_utility_batch(model, X_crash, Sigma_seed).mean().item()

print(f"Mean utility before scenario: {U_before:.4f}")
print(f"Mean utility after scenario : {U_after:.4f}")
print(f"Target utility              : {TARGET_UTILITY:.4f}")
