# ============================================================
# Predict-Then-Optimize (PTO) Backtest Pipeline (Clean Version)
# - Loads data from parquet
# - Loads trained FNN model
# - Generates OOS predictions on test (2016-2024)
# - Runs MVO + Robust MVO (kappa grid, omega modes)
# - Produces tables + plots (excess + optional total)
# ============================================================

# ----------------------------
# 0) Environment / Imports
# ----------------------------
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import cvxpy as cp


# ----------------------------
# 1) Global Config 
# ----------------------------
# Data paths
DATA_DIR = "/content/drive/MyDrive/POE/ready_data"

# Saved FNN model folder
FNN_DIR = "/content/drive/MyDrive/POE/models/fnn_v1"  # contains model_config.json, feature_columns.json, state_dict.pt

# Time split cutoffs 
TRAIN_END = 200512
VAL_END   = 201512  # test starts 201601

# Portfolio / backtest parameters
TOPK = 200
PRESELECT_FACTOR = 3
LOOKBACK = 60

LAM = 0.94
SHRINK = 0.10
RIDGE = 1e-6

GAMMA = 5.0  # risk aversion (MVO + robust both use this)

# Robustness grids
OMEGA_MODES = ["diagSigma", "identity"]
KAPPA_GRID  = [0.0, 0.1, 0.5, 1.0, 10.0]

# Rolling Sharpe window
ROLL = 36

# Inference
BATCH_SIZE_PRED = 512


# ============================================================
# 2) Data Loading + Target Patching
# ============================================================
class DataStorageEngine:
    def __init__(self, storage_dir=DATA_DIR):
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
            if key.startswith("y_"):
                loaded_dict[key] = df.iloc[:, 0]
            else:
                loaded_dict[key] = df
        return loaded_dict


def patch_targets_inplace(data: dict, clip_lower=-0.99):
    """
      - if mean magnitude suggests percent scale -> /100 (kept as you had it)
      - clip at -0.99
    """
    print("\n--- [Patch] Targets ---")
    train_mean = float(data["y_train"].mean())
    if abs(train_mean) > 0.1:
        print(f">> Detected Percentage Scale (Mean={train_mean:.2f}). Dividing all targets by 100.")
        for key in ["y_train", "y_val", "y_test"]:
            data[key] = data[key] / 100.0
    else:
        print(f">> Detected Decimal Scale (Mean={train_mean:.4f}). No scaling needed.")

    for key in ["y_train", "y_val", "y_test"]:
        pre_min = float(data[key].min())
        data[key] = data[key].clip(lower=clip_lower)
        post_min = float(data[key].min())
        print(f"{key}: Min clipped from {pre_min:.4f} to {post_min:.4f}")

    print("\nTarget stats after patch:")
    print(data["y_train"].describe()[["mean", "min", "max", "std"]])


def detect_id_col(df: pd.DataFrame) -> str:
    candidates = ["permno", "PERMNO", "asset_id", "id", "ticker", "TICKER", "gvkey", "GVKEY", "cusip", "CUSIP"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not detect an asset id column. Expected one of: {candidates}")


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


# ============================================================
# 3) Model Definition + Loader (FNN)
# ============================================================
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


def load_fnn_from_dir(load_dir: str):
    cfg_path = os.path.join(load_dir, "model_config.json")
    cols_path = os.path.join(load_dir, "feature_columns.json")
    state_path = os.path.join(load_dir, "state_dict.pt")

    for p in [cfg_path, cols_path, state_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    with open(cols_path, "r") as f:
        feature_cols = json.load(f)

    state = torch.load(state_path, map_location="cpu")

    model = AssetPricingFNN(input_dim=int(cfg["input_dim"]), dropout_rate=float(cfg["dropout_rate"]))
    model.load_state_dict(state)
    model.eval()

    print("Loaded FNN model.")
    print("Config:", cfg)
    print("Feature cols:", len(feature_cols))
    return model, feature_cols, cfg


def align_features(X_df: pd.DataFrame, feature_cols: list):
    missing = [c for c in feature_cols if c not in X_df.columns]
    extra = [c for c in X_df.columns if c not in feature_cols]
    if missing:
        raise ValueError(f"Missing {len(missing)} columns, e.g. {missing[:10]}")
    X_aligned = X_df.loc[:, feature_cols]
    return X_aligned, extra


@torch.no_grad()
def predict_in_batches(model: nn.Module, X: pd.DataFrame, batch_size=512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Prediction device: {device}")

    model = model.to(device)
    model.eval()

    Xt = torch.tensor(X.values, dtype=torch.float32)
    ds = torch.utils.data.TensorDataset(Xt)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)

    preds = []
    for (bx,) in dl:
        bx = bx.to(device)
        out = model(bx).detach().cpu().numpy().reshape(-1)
        preds.append(out)

    pred = np.concatenate(preds, axis=0)
    return pred


def r2_oos_zero(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    sse_model = np.sum((y_true - y_pred) ** 2)
    sse_zero  = np.sum((y_true) ** 2)
    return float(1.0 - sse_model / sse_zero)


# ============================================================
# 4) Date Helpers + EWMA Covariance (full-history)
# ============================================================
def shift_yyyymm(yyyymm: int, k: int) -> int:
    """Shift yyyymm by k months (k can be negative)."""
    y = yyyymm // 100
    m = yyyymm % 100
    idx = y * 12 + (m - 1) + k
    y2 = idx // 12
    m2 = idx % 12 + 1
    return int(y2 * 100 + m2)


def yyyymm_to_dt(yyyymm: int) -> pd.Timestamp:
    y = int(yyyymm // 100)
    m = int(yyyymm % 100)
    return pd.Timestamp(year=y, month=m, day=1)


def make_psd(Sigma: np.ndarray, eps=1e-10) -> np.ndarray:
    """Project symmetric matrix to PSD by eigenvalue clipping."""
    S = 0.5 * (Sigma + Sigma.T)
    vals, vecs = np.linalg.eigh(S)
    vals = np.maximum(vals, eps)
    return (vecs * vals) @ vecs.T


def ewma_cov_full_history_matrix(R: np.ndarray, lam=0.94, shrink=0.10, ridge=1e-6, psd_proj=True):
    """
    EWMA covariance for a full-history matrix R (T x N) with NO NaNs.
    """
    R = np.asarray(R, float)
    if np.isnan(R).any():
        raise ValueError("ewma_cov_full_history_matrix expects NO NaNs (full-history).")

    T, N = R.shape
    if T < 2 or N < 2:
        return None

    # weights: newest gets most weight
    exponents = np.arange(T - 1, -1, -1)
    w = (1.0 - lam) * (lam ** exponents)
    w = w / np.sum(w)

    mu = np.sum(R * w[:, None], axis=0)
    Xc = R - mu[None, :]
    Sigma = (Xc.T @ (Xc * w[:, None]))
    Sigma = 0.5 * (Sigma + Sigma.T)

    # shrink to diagonal
    diag = np.diag(np.diag(Sigma))
    Sigma = (1.0 - shrink) * Sigma + shrink * diag

    # ridge + optional PSD projection
    Sigma = Sigma + ridge * np.eye(N)
    Sigma = 0.5 * (Sigma + Sigma.T)

    if psd_proj:
        Sigma = make_psd(Sigma, eps=1e-10)
        Sigma = 0.5 * (Sigma + Sigma.T)

    return Sigma


# ============================================================
# 5) Solvers: MVO (kappa=0) and Robust (kappa>0)
# ============================================================
def solve_mvo_long_only(mu: np.ndarray, Sigma: np.ndarray, gamma: float = 5.0):
    """
    max mu'w - (gamma/2) w' Sigma w
    s.t. sum w=1, w>=0
    """
    mu = np.asarray(mu, float).reshape(-1)
    N = mu.shape[0]
    if N < 2:
        return None

    Sigma = make_psd(np.asarray(Sigma, float), eps=1e-10)

    w = cp.Variable(N)
    obj = cp.Maximize(mu @ w - (gamma / 2.0) * cp.quad_form(w, Sigma))
    cons = [cp.sum(w) == 1, w >= 0]
    prob = cp.Problem(obj, cons)

    # QP -> OSQP is natural; fallback to ECOS
    try:
        prob.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-6, eps_rel=1e-6, max_iter=20000)
    except Exception:
        prob.solve(solver=cp.ECOS, verbose=False)

    if w.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
        return None

    w_hat = np.asarray(w.value).reshape(-1)
    w_hat = np.clip(w_hat, 0, None)
    s = w_hat.sum()
    return (w_hat / s) if s > 0 else None


def solve_robust_longonly(
    mu_hat: np.ndarray,
    Sigma: np.ndarray,
    gamma: float,
    kappa: float,
    omega_mode: str = "diagSigma",  # "diagSigma" or "identity"
    solver_chain=("CLARABEL", "SCS", "ECOS"),
):
    """
    max mu'w - kappa||A w||_2 - (gamma/2) w' Sigma w
    s.t. sum w=1, w>=0

    omega_mode:
      - "diagSigma": A = diag(vol), vol_i = sqrt(Sigma_ii)
      - "identity" : A = I
    """
    mu_hat = np.asarray(mu_hat, float).reshape(-1)
    N = mu_hat.shape[0]
    if N < 2:
        return None

    Sigma = make_psd(np.asarray(Sigma, float), eps=1e-10)

    diagS = np.maximum(np.diag(Sigma), 0.0)
    vol = np.sqrt(np.maximum(diagS, 1e-12))

    if omega_mode == "diagSigma":
        A = np.diag(vol)
    elif omega_mode == "identity":
        A = np.eye(N)
    else:
        raise ValueError(f"Unknown omega_mode={omega_mode}")

    w = cp.Variable(N)
    obj = cp.Maximize(mu_hat @ w - kappa * cp.norm(A @ w, 2) - (gamma / 2.0) * cp.quad_form(w, Sigma))
    cons = [cp.sum(w) == 1, w >= 0]
    prob = cp.Problem(obj, cons)

    last_err = None
    for s in solver_chain:
        try:
            prob.solve(solver=s, warm_start=True, verbose=False)
            if w.value is not None and prob.status in ("optimal", "optimal_inaccurate"):
                w_hat = np.asarray(w.value).reshape(-1)
                w_hat = np.clip(w_hat, 0, None)
                ss = w_hat.sum()
                return (w_hat / ss) if ss > 0 else (np.ones(N) / N)
        except Exception as e:
            last_err = e

    # If everything fails, return None (caller will fallback)
    # print("Robust solver failed. Last error:", repr(last_err))
    return None


# ============================================================
# 6) PTO Backtest (MVO special case when kappa=0)
# ============================================================
def run_pto_backtest(
    df_results: pd.DataFrame,
    df_ret_all: pd.DataFrame,
    id_col: str,
    topk: int = 200,
    preselect_factor: int = 3,
    lookback: int = 60,
    lam: float = 0.94,
    shrink: float = 0.10,
    ridge: float = 1e-6,
    gamma: float = 5.0,
    kappa: float = 0.0,
    omega_mode: str = "diagSigma",
    min_assets: int = 2
):
    """
    df_results: test panel [yyyymm, id_col, pred_return, actual_return] (excess)
    df_ret_all: long history [yyyymm, id_col, ret] (excess)
    """

    need = {"yyyymm", id_col, "pred_return", "actual_return"}
    if not need.issubset(df_results.columns):
        raise ValueError(f"df_results must contain {need}, got {df_results.columns.tolist()}")

    need2 = {"yyyymm", id_col, "ret"}
    if not need2.issubset(df_ret_all.columns):
        raise ValueError(f"df_ret_all must contain {need2}, got {df_ret_all.columns.tolist()}")

    months = sorted(df_results["yyyymm"].astype(int).unique().tolist())

    perf_rows = []
    weight_rows = []

    for t in months:
        # 1) month cross-section
        df_t_all = df_results.loc[df_results["yyyymm"] == t, [id_col, "pred_return", "actual_return"]].dropna()
        if df_t_all.empty:
            continue

        preK = int(topk * preselect_factor)
        df_t_pre = df_t_all.sort_values("pred_return", ascending=False).head(preK)

        # 2) build history window [t-lookback, t-1]
        start = shift_yyyymm(int(t), -lookback)
        end   = shift_yyyymm(int(t), -1)

        df_hist = df_ret_all[(df_ret_all["yyyymm"] >= start) & (df_ret_all["yyyymm"] <= end)]
        df_hist = df_hist[df_hist[id_col].isin(df_t_pre[id_col].tolist())]

        # pivot (lookback x assets)
        R = df_hist.pivot(index="yyyymm", columns=id_col, values="ret").sort_index()

        # full-history eligibility: drop any asset with missing in window
        R_use = R.dropna(axis=1, how="any")
        eligible_assets = set(R_use.columns.tolist())

        # 3) pick final topk among eligible
        df_t = df_t_pre[df_t_pre[id_col].isin(eligible_assets)].copy()
        df_t = df_t.sort_values("pred_return", ascending=False).head(topk)

        assets = df_t[id_col].tolist()
        if len(assets) < min_assets:
            continue

        # align history matrix to final asset set (no NaNs by construction)
        R_final = R_use.reindex(columns=assets)
        if R_final.shape[0] < 2:
            continue

        Sigma = ewma_cov_full_history_matrix(
            R_final.to_numpy(dtype=float),
            lam=lam, shrink=shrink, ridge=ridge, psd_proj=True
        )
        if Sigma is None:
            # fallback equal-weight
            w_hat = np.ones(len(assets)) / len(assets)
            fallback = 1
        else:
            mu_hat = df_t["pred_return"].to_numpy(dtype=float)

            if float(kappa) == 0.0:
                w_hat = solve_mvo_long_only(mu_hat, Sigma, gamma=gamma)
            else:
                w_hat = solve_robust_longonly(mu_hat, Sigma, gamma=gamma, kappa=float(kappa), omega_mode=omega_mode)

            if w_hat is None:
                w_hat = np.ones(len(assets)) / len(assets)
                fallback = 1
            else:
                fallback = 0

        # realized excess return
        r_real = df_t["actual_return"].to_numpy(dtype=float)
        port_ret = float(np.dot(w_hat, r_real))

        # diagnostics
        hhi = float(np.sum(w_hat ** 2))
        n_eff = float(1.0 / hhi) if hhi > 0 else np.nan
        max_w = float(np.max(w_hat))
        active = int(np.sum(w_hat > 1e-12))

        perf_rows.append({
            "yyyymm": int(t),
            "n_assets": int(len(assets)),
            "port_ret": port_ret,
            "fallback": int(fallback),
            "HHI": hhi,
            "N_eff": n_eff,
            "max_w": max_w,
            "active": active,
        })

        for a, wv in zip(assets, w_hat):
            weight_rows.append({"yyyymm": int(t), id_col: a, "w": float(wv)})

    perf = pd.DataFrame(perf_rows).sort_values("yyyymm").reset_index(drop=True)
    weights = pd.DataFrame(weight_rows)
    return perf, weights


# ============================================================
# 7) Metrics + Plot Helpers (no transaction cost)
# ============================================================
def perf_stats_excess(r: np.ndarray) -> dict:
    """
      - mean_m, vol_m
      - annualized via *12 and *sqrt(12)
      - sharpe_a = mean_a/vol_a
      - cum_simple = final_wealth - 1 on (1+r)
      - max_drawdown on wealth path from (1+r)
    """
    r = np.asarray(r, float)
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return {}

    mean_m = float(r.mean())
    vol_m  = float(r.std(ddof=1))
    mean_a = float(mean_m * 12.0)
    vol_a  = float(vol_m * np.sqrt(12.0))
    sharpe_a = float(mean_a / vol_a) if vol_a > 0 else np.nan

    wealth = np.cumprod(1.0 + r)
    cum_simple = float(wealth[-1] - 1.0)

    peak = np.maximum.accumulate(wealth)
    dd = wealth / peak - 1.0
    max_dd = float(dd.min())

    return {
        "n_months": int(len(r)),
        "mean_m": mean_m,
        "vol_m": vol_m,
        "mean_a": mean_a,
        "vol_a": vol_a,
        "sharpe_a": sharpe_a,
        "cum_simple": cum_simple,
        "max_drawdown": max_dd,
        "hit_rate": float((r > 0).mean()),
        "worst_month": float(r.min()),
        "best_month": float(r.max()),
    }


def rolling_sharpe(series: pd.Series, window=36):
    m = series.rolling(window).mean()
    s = series.rolling(window).std(ddof=1)
    return (m * 12.0) / (s * np.sqrt(12.0))


def plot_wealth_paths(wealth_df: pd.DataFrame, title: str):
    plt.figure(figsize=(12, 5))
    for col in wealth_df.columns:
        if col in ["yyyymm", "date"]:
            continue
        plt.plot(wealth_df["date"], wealth_df[col].values, label=col)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Wealth (start = 1)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_rolling_sharpes(ret_df: pd.DataFrame, title: str, window=36):
    plt.figure(figsize=(12, 5))
    for col in ret_df.columns:
        if col in ["yyyymm", "date"]:
            continue
        rs = rolling_sharpe(ret_df[col].astype(float), window)
        plt.plot(ret_df["date"], rs.values, label=col)
    plt.axhline(0.0, linewidth=1.0)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(f"Rolling Sharpe (annualized), window={window}m")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 8) MAIN: Run Everything
# ============================================================
print("\n==============================")
print("A) Load data")
print("==============================")
storage = DataStorageEngine(storage_dir=DATA_DIR)
data = storage.load_dataset()

print("\n--- Shape Verification ---")
print(f"X_train: {data['X_train'].shape} | y_train: {data['y_train'].shape}")
print(f"X_val:   {data['X_val'].shape}   | y_val:   {data['y_val'].shape}")
print(f"X_test:  {data['X_test'].shape}   | y_test:  {data['y_test'].shape}")

patch_targets_inplace(data, clip_lower=-0.99)

print("\n==============================")
print("B) Strict metadata alignment")
print("==============================")
if "metadata" not in data:
    raise KeyError("Expected data['metadata'] from parquet store.")
meta_train_fixed, meta_val_fixed, meta_test_fixed = strict_metadata_alignment(data["metadata"], TRAIN_END, VAL_END)

print("Rows check:")
print("X_train rows:", len(data["X_train"]), "| meta_train:", len(meta_train_fixed))
print("X_val rows:  ", len(data["X_val"]),   "| meta_val:  ", len(meta_val_fixed))
print("X_test rows: ", len(data["X_test"]),  "| meta_test: ", len(meta_test_fixed))


if len(meta_train_fixed) != len(data["X_train"]):
    print(">> WARNING: meta_train mismatch; using slicing fallback.")
    full_meta = data["metadata"].copy()
    meta_train_fixed = full_meta.iloc[:len(data["X_train"])].copy()
    meta_val_fixed   = full_meta.iloc[len(data["X_train"]):len(data["X_train"])+len(data["X_val"])].copy()
    meta_test_fixed  = full_meta.iloc[len(data["X_train"])+len(data["X_val"]):].copy()

id_col = detect_id_col(meta_test_fixed)
print("Detected id_col:", id_col)

# Ensure yyyymm exists
if "yyyymm" not in meta_test_fixed.columns:
    raise ValueError("metadata must contain 'yyyymm'.")

# ============================================================
# C) Load FNN and predict OOS on test
# ============================================================
print("\n==============================")
print("C) Load FNN + Predict on Test")
print("==============================")
model, feature_cols, cfg = load_fnn_from_dir(FNN_DIR)

X_test_aligned, extra_cols = align_features(data["X_test"], feature_cols)
print("Aligned X_test shape:", X_test_aligned.shape, "| Extra dropped:", len(extra_cols))

pred_oos = predict_in_batches(model, X_test_aligned, batch_size=BATCH_SIZE_PRED)

# OOS R2 vs zero benchmark 
y_test_true = data["y_test"].values.astype(float)
r2 = r2_oos_zero(y_test_true, pred_oos)
print(f"R2_OOS (zero benchmark) on test: {r2:.4%}")

# Build df_results using meta_test_fixed 
df_results = meta_test_fixed[[id_col, "yyyymm"]].copy().reset_index(drop=True)
df_results["pred_return"] = pred_oos
df_results["actual_return"] = y_test_true

df_results["yyyymm"] = df_results["yyyymm"].astype(int)
print("df_results head:")
print(df_results.head())
print("df_results rows:", len(df_results), "| months:", df_results["yyyymm"].min(), "to", df_results["yyyymm"].max())

# Save predictions artifact 
df_results.to_parquet("oos_predictions_2016_2024.parquet", index=False)
print("Saved: oos_predictions_2016_2024.parquet")

# ============================================================
# D) Build realized return history df_ret_all (train+val+test)
# ============================================================
print("\n==============================")
print("D) Build df_ret_all (realized history)")
print("==============================")
df_ret_train = meta_train_fixed[[id_col, "yyyymm"]].copy()
df_ret_train["ret"] = data["y_train"].values.astype(float)

df_ret_val = meta_val_fixed[[id_col, "yyyymm"]].copy()
df_ret_val["ret"] = data["y_val"].values.astype(float)

df_ret_test = meta_test_fixed[[id_col, "yyyymm"]].copy()
df_ret_test["ret"] = data["y_test"].values.astype(float)

df_ret_all = pd.concat([df_ret_train, df_ret_val, df_ret_test], ignore_index=True)
df_ret_all["yyyymm"] = df_ret_all["yyyymm"].astype(int)

print(df_ret_all[["yyyymm", "ret"]].describe())

# ============================================================
# E) Run strategies: EqualWeight baseline + Robust grid
# ============================================================
print("\n==============================")
print("E) Backtests")
print("==============================")

# 1) EqualWeight TopK baseline (uses same selection & full-history eligibility)
def run_equal_weight_topk(df_results, df_ret_all, id_col, topk=200, preselect_factor=3, lookback=60):
    months = sorted(df_results["yyyymm"].astype(int).unique().tolist())
    rows = []

    for t in months:
        df_t_all = df_results.loc[df_results["yyyymm"] == t, [id_col, "pred_return", "actual_return"]].dropna()
        if df_t_all.empty:
            continue

        preK = int(topk * preselect_factor)
        df_t_pre = df_t_all.sort_values("pred_return", ascending=False).head(preK)

        start = shift_yyyymm(int(t), -lookback)
        end   = shift_yyyymm(int(t), -1)

        df_hist = df_ret_all[(df_ret_all["yyyymm"] >= start) & (df_ret_all["yyyymm"] <= end)]
        df_hist = df_hist[df_hist[id_col].isin(df_t_pre[id_col].tolist())]

        R = df_hist.pivot(index="yyyymm", columns=id_col, values="ret").sort_index()
        eligible = set(R.dropna(axis=1, how="any").columns.tolist())

        df_t = df_t_pre[df_t_pre[id_col].isin(eligible)].copy()
        df_t = df_t.sort_values("pred_return", ascending=False).head(topk)

        assets = df_t[id_col].tolist()
        if len(assets) < 2:
            continue

        w = np.ones(len(assets)) / len(assets)
        port_ret = float(np.dot(w, df_t["actual_return"].to_numpy(dtype=float)))

        hhi = float(np.sum(w ** 2))
        n_eff = float(1.0 / hhi) if hhi > 0 else np.nan

        rows.append({
            "yyyymm": int(t),
            "n_assets": int(len(assets)),
            "port_ret": port_ret,
            "fallback": 0,
            "HHI": hhi,
            "N_eff": n_eff,
            "max_w": float(w.max()),
            "active": int(np.sum(w > 1e-12)),
        })

    return pd.DataFrame(rows).sort_values("yyyymm").reset_index(drop=True)


# Run baseline
perf_eq = run_equal_weight_topk(
    df_results=df_results,
    df_ret_all=df_ret_all,
    id_col=id_col,
    topk=TOPK,
    preselect_factor=PRESELECT_FACTOR,
    lookback=LOOKBACK
)
print("EqualWeight months:", len(perf_eq))

# Run robust grid (includes kappa=0 -> MVO special case)
specs = {}
specs["EqualWeight_TopK"] = perf_eq

for omega_mode in OMEGA_MODES:
    for kappa in KAPPA_GRID:
        label = f"{omega_mode}_kappa_{kappa}"
        perf, wdf = run_pto_backtest(
            df_results=df_results,
            df_ret_all=df_ret_all,
            id_col=id_col,
            topk=TOPK,
            preselect_factor=PRESELECT_FACTOR,
            lookback=LOOKBACK,
            lam=LAM,
            shrink=SHRINK,
            ridge=RIDGE,
            gamma=GAMMA,
            kappa=float(kappa),
            omega_mode=omega_mode
        )
        specs[label] = perf

        # Save weights/perf (optional but submission-friendly)
        perf.to_parquet(f"perf_{label}.parquet", index=False)
        wdf.to_parquet(f"weights_{label}.parquet", index=False)

print("Completed all backtests.")


# ============================================================
# F) Summary Table (EXCESS returns)
# ============================================================
print("\n==============================")
print("F) Performance summary (excess)")
print("==============================")

summary_rows = []
for name, perf in specs.items():
    r = perf["port_ret"].to_numpy(dtype=float)
    s = perf_stats_excess(r)
    if not s:
        continue
    s["strategy"] = name
    summary_rows.append(s)

summary_df = pd.DataFrame(summary_rows)[
    ["strategy","mean_m","vol_m","mean_a","vol_a","sharpe_a","cum_simple","max_drawdown",
     "hit_rate","worst_month","best_month","n_months"]
].sort_values("strategy")

print(summary_df.to_string(index=False))
summary_df.to_csv("results_summary_excess.csv", index=False)
print("Saved: results_summary_excess.csv")


# ============================================================
# G) Plots: Wealth + Rolling Sharpe (EXCESS)
# ============================================================
print("\n==============================")
print("G) Plots (excess)")
print("==============================")

# Align all series on common months (outer merge, then you can dropna if you want strict intersection)
all_months = sorted(df_results["yyyymm"].astype(int).unique().tolist())
dates = [yyyymm_to_dt(m) for m in all_months]

wealth_excess = pd.DataFrame({"yyyymm": all_months, "date": dates})
rets_excess   = pd.DataFrame({"yyyymm": all_months, "date": dates})

for name, perf in specs.items():
    tmp = perf.set_index("yyyymm").reindex(all_months)
    r = tmp["port_ret"].to_numpy(dtype=float)
    wealth = np.cumprod(1.0 + np.nan_to_num(r, nan=0.0))  # if missing months, treat as 0 for plotting continuity
    wealth_excess[name] = wealth
    rets_excess[name] = r

plot_wealth_paths(
    wealth_excess,
    title="Wealth (EXCESS returns): EqualWeight vs Robust PTO (kappa grid, omega modes)"
)
plot_rolling_sharpes(
    rets_excess,
    title=f"Rolling {ROLL}m Sharpe (EXCESS returns): EqualWeight vs Robust PTO",
    window=ROLL
)

wealth_excess.to_csv("wealth_paths_excess_all_specs.csv", index=False)
rets_excess.to_csv("returns_excess_all_specs.csv", index=False)
print("Saved: wealth_paths_excess_all_specs.csv, returns_excess_all_specs.csv")


# ============================================================
# H) Optional: Total Returns using Rfree (if macro_final exists)
# ============================================================
print("\n==============================")
print("H) Optional total-return wealth using Rfree")
print("==============================")

if "macro_final" in data and ("Rfree" in data["macro_final"].columns) and ("yyyymm" in data["macro_final"].columns):
    rf = data["macro_final"][["yyyymm","Rfree"]].copy()
    rf["yyyymm"] = rf["yyyymm"].astype(int)
    rf = rf.rename(columns={"Rfree":"rf"}).sort_values("yyyymm")

    rf_test = rf[rf["yyyymm"].isin(all_months)].copy()
    if len(rf_test) != len(all_months):
        missing_months = sorted(set(all_months) - set(rf_test["yyyymm"].tolist()))
        raise ValueError(f"Missing rf for months: {missing_months[:10]} (count={len(missing_months)})")

    rf_map = dict(zip(rf_test["yyyymm"].tolist(), rf_test["rf"].astype(float).tolist()))
    rf_series = np.array([rf_map[m] for m in all_months], dtype=float)

    wealth_total = pd.DataFrame({"yyyymm": all_months, "date": dates})
    rets_total   = pd.DataFrame({"yyyymm": all_months, "date": dates})

    for name, perf in specs.items():
        tmp = perf.set_index("yyyymm").reindex(all_months)
        rex = tmp["port_ret"].to_numpy(dtype=float)
        rtot = np.nan_to_num(rex, nan=0.0) + rf_series
        wealth_total[name] = np.cumprod(1.0 + rtot)
        rets_total[name] = rtot

    plot_wealth_paths(
        wealth_total,
        title="Wealth (TOTAL returns = Rfree + excess): EqualWeight vs Robust PTO"
    )
    plot_rolling_sharpes(
        rets_total,
        title=f"Rolling {ROLL}m Sharpe (TOTAL returns): EqualWeight vs Robust PTO",
        window=ROLL
    )

    wealth_total.to_csv("wealth_paths_total_all_specs.csv", index=False)
    rets_total.to_csv("returns_total_all_specs.csv", index=False)
    print("Saved: wealth_paths_total_all_specs.csv, returns_total_all_specs.csv")
else:
    print("macro_final with columns ['yyyymm','Rfree'] not found. Skipping total-return plots.")
