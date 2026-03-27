"""
sigma_builder_pto_e2e.py

End-to-end, PTO/E2E-consistent Sigma (EWMA) + risk-factor U builder.

What it does:
  1) Loads 'metadata' from your ready_data via DataStorageEngine
  2) Patches metadata['excess_ret'] to match PTO/E2E scale & clipping
  3) Builds R_final: (yyyymm x permno) realized excess return matrix
  4) For a given decision month t and a candidate permno list:
       - extracts the window [t-lookback, ..., t-1]
       - enforces full-history (drops assets with any NaN in the window)
       - computes EWMA covariance Sigma (PTO/E2E)
       - shrink-to-diagonal + ridge + PSD projection
       - computes U such that ||U w||^2 = w' Sigma w (matches E2E)
  5) Returns (Sigma, U, used_assets)

Dependencies:
  - numpy, pandas
  - your PTO module must expose DataStorageEngine and make_psd
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from src.utils.helper_functions import make_psd_np
from src.modules.dataloaders import DataStorageEngine

from src.modules.pao_model_defs import PAOPortfolioModel
import torch
READY_DATA_DIR = "./Data/final_data"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def shift_yyyymm(yyyymm: int, k: int) -> int:
    y = yyyymm // 100
    m = yyyymm % 100
    idx = y * 12 + (m - 1) + k
    y2 = idx // 12
    m2 = idx % 12 + 1
    return int(y2 * 100 + m2)


def patch_excess_ret_inplace(metadata: pd.DataFrame, clip_lower: float = -0.99, col: str = "excess_ret") -> None:
    """
    Apply PTO/E2E-style target patching logic to metadata[excess_ret]:
      - If mean suggests percent scale, divide by 100
      - Clip lower tail at clip_lower
    """
    if col not in metadata.columns:
        raise ValueError(f"metadata missing column '{col}'")

    x = metadata[col].astype(float)
    mean_val = float(np.nanmean(x.values))

    # Same heuristic used in your pipeline for y_train
    if abs(mean_val) > 0.1:
        metadata[col] = x / 100.0
    else:
        metadata[col] = x

    metadata[col] = metadata[col].clip(lower=clip_lower)


def build_R_final(metadata: pd.DataFrame, id_col: str = "permno", ret_col: str = "excess_ret") -> pd.DataFrame:
    """
    Build (yyyymm x permno) realized return matrix R_final.

    Convention consistent with your pipeline:
      - Row index yyyymm corresponds to decision month t.
      - Entry is realized return over t+1 but indexed by decision month t.
    """
    need = ["yyyymm", id_col, ret_col]
    for c in need:
        if c not in metadata.columns:
            raise ValueError(f"metadata missing required column '{c}'")

    df = metadata[need].copy()
    df["yyyymm"] = df["yyyymm"].astype(int)
    df = df.dropna(subset=["yyyymm", id_col, ret_col])

    # Guard against duplicates; pivot() would fail.
    # Deterministic rule: keep first occurrence.
    df = df.drop_duplicates(subset=["yyyymm", id_col], keep="first")

    R = df.pivot_table(index="yyyymm", columns=id_col, values=ret_col, aggfunc="first").sort_index()
    return R.astype(np.float32)


def get_sigma_ewma_pto(
    R_final: pd.DataFrame,
    permnos: List[int],
    t: int,
    lookback: int = 60,
    lam: float = 0.94,
    shrink: float = 0.10,
    ridge: float = 1e-6,
) -> Tuple[np.ndarray, List[int]]:
    """
    Compute PTO/E2E-style EWMA covariance Sigma using window [t-lookback, ..., t-1].

    Enforces full-history eligibility: drop any asset with any NaN in the window.
    Returns:
      Sigma: (N,N) float64
      used_assets: list of permnos kept after filters, in the same order as Sigma axes
    """
    t = int(t)
    if t not in R_final.index:
        raise KeyError(f"t={t} not in R_final.index (min={R_final.index.min()}, max={R_final.index.max()})")

    start = shift_yyyymm(t, -lookback)
    end = shift_yyyymm(t, -1)

    R_win = R_final.loc[(R_final.index >= start) & (R_final.index <= end)]
    if R_win.shape[0] != lookback:
        raise ValueError(
            f"Window for t={t} has {R_win.shape[0]} rows, expected lookback={lookback}. "
            f"This usually means missing months in R_final index."
        )

    cols = [int(p) for p in permnos if p in R_win.columns]
    if len(cols) < 2:
        raise ValueError(f"Too few requested assets found in the window. Found={len(cols)}")

    R_sub = R_win[cols].dropna(axis=1, how="any")  # full-history filter
    X = R_sub.to_numpy(dtype=float)
    T, N = X.shape

    if N < 2:
        raise ValueError(f"After full-history filter, N={N} < 2 (too many NaNs in window).")

    # EWMA weights (matches your E2E ewma_cov_full_history_matrix)
    exponents = np.arange(T - 1, -1, -1)
    w = (1.0 - lam) * (lam ** exponents)
    w = w / np.sum(w)

    mu = np.sum(X * w[:, None], axis=0)
    Xc = X - mu[None, :]
    Sigma = (Xc.T @ (Xc * w[:, None]))
    Sigma = 0.5 * (Sigma + Sigma.T)

    # shrink-to-diagonal + ridge
    diag = np.diag(np.diag(Sigma))
    Sigma = (1.0 - shrink) * Sigma + shrink * diag
    Sigma = Sigma + ridge * np.eye(N)
    Sigma = 0.5 * (Sigma + Sigma.T)

    # PSD projection (matches PTO/E2E make_psd usage)
    Sigma = make_psd_np(Sigma)
    Sigma = 0.5 * (Sigma + Sigma.T)

    used_assets = [int(a) for a in R_sub.columns.tolist()]
    return Sigma.astype(np.float64), used_assets


def compute_sigma_factor_for_risk(Sigma: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Return U such that ||U @ w||^2 = w' Sigma w, i.e., U^T U = Sigma.
    Matches your E2E implementation.

    - Symmetrize Sigma
    - Add diagonal jitter eps
    - Try Cholesky => U = L^T
    - Fallback: symmetric sqrt via eigen decomposition
    """
    S = np.asarray(Sigma, dtype=np.float64)
    S = 0.5 * (S + S.T)
    S = S + eps * np.eye(S.shape[0], dtype=np.float64)

    try:
        L = np.linalg.cholesky(S)  # S = L L^T
        return (L.T).astype(np.float32)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(S)
        vals = np.maximum(vals, eps)
        sqrtS = vecs @ np.diag(np.sqrt(vals)) @ vecs.T
        sqrtS = 0.5 * (sqrtS + sqrtS.T)
        return sqrtS.astype(np.float32)


def build_sigma_and_U_from_ready_data(
    ready_data_dir: str,
    permnos: List[int],
    t: int,
    lookback: int = 60,
    lam: float = 0.94,
    shrink: float = 0.10,
    ridge: float = 1e-6,
    clip_lower: float = -0.99,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    One-call convenience wrapper:
      Loads metadata -> patches excess_ret -> builds R_final -> computes Sigma and U.

    Returns:
      Sigma (N,N) float64
      U     (N,N) float32
      used_assets list length N
    """
    storage = DataStorageEngine(storage_dir=ready_data_dir)
    loaded = storage.load_dataset()

    metadata = loaded["metadata"].copy()
    patch_excess_ret_inplace(metadata, clip_lower=clip_lower, col="excess_ret")

    R_final = build_R_final(metadata, id_col="permno", ret_col="excess_ret")

    Sigma, used_assets = get_sigma_ewma_pto(
        R_final=R_final,
        permnos=permnos,
        t=int(t),
        lookback=int(lookback),
        lam=float(lam),
        shrink=float(shrink),
        ridge=float(ridge),
    )

    U = compute_sigma_factor_for_risk(Sigma, eps=1e-8)
    return Sigma, U, used_assets


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    READY_DATA_DIR = "/content/drive/MyDrive/POE/ready_data"

    permnos = [10001, 10006, 10011, 10014, 10020]
    t = 201612  # decision month

    Sigma, U, used_assets = build_sigma_and_U_from_ready_data(
        ready_data_dir=READY_DATA_DIR,
        permnos=permnos,
        t=t,
        lookback=60,
        lam=0.94,
        shrink=0.10,
        ridge=1e-6,
        clip_lower=-0.99,
    )

    print("Sigma shape:", Sigma.shape)
    print("U shape    :", U.shape)
    print("Assets used:", used_assets)
    print("Check (U^T U) approx Sigma:",
          float(np.max(np.abs((U.T @ U).astype(np.float64) - Sigma))))


    
# --- 2. LOAD THE DATA ---

# A. Load
# Change Path


storage = DataStorageEngine(storage_dir="./Data/final_data", load_train=False)
data = storage.load_dataset()



def construct_C(
    model   : PAOPortfolioModel,
    df      : pd.DataFrame, #Considered to contain all the interaction terms
    meta_df : pd.DataFrame,
    date    : int, #in yyyymm format
    K       : int, # the firm characteristics of top/bottom K predictions are given 
    bestK   : bool = True #otherwise we take the worst K
    ):
    dd = df[meta_df['yyyymm']==date] #date data
    dm = meta_df[meta_df['yyyymm']==date] #date meta, so that we capture firm ids
    dd_tensor = torch.tensor(dd.to_numpy())
    with torch.no_grad():
        predictions = model.predictor(dd_tensor)
    top3Kindices = torch.argsort(predictions, descending=bestK)[:3*K]
    firm_ids = dm.iloc[top3Kindices]['permno']
    firm_rets = dm.iloc[top3Kindices]['excess_ret']
    permnos = [int(a) for a in firm_ids]
    firm_chars = dd_tensor[top3Kindices][:, :140]
    Sigma, U, used_assets = build_sigma_and_U_from_ready_data(
        ready_data_dir=READY_DATA_DIR,
        permnos=permnos,
        t=date,
        lookback=60,
        lam=0.94,
        shrink=0.10,
        ridge=1e-6,
        clip_lower=-0.99,
    )
    positions = [i for (i,val) in enumerate(firm_ids) if val in used_assets]
    C_t = firm_chars[positions[:K]]
    rets_t = torch.tensor(firm_rets.iloc[positions].iloc[:K].to_numpy())
    return torch.tensor(Sigma[:K,:K], dtype = torch.float32), C_t, rets_t, used_assets[:K]

from typing import Dict, Union

def construct_C2(
    data : Dict[str, pd.DataFrame], #Considered to contain all the interaction terms
    date    : Union[float, int],
    K       : int
):
    LB = 60 #lookback
    meta_df = data['metadata']
    meta_df['_orig_index'] = meta_df.index
    g = meta_df[meta_df['yyyymm'].between(shift_yyyymm(date, -LB), shift_yyyymm(date,-1))].groupby('permno')['excess_ret']

    valid_today = meta_df.loc[meta_df['yyyymm'] == date, 'permno']

    permnos = (
        g.mean()
        .where(g.count() == LB)              # full lookback
        .loc[lambda x: x.index.isin(valid_today)]  # exists at date
        .sort_values(ascending=False, na_position='last')
        .head(K).index
    )
    
    meta_df_pn = meta_df.set_index('permno')
    max_returners = meta_df_pn[meta_df_pn['yyyymm']==date].loc[permnos]
    df = data['X_test']
    dd = df.loc[max_returners['_orig_index']]
    
    Sigma, U, used_assets = build_sigma_and_U_from_ready_data(
        ready_data_dir=READY_DATA_DIR,
        permnos=permnos,
        t=date,
        lookback=LB,
        lam=0.94,
        shrink=0.10,
        ridge=1e-6,
        clip_lower=-0.99,
    )
    assert len(used_assets)== K, "number of used assets is different"

    dd_tensor = torch.tensor(dd.to_numpy())
    C_t = dd_tensor[:, :140]
    rets_t = torch.tensor(max_returners['excess_ret'].to_numpy())
    return torch.tensor(Sigma, dtype = torch.float32), C_t, rets_t, permnos.tolist()

    
    