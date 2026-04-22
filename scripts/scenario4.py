import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import Dict, List, Optional
import warnings
import time

import numpy as np
import pandas as pd
import torch

from end2endportfolio.src import langevin
from src.modules.probe_eval import (
    A,
    Sigma_inv,
    c,
    AllocationPipeline,
    G_contrast_function,
    evaluate,
)
from src.modules.runtime_model_loader import load_exact_e2e_model_from_run
from src.modules.runtime_regime import MACRO_ORDER, regime_classifier
from src.utils.helper_functions import sqrt_decay
from src.data.macro_scaler import MacroScaler
from src.modules.runtime_components import FeatureScenarioBuilder

_RUNTIME = ROOT / "runtime_universe500"
_macro_scaler = MacroScaler.load(_RUNTIME / "scaler")
macro_mean: torch.Tensor = _macro_scaler.mean  # shape [9], float32
macro_std:  torch.Tensor = _macro_scaler.std   # shape [9], float32

# ── Universe500 runtime data (firm features + returns) ────────────────────────
_DATA = _RUNTIME / "data"
_meta_all  = pd.read_parquet(_DATA / "metadata.parquet")
_meta_test = pd.read_parquet(_DATA / "metadata_test.parquet")
_X_test    = pd.read_parquet(_DATA / "X_test.parquet")
_meta_all["yyyymm"]  = _meta_all["yyyymm"].astype(int)
_meta_test["yyyymm"] = _meta_test["yyyymm"].astype(int)
_feature_names      = tuple(np.load(_DATA / "feature_names.npy",      allow_pickle=True).tolist())
_firm_feature_names = tuple(np.load(_DATA / "firm_feature_names.npy", allow_pickle=True).tolist())
_macro_predictors   = tuple(np.load(_DATA / "macro_predictors.npy",   allow_pickle=True).tolist())
_macro_final_21a    = pd.read_parquet(_DATA / "macro_final.parquet")
_macro_final_21a["yyyymm"] = _macro_final_21a["yyyymm"].astype(int)


def _ewma_covariance(
    returns_matrix: np.ndarray,
    ewma_lambda: float = 0.94,
    shrinkage: float = 0.10,
    ridge: float = 1e-6,
) -> np.ndarray:
    from src.utils.helper_functions import make_psd_np
    n = returns_matrix.shape[0]
    exp = np.arange(n - 1, -1, -1)
    w = (1.0 - ewma_lambda) * (ewma_lambda ** exp)
    w /= w.sum()
    mu = np.sum(returns_matrix * w[:, None], axis=0)
    c_ = returns_matrix - mu
    sigma = c_.T @ (c_ * w[:, None])
    sigma = 0.5 * (sigma + sigma.T)
    diag = np.diag(np.diag(sigma))
    sigma = (1.0 - shrinkage) * sigma + shrinkage * diag + ridge * np.eye(sigma.shape[0])
    sigma = make_psd_np(sigma)
    return 0.5 * (sigma + sigma.T)


def construct_C2_21a(date: int, K: int):
    from src.utils.dates import shift_yyyymm
    lookback = 60
    # Select K permnos with full 60-month return history ending before date
    hist = _meta_all[
        _meta_all["yyyymm"].between(shift_yyyymm(date, -lookback), shift_yyyymm(date, -1))
    ].groupby("permno")["excess_ret"]
    valid_today = _meta_test.loc[_meta_test["yyyymm"] == date, "permno"]
    permnos = (
        hist.mean()
        .where(hist.count() == lookback)
        .loc[lambda s: s.index.isin(valid_today)]
        .sort_values(ascending=False, na_position="last")
        .head(K)
        .index.tolist()
    )
    if len(permnos) != K:
        raise ValueError(f"construct_C2_21a: got {len(permnos)} assets for {date}, expected {K}.")

    # EWMA covariance from return history
    hist_wide = (
        _meta_all[
            (_meta_all["yyyymm"] >= shift_yyyymm(date, -lookback))
            & (_meta_all["yyyymm"] <= shift_yyyymm(date, -1))
            & (_meta_all["permno"].isin(permnos))
        ]
        .pivot(index="yyyymm", columns="permno", values="excess_ret")
        .sort_index()[permnos]
        .dropna(axis=1, how="any")
    )
    sigma = _ewma_covariance(hist_wide.to_numpy(dtype=float))
    Sigma = torch.tensor(sigma.astype(np.float32), dtype=torch.float32)

    # Firm features at date from X_test (no FeatureScaler — 21April models don't use one)
    today_rows = _meta_test[(_meta_test["yyyymm"] == date) & (_meta_test["permno"].isin(permnos))]
    today_rows = today_rows.drop_duplicates("permno")
    # Map permno → original integer row index shared with X_test
    perm_to_row = {int(row["permno"]): idx for idx, row in today_rows.iterrows()}
    orig_indices = [perm_to_row[p] for p in permnos]

    feature_frame = _X_test.loc[orig_indices, list(_firm_feature_names)].copy()
    feature_frame.index = range(len(feature_frame))  # reset for FeatureScenarioBuilder

    C_t = FeatureScenarioBuilder(
        feature_frame=feature_frame,
        feature_names=_feature_names,
        macro_predictors=_macro_predictors,
        scaler=None,       # 21April models trained without FeatureScaler
        macro_scaler=None, # m is already standardized when passed from the chain
    )

    rets_t = torch.tensor(
        today_rows.set_index("permno").loc[permnos, "excess_ret"].to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )
    return Sigma, C_t, rets_t, permnos

warnings.filterwarnings(
    "ignore",
    message="Converting A to a CSC \\(compressed sparse column\\) matrix; may take a while\\.",
    category=UserWarning,
)


def print_traj(m_traj: torch.Tensor):
    # un-standardize so printed values are in raw macro units
    raw = np.stack([_unstd(m) for m in m_traj])
    df = pd.DataFrame(raw, columns=MACRO_ORDER)
    print(df.describe())
    return df


ASSET_SIZE = 30

DATE = 202004

LAMBDA = 10.0

KAPPA = 0.5 

GAMMA = 0.5 # controls the strength of the contrast in the G function; higher gamma → stronger contrast, more aggressive moves to increase the return gap (for "distinct_return" contrast)

L2REG = 0.05

ETA = 0.005   # initial eta; effective step = ETA/BETA = 0.005/20 = 2.5e-4 (in calibrated range)

BETA = 20.0
N_STEPS = 500
RANDOM_START_SCALE = 0.005
ADAPT_STEPS = 80
TARGET_ACCEPT = 0.57
ETA_MIN = 0.0005  # floor: effective step = 2.5e-5
ETA_MAX = 0.1000  # ceiling: effective step = 5e-3; wider range so adapter isn't capped
ADAPT_GAIN = 1.0

# ── Regularizer mode (switch to change scenario story) ────────────────────────
# "var1" : penalize distance from VAR(1) one-step prediction A@anchor+c
#           Story: "find plausible next-month states from the anchor that maximise SC-WW gap"
# "l2"   : penalize squared L2 distance from the anchor itself
#           Story: "find nearby variants of the anchor month that flip the SC-WW outcome"
REG_MODE = "var1"


# SC trained on stable months, WW trained on stress months
run_dir_summer_child = _RUNTIME / "models/summer_child"
run_dir_winter_wolf  = _RUNTIME / "models/winter_wolf"
summer_model, summer_cfg = load_exact_e2e_model_from_run(run_dir_summer_child)
winter_model, winter_cfg = load_exact_e2e_model_from_run(run_dir_winter_wolf)

Sigma, C_t, rets_t, permnos = construct_C2_21a(DATE, ASSET_SIZE)

summer_pi = AllocationPipeline(summer_model, Sigma)
winter_pi = AllocationPipeline(winter_model, Sigma)

macro_df: pd.DataFrame = _macro_final_21a
macro_columns = list(MACRO_ORDER)
# Standardize anchor: MALA operates in standardized macro space (mirror configured_runner.py)
m0_raw = torch.tensor(
    macro_df.loc[macro_df["yyyymm"] == DATE, macro_columns].to_numpy(dtype=np.float32, copy=True)[0],
    dtype=torch.float32,
)
m0 = (m0_raw - macro_mean) / macro_std  # standardized anchor

# Bounds also in standardized space
macro_arr = torch.tensor(macro_df[macro_columns].to_numpy(dtype=np.float32), dtype=torch.float32)
macro_std_hist = macro_arr.std(dim=0)
macro_min_std  = (macro_arr.min(dim=0).values - macro_mean) / macro_std
macro_max_std  = (macro_arr.max(dim=0).values - macro_mean) / macro_std
a = macro_min_std - macro_std_hist / macro_std
b = macro_max_std + macro_std_hist / macro_std

# m is already standardized — do NOT pass macro_mean/macro_std (would double-standardize)
results_summer, w_summer = evaluate(m0, C_t, rets_t, Sigma, summer_pi)
results_winter, w_winter = evaluate(m0, C_t, rets_t, Sigma, winter_pi)

return_summer = results_summer[0].item()
sharpe_summer = results_summer[2].item()
return_winter = results_winter[0].item()
sharpe_winter = results_winter[2].item()


def _build_reg_fn(mode: str):
    if mode == "var1":
        return None  # G_contrast_function builds VAR(1) mahalonobis internally
    if mode == "l2":
        def _l2(m: torch.Tensor) -> torch.Tensor:
            diff = m - m0  # penalize distance from anchor in standardized space
            return diff @ diff
        return _l2
    raise ValueError(f"Unknown REG_MODE: {mode!r}")


G, gradG = G_contrast_function(
    summer_pi,
    winter_pi,
    C_t,
    rets_t,
    "distinct_return",
    anchor=m0,
    l2reg=L2REG,
    reg_fn=_build_reg_fn(REG_MODE),
    gamma=GAMMA,
)

def mahalanobis_squared(x: torch.Tensor, anchor: torch.Tensor) -> float:
    """Always reports VAR(1) Mahalanobis for interpretability, regardless of REG_MODE."""
    diff = x.detach().float() - (A @ anchor.detach().float() + c)
    return float((diff @ (Sigma_inv @ diff)).item())


def l2_squared(x: torch.Tensor, anchor: torch.Tensor) -> float:
    diff = x.detach().float() - anchor.detach().float()
    return float((diff @ diff).item())


def regime_summary(macro_state: torch.Tensor) -> Dict[str, object]:
    # regime_classifier expects raw macro — un-standardize first
    raw = _unstd(macro_state)
    result = regime_classifier.classify(raw)
    return {
        "label": result["label"],
        "probabilities": result["probabilities"],
    }


def format_regime(regime_info: Dict[str, object]) -> str:
    probs = regime_info["probabilities"]
    assert isinstance(probs, dict)
    ordered = sorted(probs.items(), key=lambda item: item[1], reverse=True)
    top = ", ".join(f"{name}={value:.2%}" for name, value in ordered[:2])
    return f"{regime_info['label']} ({top})"


def evaluate_pair(macro_state: torch.Tensor) -> Dict[str, float]:
    # macro_state is standardized — no macro_mean/macro_std needed
    summer_results, _ = evaluate(macro_state, C_t, rets_t, Sigma, summer_pi)
    winter_results, _ = evaluate(macro_state, C_t, rets_t, Sigma, winter_pi)
    summer_return = float(summer_results[0].item())
    winter_return = float(winter_results[0].item())
    summer_sharpe = float(summer_results[2].item())
    winter_sharpe = float(winter_results[2].item())
    winner = "summer_child" if summer_return >= winter_return else "winter_wolf"
    return {
        "summer_return": summer_return,
        "winter_return": winter_return,
        "summer_sharpe": summer_sharpe,
        "winter_sharpe": winter_sharpe,
        "return_gap": summer_return - winter_return,
        "sharpe_gap": summer_sharpe - winter_sharpe,
        "winner": winner,
    }


def effective_sample_size(series: np.ndarray) -> float:
    values = np.asarray(series, dtype=np.float64)
    n = values.shape[0]
    if n < 3:
        return float(n)
    centered = values - values.mean()
    variance = np.dot(centered, centered) / n
    if not np.isfinite(variance) or variance <= 1e-12:
        return float(n)
    rho_sum = 0.0
    max_lag = min(n - 1, max(1, n // 2))
    for lag in range(1, max_lag + 1):
        rho = np.dot(centered[:-lag], centered[lag:]) / ((n - lag) * variance)
        if not np.isfinite(rho) or rho <= 0:
            break
        rho_sum += rho
    ess = n / (1.0 + 2.0 * rho_sum)
    return float(max(1.0, min(float(n), ess)))


def summarize_ess(traj: torch.Tensor, accepted_flags: List[bool]) -> Dict[str, float]:
    values = traj.detach().cpu().numpy()
    raw_ess = np.asarray([effective_sample_size(values[:, idx]) for idx in range(values.shape[1])], dtype=np.float64)
    state_change_count = 1 + int(sum(bool(flag) for flag in accepted_flags))
    ess_values = np.minimum(raw_ess, float(state_change_count))
    return {
        "accepted_moves": float(sum(bool(flag) for flag in accepted_flags)),
        "state_change_count": float(state_change_count),
        "raw_ess_min": float(raw_ess.min()),
        "raw_ess_mean": float(raw_ess.mean()),
        "raw_ess_max": float(raw_ess.max()),
        "ess_min": float(ess_values.min()),
        "ess_mean": float(ess_values.mean()),
        "ess_max": float(ess_values.max()),
    }


def _unstd(t: torch.Tensor) -> np.ndarray:
    return (t.detach().float() * macro_std + macro_mean).cpu().numpy()

def macro_shift_frame(start: torch.Tensor, final: torch.Tensor) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "macro": macro_columns,
            "anchor": _unstd(m0),
            "start":  _unstd(start),
            "final":  _unstd(final),
        }
    )
    frame["shift_vs_anchor"] = frame["final"] - frame["anchor"]
    frame["shift_vs_start"]  = frame["final"] - frame["start"]
    return frame


def run_tuned_mala_chain(
    start: torch.Tensor,
    n_steps: int,
):
    eta_base = float(ETA)
    x = start.detach().clone().to(dtype=m0.dtype)
    traj = []
    accepted: List[bool] = []
    accept_probs: List[float] = []
    log_accept_ratios: List[float] = []
    eta_history: List[float] = []

    for step in range(n_steps):
        lr = (eta_base / BETA) * sqrt_decay(1.0)(step)
        hyps = (G, gradG, lr, BETA, a, b)
        x = x.detach().requires_grad_(True)
        x, step_info = langevin.torch_MALA_step(x, hyps, return_info=True)
        traj.append(x.detach())
        accepted.append(step_info["accepted"])
        accept_probs.append(step_info["accept_prob"])
        log_accept_ratios.append(step_info["log_accept_ratio"])
        eta_history.append(eta_base)

        if step < ADAPT_STEPS:
            gain = ADAPT_GAIN / np.sqrt(step + 1.0)
            eta_base *= float(np.exp(gain * (step_info["accept_prob"] - TARGET_ACCEPT)))
            eta_base = float(np.clip(eta_base, ETA_MIN, ETA_MAX))

    info = {
        "accepted": accepted,
        "accept_probs": accept_probs,
        "log_accept_ratios": log_accept_ratios,
        "accept_rate": float(np.mean(accepted)) if accepted else 0.0,
        "mean_accept_prob": float(np.mean(accept_probs)) if accept_probs else 0.0,
        "n_steps": int(n_steps),
        "eta_init": float(ETA),
        "eta_final": float(eta_base),
        "eta_mean": float(np.mean(eta_history)) if eta_history else float(ETA),
        "eta_history": eta_history,
    }
    return x.detach(), torch.stack(traj), info


def print_run_header(n_seeds: int, n_steps: int) -> None:
    anchor_regime = regime_summary(m0)
    eval_summary = regime_classifier.eval_summary["test"]
    print("=" * 88)
    print("Scenario 4 | Universe500 root runtime")
    print(
        f"Question: find plausible macro states near {DATE} where SummerChild and WinterWolf "
        "produce distinct realized returns on the same 30-name Universe500 basket."
    )
    reg_desc = (
        f"VAR1-Mahalanobis²(m, A@anchor+c)"
        if REG_MODE == "var1"
        else f"L2²(m, anchor)"
    )
    print(
        f"Objective: exp(-100 * (r_SC - r_WW)^2) + {L2REG:.4g} * {reg_desc}  [REG_MODE={REG_MODE!r}]"
    )
    print(
        f"Chain: MALA, steps={n_steps}, seeds={n_seeds}, beta={BETA:.2f}, "
        f"eta0={ETA / BETA:.6f}, random_start_scale={RANDOM_START_SCALE:.4f}, "
        f"adapt_steps={ADAPT_STEPS}, target_accept={TARGET_ACCEPT:.2f}, clipped_to=[a,b]"
    )
    print(
        f"Models: SC={run_dir_summer_child.name} (lambda={summer_cfg['lambda']}, "
        f"kappa={summer_cfg['kappa']}), WW={run_dir_winter_wolf.name} "
        f"(lambda={winter_cfg['lambda']}, kappa={winter_cfg['kappa']})"
    )
    print(
        f"Anchor month {DATE}: SC return={return_summer:.2%}, SC Sharpe={sharpe_summer:.2f}, "
        f"WW return={return_winter:.2%}, WW Sharpe={sharpe_winter:.2f}"
    )
    print(f"Anchor classifier regime: {format_regime(anchor_regime)}")
    print(
        f"Regime classifier held-out test: accuracy={eval_summary['accuracy']:.3f}, "
        f"macro_f1={eval_summary['macro_f1']:.3f}, months={eval_summary['n_months']}"
    )
    print("=" * 88)


def print_seed_summary(seed_idx: int, summary: Dict[str, object]) -> None:
    print(
        f"[seed {seed_idx:02d}] accept_rate={summary['accept_rate']:.1%}, "
        f"mean_accept_prob={summary['mean_accept_prob']:.1%}, "
        f"eta(init/mean/final)={summary['eta_init']:.6f}/{summary['eta_mean']:.6f}/{summary['eta_final']:.6f}, "
        f"accepted_moves={int(summary['accepted_moves'])}, "
        f"state_changes={int(summary['state_change_count'])}, "
        f"ESS cap(min/mean/max)={summary['ess_min']:.1f}/{summary['ess_mean']:.1f}/{summary['ess_max']:.1f}"
    )
    print(
        f"[seed {seed_idx:02d}] start regime={format_regime(summary['start_regime'])}, "
        f"final regime={format_regime(summary['final_regime'])}"
    )
    print(
        f"[seed {seed_idx:02d}] VAR1-Mah: start={summary['start_mah_dist']:.3f} "
        f"(sq={summary['start_mah2']:.3f}), final={summary['final_mah_dist']:.3f} "
        f"(sq={summary['final_mah2']:.3f}) | "
        f"L2(std): start={summary['start_l2_dist']:.4f}, final={summary['final_l2_dist']:.4f}"
    )
    print(
        f"[seed {seed_idx:02d}] final winner={summary['winner']}, "
        f"return_gap(SC-WW)={summary['return_gap']:.4%}, "
        f"sharpe_gap(SC-WW)={summary['sharpe_gap']:.4f}"
    )
    frame = summary["macro_shift_frame"]
    assert isinstance(frame, pd.DataFrame)
    print(frame.to_string(index=False, float_format=lambda value: f"{value: .4f}"))
    print("-" * 88)


def main(
    n_seeds: int = 20,
    lasts: Optional[List[torch.Tensor]] = None,
    n_steps: int = N_STEPS,
):
    m_trajs_contrast = []
    m_lasts_contrast = []
    seed_summaries = []
    if lasts is not None:
        n_seeds = len(lasts)

    print_run_header(n_seeds=n_seeds, n_steps=n_steps)

    for i in range(n_seeds):
        if lasts is None:
            m_start = m0 + (b - a) * RANDOM_START_SCALE * torch.rand((9,), dtype=m0.dtype, device=m0.device)
        else:
            m_start = lasts[i].detach().clone().to(dtype=m0.dtype)
        print(f"[seed {i + 1:02d}] running contrastive trajectory from start state")
        seed_start_time = time.time()
        try:
            m_last, m_traj, chain_info = run_tuned_mala_chain(m_start, n_steps)
        except KeyboardInterrupt:
            elapsed = time.time() - seed_start_time
            print(f"[seed {i + 1:02d}] interrupted after {elapsed:.1f}s; returning completed seeds only")
            break
        m_lasts_contrast.append(m_last)
        m_trajs_contrast.append(m_traj)

        ess_summary = summarize_ess(m_traj, chain_info["accepted"])
        start_mah2 = mahalanobis_squared(m_start, m0)
        final_mah2 = mahalanobis_squared(m_last, m0)
        start_l2 = l2_squared(m_start, m0)
        final_l2 = l2_squared(m_last, m0)
        final_eval = evaluate_pair(m_last)
        summary = {
            "seed": i + 1,
            "start_regime": regime_summary(m_start),
            "final_regime": regime_summary(m_last),
            "start_mah2": start_mah2,
            "final_mah2": final_mah2,
            "start_mah_dist": start_mah2 ** 0.5,
            "final_mah_dist": final_mah2 ** 0.5,
            "start_l2": start_l2,
            "final_l2": final_l2,
            "start_l2_dist": start_l2 ** 0.5,
            "final_l2_dist": final_l2 ** 0.5,
            "accept_rate": chain_info["accept_rate"],
            "mean_accept_prob": chain_info["mean_accept_prob"],
            "eta_init": chain_info["eta_init"],
            "eta_mean": chain_info["eta_mean"],
            "eta_final": chain_info["eta_final"],
            "macro_shift_frame": macro_shift_frame(m_start, m_last),
            **ess_summary,
            **final_eval,
        }
        seed_summaries.append(summary)
        summary["elapsed_seconds"] = time.time() - seed_start_time
        print_seed_summary(i + 1, summary)

    return {
        "date": DATE,
        "gamma": LAMBDA,
        "kappa": KAPPA,
        "n_assets": ASSET_SIZE,
        "beta": BETA,
        "m_trajs_contrast": m_trajs_contrast,
        "m_lasts_contrast": m_lasts_contrast,
        "seed_summaries": seed_summaries,
    }


if __name__ == "__main__":
    main()
