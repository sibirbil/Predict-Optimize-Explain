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
from src.modules.runtime_paths import MODELS_DIR
from src.modules.runtime_regime import MACRO_ORDER, regime_classifier
from src.modules.runtime_universe500 import construct_C2, data
from src.utils.helper_functions import sqrt_decay

warnings.filterwarnings(
    "ignore",
    message="Converting A to a CSC \\(compressed sparse column\\) matrix; may take a while\\.",
    category=UserWarning,
)


def print_traj(m_traj: torch.Tensor):
    df = pd.DataFrame(m_traj.detach().cpu().numpy(), columns=MACRO_ORDER)
    print(df.describe())
    return df


ASSET_SIZE = 30
DATE = 202003
LAMBDA = 10.0
KAPPA = 1.0
L2REG = 0.002
ETA = 0.0015
BETA = 8.0
N_STEPS = 500
RANDOM_START_SCALE = 0.05


run_dir_summer_child = MODELS_DIR / "summer_child"
summer_model, summer_cfg = load_exact_e2e_model_from_run(run_dir_summer_child)

run_dir_winter_wolf = MODELS_DIR / "winter_wolf"
winter_model, winter_cfg = load_exact_e2e_model_from_run(run_dir_winter_wolf)

Sigma, C_t, rets_t, permnos = construct_C2(data, DATE, ASSET_SIZE)

summer_pi = AllocationPipeline(summer_model, Sigma)
winter_pi = AllocationPipeline(winter_model, Sigma)

macro_df: pd.DataFrame = data["macro_final"]
macro_columns = list(MACRO_ORDER)
m0 = torch.tensor(
    macro_df.loc[macro_df["yyyymm"] == DATE, macro_columns].to_numpy(dtype=np.float32, copy=True)[0],
    dtype=torch.float32,
)
a = torch.tensor(macro_df[macro_columns].min().to_numpy(dtype=np.float32) - macro_df[macro_columns].std().to_numpy(dtype=np.float32), dtype=torch.float32)
b = torch.tensor(macro_df[macro_columns].max().to_numpy(dtype=np.float32) + macro_df[macro_columns].std().to_numpy(dtype=np.float32), dtype=torch.float32)

results_summer, w_summer = evaluate(m0, C_t, rets_t, Sigma, summer_pi)
results_winter, w_winter = evaluate(m0, C_t, rets_t, Sigma, winter_pi)

return_summer = results_summer[0].item()
sharpe_summer = results_summer[2].item()
return_winter = results_winter[0].item()
sharpe_winter = results_winter[2].item()


G, gradG = G_contrast_function(
    summer_pi,
    winter_pi,
    C_t,
    rets_t,
    "distinct_return",
    anchor=m0,
    l2reg=L2REG,
)

hypsG = G, gradG, lambda t: (ETA / BETA) * sqrt_decay(1.0)(t), BETA


def mahalanobis_squared(x: torch.Tensor, anchor: torch.Tensor) -> float:
    A_t = torch.tensor(A, dtype=torch.float32)
    c_t = torch.tensor(c, dtype=torch.float32)
    sigma_inv_t = torch.tensor(Sigma_inv, dtype=torch.float32)
    diff = x.detach().float() - (A_t @ anchor.detach().float() + c_t)
    return float((diff @ (sigma_inv_t @ diff)).item())


def regime_summary(macro_state: torch.Tensor) -> Dict[str, object]:
    result = regime_classifier.classify(macro_state.detach().cpu().numpy())
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


def macro_shift_frame(start: torch.Tensor, final: torch.Tensor) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "macro": macro_columns,
            "anchor": m0.detach().cpu().numpy(),
            "start": start.detach().cpu().numpy(),
            "final": final.detach().cpu().numpy(),
        }
    )
    frame["shift_vs_anchor"] = frame["final"] - frame["anchor"]
    frame["shift_vs_start"] = frame["final"] - frame["start"]
    return frame


def print_run_header(n_seeds: int, n_steps: int) -> None:
    anchor_regime = regime_summary(m0)
    eval_summary = regime_classifier.eval_summary["test"]
    print("=" * 88)
    print("Scenario 4 | Universe500 root runtime")
    print(
        f"Question: find plausible macro states near {DATE} where SummerChild and WinterWolf "
        "produce distinct realized returns on the same 30-name Universe500 basket."
    )
    print(
        f"Objective: exp(-100 * (r_SC - r_WW)^2) + {L2REG:.4g} * Mahalanobis^2("
        "m, A @ anchor + c)"
    )
    print(
        f"Chain: MALA, steps={n_steps}, seeds={n_seeds}, beta={BETA:.2f}, "
        f"eta0={ETA / BETA:.6f}, random_start_scale={RANDOM_START_SCALE:.2f}"
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
        f"accepted_moves={int(summary['accepted_moves'])}, "
        f"state_changes={int(summary['state_change_count'])}, "
        f"ESS cap(min/mean/max)={summary['ess_min']:.1f}/{summary['ess_mean']:.1f}/{summary['ess_max']:.1f}"
    )
    print(
        f"[seed {seed_idx:02d}] start regime={format_regime(summary['start_regime'])}, "
        f"final regime={format_regime(summary['final_regime'])}"
    )
    print(
        f"[seed {seed_idx:02d}] Mahalanobis: start={summary['start_mah_dist']:.3f} "
        f"(sq={summary['start_mah2']:.3f}), final={summary['final_mah_dist']:.3f} "
        f"(sq={summary['final_mah2']:.3f})"
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
            m_last, m_traj, chain_info = langevin.torch_MALA_chain(m_start, hypsG, n_steps, return_info=True)
        except KeyboardInterrupt:
            elapsed = time.time() - seed_start_time
            print(f"[seed {i + 1:02d}] interrupted after {elapsed:.1f}s; returning completed seeds only")
            break
        m_lasts_contrast.append(m_last)
        m_trajs_contrast.append(m_traj)

        ess_summary = summarize_ess(m_traj, chain_info["accepted"])
        start_mah2 = mahalanobis_squared(m_start, m0)
        final_mah2 = mahalanobis_squared(m_last, m0)
        final_eval = evaluate_pair(m_last)
        summary = {
            "seed": i + 1,
            "start_regime": regime_summary(m_start),
            "final_regime": regime_summary(m_last),
            "start_mah2": start_mah2,
            "final_mah2": final_mah2,
            "start_mah_dist": start_mah2 ** 0.5,
            "final_mah_dist": final_mah2 ** 0.5,
            "accept_rate": chain_info["accept_rate"],
            "mean_accept_prob": chain_info["mean_accept_prob"],
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
