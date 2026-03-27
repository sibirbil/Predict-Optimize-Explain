#!/usr/bin/env python3
"""VAR(1) macro prior + Mahalanobis diagnostics.

This script:
1) loads monthly macro data,
2) fits plain VAR(1) by OLS,
3) saves prior artifacts (c, A, Sigma),
4) computes Mahalanobis/log-density diagnostics,
5) writes a small set of plots and tables.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_PATH = Path("universe_500/macro_final.parquet")
DATE_COL = "yyyymm"
MACRO_COLS = ["dp", "ep", "bm", "ntis", "tbl", "tms", "dfy", "svar", "infl"]

OUTPUT_DIR = Path("artifacts/var1")
SIGMA_JITTER = 1e-10
RANDOM_SEED = 42
WARMUP_MONTHS = 120


def load_monthly_macro() -> tuple[np.ndarray, np.ndarray]:
    """Load monthly macro matrix in fixed variable order."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    df = pd.read_parquet(DATA_PATH)
    needed = [DATE_COL, *MACRO_COLS]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[needed].copy()
    df[DATE_COL] = pd.to_numeric(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=needed).copy()
    df[DATE_COL] = df[DATE_COL].round().astype(int)
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    dates = df[DATE_COL].to_numpy(dtype=int)
    states = df[MACRO_COLS].to_numpy(dtype=float)
    if len(states) < 3:
        raise ValueError("Need at least 3 months to fit VAR(1).")
    return dates, states


def fit_var1(states: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit m_{t+1} = c + A m_t + eps by plain OLS."""
    x_t = states[:-1]
    x_tp1 = states[1:]

    design = np.column_stack([np.ones(len(x_t)), x_t])
    beta, *_ = np.linalg.lstsq(design, x_tp1, rcond=None)

    c = beta[0, :]
    a = beta[1:, :].T
    residuals = x_tp1 - design @ beta
    sigma = np.cov(residuals, rowvar=False, ddof=1)
    return c, a, sigma


def conditional_mean(c: np.ndarray, a: np.ndarray, m_t: np.ndarray) -> np.ndarray:
    return c + a @ m_t


def mahalanobis(m_next: np.ndarray, mu: np.ndarray, sigma_inv: np.ndarray) -> tuple[float, float]:
    diff = m_next - mu
    md2 = float(diff.T @ sigma_inv @ diff)
    return float(np.sqrt(max(md2, 0.0))), md2


def gaussian_logpdf(m_next: np.ndarray, mu: np.ndarray, sigma_inv: np.ndarray, sigma_logdet: float) -> float:
    diff = m_next - mu
    dim = len(diff)
    md2 = float(diff.T @ sigma_inv @ diff)
    return float(-0.5 * (dim * np.log(2.0 * np.pi) + sigma_logdet + md2))


def historical_scores(
    dates: np.ndarray,
    states: np.ndarray,
    c: np.ndarray,
    a: np.ndarray,
    sigma: np.ndarray,
) -> pd.DataFrame:
    """In-sample one-step Mahalanobis/log-density series."""
    sigma_stable = sigma + SIGMA_JITTER * np.eye(sigma.shape[0])
    sigma_inv = np.linalg.pinv(sigma_stable)
    sign, sigma_logdet = np.linalg.slogdet(sigma_stable)
    if sign <= 0:
        raise ValueError("Covariance is not positive definite after jitter.")

    rows = []
    for i in range(1, len(states)):
        mu = conditional_mean(c, a, states[i - 1])
        md, md2 = mahalanobis(states[i], mu, sigma_inv)
        rows.append(
            {
                DATE_COL: int(dates[i]),
                "mahalanobis": md,
                "mahalanobis_sq": md2,
                "gaussian_logpdf": gaussian_logpdf(states[i], mu, sigma_inv, sigma_logdet),
            }
        )
    return pd.DataFrame(rows)


def recursive_keyvar_forecasts(dates: np.ndarray, states: np.ndarray) -> pd.DataFrame:
    """Expanding-window one-step forecasts for key variables."""
    start = max(2, min(WARMUP_MONTHS, len(states) - 2))
    rows = []
    for i in range(start, len(states)):
        c_i, a_i, _ = fit_var1(states[:i])
        mu = conditional_mean(c_i, a_i, states[i - 1])
        row = {DATE_COL: int(dates[i])}
        for j, var in enumerate(MACRO_COLS):
            row[f"realized_{var}"] = float(states[i, j])
            row[f"predicted_{var}"] = float(mu[j])
        rows.append(row)
    return pd.DataFrame(rows)


def save_outputs(
    dates: np.ndarray,
    states: np.ndarray,
    c: np.ndarray,
    a: np.ndarray,
    sigma: np.ndarray,
    df_md: pd.DataFrame,
    df_fcst: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sigma_stable = sigma + SIGMA_JITTER * np.eye(sigma.shape[0])
    sigma_inv = np.linalg.pinv(sigma_stable)
    sign, sigma_logdet = np.linalg.slogdet(sigma_stable)
    if sign <= 0:
        raise ValueError("Covariance is not positive definite after jitter.")

    # Core artifacts
    np.save(OUTPUT_DIR / "intercept_c.npy", c)
    np.save(OUTPUT_DIR / "transition_A.npy", a)
    np.save(OUTPUT_DIR / "innovation_covariance_Sigma.npy", sigma_stable)
    np.save(OUTPUT_DIR / "innovation_covariance_inv.npy", sigma_inv)

    pd.DataFrame({"variable": MACRO_COLS, "c": c}).to_csv(OUTPUT_DIR / "intercept_c.csv", index=False)
    pd.DataFrame(a, index=MACRO_COLS, columns=MACRO_COLS).to_csv(OUTPUT_DIR / "transition_A.csv")
    pd.DataFrame(sigma_stable, index=MACRO_COLS, columns=MACRO_COLS).to_csv(OUTPUT_DIR / "innovation_covariance_Sigma.csv")

    # Tables
    df_md.to_csv(OUTPUT_DIR / "historical_mahalanobis.csv", index=False)
    top = df_md.sort_values("mahalanobis", ascending=False).head(20).copy().reset_index(drop=True)
    top.insert(0, "rank", np.arange(1, len(top) + 1))
    top_legacy = top.rename(columns={DATE_COL: "month"})
    top_legacy.to_csv(OUTPUT_DIR / "top_mah_outliers.csv", index=False)
    top.to_csv(OUTPUT_DIR / "top_mahalanobis_outliers.csv", index=False)

    # Last-month prior and samples
    ref_month = int(dates[-1])
    mu_last = conditional_mean(c, a, states[-1])
    pd.DataFrame({"variable": MACRO_COLS, "mu_next": mu_last}).to_csv(
        OUTPUT_DIR / "last_month_conditional_mean.csv", index=False
    )
    rng = np.random.default_rng(RANDOM_SEED)
    samples = rng.multivariate_normal(mu_last, sigma_stable, size=10)
    pd.DataFrame(samples, columns=MACRO_COLS).to_csv(OUTPUT_DIR / "example_prior_samples.csv", index=False)

    # Plots
    p50 = float(df_md["mahalanobis"].quantile(0.50))
    p90 = float(df_md["mahalanobis"].quantile(0.90))
    p95 = float(df_md["mahalanobis"].quantile(0.95))

    plt.figure(figsize=(7, 4.5))
    plt.hist(df_md["mahalanobis"], bins=30, edgecolor="black")
    plt.title("Historical One-Step Mahalanobis Distances")
    plt.xlabel("Mahalanobis distance")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mahalanobis_hist.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.hist(df_md["mahalanobis"], bins=30, edgecolor="black")
    plt.axvline(p50, color="black", linestyle="--", linewidth=1.4, label="Median")
    plt.axvline(p90, color="tab:orange", linestyle="--", linewidth=1.4, label="P90")
    plt.axvline(p95, color="tab:red", linestyle="--", linewidth=1.4, label="P95")
    plt.title("Mahalanobis Distances with Percentiles")
    plt.xlabel("Mahalanobis distance")
    plt.ylabel("Count")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mah_histogram_with_percentiles.png", dpi=150)
    plt.close()

    month_dt = pd.to_datetime(df_md[DATE_COL].astype(str) + "01", format="%Y%m%d")
    plt.figure(figsize=(9, 4.5))
    plt.plot(month_dt, df_md["mahalanobis"], linewidth=1.4, color="tab:blue")
    plt.axhline(p90, color="tab:orange", linestyle="--", linewidth=1.2, label="P90")
    plt.axhline(p95, color="tab:red", linestyle="--", linewidth=1.2, label="P95")
    plt.title("Mahalanobis Distance Over Time")
    plt.xlabel("Month")
    plt.ylabel("Mahalanobis distance")
    plt.legend(frameon=False, loc="upper left")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mah_timeseries.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(np.sort(df_md["mahalanobis"].to_numpy()), linewidth=1.8)
    plt.title("Sorted Historical Mahalanobis Distances")
    plt.xlabel("Order statistic index")
    plt.ylabel("Mahalanobis distance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mahalanobis_sorted.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6.8, 5.8))
    im = plt.imshow(sigma_stable, aspect="auto")
    plt.title("Innovation Covariance Heatmap")
    plt.xticks(np.arange(len(MACRO_COLS)), MACRO_COLS, rotation=45, ha="right")
    plt.yticks(np.arange(len(MACRO_COLS)), MACRO_COLS)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Covariance value")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "innovation_cov_heatmap.png", dpi=150)
    plt.close()

    std = np.sqrt(np.diag(sigma_stable))
    corr = np.nan_to_num(sigma_stable / np.outer(std, std), nan=0.0, posinf=0.0, neginf=0.0)
    plt.figure(figsize=(6.8, 5.8))
    im = plt.imshow(corr, aspect="auto", vmin=-1.0, vmax=1.0, cmap="coolwarm")
    plt.title("Innovation Correlation Heatmap")
    plt.xticks(np.arange(len(MACRO_COLS)), MACRO_COLS, rotation=45, ha="right")
    plt.yticks(np.arange(len(MACRO_COLS)), MACRO_COLS)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Correlation")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "innovation_corr_heatmap.png", dpi=150)
    plt.close()

    key_vars = [v for v in ["dfy", "svar", "tbl", "infl"] if v in MACRO_COLS]
    if key_vars and not df_fcst.empty:
        month_fcst = pd.to_datetime(df_fcst[DATE_COL].astype(str) + "01", format="%Y%m%d")
        fig, axes = plt.subplots(len(key_vars), 1, figsize=(10, 2.2 * len(key_vars)), sharex=True)
        if len(key_vars) == 1:
            axes = [axes]
        for ax, var in zip(axes, key_vars):
            ax.plot(month_fcst, df_fcst[f"realized_{var}"], linewidth=1.4, label=f"Realized {var}")
            ax.plot(month_fcst, df_fcst[f"predicted_{var}"], linewidth=1.2, linestyle="--", label=f"Predicted {var}")
            ax.set_ylabel(var)
            ax.legend(frameon=False, loc="upper left")
        axes[-1].set_xlabel("Month")
        fig.suptitle("Predicted vs Realized Next-Month Values (Recursive VAR(1))", y=0.995)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "predicted_vs_realized_keyvars_recursive.png", dpi=150)
        plt.close(fig)

    summary = {
        "data_path": str(DATA_PATH),
        "date_column": DATE_COL,
        "macro_columns": MACRO_COLS,
        "n_months_used": int(len(states)),
        "sample_start": int(dates.min()),
        "sample_end": int(dates.max()),
        "n_transitions": int(len(states) - 1),
        "sigma_jitter": SIGMA_JITTER,
        "sigma_logdet": float(sigma_logdet),
        "reference_month": ref_month,
        "mahalanobis_mean": float(df_md["mahalanobis"].mean()),
        "mahalanobis_median": float(df_md["mahalanobis"].median()),
        "mahalanobis_p95": float(df_md["mahalanobis"].quantile(0.95)),
        "recursive_warmup_months": WARMUP_MONTHS,
        "recursive_forecast_n_months": int(len(df_fcst)),
        "regularizer_note": "E_var = 0.5*(m_next-mu)' Sigma^{-1} (m_next-mu) in generation objective.",
    }
    with (OUTPUT_DIR / "var1_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    dates, states = load_monthly_macro()
    c, a, sigma = fit_var1(states)
    df_md = historical_scores(dates, states, c, a, sigma)
    df_fcst = recursive_keyvar_forecasts(dates, states)
    save_outputs(dates, states, c, a, sigma, df_md, df_fcst)

    print("VAR(1) fit complete")
    print(f"Data: {DATA_PATH}")
    print(f"Columns: {MACRO_COLS}")
    print(f"Months used: {len(states)} ({dates.min()} to {dates.max()})")
    print(f"Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
