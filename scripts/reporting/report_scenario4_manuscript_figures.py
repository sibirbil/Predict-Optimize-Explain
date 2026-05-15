"""Manuscript figures for Scenario 4 (sc_beats_ww, anchor 202004).

Figures (matching v3 story-figure aesthetics):
  1. Macro density grid         – identical style to e2e_macro_density_grid, no header
  2. NFCI analogs               – full-history style, no header/note, smaller dots
  3. PCA + t-SNE regime cloud   – regime-labeled, generated NOT in fit, no header
  4. Regime probability + SC beats WW bars
  5. MALA trace plots (appendix)
"""
from __future__ import annotations

import csv, json, sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from report_story_figures_v2 import (
    MACRO_COLS, REGIME_COLORS, MODEL_COLORS,
    historical_std_frame, anchor_std, load_regime_nfci_panel,
    load_s4, save_figure, yyyymm_to_datetime,
)
from report_story_figures_v3 import (
    nearest_analogs, historical_tsne_with_nn_overlay, anchor_probs,
)
from src.utils.plotting import PAPER_COLORS, macro_density_grid, set_publication_style
from src.data.macro_scaler import MacroScaler

RUN = ROOT / "scenario_outputs" / "scenario4_202004" / "runs" / "20260509_072049"
OUT = ROOT / "submission_plots" / "scenario4_manuscript_figures"
ANCHOR = 202004
REGIME_ORDER = ["financial_stress", "contraction", "expansion"]
PROB_COLS = ["prob_financial_stress", "prob_contraction", "prob_expansion"]
PROB_LABELS = {
    "prob_financial_stress": "Financial\nstress",
    "prob_contraction": "Contraction",
    "prob_expansion": "Expansion",
}
SEED_COLORS = ["#157f78", "#f28e2b", "#4c78a8", "#9b6eb7"]
RNG_SEED = 20260509


# ── helpers ──────────────────────────────────────────────────────────────────

def _n(d: Path, p: str) -> Path:
    m = sorted(d.glob(p))
    if not m:
        raise FileNotFoundError(f"No {p} in {d}")
    return m[-1]


def _rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def raw_anchor(historical: pd.DataFrame) -> np.ndarray:
    row = historical[historical["yyyymm"].astype(int).eq(ANCHOR)]
    if row.empty:
        raise ValueError(f"Anchor {ANCHOR} not found")
    return row[MACRO_COLS].iloc[0].to_numpy(dtype=float)


# ── load & validate ──────────────────────────────────────────────────────────

def load():
    P = {k: _n(RUN, p) for k, p in [
        ("config", "config_*.json"),
        ("final", "final_state_diagnostics_*.csv"),
        ("seed", "seed_summary_*.csv"),
        ("sample", "generated_macro_sample_postburnin_*.csv"),
        ("historical", "historical_macro_panel_*.csv"),
        ("transitions", "regime_transitions_*.csv"),
        ("metadata", "trajectory_tensor_metadata_*.json"),
    ]}
    d = {
        "paths": P,
        "config": json.loads(P["config"].read_text()),
        "metadata": json.loads(P["metadata"].read_text()),
        "final": pd.read_csv(P["final"]),
        "seed": pd.read_csv(P["seed"]),
        "sample": pd.read_csv(P["sample"]),
        "historical": pd.read_csv(P["historical"]),
        "transitions": pd.read_csv(P["transitions"]),
    }
    d["regime_labels"] = pd.read_csv(ROOT / "runtime_universe500" / "regime" / "regime_label_panel.csv")
    d["traj_full"] = np.load(RUN / "trajectories_standardized_3d_20260509_072049.npy")
    return d


def validate(d):
    c, m, f, s = d["config"], d["metadata"], d["final"], d["sample"]
    for ok, msg in [
        (m["scenario"] == "scenario4", "scenario"),
        (int(c["DATE"]) == 202004, "DATE"),
        (int(c["N_SEEDS"]) == 4, "N_SEEDS"),
        (int(c["N_STEPS"]) == 20000, "N_STEPS"),
        (c["CONTRAST_FUNCTION"] == "sc_beats_ww", "CF"),
        (len(f) == 4, "final rows"),
        (f["winner"].eq("summer_child").all(), "winners"),
        (list(m["full_shape"]) == [4, 20000, 9], "shape"),
        (len(s) == 8000, "sample rows"),
    ]:
        assert ok, f"FAIL: {msg}"
    print("✓ Validation passed")


# ── FIG 1: Macro density grid (v3 style, no header) ─────────────────────────

def fig1_macro_density(d):
    """Reuse the repo's macro_density_grid utility, matching e2e style exactly."""
    fig = macro_density_grid(
        historical_df=d["historical"][MACRO_COLS],
        generated_df=d["sample"][MACRO_COLS],
        anchor=raw_anchor(d["historical"]),
        columns=MACRO_COLS,
        title=None,  # no header
    )
    fig.set_size_inches(11.5, 8.0, forward=True)
    return save_figure(fig, OUT, "s4_macro_density_grid")


# ── FIG 2: NFCI analogs (full-history, no header/note, smaller dots) ────────

def fig2_nfci_analogs(d):
    set_publication_style()
    top = nearest_analogs(d["final"], d["historical"], ANCHOR, top_k=5)
    top.to_csv(OUT / "s4_top5_neighbors_full.csv", index=False)
    panel = load_regime_nfci_panel()
    anchor_date = yyyymm_to_datetime([ANCHOR]).iloc[0]
    anchor_nfci = float(panel.loc[panel["yyyymm"].astype(int).eq(ANCHOR), "NFCI"].iloc[0])

    fig, ax = plt.subplots(figsize=(10.6, 4.6))
    ax.plot(panel["date"], panel["NFCI"], color="#1f1f1f", linewidth=1.25, label="NFCI")
    ax.axhline(0.0, color="#73777d", linestyle="--", linewidth=0.85)
    ax.axvline(anchor_date, color=PAPER_COLORS["anchor"], linestyle="--", linewidth=1.2, label=f"{ANCHOR} anchor")

    # smaller circles than reference
    sizes = 32 + 12 * np.sqrt(top["nearest_final_state_count"].clip(lower=1).to_numpy(dtype=float))
    ax.scatter(
        top["date"], top["NFCI"], s=sizes,
        color=PAPER_COLORS["generated"], edgecolors="white", linewidth=0.85,
        zorder=5, label="Top analogs",
    )

    # annotations with manual offsets to avoid overlap and arrowprops for clarity
    for row in top.itertuples(index=False):
        yyyymm = int(row.yyyymm)
        regime_val = d["regime_labels"].loc[d["regime_labels"]["yyyymm"].astype(int).eq(yyyymm), "regime"].iloc[0]
        regime_str = regime_val.replace("_", " ").title()
        
        if yyyymm == 201512:
            xytext = (-30, 25)
        elif yyyymm == 201812:
            xytext = (-30, -25)
        elif yyyymm == 202005:
            xytext = (0, 35)
        elif yyyymm == 202010:
            xytext = (25, -30)
        elif yyyymm == 202211:
            xytext = (30, 25)
        else:
            xytext = (0, 25)
            
        ax.annotate(
            f"{yyyymm} (#{int(row.rank)})\n{regime_str}",
            xy=(row.date, row.NFCI),
            xytext=xytext,
            textcoords="offset points",
            arrowprops={"arrowstyle": "-", "color": PAPER_COLORS["grid"], "lw": 0.8},
            ha="center", fontsize=7.0,
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white",
                  "edgecolor": PAPER_COLORS["grid"], "alpha": 0.94},
        )
    # no subtitle/note box at all
    ax.set_ylabel("NFCI")
    ax.xaxis.set_major_locator(mdates.YearLocator(4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", color=PAPER_COLORS["grid"], alpha=0.60, linewidth=0.6)
    ax.legend(loc="upper right", frameon=False, fontsize=8.0)
    return save_figure(fig, OUT, "s4_nfci_analogs_full_history")


# ── FIG 3: PCA + t-SNE regime cloud (no header, generated NOT in fit) ───────

def fig3_regime_cloud(d):
    set_publication_style()
    hist_std = historical_std_frame(d["historical"])
    cols = [f"{col}_std" for col in MACRO_COLS]
    sample_plot = d["sample"].sample(n=min(260, len(d["sample"])), random_state=23) \
        if len(d["sample"]) > 260 else d["sample"].copy()
    hist_x = hist_std[cols].to_numpy(dtype=float)
    sample_x = sample_plot[cols].to_numpy(dtype=float)
    final_x = d["final"][cols].to_numpy(dtype=float)
    anchor_x = anchor_std(d["historical"], ANCHOR)[cols].to_numpy(dtype=float)

    # PCA fit on historical only (generated NOT included)
    pca = PCA(n_components=2, random_state=0).fit(hist_x)
    pca_parts = {
        "hist": pca.transform(hist_x),
        "sample": pca.transform(sample_x),
        "final": pca.transform(final_x),
        "anchor": pca.transform(anchor_x[None, :]),
    }

    # t-SNE fit on historical only, generated placed by NN overlay
    tsne_parts = historical_tsne_with_nn_overlay(
        hist_x,
        {"sample": sample_x, "final": final_x, "anchor": anchor_x},
        random_state=29, perplexity=20, n_iter=650, k=8,
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.1))
    for ax, parts, title in zip(axes, [pca_parts, tsne_parts], ["PCA regime cloud", "t-SNE regime cloud"]):
        for regime in ["expansion", "contraction", "financial_stress"]:
            idx = hist_std["regime"].eq(regime).to_numpy()
            cloud = parts["hist"][idx]
            ax.scatter(
                cloud[:, 0], cloud[:, 1], s=20,
                color=REGIME_COLORS[regime], alpha=0.22,
                linewidths=0, rasterized=True,
            )
            center = np.median(cloud, axis=0)
            
            # Manual overlap fix for PCA regime labels
            if title.startswith("PCA"):
                if regime == "contraction":
                    center[1] -= 0.65
                elif regime == "expansion":
                    center[1] += 0.55

            ax.text(
                center[0], center[1],
                regime.replace("_", " "),
                ha="center", va="center", fontsize=7.4, weight="bold",
                color=REGIME_COLORS[regime],
                bbox={"boxstyle": "round,pad=0.18", "facecolor": "white",
                      "edgecolor": REGIME_COLORS[regime], "alpha": 0.78},
            )
        ax.scatter(
            parts["sample"][:, 0], parts["sample"][:, 1],
            s=9, color="#343a40", alpha=0.18, linewidths=0, rasterized=True,
        )
        ax.scatter(
            parts["final"][:, 0], parts["final"][:, 1],
            s=74, marker="D", color=PAPER_COLORS["generated"],
            edgecolors="white", linewidth=0.65, zorder=4,
        )
        ax.scatter(
            parts["anchor"][:, 0], parts["anchor"][:, 1],
            s=150, marker="X", color=PAPER_COLORS["anchor"],
            edgecolors="white", linewidth=0.85, zorder=5,
        )
        ax.set_title(title, loc="left", fontsize=11.3, pad=7)
        ax.grid(color=PAPER_COLORS["grid"], alpha=0.52, linewidth=0.55)
        if title.startswith("PCA"):
            ax.set_xlabel(f"PC1 ({100 * pca.explained_variance_ratio_[0]:.1f}%)")
            ax.set_ylabel(f"PC2 ({100 * pca.explained_variance_ratio_[1]:.1f}%)")
        else:
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")

    handles = [
        mlines.Line2D([], [], linestyle="", marker="o", markersize=6, color=REGIME_COLORS["expansion"], label="Historical expansion"),
        mlines.Line2D([], [], linestyle="", marker="o", markersize=6, color=REGIME_COLORS["contraction"], label="Historical contraction"),
        mlines.Line2D([], [], linestyle="", marker="o", markersize=6, color=REGIME_COLORS["financial_stress"], label="Historical stress"),
        mlines.Line2D([], [], linestyle="", marker="o", markersize=5, color="#343a40", label="Generated sample"),
        mlines.Line2D([], [], linestyle="", marker="D", markersize=6, color=PAPER_COLORS["generated"], label="Final scenario states"),
        mlines.Line2D([], [], linestyle="", marker="X", markersize=8, color=PAPER_COLORS["anchor"], label="Anchor"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=7.0, bbox_to_anchor=(0.5, 0.0))
    fig.text(0.5, 0.055, "t-SNE is fit on historical macro states only; scenario points are overlaid by nearest-neighbor placement.", ha="center", fontsize=7.5, color="#5f6368")
    fig.tight_layout(rect=(0.0, 0.13, 1.0, 0.97))
    return save_figure(fig, OUT, "s4_regime_class_cloud_pca_tsne")


# ── FIG 4: Regime probability + SC beats WW bars ────────────────────────────

def fig4_regime_and_return(d):
    """Two panels: (a) regime bars anchor vs generated, (b) SC vs WW return bars."""
    set_publication_style()
    cfg, fin, sam = d["config"], d["final"], d["sample"]
    ap = anchor_probs(ANCHOR)
    colors = [REGIME_COLORS["financial_stress"], REGIME_COLORS["contraction"], REGIME_COLORS["expansion"]]

    # Regime sample shares (mean of multinomial probabilities)
    gen_vals = [sam[col].mean() * 100.0 for col in PROB_COLS]

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), gridspec_kw={"width_ratios": [1.2, 1.0]})

    # ── Panel A: Regime probability bars (anchor vs generated sample) ──
    ax = axes[0]
    x = np.arange(3)
    width = 0.35
    anchor_vals = [100 * ap[col] for col in PROB_COLS]
    ax.bar(x - width / 2, anchor_vals, width, color=colors, alpha=0.88, edgecolor="white", linewidth=0.4)
    ax.bar(x + width / 2, gen_vals, width, color="white", edgecolor=colors, linewidth=1.0, hatch="////")
    for i, (av, gv) in enumerate(zip(anchor_vals, gen_vals)):
        ax.text(i - width / 2, av + 2.5, f"{av:.0f}%", ha="center", fontsize=7.0, fontweight="bold")
        ax.text(i + width / 2, gv + 2.5, f"{gv:.0f}%", ha="center", fontsize=7.0)
    ax.set_xticks(x)
    ax.set_xticklabels([PROB_LABELS[col] for col in PROB_COLS], fontsize=8)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Probability / share (%)")
    ax.set_title("Regime shift: anchor → generated", loc="left", fontsize=10.5, pad=7)
    import matplotlib.patches as mpatches
    h_anchor = mpatches.Patch(facecolor="#999999", edgecolor="white", linewidth=0.5, label="Anchor (solid)")
    h_gen = mpatches.Patch(facecolor="white", edgecolor="#999999", hatch="////", linewidth=1.0, label="Generated (hatched)")
    ax.legend(handles=[h_anchor, h_gen], loc="upper right", frameon=False, fontsize=7.5)
    ax.grid(axis="y", color=PAPER_COLORS["grid"], alpha=0.55, linewidth=0.55)

    # ── Panel B: SC vs WW return bars (anchor + 4 seeds) ──
    ax = axes[1]
    # Anchor returns
    sc_anchor = float(cfg["ANCHOR_SUMMER_RETURN"]) * 100
    ww_anchor = float(cfg["ANCHOR_WINTER_RETURN"]) * 100
    # Final returns per seed
    sc_finals = (fin["summer_return"] * 100).to_numpy(dtype=float)
    ww_finals = (fin["winter_return"] * 100).to_numpy(dtype=float)

    labels = ["Anchor"] + [f"Seed {i+1}" for i in range(len(fin))]
    sc_vals = np.concatenate([[sc_anchor], sc_finals])
    ww_vals = np.concatenate([[ww_anchor], ww_finals])
    x = np.arange(len(labels))
    w = 0.34
    ax.bar(x - w / 2, sc_vals, w, color=MODEL_COLORS["summer_child"], alpha=0.88, label="Summer Child")
    ax.bar(x + w / 2, ww_vals, w, color=MODEL_COLORS["winter_wolf"], alpha=0.88, label="Winter Wolf")
    ax.axhline(0, color=PAPER_COLORS["text"], linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5, rotation=15, ha="right")
    ax.set_ylabel("Realized return (%)")
    ax.set_title("SC beats WW in every seed", loc="left", fontsize=10.5, pad=7)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.grid(axis="y", color=PAPER_COLORS["grid"], alpha=0.55, linewidth=0.55)

    return save_figure(fig, OUT, "s4_regime_and_return_story")


# ── FIG 5: MALA trace plots (appendix) ──────────────────────────────────────

def fig5_mala_traces(d):
    set_publication_style()
    traj = d["traj_full"]  # [4, 20000, 9]
    n_seeds, n_steps, _ = traj.shape
    burn = n_steps // 2
    steps = np.arange(n_steps)

    fig, axes = plt.subplots(3, 3, figsize=(11.0, 7.0))
    fig.subplots_adjust(hspace=0.42, wspace=0.28)

    for i, (ax, mc) in enumerate(zip(axes.ravel(), MACRO_COLS)):
        for s in range(n_seeds):
            ax.plot(steps, traj[s, :, i], color=SEED_COLORS[s], lw=0.3, alpha=0.75, rasterized=True)
        ax.axvline(burn, color="#888888", lw=0.7, ls="--", zorder=3)
        ax.set_title(mc, fontsize=9, fontweight="bold", pad=3)
        ax.tick_params(axis="both", labelsize=6.5)
        if i >= 6:
            ax.set_xlabel("MALA step", fontsize=7.5)
        if i % 3 == 0:
            ax.set_ylabel("Std. value", fontsize=7.5)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
        ax.xaxis.set_major_locator(mticker.MaxNLocator(4))
        ax.grid(color=PAPER_COLORS["grid"], alpha=0.45, linewidth=0.5)

    h = [mlines.Line2D([], [], color=SEED_COLORS[s], lw=1.2, label=f"Chain {s + 1}") for s in range(n_seeds)]
    h.append(mlines.Line2D([], [], color="#888888", lw=0.7, ls="--", label="Burn-in (10k)"))
    fig.legend(handles=h, loc="lower center", ncol=5, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.97))
    return save_figure(fig, OUT, "s4_mala_traces")


# ── notes / manifest ────────────────────────────────────────────────────────

def write_notes(d):
    c, f, s, sam, md = d["config"], d["final"], d["seed"], d["sample"], d["metadata"]
    lines = [
        "# Scenario 4 Manuscript Figure Notes", "",
        f"**Run:** `{_rel(RUN)}`", "",
        "## Configuration",
        f"- scenario4, anchor 202004, objective sc_beats_ww",
        f"- Seeds: 4, steps: 20,000, full shape {md['full_shape']}, post-burn-in {md['postburnin_shape']}",
        f"- ETA={c['ETA']}, BETA={c['BETA']}, L2REG={c['L2REG']}", "",
        "## Target Achievement",
        f"- Anchor SC−WW gap: {100 * c['ANCHOR_RETURN_GAP']:+.2f} pp",
        f"- Final: {int(f['winner'].eq('summer_child').sum())}/4 winner=summer_child",
        f"- Median final gap: {100 * f['return_gap'].median():+.2f} pp", "",
        "## Acceptance & ESS",
        f"- Accept rates: mean {s['accept_rate'].mean():.4f}, range [{s['accept_rate'].min():.4f}, {s['accept_rate'].max():.4f}]", "",
        "## Regime Diagnostics",
        f"- Final hard regimes: {dict(f['regime'].value_counts())}",
        f"- Generated hard regimes: {dict(sam['regime'].value_counts())}", "",
        "## VAR(1) Mahalanobis Caveat",
        f"- Median final Mah. distance: {f['mah_dist'].median():.1f}",
        "- **Extreme.** Frame as stress-escape counterfactual, not VAR-plausible forecast.", "",
        "## Manuscript Recommendation",
        "- **Main:** Fig 1 (density), Fig 3 (PCA/t-SNE), Fig 4 (regime+return)",
        "- **Appendix:** Fig 2 (NFCI analogs), Fig 5 (traces)",
    ]
    (OUT / "figure_notes.md").write_text("\n".join(lines) + "\n")


def write_latex():
    t = r"""\begin{figure}[t]
  \centering
  \includegraphics[width=\textwidth]{s4_macro_density_grid.pdf}
  \caption{Scenario~4 generated macro density versus historical panel.
    Shaded areas show historical (grey) and generated (teal) marginal distributions.
    Dashed red line marks the April~2020 anchor; dotted teal line marks the generated mean.}
  \label{fig:s4-density}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=\textwidth]{s4_regime_class_cloud_pca_tsne.pdf}
  \caption{PCA and t-SNE regime clouds with scenario overlay.
    Historical months are colored by diagnostic regime.
    Generated states cluster in the contraction region; the anchor starts in financial stress.
    t-SNE is fit on historical data only; scenario points are placed by nearest-neighbor interpolation.}
  \label{fig:s4-pca-tsne}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=\textwidth]{s4_regime_and_return_story.pdf}
  \caption{Scenario~4 regime probability shift and return reversal.
    (a)~Regime probability bars show financial stress collapses from the anchor to the generated sample.
    (b)~SummerChild beats WinterWolf in all four seeds.}
  \label{fig:s4-regime-return}
\end{figure}
"""
    (OUT / "latex_include_snippet.tex").write_text(t)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    d = load()
    validate(d)
    OUT.mkdir(parents=True, exist_ok=True)

    pdf, png = fig1_macro_density(d)
    print(f"  Fig 1 (density): {pdf}")

    pdf, png = fig2_nfci_analogs(d)
    print(f"  Fig 2 (NFCI): {pdf}")

    pdf, png = fig3_regime_cloud(d)
    print(f"  Fig 3 (PCA/t-SNE): {pdf}")

    pdf, png = fig4_regime_and_return(d)
    print(f"  Fig 4 (regime+return): {pdf}")

    pdf, png = fig5_mala_traces(d)
    print(f"  Fig 5 (traces): {pdf}")

    write_notes(d)
    write_latex()
    print(f"\n✓ 5 figures → {_rel(OUT)}")


if __name__ == "__main__":
    main()
