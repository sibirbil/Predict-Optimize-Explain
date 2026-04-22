from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .universe_500 import Universe500Bundle


def _yyyymm_range(series: pd.Series) -> tuple[int, int]:
    return int(series.min()), int(series.max())


def build_split_summary(bundle: Universe500Bundle) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split in (bundle.train, bundle.val, bundle.test):
        merged = pd.concat(
            [
                split.metadata.reset_index(drop=True),
                split.y.reset_index(drop=True),
            ],
            axis=1,
        )
        start_yyyymm, end_yyyymm = _yyyymm_range(split.metadata["yyyymm"])
        rows.append(
            {
                "split": split.name,
                "rows": int(len(split.X)),
                "feature_columns": int(split.X.shape[1]),
                "target_columns": int(split.y.shape[1]),
                "metadata_columns": int(split.metadata.shape[1]),
                "start_yyyymm": start_yyyymm,
                "end_yyyymm": end_yyyymm,
                "n_months": int(split.metadata["yyyymm"].nunique()),
                "n_unique_permnos": int(split.metadata["permno"].nunique()),
                "target_mean": float(merged["target"].mean()),
                "target_std": float(merged["target"].std()),
                "ret_tplus1_mean": float(merged["ret_tplus1"].mean()),
                "excess_ret_mean": float(merged["excess_ret"].mean()),
                "positive_target_share": float((merged["target"] > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def build_feature_contract_summary(bundle: Universe500Bundle) -> dict[str, object]:
    total_features = len(bundle.feature_names)
    firm_features = len(bundle.firm_feature_names)
    macro_predictors = len(bundle.macro_predictors)
    interaction_features = total_features - firm_features
    interaction_feature_names = [name for name in bundle.feature_names if "_x_" in name]
    return {
        "total_feature_names": total_features,
        "firm_feature_names": firm_features,
        "macro_predictors": macro_predictors,
        "expected_interaction_features": firm_features * macro_predictors,
        "observed_interaction_features": interaction_features,
        "observed_interaction_features_with_name_pattern": len(interaction_feature_names),
        "selected_permnos_count": len(bundle.selected_permnos),
        "feature_name_sample": list(bundle.feature_names[:15]),
        "firm_feature_name_sample": list(bundle.firm_feature_names[:15]),
        "macro_predictor_names": list(bundle.macro_predictors),
    }


def build_month_coverage(bundle: Universe500Bundle) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split in (bundle.train, bundle.val, bundle.test):
        month_counts = (
            split.metadata.groupby("yyyymm", sort=True)["permno"]
            .nunique()
            .rename("n_permnos")
            .reset_index()
        )
        month_counts.insert(0, "split", split.name)
        rows.append(month_counts)
    return pd.concat(rows, ignore_index=True)


def build_macro_summary(bundle: Universe500Bundle) -> pd.DataFrame:
    macro = bundle.macro_final.copy()
    rows: list[dict[str, object]] = []
    for column in macro.columns:
        if column == "yyyymm":
            continue
        values = macro[column]
        rows.append(
            {
                "column": column,
                "dtype": str(values.dtype),
                "non_null_count": int(values.notna().sum()),
                "null_count": int(values.isna().sum()),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def _summarize_feature_frame(
    frame: pd.DataFrame,
    split: str,
    source: str,
) -> pd.DataFrame:
    summary = frame.agg(["mean", "std", "min", "max"]).T.reset_index()
    summary.columns = ["feature", "mean", "std", "min", "max"]
    summary.insert(0, "source", source)
    summary.insert(0, "split", split)
    summary["null_count"] = frame.isna().sum().to_numpy(dtype=int)
    return summary


def build_normalized_firm_feature_summary(bundle: Universe500Bundle) -> pd.DataFrame:
    firm_cols = list(bundle.firm_feature_names)
    outputs: list[pd.DataFrame] = []
    for split in (bundle.train, bundle.val, bundle.test):
        outputs.append(
            _summarize_feature_frame(
                frame=split.X[firm_cols],
                split=split.name,
                source="normalized_firm_features",
            )
        )
    return pd.concat(outputs, ignore_index=True)


def build_standardized_macro_summary(bundle: Universe500Bundle) -> pd.DataFrame:
    macro_cols = list(bundle.macro_predictors)
    macro = bundle.macro_final[["yyyymm", *macro_cols]].copy()
    train_end = int(bundle.train.metadata["yyyymm"].max())
    train_mask = macro["yyyymm"] <= train_end
    train_macro = macro.loc[train_mask, macro_cols]

    means = train_macro.mean()
    stds = train_macro.std().replace(0.0, 1.0)

    macro_z = macro.copy()
    macro_z[macro_cols] = (macro_z[macro_cols] - means) / stds

    rows: list[dict[str, object]] = []
    for source_name, frame in [
        ("raw_macro_features", macro),
        ("train_standardized_macro_features", macro_z),
    ]:
        for column in macro_cols:
            values = frame[column]
            rows.append(
                {
                    "source": source_name,
                    "column": column,
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "train_mean": float(means[column]),
                    "train_std": float(stds[column]),
                    "train_end_yyyymm": train_end,
                }
            )
    return pd.DataFrame(rows)


def build_interaction_feature_summary(bundle: Universe500Bundle) -> pd.DataFrame:
    interaction_cols = [name for name in bundle.feature_names if "_x_" in name]
    outputs: list[pd.DataFrame] = []
    for split in (bundle.train, bundle.val, bundle.test):
        summary = _summarize_feature_frame(
            frame=split.X[interaction_cols],
            split=split.name,
            source="interaction_features",
        )
        parts = summary["feature"].str.split("_x_", n=1, expand=True)
        summary.insert(2, "base_feature", parts[0])
        summary.insert(3, "macro_feature", parts[1])
        outputs.append(summary)
    return pd.concat(outputs, ignore_index=True)


def build_construction_audit(bundle: Universe500Bundle) -> pd.DataFrame:
    macro_cols = list(bundle.macro_predictors)
    base_cols = list(bundle.firm_feature_names)
    macro = bundle.macro_final[["yyyymm", *macro_cols]].copy().set_index("yyyymm")

    train_end = int(bundle.train.metadata["yyyymm"].max())
    train_macro = macro.loc[macro.index <= train_end, macro_cols]
    means = train_macro.mean()
    stds = train_macro.std().replace(0.0, 1.0)
    macro_z = (macro[macro_cols] - means) / stds

    rows: list[dict[str, object]] = []
    for split in (bundle.train, bundle.val, bundle.test):
        yyyymm = split.metadata["yyyymm"].astype(int)
        base_matrix = split.X[base_cols].to_numpy(dtype=np.float64, copy=False)

        for macro_col in macro_cols:
            inter_cols = [f"{base}_x_{macro_col}" for base in base_cols]
            observed = split.X[inter_cols].to_numpy(dtype=np.float64, copy=False)

            raw_macro_values = yyyymm.map(macro[macro_col]).to_numpy(dtype=np.float64)
            z_macro_values = yyyymm.map(macro_z[macro_col]).to_numpy(dtype=np.float64)

            expected_raw = base_matrix * raw_macro_values[:, None]
            expected_z = base_matrix * z_macro_values[:, None]

            raw_abs_err = np.abs(observed - expected_raw)
            z_abs_err = np.abs(observed - expected_z)

            mean_abs_err_raw = float(raw_abs_err.mean())
            mean_abs_err_train_standardized = float(z_abs_err.mean())

            rows.append(
                {
                    "split": split.name,
                    "macro_feature": macro_col,
                    "n_rows": int(len(split.X)),
                    "n_interaction_columns": int(len(inter_cols)),
                    "mean_abs_err_raw_macro": mean_abs_err_raw,
                    "max_abs_err_raw_macro": float(raw_abs_err.max()),
                    "mean_abs_err_train_standardized_macro": mean_abs_err_train_standardized,
                    "max_abs_err_train_standardized_macro": float(z_abs_err.max()),
                    "inferred_interaction_macro_source": (
                        "raw_macro"
                        if mean_abs_err_raw < mean_abs_err_train_standardized
                        else "train_standardized_macro"
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_inventory(bundle: Universe500Bundle) -> dict[str, object]:
    root = bundle.root
    file_inventory = sorted(p.name for p in root.iterdir() if p.is_file())
    macro_start_yyyymm, macro_end_yyyymm = _yyyymm_range(bundle.macro_final["yyyymm"])
    inventory = {
        "dataset_root": str(root),
        "files": file_inventory,
        "macro_final_rows": int(len(bundle.macro_final)),
        "macro_final_columns": list(bundle.macro_final.columns),
        "macro_final_start_yyyymm": macro_start_yyyymm,
        "macro_final_end_yyyymm": macro_end_yyyymm,
        "has_macro_fred_supplement": bundle.macro_fred_supplement is not None,
        "macro_fred_columns": (
            list(bundle.macro_fred_supplement.columns)
            if bundle.macro_fred_supplement is not None
            else []
        ),
    }
    return inventory


def write_markdown_overview(
    output_dir: Path,
    bundle: Universe500Bundle,
    split_summary: pd.DataFrame,
    feature_contract: dict[str, object],
    inventory: dict[str, object],
    construction_audit: pd.DataFrame,
) -> None:
    raw_best = construction_audit["mean_abs_err_raw_macro"].max()
    std_best = construction_audit["mean_abs_err_train_standardized_macro"].min()
    inferred_sources = sorted(construction_audit["inferred_interaction_macro_source"].unique().tolist())
    if inferred_sources == ["raw_macro"]:
        conclusion = (
            "`X_*` interactions are consistent with `firm_normalized x raw_macro`, "
            "not `firm_normalized x train_standardized_macro`."
        )
    elif inferred_sources == ["train_standardized_macro"]:
        conclusion = (
            "`X_*` interactions are consistent with `firm_normalized x train_standardized_macro`, "
            "not `firm_normalized x raw_macro`."
        )
    else:
        conclusion = (
            "Interaction construction is mixed across splits/macros; inspect "
            "`construction_audit.csv` directly."
        )
    lines = [
        "# Universe 500 EDA Overview",
        "",
        "## Scope",
        f"- Dataset root: `{bundle.root}`",
        f"- Total feature columns: `{feature_contract['total_feature_names']}`",
        f"- Firm feature columns: `{feature_contract['firm_feature_names']}`",
        f"- Macro predictors: `{feature_contract['macro_predictors']}`",
        f"- Expected interactions: `{feature_contract['expected_interaction_features']}`",
        f"- Observed interactions: `{feature_contract['observed_interaction_features']}`",
        "",
        "## Splits",
    ]
    for row in split_summary.to_dict(orient="records"):
        lines.append(
            "- "
            + (
                f"{row['split']}: rows={row['rows']}, months={row['n_months']}, "
                f"permnos={row['n_unique_permnos']}, range={row['start_yyyymm']}..{row['end_yyyymm']}, "
                f"target_mean={row['target_mean']:.6f}, target_std={row['target_std']:.6f}"
            )
        )
    lines += [
        "",
        "## Construction Audit",
        "- Firm block in `X_*`: normalized firm characteristics, approximately bounded in `[-1, 1]`.",
        "- Interaction block audit compares observed interactions against two candidates:",
        "  `firm_normalized x raw_macro` and `firm_normalized x train_standardized_macro`.",
        f"- Inferred interaction macro source(s): `{', '.join(inferred_sources)}`",
        f"- Worst split/macro mean absolute error vs raw macro: `{raw_best:.12f}`",
        f"- Best split/macro mean absolute error vs train-standardized macro: `{std_best:.12f}`",
        f"- Conclusion: {conclusion}",
        "",
        "## Macro Table",
        f"- macro_final rows: `{inventory['macro_final_rows']}`",
        f"- macro_final range: `{inventory['macro_final_start_yyyymm']}`..`{inventory['macro_final_end_yyyymm']}`",
        f"- FRED supplement present: `{inventory['has_macro_fred_supplement']}`",
        "",
        "## Outputs",
        "- `inventory.json`",
        "- `split_summary.csv`",
        "- `feature_contract.json`",
        "- `month_coverage.csv`",
        "- `macro_summary.csv`",
        "- `normalized_firm_feature_summary.csv`",
        "- `standardized_macro_summary.csv`",
        "- `interaction_feature_summary.csv`",
        "- `construction_audit.csv`",
        "- `README.md`",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def write_eda_outputs(output_dir: Path, bundle: Universe500Bundle) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    split_summary = build_split_summary(bundle)
    feature_contract = build_feature_contract_summary(bundle)
    month_coverage = build_month_coverage(bundle)
    macro_summary = build_macro_summary(bundle)
    normalized_firm_feature_summary = build_normalized_firm_feature_summary(bundle)
    standardized_macro_summary = build_standardized_macro_summary(bundle)
    interaction_feature_summary = build_interaction_feature_summary(bundle)
    construction_audit = build_construction_audit(bundle)
    inventory = build_inventory(bundle)

    split_summary.to_csv(output_dir / "split_summary.csv", index=False)
    month_coverage.to_csv(output_dir / "month_coverage.csv", index=False)
    macro_summary.to_csv(output_dir / "macro_summary.csv", index=False)
    normalized_firm_feature_summary.to_csv(
        output_dir / "normalized_firm_feature_summary.csv",
        index=False,
    )
    standardized_macro_summary.to_csv(
        output_dir / "standardized_macro_summary.csv",
        index=False,
    )
    interaction_feature_summary.to_csv(
        output_dir / "interaction_feature_summary.csv",
        index=False,
    )
    construction_audit.to_csv(
        output_dir / "construction_audit.csv",
        index=False,
    )
    (output_dir / "inventory.json").write_text(
        json.dumps(inventory, indent=2), encoding="utf-8"
    )
    (output_dir / "feature_contract.json").write_text(
        json.dumps(feature_contract, indent=2), encoding="utf-8"
    )
    write_markdown_overview(
        output_dir,
        bundle,
        split_summary,
        feature_contract,
        inventory,
        construction_audit,
    )
