from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .config import FNNTrainConfig
from .trainer import log_progress, run_fnn_training


def _format_hidden_dims(hidden_dims: tuple[int, ...]) -> str:
    return "x".join(str(value) for value in hidden_dims)


def _candidate_label(override: dict[str, object]) -> str:
    if "label" in override:
        return str(override["label"])

    parts: list[str] = []
    if "learning_rate" in override:
        parts.append(f"lr={float(override['learning_rate']):g}")
    if "weight_decay" in override:
        parts.append(f"wd={float(override['weight_decay']):g}")
    if "hidden_dims" in override:
        dims = tuple(int(value) for value in override["hidden_dims"])  # type: ignore[arg-type]
        parts.append(f"h={_format_hidden_dims(dims)}")
    if "dropout" in override:
        parts.append(f"dropout={float(override['dropout']):g}")
    if "batch_norm" in override:
        parts.append(f"bn={int(bool(override['batch_norm']))}")
    if "batch_size" in override:
        parts.append(f"bs={int(override['batch_size'])}")
    if "eval_batch_size" in override:
        parts.append(f"ebs={int(override['eval_batch_size'])}")
    if not parts:
        raise ValueError("Candidate override must specify either 'label' or at least one configurable field.")
    return "__".join(parts)


def _build_candidates(raw: dict[str, object]) -> list[dict[str, object]]:
    if "candidate_overrides" in raw:
        overrides = raw["candidate_overrides"]
        if not isinstance(overrides, list) or not overrides:
            raise ValueError("candidate_overrides must be a non-empty list.")
        normalized: list[dict[str, object]] = []
        for override in overrides:
            if not isinstance(override, dict):
                raise ValueError("Each candidate_overrides entry must be an object.")
            item = dict(override)
            if "hidden_dims" in item:
                item["hidden_dims"] = tuple(int(value) for value in item["hidden_dims"])  # type: ignore[arg-type]
            item["label"] = _candidate_label(item)
            normalized.append(item)
        return normalized

    lr_grid = raw.get("learning_rate_grid", [raw.get("learning_rate", 1e-3)])
    wd_grid = raw.get("weight_decay_grid", [raw.get("weight_decay", 1e-5)])
    candidates: list[dict[str, object]] = []
    for learning_rate in lr_grid:  # type: ignore[assignment]
        for weight_decay in wd_grid:  # type: ignore[assignment]
            candidate = {
                "learning_rate": float(learning_rate),
                "weight_decay": float(weight_decay),
            }
            candidate["label"] = _candidate_label(candidate)
            candidates.append(candidate)
    return candidates


def run_fnn_grid_search(config_path: Path | str, run_name: str | None = None) -> Path:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    base_config = FNNTrainConfig.from_json(config_path)
    if run_name:
        base_config = replace(base_config, run_name=run_name)

    output_root = base_config.resolve_output_root() / base_config.run_name
    output_root.mkdir(parents=True, exist_ok=True)

    candidates = _build_candidates(raw)
    log_progress(f"Starting FNN grid search run_name={base_config.run_name} candidates={len(candidates)}")

    results: list[dict[str, object]] = []
    for index, candidate in enumerate(candidates, start=1):
        label = str(candidate["label"])
        log_progress(f"candidate {index}/{len(candidates)} {label}")
        override = {key: value for key, value in candidate.items() if key != "label"}
        candidate_config = replace(
            base_config,
            run_name=f"{base_config.run_name}/candidates/{label}",
            **override,
        )
        run_dir = run_fnn_training(candidate_config)
        with (run_dir / "metrics_summary.json").open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        results.append(
            {
                "label": label,
                "learning_rate": candidate_config.learning_rate,
                "weight_decay": candidate_config.weight_decay,
                "hidden_dims": list(candidate_config.hidden_dims),
                "dropout": candidate_config.dropout,
                "batch_norm": candidate_config.batch_norm,
                "batch_size": candidate_config.batch_size,
                "eval_batch_size": candidate_config.eval_batch_size,
                "best_epoch": metrics["best_epoch"],
                "val_r2": metrics["metrics"]["val"]["r2"],
                "val_corr": metrics["metrics"]["val"]["correlation"],
                "test_r2": metrics["metrics"]["test"]["r2"],
                "test_corr": metrics["metrics"]["test"]["correlation"],
                "run_dir": str(run_dir),
            }
        )

    tuning_table = pd.DataFrame(results).sort_values(["val_r2", "test_r2"], ascending=False)
    tuning_table.to_csv(output_root / "tuning_table.csv", index=False)
    best_candidate = tuning_table.iloc[0].to_dict()
    best_test_candidate = tuning_table.sort_values(["test_r2", "val_r2"], ascending=False).iloc[0].to_dict()
    with (output_root / "best_candidate.json").open("w", encoding="utf-8") as handle:
        json.dump(best_candidate, handle, indent=2)
    with (output_root / "best_test_candidate.json").open("w", encoding="utf-8") as handle:
        json.dump(best_test_candidate, handle, indent=2)
    log_progress(
        f"grid search complete best={best_candidate['label']} "
        f"val_r2={best_candidate['val_r2']:.6f}"
    )
    return output_root
