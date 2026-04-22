from __future__ import annotations

import itertools
import json
import logging
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.configs.model.fnn import FNNTrainConfig
from src.modeling.fnn.trainer import run_fnn_training


@dataclass(frozen=True)
class FNNCandidate:
    learning_rate: float
    weight_decay: float
    
    def label(self) -> str:
        return f"lr={self.learning_rate:g}__wd={self.weight_decay:g}"


def log_progress(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def run_fnn_grid_search(config_path: Path | str, run_name: str | None = None) -> Path:
    """
    Extends FNNTrainConfig by reading learning_rate_grid and weight_decay_grid
    and running a sequential grid search.
    """
    base_path = Path(config_path)
    with base_path.open("r", encoding="utf-8") as handle:
        raw_config = json.load(handle)
        
    lr_grid = raw_config.get("learning_rate_grid", [raw_config.get("learning_rate", 1e-3)])
    wd_grid = raw_config.get("weight_decay_grid", [raw_config.get("weight_decay", 1e-5)])
    
    config_obj = FNNTrainConfig.from_json(base_path)
    if run_name:
        config_obj = replace(config_obj, run_name=run_name)
        
    output_root = config_obj.resolve_output_root() / config_obj.run_name
    output_root.mkdir(parents=True, exist_ok=True)
    
    candidates = [
        FNNCandidate(learning_rate=lr, weight_decay=wd)
        for lr, wd in itertools.product(lr_grid, wd_grid)
    ]
    
    log_progress(f"Starting FNN Grid Search for {config_obj.run_name} with {len(candidates)} candidates.")
    
    results = []
    for i, candidate in enumerate(candidates, start=1):
        candidate_run_name = f"{config_obj.run_name}_{candidate.label()}"
        log_progress(f"Candidate {i}/{len(candidates)}: {candidate.label()}")
        
        # Override config for this candidate
        candidate_config = replace(
            config_obj,
            run_name=f"{config_obj.run_name}/candidates/{candidate.label()}",
            learning_rate=candidate.learning_rate,
            weight_decay=candidate.weight_decay,
        )
        
        try:
            run_dir = run_fnn_training(candidate_config)
            with (run_dir / "metrics_summary.json").open("r", encoding="utf-8") as h:
                metrics = json.load(h)
                
            results.append({
                "label": candidate.label(),
                "learning_rate": candidate.learning_rate,
                "weight_decay": candidate.weight_decay,
                "val_mse": metrics["metrics"]["val"]["mse"],
                "val_r2": metrics["metrics"]["val"]["r2"],
                "val_r2_zero": metrics["metrics"]["val"]["r2_zero"],
                "val_correlation": metrics["metrics"]["val"]["correlation"],
                "test_mse": metrics["metrics"]["test"]["mse"],
                "test_r2": metrics["metrics"]["test"]["r2"],
                "test_correlation": metrics["metrics"]["test"]["correlation"],
                "best_epoch": metrics["best_epoch"],
                "run_dir": str(run_dir)
            })
        except Exception as e:
            log_progress(f"Candidate {candidate.label()} failed: {e}")
            
    tuning_table = pd.DataFrame(results).sort_values("val_r2", ascending=False)
    tuning_table.to_csv(output_root / "tuning_table.csv", index=False)
    
    best_candidate = tuning_table.iloc[0]
    log_progress(f"Best Candidate: {best_candidate['label']} with Val R2: {best_candidate['val_r2']:.6f}")
    
    # Save a pointer to the best model
    with (output_root / "best_candidate.json").open("w", encoding="utf-8") as h:
        json.dump(best_candidate.to_dict(), h, indent=2)
        
    return output_root
