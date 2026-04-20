from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch

from src.modules.runtime_paths import REGIME_DIR


MACRO_ORDER = ("dp", "ep", "bm", "ntis", "tbl", "tms", "dfy", "svar", "infl")


@dataclass
class RuntimeRegimeClassifier:
    classes: tuple[str, ...]
    coef: np.ndarray
    intercept: np.ndarray
    macro_means: np.ndarray
    macro_stds: np.ndarray
    label_panel: pd.DataFrame
    probability_panel: pd.DataFrame
    eval_summary: Dict[str, object]

    @classmethod
    def load(cls, regime_dir: Path = REGIME_DIR) -> "RuntimeRegimeClassifier":
        classifier = json.loads((regime_dir / "classifier_summary.json").read_text(encoding="utf-8"))
        thresholds = json.loads((regime_dir / "threshold_summary.json").read_text(encoding="utf-8"))
        eval_summary = json.loads((regime_dir / "evaluation" / "eval_summary.json").read_text(encoding="utf-8"))
        label_panel = pd.read_csv(regime_dir / "regime_label_panel.csv")
        probability_panel = pd.read_csv(regime_dir / "regime_probability_panel.csv")
        return cls(
            classes=tuple(classifier["classes"]),
            coef=np.asarray(classifier["coef"], dtype=np.float64),
            intercept=np.asarray(classifier["intercept"], dtype=np.float64),
            macro_means=np.asarray([thresholds["macro_means"][name] for name in MACRO_ORDER], dtype=np.float64),
            macro_stds=np.asarray([thresholds["macro_stds"][name] for name in MACRO_ORDER], dtype=np.float64),
            label_panel=label_panel,
            probability_panel=probability_panel,
            eval_summary=eval_summary,
        )

    def zscore(self, macro_state: torch.Tensor | np.ndarray) -> np.ndarray:
        values = np.asarray(macro_state, dtype=np.float64)
        return (values - self.macro_means) / self.macro_stds

    def predict_proba(self, macro_state: torch.Tensor | np.ndarray) -> Dict[str, float]:
        z = self.zscore(macro_state)
        logits = self.intercept + self.coef @ z
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs = probs / probs.sum()
        return {name: float(value) for name, value in zip(self.classes, probs)}

    def classify(self, macro_state: torch.Tensor | np.ndarray) -> Dict[str, object]:
        probs = self.predict_proba(macro_state)
        label = max(probs, key=probs.get)
        return {"label": label, "probabilities": probs}

    @cached_property
    def label_lookup(self) -> Dict[int, str]:
        panel = self.label_panel.copy()
        panel["yyyymm"] = panel["yyyymm"].astype(int)
        return dict(zip(panel["yyyymm"], panel["regime"]))

    def historical_label(self, yyyymm: int) -> str | None:
        return self.label_lookup.get(int(yyyymm))


regime_classifier = RuntimeRegimeClassifier.load()


__all__ = ["MACRO_ORDER", "RuntimeRegimeClassifier", "regime_classifier"]
