from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch

from src.modules.runtime_paths import REGIME_DIR


MACRO_ORDER = ("dp", "ep", "bm", "ntis", "tbl", "tms", "dfy", "svar", "infl")


@dataclass
class RuntimeRegimeClassifier:
    classes: tuple[str, ...]
    macro_means: np.ndarray
    macro_stds: np.ndarray
    label_panel: pd.DataFrame
    probability_panel: pd.DataFrame
    eval_summary: Dict[str, object]
    # sklearn model loaded from pickle (decision tree or any sklearn classifier)
    _sklearn_model: Optional[Any] = field(default=None, repr=False)
    _label_encoder: Optional[Any] = field(default=None, repr=False)
    _feature_names: tuple[str, ...] = field(default=(), repr=False)
    # Legacy logistic regression weights (kept for backward compat if no pickle)
    _coef: Optional[np.ndarray] = field(default=None, repr=False)
    _intercept: Optional[np.ndarray] = field(default=None, repr=False)

    @classmethod
    def load(cls, regime_dir: Path = REGIME_DIR) -> "RuntimeRegimeClassifier":
        classifier_meta = json.loads((regime_dir / "classifier_summary.json").read_text(encoding="utf-8"))
        thresholds = json.loads((regime_dir / "threshold_summary.json").read_text(encoding="utf-8"))
        eval_summary = json.loads((regime_dir / "evaluation" / "eval_summary.json").read_text(encoding="utf-8"))
        label_panel = pd.read_csv(regime_dir / "regime_label_panel.csv")
        probability_panel = pd.read_csv(regime_dir / "regime_probability_panel.csv")

        macro_means = np.asarray([thresholds["macro_means"][name] for name in MACRO_ORDER], dtype=np.float64)
        macro_stds = np.asarray([thresholds["macro_stds"][name] for name in MACRO_ORDER], dtype=np.float64)
        classes = tuple(classifier_meta["classes"])

        # Try loading sklearn model from pickle (decision tree format)
        sklearn_model = None
        label_encoder = None
        feature_names: tuple[str, ...] = ()
        coef = None
        intercept = None

        pkl_path = regime_dir / "classifier_model.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                bundle = pickle.load(f)
            sklearn_model = bundle["model"]
            label_encoder = bundle["label_encoder"]
            feature_names = tuple(classifier_meta.get("feature_names", []))
        else:
            # Legacy logistic regression fallback
            coef = np.asarray(classifier_meta.get("coef", []), dtype=np.float64)
            intercept = np.asarray(classifier_meta.get("intercept", []), dtype=np.float64)

        obj = cls(
            classes=classes,
            macro_means=macro_means,
            macro_stds=macro_stds,
            label_panel=label_panel,
            probability_panel=probability_panel,
            eval_summary=eval_summary,
        )
        obj._sklearn_model = sklearn_model
        obj._label_encoder = label_encoder
        obj._feature_names = feature_names
        obj._coef = coef
        obj._intercept = intercept
        return obj

    def zscore(self, macro_state: torch.Tensor | np.ndarray) -> np.ndarray:
        values = np.asarray(macro_state, dtype=np.float64)
        return (values - self.macro_means) / self.macro_stds

    def predict_proba(self, macro_state: torch.Tensor | np.ndarray) -> Dict[str, float]:
        z = self.zscore(macro_state)

        if self._sklearn_model is not None:
            # Build feature vector in the order the model expects (9 z-scores)
            z_row = z.reshape(1, -1).astype(np.float32)
            raw_proba = self._sklearn_model.predict_proba(z_row)[0]
            # label_encoder.classes_ gives int codes; map back to regime names
            le_classes = [self._label_encoder.inverse_transform([i])[0] for i in range(len(raw_proba))]
            proba_map = {name: float(p) for name, p in zip(le_classes, raw_proba)}
            # Return in canonical REGIME_CLASSES order
            return {name: proba_map.get(name, 0.0) for name in self.classes}

        # Legacy: logistic regression manual softmax
        logits = self._intercept + self._coef @ z
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
