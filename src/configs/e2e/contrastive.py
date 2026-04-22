from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.utils.paths import resolve_repo_path

DEFAULT_CONTRASTIVE_E2E_CONFIG_PATH = Path("src/configs/e2e/standardized_sc.json")


@dataclass(frozen=True)
class ContrastiveE2EConfig:
    run_name: str = "summerchild_winterwolf_current_recipe"
    output_root: str = "batuhan/artifacts/e2e_contrastive"
    base_e2e_config_path: str = "src/configs/e2e/stdz_exact_layer_baseline.json"
    base_e2e_run_name: str = "e2e_v2_full_month_20260404"
    base_e2e_output_root: str = "batuhan/artifacts/e2e"
    pto_baseline_run_name: str = "pto_robust_v1_staged_20260404"
    pto_output_root: str = "batuhan/artifacts/pto"
    stable_rule_name: str = "macro_financial_stress_v1"
    stress_quantile: float = 0.67
    dfy_weight: float = 1.0
    svar_weight: float = 1.0
    tms_weight: float = -1.0
    dfy_z_cap: float = 1.0
    svar_z_cap: float = 1.0
    rolling_sharpe_window: int = 36
    variant_construction: str = "legacy_contrastive"
    balance_selection_rule: str = "chronological_even_thinning"
    # --- exact-layer overrides ---
    override_epochs: int | None = None
    override_patience: int | None = None
    override_kappa: float | None = None
    # --- enhanced mask: NFCI + recession ---
    use_nfci: bool = False
    nfci_csv_path: str = "batuhan/NFCI (1).csv"
    nfci_stress_percentile: float = 0.80
    use_recession: bool = False
    recession_md_path: str = "batuhan/Recessiondating.md"
    # --- kappa sensitivity ---
    kappa_sensitivity_grid: tuple[float, ...] | None = None
    # --- frozen E2E benchmark for comparison ---
    frozen_e2e_run_name: str | None = None
    frozen_e2e_output_root: str | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> "ContrastiveE2EConfig":
        with resolve_repo_path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for field_name in ("kappa_sensitivity_grid",):
            if field_name in payload and payload[field_name] is not None:
                payload[field_name] = tuple(payload[field_name])
        return cls(**payload)

    def resolve_output_root(self) -> Path:
        return resolve_repo_path(self.output_root)

    def resolve_base_e2e_config_path(self) -> Path:
        return resolve_repo_path(self.base_e2e_config_path)

    def resolve_base_e2e_run_dir(self) -> Path:
        return resolve_repo_path(self.base_e2e_output_root) / self.base_e2e_run_name

    def resolve_pto_run_dir(self) -> Path:
        return resolve_repo_path(self.pto_output_root) / self.pto_baseline_run_name

    def resolve_frozen_e2e_run_dir(self) -> Path | None:
        if self.frozen_e2e_run_name is None or self.frozen_e2e_output_root is None:
            return None
        return resolve_repo_path(self.frozen_e2e_output_root) / self.frozen_e2e_run_name

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
