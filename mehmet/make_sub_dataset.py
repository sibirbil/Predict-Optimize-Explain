import os
from pathlib import Path
import numpy as np
import pandas as pd
import json

# ============================================================
# CONFIG
# ============================================================
FULL_DATA_DIR = Path(r"./Data/final_data")          # <- supervisor sets this
OUT_DIR       = Path(r"./Data/final_data_sub")   # <- supervisor sets this

N_ASSETS = 300

# Use the test window you used (201601–202411)
TEST_MIN_YYYYMM = 201601
TEST_MAX_YYYYMM = 202411

# If True: keep only assets with full monthly coverage in test window
REQUIRE_FULL_COVERAGE = True

RANDOM_SEED = 123  # reproducible universe selection
# ============================================================


class DataStorageEngine:
    def __init__(self, storage_dir):
        self.storage_dir = Path(storage_dir)
        print(f"Initializing Loader from: {self.storage_dir}")

    def load_dataset(self):
        print("\n--- Loading Data ---")
        loaded_dict = {}
        files = list(self.storage_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found in {self.storage_dir}")

        for file_path in files:
            key = file_path.stem
            print(f"Loading {key}...")
            df = pd.read_parquet(file_path)
            if key.startswith("y_"):
                loaded_dict[key] = df.iloc[:, 0]
            else:
                loaded_dict[key] = df
        return loaded_dict


def patch_targets(data):
    print("\n--- Patching Data Targets ---")

    train_mean = float(data["y_train"].mean())
    if abs(train_mean) > 0.1:
        print(f">> Detected Percentage Scale (Mean={train_mean:.2f}). Dividing all targets by 100.")
        for key in ["y_train", "y_val", "y_test"]:
            data[key] = data[key] / 100.0
    else:
        print(f">> Detected Decimal Scale (Mean={train_mean:.4f}). No scaling needed.")

    for key in ["y_train", "y_val", "y_test"]:
        pre_min = float(data[key].min())
        data[key] = data[key].clip(lower=-0.99)
        post_min = float(data[key].min())
        print(f"{key}: Min clipped from {pre_min:.4f} to {post_min:.4f}")

    print("\nTarget Stats after Patch (train):")
    desc = data["y_train"].describe()
    print(desc[["mean", "min", "max", "std"]])


def pick_universe_from_test(metadata, N, require_full_coverage=True, seed=123):
    """
    Universe selection based on TEST window in metadata.
    Returns: months_sorted, chosen_permnos (length N)
    """
    md = metadata.copy()
    # Ensure types
    md["yyyymm"] = md["yyyymm"].astype(int)
    md["permno"] = md["permno"].astype(int)

    md_test = md[(md["yyyymm"] >= TEST_MIN_YYYYMM) & (md["yyyymm"] <= TEST_MAX_YYYYMM)].copy()
    months = np.sort(md_test["yyyymm"].unique())
    T = len(months)

    if require_full_coverage:
        cnt = md_test.groupby("permno")["yyyymm"].nunique()
        eligible = cnt[cnt == T].index.to_numpy()
        print(f"Eligible (full coverage) assets in test window: {len(eligible)}")
    else:
        # relaxed: at least 80% coverage
        cnt = md_test.groupby("permno")["yyyymm"].nunique()
        eligible = cnt[cnt >= int(0.8 * T)].index.to_numpy()
        print(f"Eligible (>=80% coverage) assets in test window: {len(eligible)}")

    if len(eligible) < N:
        raise ValueError(f"Not enough eligible assets for N={N}. Eligible={len(eligible)}. "
                         f"Set REQUIRE_FULL_COVERAGE=False or lower N.")

    rng = np.random.default_rng(seed)
    chosen = rng.choice(eligible, size=N, replace=False)
    chosen = np.sort(chosen)

    print(f"Chosen universe size: {len(chosen)}")
    print(f"Test months T: {T} | min/max: {months.min()} {months.max()}")
    return months, chosen


def subset_panel_by_permno(data, permnos):
    """
    Subset X_*, y_* by permno using index alignment via metadata.
    Assumes X_* indices correspond to rows in metadata (as in your pipeline).
    """
    md = data["metadata"].copy()
    md["permno"] = md["permno"].astype(int)

    keep_idx = md.index[md["permno"].isin(set(map(int, permnos)))]

    # Subset X_train/val/test + y_train/val/test by index intersection
    out = {}
    for split in ["train", "val", "test"]:
        Xk = f"X_{split}"
        yk = f"y_{split}"
        if Xk in data:
            out[Xk] = data[Xk].loc[data[Xk].index.intersection(keep_idx)].copy()
        if yk in data:
            out[yk] = data[yk].loc[data[yk].index.intersection(keep_idx)].copy()

    # Also subset metadata and firm_clean for those permnos
    out["metadata"] = data["metadata"].loc[keep_idx].copy()
    out["firm_clean"] = data["firm_clean"].loc[data["firm_clean"]["permno"].astype(int).isin(set(map(int, permnos)))].copy()

    # Macro stays as-is (small)
    out["macro_final"] = data["macro_final"].copy()

    return out


def save_sub_dataset(sub, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Saving sub dataset ---")
    for k, v in sub.items():
        fp = out_dir / f"{k}.parquet"
        if isinstance(v, pd.Series):
            v.to_frame(name=v.name if v.name is not None else "target").to_parquet(fp)
        else:
            v.to_parquet(fp)
        print("Saved:", fp, "| shape:", getattr(v, "shape", None))

    # Quick manifest
    manifest = {
        "N_ASSETS": N_ASSETS,
        "TEST_MIN_YYYYMM": TEST_MIN_YYYYMM,
        "TEST_MAX_YYYYMM": TEST_MAX_YYYYMM,
        "REQUIRE_FULL_COVERAGE": REQUIRE_FULL_COVERAGE,
        "RANDOM_SEED": RANDOM_SEED,
        "files": sorted([p.name for p in out_dir.glob("*.parquet")]),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Saved:", out_dir / "manifest.json")


def main():
    # 1) Load full dataset
    storage = DataStorageEngine(FULL_DATA_DIR)
    data = storage.load_dataset()

    # 2) Basic verification
    print("\n--- Shapes ---")
    print("X_train:", data["X_train"].shape, "| y_train:", data["y_train"].shape)
    print("X_val  :", data["X_val"].shape,   "| y_val  :", data["y_val"].shape)
    print("X_test :", data["X_test"].shape,  "| y_test :", data["y_test"].shape)
    print("metadata:", data["metadata"].shape, "| firm_clean:", data["firm_clean"].shape, "| macro_final:", data["macro_final"].shape)

    # 3) Patch targets (same logic as your notebook)
    patch_targets(data)

    # 4) Choose universe from metadata in test window
    months, permnos = pick_universe_from_test(
        metadata=data["metadata"],
        N=N_ASSETS,
        require_full_coverage=REQUIRE_FULL_COVERAGE,
        seed=RANDOM_SEED
    )

    # 5) Build sub dataset
    sub = subset_panel_by_permno(data, permnos)

    # 6) Sanity checks
    print("\n--- Sanity checks (sub) ---")
    for split in ["train", "val", "test"]:
        Xk, yk = f"X_{split}", f"y_{split}"
        if Xk in sub and yk in sub:
            print(f"{split}: X {sub[Xk].shape} | y {sub[yk].shape} | same_index={sub[Xk].index.equals(sub[yk].index)}")

    # 7) Save
    save_sub_dataset(sub, OUT_DIR)

    print("\nDone. Supervisor can now work from:", OUT_DIR)


if __name__ == "__main__":
    main()
