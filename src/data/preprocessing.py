"""
ICU Mortality Risk Tracker
Pipeline: raw CSV → (N, 48, 24) tensor
"""

from __future__ import annotations

import os
import glob
import pickle
import random
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# 0. CONFIG
def set_random_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def resolve_paths() -> Tuple[str, str]:
    """Auto-detect Kaggle vs local environment."""
    if os.path.exists("/kaggle/input"):
        candidates = glob.glob("/kaggle/input/**/*.csv", recursive=True)
        if not candidates:
            raise FileNotFoundError("Không tìm thấy CSV trong /kaggle/input/.")
        return candidates[0], "/kaggle/working/processed"
    return "RawData.csv", "processed"


@dataclass
class PipelineConfig:
    data_path:   str = ""
    output_dir:  str = ""
    random_seed: int = 42
    n_hours:     int = 48

    vitals: List[str] = field(default_factory=lambda: [
        "heart_rate", "spo2", "map", "resp_rate", "temperature", "gcs_total",
    ])
    urine: List[str] = field(default_factory=lambda: ["urine_output_ml"])
    labs:  List[str] = field(default_factory=lambda: [
        "creatinine", "bun", "wbc", "hemoglobin", "lactate",
    ])

    ffill_vitals: int = 4
    ffill_urine:  int = 1
    ffill_labs:   int = 24

    iqr_multiplier: float = 3.0

    clinical_bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "heart_rate"     : (0,   300),
        "spo2"           : (50,  100),
        "map"            : (0,   200),
        "resp_rate"      : (0,   80),
        "temperature"    : (25,  45),
        "gcs_total"      : (3,   15),
        "urine_output_ml": (0,   2500),
        "creatinine"     : (0,   30),
        "bun"            : (0,   200),
        "wbc"            : (0,   200),
        "hemoglobin"     : (0,   25),
        "lactate"        : (0,   30),
    })

    train_ratio: float = 0.70
    val_ratio:   float = 0.15
    test_ratio:  float = 0.15

    @property
    def feature_cols(self) -> List[str]:
        return self.vitals + self.urine + self.labs

    @property
    def n_features(self) -> int:
        return len(self.feature_cols)

    @property
    def ffill_limits(self) -> Dict[str, int]:
        return {
            **{f: self.ffill_vitals for f in self.vitals},
            **{f: self.ffill_urine  for f in self.urine},
            **{f: self.ffill_labs   for f in self.labs},
        }



# 1. Mask from raw data
def b1_create_mask(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    return pd.DataFrame(
        {f"{col}_mask": df[col].notna().astype(np.int8)
         for col in cfg.feature_cols},
        index=df.index,
    )

# 2. Outlier: clipping + IQR×3
# Notes: Outlier IQR to NaN + mask=0
def b2_handle_outliers(
    df: pd.DataFrame,
    mask_df: pd.DataFrame,
    cfg: PipelineConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df, mask_df = df.copy(), mask_df.copy()

    # Pass 1: clip
    for col, (lo, hi) in cfg.clinical_bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lo, upper=hi)

    # Pass 2: IQR
    for col in cfg.feature_cols:
        series = df[col].dropna()
        if len(series) < 4:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lo = q1 - cfg.iqr_multiplier * iqr
        hi = q3 + cfg.iqr_multiplier * iqr
        outlier = (df[col] < lo) | (df[col] > hi)
        if outlier.any():
            df.loc[outlier, col] = np.nan
            mask_df.loc[outlier, f"{col}_mask"] = 0

    return df, mask_df


# 3.Forward-fill per patient
# Notes: Không update mask, ffill không tạo observation mới, groupby stay_id
def b3_forward_fill(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    df = df.copy().sort_values(["stay_id", "time_step_t"])
    # Gom features cùng limit để gọi ffill/limit
    limit_groups: Dict[int, List[str]] = {}
    for col, lim in cfg.ffill_limits.items():
        limit_groups.setdefault(lim, []).append(col)

    for lim, cols in limit_groups.items():
        df[cols] = df.groupby("stay_id")[cols].ffill(limit=lim)

    return df

# 4. Reshape long-format to tensor (N, 48, 12)
# Notes: LOS<48h được pad NaN
def b4_reshape_to_tensor(
    df: pd.DataFrame,
    mask_df: pd.DataFrame,
    cfg: PipelineConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Filter time_step ngoài [0, 47]
    df = df[(df["time_step_t"] >= 0) & (df["time_step_t"] < cfg.n_hours)]
    mask_df = mask_df.loc[df.index]

    # Deduplicate
    keep_idx = df.drop_duplicates(subset=["stay_id", "time_step_t"], keep="first").index
    df = df.loc[keep_idx]
    mask_df = mask_df.loc[keep_idx]

    # Allocate tensors
    stay_ids = sorted(df["stay_id"].unique())
    N, T, F = len(stay_ids), cfg.n_hours, cfg.n_features

    X = np.full((N, T, F), np.nan, dtype=np.float32)
    M = np.zeros((N, T, F),         dtype=np.int8)

    # Attach mask cols
    df_full = df.copy()
    mask_cols = [f"{c}_mask" for c in cfg.feature_cols]
    for mc in mask_cols:
        df_full[mc] = mask_df[mc].values

    # Vectorized scatter
    stay_id_to_idx = {sid: i for i, sid in enumerate(stay_ids)}
    p_idx = df_full["stay_id"].map(stay_id_to_idx).values.astype(np.int64)
    t_idx = df_full["time_step_t"].values.astype(np.int64)

    X[p_idx, t_idx, :] = df_full[cfg.feature_cols].values.astype(np.float32)
    M[p_idx, t_idx, :] = df_full[mask_cols].values.astype(np.int8)

    # Labels: 1 per patient
    df_full["_pidx"] = p_idx
    y = df_full.groupby("_pidx")["hospital_expire_flag"].first().values.astype(np.int32)

    return X, M, y, np.array(stay_ids, dtype=np.int64)

# 5. Stratified split by stay_id (70/15/15)
def b5_split(
    X: np.ndarray, M: np.ndarray, y: np.ndarray, stay_ids: np.ndarray,
    cfg: PipelineConfig,
) -> Dict[str, Dict[str, np.ndarray]]:
    n = len(y)
    indices = np.arange(n)

    if n < 6:
        rng = np.random.default_rng(cfg.random_seed)
        shuffled = rng.permutation(indices)
        n_test  = max(1, int(round(n * cfg.test_ratio)))
        n_val   = max(1, int(round(n * cfg.val_ratio)))
        n_train = n - n_val - n_test
        if n_train < 1:
            n_train, n_val, n_test = 1, 1, n - 2
        idx_train = shuffled[:n_train]
        idx_val   = shuffled[n_train:n_train + n_val]
        idx_test  = shuffled[n_train + n_val:]
    else:
        try:
            idx_train, idx_temp = train_test_split(
                indices, test_size=(cfg.val_ratio + cfg.test_ratio),
                stratify=y, random_state=cfg.random_seed,
            )
            try:
                idx_val, idx_test = train_test_split(
                    idx_temp, test_size=0.5,
                    stratify=y[idx_temp], random_state=cfg.random_seed,
                )
            except ValueError:
                idx_val, idx_test = train_test_split(
                    idx_temp, test_size=0.5, random_state=cfg.random_seed,
                )
        except ValueError:
            idx_train, idx_temp = train_test_split(
                indices, test_size=(cfg.val_ratio + cfg.test_ratio),
                random_state=cfg.random_seed,
            )
            idx_val, idx_test = train_test_split(
                idx_temp, test_size=0.5, random_state=cfg.random_seed,
            )

    return {
        name: {"X": X[idx], "M": M[idx], "y": y[idx], "stay_ids": stay_ids[idx]}
        for name, idx in [("train", idx_train), ("val", idx_val), ("test", idx_test)]
    }

# B6 — Mean imputation 
# Notes: train chỉ dùng mean, val/test dùng train mean
def compute_train_means(X_train: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    flat = X_train.reshape(-1, cfg.n_features)
    means = np.nanmean(flat, axis=0)
    means = np.where(np.isnan(means), 0.0, means)
    return means.astype(np.float32)


def apply_mean_imputation(X: np.ndarray, train_means: np.ndarray) -> np.ndarray:
    X = X.copy()
    nan_mask = np.isnan(X)
    X[nan_mask] = np.broadcast_to(train_means, X.shape)[nan_mask]
    return X

# 7. StandardScaler fit on observed train values(mask=1)
# Notes: Không fit trên data đã impute -> tránh variance shrinkage
def b7_fit_scaler_on_observed(
    X_train: np.ndarray,
    M_train: np.ndarray,
    cfg: PipelineConfig,
) -> StandardScaler:
    F = cfg.n_features
    masked = np.where(M_train == 1, X_train, np.nan)
    flat   = masked.reshape(-1, F)

    means = np.nanmean(flat, axis=0)
    stds  = np.nanstd (flat, axis=0, ddof=0)

    means = np.where(np.isnan(means), 0.0, means).astype(np.float64)
    stds  = np.where(np.isnan(stds) | (stds < 1e-8), 1.0, stds).astype(np.float64)

    scaler = StandardScaler()
    scaler.mean_           = means
    scaler.scale_          = stds
    scaler.var_            = stds ** 2
    scaler.n_features_in_  = F
    scaler.n_samples_seen_ = X_train.shape[0] * X_train.shape[1]
    return scaler


def apply_scaler(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    N, T, F = X.shape
    flat = X.reshape(-1, F)
    scaled = (flat - scaler.mean_) / scaler.scale_
    return scaled.reshape(N, T, F).astype(np.float32)


# 8. Concat [X_norm, M] → (N, 48, 24), export
def b8_concat_and_export(
    splits: Dict[str, Dict[str, np.ndarray]],
    scaler: StandardScaler,
    cfg: PipelineConfig,
) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)

    for name, data in splits.items():
        Xm = np.concatenate([data["X_norm"], data["M"].astype(np.float32)], axis=-1)
        np.save(os.path.join(cfg.output_dir, f"Xm_{name}.npy"), Xm)
        np.save(os.path.join(cfg.output_dir, f"y_{name}.npy"),  data["y"])
        pd.DataFrame({"stay_id": data["stay_ids"]}).to_csv(
            os.path.join(cfg.output_dir, f"stay_ids_{name}.csv"), index=False
        )

    with open(os.path.join(cfg.output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)


# RUN PIPELINE
def run_pipeline(cfg: PipelineConfig) -> Dict[str, Dict[str, np.ndarray]]:
    """Run B1-8, return splits dict."""
    df = pd.read_csv(cfg.data_path, parse_dates=["chart_hour"])

    # B1: mask
    mask_df = b1_create_mask(df, cfg)

    # B2: outliers
    df_clean, mask_df = b2_handle_outliers(df, mask_df, cfg)

    # B3: forward-fill
    df_ffilled = b3_forward_fill(df_clean, cfg)

    # B4: reshape
    X, M, y, stay_ids = b4_reshape_to_tensor(df_ffilled, mask_df, cfg)

    # B5: split
    splits = b5_split(X, M, y, stay_ids, cfg)

    # B6: mean imputation
    train_means = compute_train_means(splits["train"]["X"], cfg)
    for name in ["train", "val", "test"]:
        splits[name]["X"] = apply_mean_imputation(splits[name]["X"], train_means)

    # B7: scaler
    scaler = b7_fit_scaler_on_observed(splits["train"]["X"], splits["train"]["M"], cfg)
    for name in ["train", "val", "test"]:
        splits[name]["X_norm"] = apply_scaler(splits[name]["X"], scaler)

    # B8: export
    b8_concat_and_export(splits, scaler, cfg)

    return splits

if __name__ == "__main__":
    set_random_seeds(42)
    data_path, output_dir = resolve_paths()
    cfg = PipelineConfig(data_path=data_path, output_dir=output_dir)
    run_pipeline(cfg)
    print(f"Done. Output: {os.path.abspath(cfg.output_dir)}/")