"""
ICU Mortality Risk Tracker
Pipeline: raw CSV → (N, 48, 24) tensor

Changelog:
    - Data leakage: B2 chia thành B2a (clip, run trước split) và B2b (IQR, run sau split, chỉ dùng cho train bounds)
    - clinical_bounds: điều chỉnh theo đề xuất, ngoại trừ spo2 và gcs
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
    """Cố định random seed cho toàn bộ pipeline để đảm bảo tái lập kết quả"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def resolve_paths() -> Tuple[str, str]:
    """
    Tự động phát hiện môi trường chạy (Kaggle hoặc local)

    Returns:
        data_path  : đường dẫn tới file CSV đầu vào
        output_dir : thư mục lưu file output
    """
    if os.path.exists("/kaggle/input"):
        candidates = glob.glob("/kaggle/input/**/*.csv", recursive=True)
        if not candidates:
            raise FileNotFoundError("Không tìm thấy file.")
        return candidates[0], "/kaggle/working/processed"
    return "RawData.csv", "processed"


@dataclass
class PipelineConfig:
    """
    Tập trung toàn bộ hyperparameter của pipeline
    Thay đổi data_path để chạy trên dataset khác
    """
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
        "heart_rate"     : (50,  220),
        "spo2"           : (80,  100), # fix: spo2 không thể vượt ngưỡng 100
        "map"            : (40,  150),
        "resp_rate"      : (7,   40),
        "temperature"    : (35,  40),
        "gcs_total"      : (3,   15), # fix: gcs ngoài range này là lỗi nhập liệu, bắt buộc phải clip
        "urine_output_ml": (0,   600),
        "creatinine"     : (0,   8),
        "bun"            : (0,   140),
        "wbc"            : (0,   50),
        "hemoglobin"     : (5,   20),
        "lactate"        : (0,   14),
    })

    train_ratio: float = 0.70
    val_ratio:   float = 0.15
    test_ratio:  float = 0.15

    @property
    def feature_cols(self) -> List[str]:
        """Trả về danh sách 12 features theo thứ tự: vitals -> urine -> labs"""
        return self.vitals + self.urine + self.labs

    @property
    def n_features(self) -> int:
        """Số lượng features"""
        return len(self.feature_cols)

    @property
    def ffill_limits(self) -> Dict[str, int]:
        """Ánh xạ tên feature tới giới hạn forward-fill"""
        return {
            **{f: self.ffill_vitals for f in self.vitals},
            **{f: self.ffill_urine  for f in self.urine},
            **{f: self.ffill_labs   for f in self.labs},
        }


# 1. Mask from raw data
def b1_create_mask(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """
    Tạo mask nhị phân từ dữ liệu thô, phải chạy trước mọi bước xử lý khác để phản ánh đúng sự kiện đo lường gốc
    M = 1 nếu giá trị được đo thực tế, M = 0 nếu NaN
    
    Args:
        df  : DataFrame thô chưa qua bất kỳ xử lý nào
        cfg : cấu hình pipeline

    Returns:
        DataFrame cùng index với df, gồm 12 cột <feature>_mask kiểu int8
    """
    return pd.DataFrame(
        {f"{col}_mask": df[col].notna().astype(np.int8)
         for col in cfg.feature_cols},
        index=df.index,
    )


# 2. Outlier: clipping + IQR×3

# Pass 1: clipping theo clinical bounds
# Notes: clinical_bounds là hard limits từ kiến thức lâm sàng, không học từ data nên có leakage
def b2a_clip_clinical(
    df: pd.DataFrame,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    """
    Clip giá trị theo ngưỡng lâm sàng cứng để loại data-entry errors
    Clip không thay đổi mask (giá trị bị clip vẫn được xem là đã được đo, mask=1)

    Args:
        df  : DataFrame sau khi đã tạo mask (B1)
        cfg : cấu hình pipeline

    Returns:
        DataFrame đã clip
    """
    df = df.copy()
    for col, (lo, hi) in cfg.clinical_bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lo, upper=hi)
    return df
    
# Pass 2: IQR × 3 với bounds chỉ tính trên train
# Notes: Fix Bug S01-01: chuyển IQR sang sau split để tránh leakage
def compute_iqr_bounds(
    X_train: np.ndarray,
    M_train: np.ndarray,
    cfg: PipelineConfig,
) -> Dict[str, Tuple[float, float]]:
    """
    Tính IQR bounds cho từng feature chỉ trên train set, dùng observed values (mask=1)
    Tính sau split để tránh val/test ảnh hưởng đến ngưỡng outlier detection

    Args:
        X_train : tensor train từ B5 (N_train, 48, 12), có thể chứa NaN
        M_train : mask tensor train (N_train, 48, 12)
        cfg     : cấu hình pipeline

    Returns:
        Dict {feature_name: (lo, hi)} - IQR bounds tính trên train
        Feature không đủ data hoặc IQR=0 không có trong dict (sẽ bỏ qua khi apply)
    """
    bounds: Dict[str, Tuple[float, float]] = {}
    for f, col in enumerate(cfg.feature_cols):
        observed = X_train[:, :, f][M_train[:, :, f] == 1]
        observed = observed[~np.isnan(observed)]
        if len(observed) < 4:
            continue
        q1, q3 = np.quantile(observed, 0.25), np.quantile(observed, 0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lo = float(q1 - cfg.iqr_multiplier * iqr)
        hi = float(q3 + cfg.iqr_multiplier * iqr)
        bounds[col] = (lo, hi)
    return bounds


def apply_iqr_outlier(
    X: np.ndarray,
    M: np.ndarray,
    iqr_bounds: Dict[str, Tuple[float, float]],
    cfg: PipelineConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Áp dụng IQR bounds (tính từ train) lên tensor X và M

    Outlier chuyển giá trị thành NaN, mask tương ứng bằng 0
    Train, val, test đều dùng chung 1 bounds

    Args:
        X          : tensor (N, 48, 12).
        M          : mask tensor (N, 48, 12).
        iqr_bounds : bounds từ compute_iqr_bounds.
        cfg        : cấu hình pipeline.

    Returns:
        (X_clean, M_updated) cùng shape với input
    """
    X = X.copy()
    M = M.copy()
    for f, col in enumerate(cfg.feature_cols):
        if col not in iqr_bounds:
            continue
        lo, hi = iqr_bounds[col]
        Xf = X[:, :, f]
        outlier = ~np.isnan(Xf) & ((Xf < lo) | (Xf > hi))
        if outlier.any():
            X[:, :, f][outlier] = np.nan
            M[:, :, f][outlier] = 0
    return X, M


# 3. Forward-fill per patient
# Notes: Không update mask, ffill không tạo observation mới, groupby stay_id
def b3_forward_fill(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """
    Forward-fill theo từng bệnh nhân (groupby stay_id), có giới hạn theo nhóm feature
    Giới hạn phản ánh tần suất đo lường thực tế trong ICU: vitals 4h | urine 1h | labs 24h
    Mask không được cập nhật - giá trị được fill vẫn giữ M = 0 vì forward-fill không tạo ra sự kiện đo lường mới
    
    Args:
        df  : DataFrame sau B2
        cfg : cấu hình pipeline

    Returns:
        DataFrame đã forward-fill, sắp xếp theo (stay_id, time_step_t)
    """
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
    """
    Chuyển DataFrame dạng long-format sang tensor 3 chiều

    Validations trước khi reshape:
    - Lọc bỏ hàng có time_step_t ngoài [0, 47]
    - Deduplicate (stay_id, time_step_t), giữ hàng đầu tiên
    - Bệnh nhân LOS < 48h được pad NaN cho giờ thiếu (không drop)

    Args:
        df      : DataFrame sau B3.
        mask_df : mask DataFrame từ B1 (đã cập nhật qua B2)
        cfg     : cấu hình pipeline

    Returns:
        X        : (N, 48, 12) float32 - features, có thể chứa NaN
        M        : (N, 48, 12) int8    - mask {0, 1}
        y        : (N,)        int32   - nhãn tử vong
        stay_ids : (N,)        int64   - stay ID để truy vết bệnh nhân gốc
    """
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
    """
    Chia dữ liệu thành train/val/test theo stay_id, có stratify

    Chia theo stay_id (không theo row) để tránh patient leakage
    Stratify theo hospital_expire_flag để giữ tỷ lệ tử vong ~11.2% trong cả 3 tập

    Với N < 6: fallback sang chia thủ công 1 bệnh nhân/tập (dùng cho chạy thử trên sample nhỏ)

    Args:
        X, M, y  : tensor features, mask, nhãn từ B4
        stay_ids : mảng stay ID tương ứng
        cfg      : cấu hình pipeline

    Returns:
        Dict gồm 3 key "train", "val", "test", mỗi key chứa {"X", "M", "y", "stay_ids"}
    """
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
    """
    Tính mean per-feature trên tập train, bỏ qua NaN

    Args:
        X_train : tensor train (N_train, 48, 12), có thể chứa NaN
        cfg     : cấu hình pipeline

    Returns:
        Mảng (12,) float32 - mean của mỗi feature
        Feature toàn NaN được gán mean = 0
    """
    flat = X_train.reshape(-1, cfg.n_features)
    means = np.nanmean(flat, axis=0)
    means = np.where(np.isnan(means), 0.0, means)
    return means.astype(np.float32)


def apply_mean_imputation(X: np.ndarray, train_means: np.ndarray) -> np.ndarray:
    """
    Fill NaN bằng train_means. Val và test phải dùng train_means từ compute_train_means, 
    không tự tính lại để tránh distribution leakage.

    Args:
        X           : tensor (N, 48, 12), có thể chứa NaN
        train_means : mảng (12,) từ compute_train_means

    Returns:
        Tensor cùng shape, không còn NaN
    """
    X = X.copy()
    nan_mask = np.isnan(X)
    X[nan_mask] = np.broadcast_to(train_means, X.shape)[nan_mask]
    return X


# 7. StandardScaler fit on observed train values (mask=1)
# Notes: Không fit trên data đã impute -> tránh variance shrinkage
def b7_fit_scaler_on_observed(
    X_train: np.ndarray,
    M_train: np.ndarray,
    cfg: PipelineConfig,
) -> StandardScaler:
    """
    Fit StandardScaler chỉ trên các vị trí được đo thực tế (mask = 1) của tập train.

    Không fit trên toàn bộ data đã impute vì
    imputation kéo nhiều giá trị về mean 
    -> variance co lại
    -> scaler.scale_ nhỏ hơn thực tế 
    -> giá trị thật sau transform bị inflate, vượt ngưỡng [-3, 3]

    Args:
        X_train : tensor train sau B6, không còn NaN
        M_train : mask tensor train (N_train, 48, 12)
        cfg     : cấu hình pipeline

    Returns:
        StandardScaler đã có mean_ và scale_ phản ánh phân phối thật của giá trị được đo. Dùng để transform cả 3 tập
    """
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
    """
    Chuẩn hóa tensor X bằng scaler đã fit

    Args:
        X      : tensor (N, 48, 12) chưa chuẩn hóa
        scaler : StandardScaler từ b7_fit_scaler_on_observed

    Returns:
        Tensor (N, 48, 12) float32 đã chuẩn hóa
    """
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
    """
    Ghép X_norm và M thành tensor cuối (N, 48, 24), xuất 10 files

    Layout channel: 0–11 = features đã chuẩn hóa, 12–23 = mask {0, 1}

    Files xuất ra cfg.output_dir:
        Xm_{train,val,test}.npy        (N, 48, 24) float32
        y_{train,val,test}.npy         (N,)        int32
        stay_ids_{train,val,test}.csv  (N, 1)
        scaler.pkl

    Args:
        splits : dict từ b5_split, đã có key "X_norm" sau B7
        scaler : StandardScaler đã fit từ B7
        cfg    : cấu hình pipeline
    """
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
    """
    Thực thi toàn bộ pipeline B1–B8 theo thứ tự cố định
    Thứ tự sau fix:
        B1 -> B2a (clip) -> B3 -> B4 -> B5 (split) -> B2b (IQR train-only) -> B6 -> B7 -> B8
    IQR detection được tách thành 2 hàm và đẩy xuống sau split để bounds chỉ học từ train, không leak từ val/test

    Args:
        cfg : cấu hình pipeline

    Returns:
        Dict splits {"train", "val", "test"}, mỗi tập chứa
        X, M, y, stay_ids, X_norm sau khi xử lý đầy đủ
    """
    df = pd.read_csv(cfg.data_path, parse_dates=["chart_hour"])

    # B1: mask
    mask_df = b1_create_mask(df, cfg)

    # B2a: clip clinical bounds
    df_clipped = b2a_clip_clinical(df, cfg)

    # B3: forward-fill
    df_ffilled = b3_forward_fill(df_clipped, cfg)

    # B4: reshape
    X, M, y, stay_ids = b4_reshape_to_tensor(df_ffilled, mask_df, cfg)

    # B5: split - chạy trước IQR để tránh leakage
    splits = b5_split(X, M, y, stay_ids, cfg)

    # B2b: IQR - tính bounds chỉ trên train, áp dụng cho cả 3 splits
    iqr_bounds = compute_iqr_bounds(splits["train"]["X"], splits["train"]["M"], cfg)
    for name in ["train", "val", "test"]:
        splits[name]["X"], splits[name]["M"] = apply_iqr_outlier(
            splits[name]["X"], splits[name]["M"], iqr_bounds, cfg
        )

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
