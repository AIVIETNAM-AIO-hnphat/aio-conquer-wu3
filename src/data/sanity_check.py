import sys, os
import numpy as np

output_dir = sys.argv[1] if len(sys.argv) > 1 else "processed"

all_ok = True

for split in ["train", "val", "test"]:
    Xm = np.load(os.path.join(output_dir, f"Xm_{split}.npy"))
    y  = np.load(os.path.join(output_dir, f"y_{split}.npy"))
    n  = len(y)
    feat = Xm[:, :, :12]
    mask = Xm[:, :, 12:]

    issues = []

    if np.isnan(Xm).any():
        issues.append(f"con NaN ({int(np.isnan(Xm).sum())})")

    if Xm.shape[1:] != (48, 24):
        issues.append(f"shape sai ({Xm.shape})")

    if not set(np.unique(mask).tolist()).issubset({0.0, 1.0}):
        issues.append("mask co gia tri ngoai 0/1")

    abs_max = max(abs(float(feat.min())), abs(float(feat.max())))
    if not np.isfinite(feat).all() or abs_max > 20:
        issues.append(f"value range bat thuong (max |z| = {abs_max:.1f})")

    mort = float(y.mean()) * 100
    tol = 100 if n < 10 else 10 if n < 100 else 2
    if abs(mort - 11.2) > tol:
        issues.append(f"mortality lech ({mort:.1f}%)")

    if issues:
        all_ok = False
        print(f"{split}: FAIL — " + ", ".join(issues))
    else:
        print(f"{split}: OK (N={n}, mortality={mort:.1f}%)")

print()
print("PASSED" if all_ok else "FAILED")