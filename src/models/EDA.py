import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from scipy.stats import mannwhitneyu
from matplotlib.backends.backend_pdf import PdfPages

# Load dữ liệu
X_train = np.load(r"D:\AIO_Conquer\aio-conquer-wu3\src\data\processed\Xm_train.npy")
y_train = np.load(r"D:\AIO_Conquer\aio-conquer-wu3\src\data\processed\y_train.npy")
ids_train = pd.read_csv(r"D:\AIO_Conquer\aio-conquer-wu3\src\data\processed\stay_ids_train.csv")

X_val = np.load(r"D:\AIO_Conquer\aio-conquer-wu3\src\data\processed\Xm_val.npy")
y_val = np.load(r"D:\AIO_Conquer\aio-conquer-wu3\src\data\processed\y_val.npy")
ids_val = pd.read_csv(r"D:\AIO_Conquer\aio-conquer-wu3\src\data\processed\stay_ids_val.csv")

X_test = np.load(r"D:\AIO_Conquer\aio-conquer-wu3\src\data\processed\Xm_test.npy")
y_test = np.load(r"D:\AIO_Conquer\aio-conquer-wu3\src\data\processed\y_test.npy")
ids_test = pd.read_csv(r"D:\AIO_Conquer\aio-conquer-wu3\src\data\processed\stay_ids_test.csv")

with open(r"D:\AIO_Conquer\aio-conquer-wu3\src\data\processed\scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Cohort overview
def cohort_overview(ids, y):
    df = ids.copy()
    df["outcome"] = y
    print(df.groupby("gender")["outcome"].mean())
    sns.histplot(df["anchor_age"], bins = 30, kde = True)
    plt.title("Age distribution")
    plt.show()

# Missingness Heatmap
def missingness_heatmap(X):
    missing = np.isnan(X).mean(axis=0)
    sns.heatmap(missing.T, cmap="Reds", cbar=True)
    plt.xlabel("Time (hours)")
    plt.ylabel("Missingness heatmap")
    plt.title("Missingness heatmap")
    plt.show()

# Temporal trajectory
def temporal_trajectory(X, y, feature_idx, feature_name):
    for outcome in [0,1]:
        mask = (y == outcome)
        mean = np.nanmean(X[mask, :, feature_idx], axis=0)
        std = np.nanstd(X[mask, :, feature_idx], axis=0)
        plt.plot(mean, label=f"{feature_name}, outcome={outcome}")
        plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2)
    plt.legend()
    plt.title(f"Temporal trajectory of {feature_name}")
    plt.show()

# Statistical tests
def statistical_tests(X, y):
    results = []
    for f in range (X.shape[2]):
        group0 = X[y==0, :, f].flatten()
        group1 = X[y==1, :, f].flatten()
        stat, p = mannwhitneyu(group0[~np.isnan(group0)], group1[~np.isnan(group1)])
        results.append((f, p))
    return pd.DataFrame(results, columns=["feature_idx", "p_value"])

# Class imbalance
def class_imbalance(y):
    unique, counts = np.unique(y, return_counts=True)
    print(dict(zip(unique, counts)))
    sns.countplot(x=y)
    plt.title("Class distribution")
    plt.show()

if __name__ == "__main__":
    # cohort_overview(ids_train, y_train)
    missingness_heatmap(X_train)
    temporal_trajectory(X_train, y_train, feature_idx=0, feature_name="MAP")
    stats = statistical_tests(X_train, y_train)
    print(stats.head())
    class_imbalance(y_train)