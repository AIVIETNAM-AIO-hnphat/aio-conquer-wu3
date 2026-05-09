import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import shap
import torch
import torch.nn as nn
import torch.optim as optim
from captum.attr import IntegratedGradients

# -----------------------------
# 1. Load dữ liệu
# -----------------------------
DATA_DIR = r"D:\AIO_Conquer\aio-conquer-wu3\dataset"
X_train = np.load(f"{DATA_DIR}\\Xm_train.npy")
y_train = np.load(f"{DATA_DIR}\\y_train.npy")
X_val = np.load(f"{DATA_DIR}\\Xm_val.npy")
y_val = np.load(f"{DATA_DIR}\\y_val.npy")

# -----------------------------
# 2. Baseline LightGBM
# -----------------------------
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)

lgb_model = lgb.LGBMClassifier()
lgb_model.fit(X_train_flat, y_train)
y_pred_lgb = lgb_model.predict_proba(X_val_flat)[:,1]

print("LightGBM AUROC:", roc_auc_score(y_val, y_pred_lgb))
print("LightGBM AUPRC:", average_precision_score(y_val, y_pred_lgb))

# Calibration curve
prob_true, prob_pred = calibration_curve(y_val, y_pred_lgb, n_bins=10)
plt.plot(prob_pred, prob_true, marker='o')
plt.title("Calibration curve - LightGBM")
plt.show()

# SHAP analysis for LightGBM
explainer = shap.TreeExplainer(lgb_model)
shap_values = explainer.shap_values(X_val_flat)
shap.summary_plot(shap_values, X_val_flat)

# -----------------------------
# 3. SOFA score baseline
# -----------------------------
def sofa_score(map_val, gcs_val, creat_val):
    score = 0
    if map_val < 70: score += 1
    if gcs_val < 13: score += 1
    if gcs_val < 6: score += 2
    if creat_val > 1.2: score += 1
    if creat_val > 2.0: score += 2
    return score

map_idx, gcs_idx, creat_idx = 0, 1, 2
sofa_scores = [sofa_score(X_val[i,:,map_idx].mean(),
                          X_val[i,:,gcs_idx].mean(),
                          X_val[i,:,creat_idx].mean()) for i in range(len(X_val))]
print("SOFA AUROC:", roc_auc_score(y_val, sofa_scores))
print("SOFA AUPRC:", average_precision_score(y_val, sofa_scores))

# -----------------------------
# 4. Main model: LSTM dual-input
# -----------------------------
class DualInputRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.value_rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.mask_rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim*2, 1)

    def forward(self, x_val, x_mask):
        _, (h_val, _) = self.value_rnn(x_val)
        _, (h_mask, _) = self.mask_rnn(x_mask)
        h = torch.cat([h_val[-1], h_mask[-1]], dim=1)
        return torch.sigmoid(self.fc(h))

X_train_mask = np.ones_like(X_train)
X_val_mask = np.ones_like(X_val)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DualInputRNN(input_dim=X_train.shape[2], hidden_dim=64).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
X_train_mask_t = torch.tensor(X_train_mask, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)

for epoch in range(5):
    optimizer.zero_grad()
    outputs = model(X_train_t, X_train_mask_t).squeeze()
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}, Loss {loss.item()}")

X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
X_val_mask_t = torch.tensor(X_val_mask, dtype=torch.float32).to(device)
y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

y_pred_lstm = model(X_val_t, X_val_mask_t).detach().cpu().numpy().squeeze()
print("LSTM AUROC:", roc_auc_score(y_val, y_pred_lstm))
print("LSTM AUPRC:", average_precision_score(y_val, y_pred_lstm))

# Calibration curve LSTM
prob_true, prob_pred = calibration_curve(y_val, y_pred_lstm, n_bins=10)
plt.plot(prob_pred, prob_true, marker='o')
plt.title("Calibration curve - LSTM")
plt.show()

# -----------------------------
# 5. Per-timestep AUROC
# -----------------------------
auroc_per_timestep = []
for t in range(X_val.shape[1]):
    # Flatten features at timestep t
    X_val_timestep = X_val[:,t,:]
    model_step = lgb.LGBMClassifier()
    model_step.fit(X_train[:,t,:], y_train)
    y_pred_step = model_step.predict_proba(X_val_timestep)[:,1]
    auroc_per_timestep.append(roc_auc_score(y_val, y_pred_step))
print("Per-timestep AUROC:", auroc_per_timestep)

# -----------------------------
# 6. SHAP cho LSTM bằng Captum
# -----------------------------
ig = IntegratedGradients(model)

# Chọn một batch nhỏ để phân tích
X_batch = X_val_t[:50]
X_mask_batch = X_val_mask_t[:50]

# Captum yêu cầu forward function chỉ nhận một input → wrap lại
def forward_func(inputs):
    return model(inputs, X_mask_batch).squeeze()

attributions = ig.attribute(X_batch, target=None, n_steps=50)
attr_np = attributions.detach().cpu().numpy()

# Trung bình theo timestep để xem feature importance
feature_importance = attr_np.mean(axis=(0,1))
plt.bar(range(len(feature_importance)), feature_importance)
plt.title("Feature importance (Captum IG)")
plt.show()
