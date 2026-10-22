"""
ECG Anomaly Detection — Train Autoencoder Model
===================================================
This script trains a Deep Neural Autoencoder on NORMAL ECG signals.
At inference time, anomalies are detected by high reconstruction error (MSE).

Dataset: ecg.csv (4998 samples × 140 features + 1 label)
  Label 1.0 = Normal heartbeat
  Label 0.0 = Anomalous heartbeat
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# ── Configuration ──────────────────────────────────────────────────────────────
SEED = 42
TEST_SIZE = 0.2
EPOCHS = 50
BATCH_SIZE = 32
LR = 2e-3
WEIGHT_DECAY = 1e-5

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "ecg.csv")
MODEL_DIR = os.path.join(BASE_DIR, "saved_model")
MODEL_PATH = os.path.join(MODEL_DIR, "ecg_autoencoder.pt")
SCALER_PATH = os.path.join(BASE_DIR, "scaler_params.json")
THRESHOLD_PATH = os.path.join(BASE_DIR, "threshold.json")
HISTORY_PATH = os.path.join(BASE_DIR, "training_history.json")

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cpu")
print(f"Using device: {device}")

# ── 1. Load & prepare data ────────────────────────────────────────────────────
print("Loading dataset …")
df = pd.read_csv(DATA_PATH, header=None)
print(f"  Shape: {df.shape}")

# Last column is the label
data = df.iloc[:, :-1].values.astype(np.float32)
labels = df.iloc[:, -1].values.astype(np.float32)

# Split normal vs anomalous
normal_data = data[labels == 1.0]
anomaly_data = data[labels == 0.0]
print(f"  Normal: {len(normal_data)}, Anomaly: {len(anomaly_data)}")

# ── 2. Train/validation split (normal data only) ──────────────────────────────
X_train, X_val = train_test_split(normal_data, test_size=TEST_SIZE, random_state=SEED)

# Scale using min-max of normal training data
min_val = float(np.min(X_train))
max_val = float(np.max(X_train))
val_range = max_val - min_val

X_train_s = (X_train - min_val) / val_range
X_val_s = (X_val - min_val) / val_range
anomaly_s = (anomaly_data - min_val) / val_range

# Save scaler parameters for inference
scaler_params = {
    "min_val": min_val,
    "max_val": max_val,
    "val_range": val_range
}
with open(SCALER_PATH, "w") as f:
    json.dump(scaler_params, f, indent=2)
print(f"  Scaler saved → {SCALER_PATH}")

# ── 3. Build Autoencoder Architecture ──────────────────────────────────────────
print("\nBuilding Deep Autoencoder …")


class ECGAutoencoder(nn.Module):
    def __init__(self, input_dim=140, latent_dim=16):
        super().__init__()
        # Encoder: 140 -> 64 -> 32 -> 16
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1),
            nn.Linear(32, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(0.01)
        )
        # Decoder: 16 -> 32 -> 64 -> 140
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


autoencoder = ECGAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(autoencoder.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

train_tensor = torch.tensor(X_train_s, dtype=torch.float32).to(device)
val_tensor = torch.tensor(X_val_s, dtype=torch.float32).to(device)
anomaly_tensor = torch.tensor(anomaly_s, dtype=torch.float32).to(device)

train_loader = DataLoader(TensorDataset(train_tensor), batch_size=BATCH_SIZE, shuffle=True)

# ── 4. Train ──────────────────────────────────────────────────────────────────
print("\nTraining …")
train_loss_history = []
val_loss_history = []
best_val_loss = float("inf")
best_state = None

for epoch in range(EPOCHS):
    autoencoder.train()
    running_loss = 0.0
    for (batch_x,) in train_loader:
        optimizer.zero_grad()
        recon = autoencoder(batch_x)
        loss = criterion(recon, batch_x)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(batch_x)

    epoch_train_loss = running_loss / len(train_tensor)

    # Validation
    autoencoder.eval()
    with torch.no_grad():
        val_recon = autoencoder(val_tensor)
        epoch_val_loss = criterion(val_recon, val_tensor).item()

    train_loss_history.append(float(epoch_train_loss))
    val_loss_history.append(float(epoch_val_loss))
    scheduler.step(epoch_val_loss)

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        best_state = autoencoder.state_dict().copy()

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"  Epoch [{epoch+1:2d}/{EPOCHS}] Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}")

# Restore best model weights
if best_state is not None:
    autoencoder.load_state_dict(best_state)

# Save training history
with open(HISTORY_PATH, "w") as f:
    json.dump({"loss": train_loss_history, "val_loss": val_loss_history}, f, indent=2)

# ── 5. Compute threshold ──────────────────────────────────────────────────────
print("\nComputing anomaly threshold …")
autoencoder.eval()
with torch.no_grad():
    train_pred = autoencoder(train_tensor).cpu().numpy()
    val_pred = autoencoder(val_tensor).cpu().numpy()
    anomaly_pred = autoencoder(anomaly_tensor).cpu().numpy()

train_mse = np.mean(np.power(X_train_s - train_pred, 2), axis=1)
val_mse = np.mean(np.power(X_val_s - val_pred, 2), axis=1)
anomaly_mse = np.mean(np.power(anomaly_s - anomaly_pred, 2), axis=1)

# Threshold: mean + 2*std of normal reconstruction error
threshold = float(np.mean(train_mse) + 2.0 * np.std(train_mse))
print(f"  Normal Train MSE — mean: {np.mean(train_mse):.6f}, std: {np.std(train_mse):.6f}")
print(f"  Normal Val   MSE — mean: {np.mean(val_mse):.6f}, std: {np.std(val_mse):.6f}")
print(f"  Anomaly      MSE — mean: {np.mean(anomaly_mse):.6f}, std: {np.std(anomaly_mse):.6f}")
print(f"  Threshold: {threshold:.6f}")

# Evaluate accuracy on validation normals + all anomalies
normal_correct = int(np.sum(val_mse <= threshold))
normal_total = int(len(val_mse))
anomaly_correct = int(np.sum(anomaly_mse > threshold))
anomaly_total = int(len(anomaly_mse))

accuracy = (normal_correct + anomaly_correct) / (normal_total + anomaly_total)
precision = anomaly_correct / (anomaly_correct + (normal_total - normal_correct)) if (anomaly_correct + (normal_total - normal_correct)) > 0 else 0
recall = anomaly_correct / anomaly_total if anomaly_total > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n  Overall Accuracy: {accuracy:.4f}")
print(f"  Normal detected correctly: {normal_correct}/{normal_total} ({normal_correct/normal_total:.2%})")
print(f"  Anomaly detected correctly: {anomaly_correct}/{anomaly_total} ({anomaly_correct/anomaly_total:.2%})")
print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

threshold_info = {
    "threshold": threshold,
    "normal_mse_mean": float(np.mean(train_mse)),
    "normal_mse_std": float(np.std(train_mse)),
    "anomaly_mse_mean": float(np.mean(anomaly_mse)),
    "anomaly_mse_std": float(np.std(anomaly_mse)),
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "normal_correct": normal_correct,
    "normal_total": normal_total,
    "anomaly_correct": anomaly_correct,
    "anomaly_total": anomaly_total,
}
with open(THRESHOLD_PATH, "w") as f:
    json.dump(threshold_info, f, indent=2)

# ── 6. Save model ─────────────────────────────────────────────────────────────
os.makedirs(MODEL_DIR, exist_ok=True)
torch.save(autoencoder.state_dict(), MODEL_PATH)
print(f"\n✅ Model saved → {MODEL_PATH}")
print(f"✅ Scaler saved → {SCALER_PATH}")
print(f"✅ Threshold saved → {THRESHOLD_PATH}")
print(f"✅ Training history saved → {HISTORY_PATH}")
print("\nDone! Run `python app.py` to start the interactive dashboard.")
