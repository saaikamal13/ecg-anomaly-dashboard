"""
ECG Anomaly Detection — Interactive Web Dashboard
====================================================
A Flask + Plotly dashboard that lets you:
  • Browse & classify ECG signals from the dataset
  • Upload custom ECG data for real-time anomaly detection
  • Visualize reconstruction error & model performance
  • Explore training history & model metrics
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from flask import Flask, render_template, request, jsonify

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "ecg.csv")
MODEL_DIR = os.path.join(BASE_DIR, "saved_model")
MODEL_PATH = os.path.join(MODEL_DIR, "ecg_autoencoder.pt")
SCALER_PATH = os.path.join(BASE_DIR, "scaler_params.json")
THRESHOLD_PATH = os.path.join(BASE_DIR, "threshold.json")
HISTORY_PATH = os.path.join(BASE_DIR, "training_history.json")


# ── Model Definition ───────────────────────────────────────────────────────────
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


# ── Load assets ────────────────────────────────────────────────────────────────
print("Loading model and data …")
device = torch.device("cpu")
model = ECGAutoencoder().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

with open(SCALER_PATH) as f:
    sp = json.load(f)
    min_val = float(sp["min_val"])
    max_val = float(sp["max_val"])
    val_range = float(sp["val_range"])

with open(THRESHOLD_PATH) as f:
    threshold_info = json.load(f)
    THRESHOLD = float(threshold_info["threshold"])

with open(HISTORY_PATH) as f:
    training_history = json.load(f)

# Load dataset
df = pd.read_csv(DATA_PATH, header=None)
all_data = df.iloc[:, :-1].values.astype(np.float32)
all_labels = df.iloc[:, -1].values.astype(np.float32)

# Pre-compute reconstruction errors for all samples
all_scaled = (all_data - min_val) / val_range
all_tensor = torch.tensor(all_scaled, dtype=torch.float32).to(device)

with torch.no_grad():
    all_preds_scaled = model(all_tensor).cpu().numpy()

all_mse = np.mean(np.power(all_scaled - all_preds_scaled, 2), axis=1)

print(f"Loaded {len(all_data)} samples. Threshold: {THRESHOLD:.6f}")

# ── Flask App ──────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"),
            static_folder=os.path.join(BASE_DIR, "static"))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/sample/<int:idx>")
def get_sample(idx):
    """Return a single ECG sample with prediction."""
    if idx < 0 or idx >= len(all_data):
        return jsonify({"error": "Index out of range"}), 400

    signal = all_data[idx].tolist()
    scaled = (all_data[idx] - min_val) / val_range
    inp = torch.tensor(scaled.reshape(1, -1), dtype=torch.float32).to(device)

    with torch.no_grad():
        recon_scaled = model(inp).cpu().numpy().reshape(-1)

    reconstructed = (recon_scaled * val_range + min_val).tolist()
    mse = float(all_mse[idx])
    true_label = "Normal" if all_labels[idx] == 1.0 else "Anomaly"
    predicted = "Normal" if mse <= THRESHOLD else "Anomaly"

    return jsonify({
        "index": idx,
        "signal": signal,
        "reconstructed": reconstructed,
        "mse": mse,
        "threshold": THRESHOLD,
        "true_label": true_label,
        "predicted_label": predicted,
        "is_correct": true_label == predicted,
    })


@app.route("/api/random_sample")
def random_sample():
    """Return a random sample."""
    label_filter = request.args.get("label", None)
    if label_filter == "normal":
        candidates = np.where(all_labels == 1.0)[0]
    elif label_filter == "anomaly":
        candidates = np.where(all_labels == 0.0)[0]
    else:
        candidates = np.arange(len(all_labels))
    idx = int(np.random.choice(candidates))
    return get_sample(idx)


@app.route("/api/predict", methods=["POST"])
def predict_custom():
    """Accept custom ECG data (140 comma-separated values) and predict."""
    try:
        body = request.get_json()
        values = body.get("values", [])
        if len(values) != 140:
            return jsonify({"error": f"Expected 140 values, got {len(values)}"}), 400

        signal = np.array(values, dtype=np.float32)
        scaled = (signal - min_val) / val_range
        inp = torch.tensor(scaled.reshape(1, -1), dtype=torch.float32).to(device)

        with torch.no_grad():
            pred_scaled = model(inp).cpu().numpy().reshape(-1)

        reconstructed = (pred_scaled * val_range + min_val).tolist()
        mse = float(np.mean(np.power(scaled - pred_scaled, 2)))
        predicted = "Normal" if mse <= THRESHOLD else "Anomaly"

        return jsonify({
            "signal": signal.tolist(),
            "reconstructed": reconstructed,
            "mse": mse,
            "threshold": THRESHOLD,
            "predicted_label": predicted,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/overview")
def overview():
    """Return dataset overview & model performance metrics."""
    normal_mse = all_mse[all_labels == 1.0].tolist()
    anomaly_mse = all_mse[all_labels == 0.0].tolist()

    # Confusion matrix:
    # Actual Normal (label 1.0) -> predicted Normal (mse <= THRESHOLD) is TN, predicted Anomaly is FP
    # Actual Anomaly (label 0.0) -> predicted Anomaly (mse > THRESHOLD) is TP, predicted Normal is FN
    predicted_is_anomaly = (all_mse > THRESHOLD).astype(int)
    actual_is_anomaly = (all_labels == 0.0).astype(int)

    tp = int(np.sum((predicted_is_anomaly == 1) & (actual_is_anomaly == 1)))
    tn = int(np.sum((predicted_is_anomaly == 0) & (actual_is_anomaly == 0)))
    fp = int(np.sum((predicted_is_anomaly == 1) & (actual_is_anomaly == 0)))
    fn = int(np.sum((predicted_is_anomaly == 0) & (actual_is_anomaly == 1)))

    return jsonify({
        "total_samples": len(all_data),
        "normal_count": int(np.sum(all_labels == 1.0)),
        "anomaly_count": int(np.sum(all_labels == 0.0)),
        "threshold": THRESHOLD,
        "threshold_info": threshold_info,
        "normal_mse": normal_mse,
        "anomaly_mse": anomaly_mse,
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    })


@app.route("/api/training_history")
def get_training_history():
    """Return training loss history."""
    return jsonify(training_history)


@app.route("/api/batch_samples")
def batch_samples():
    """Return multiple sample indices with their MSE and labels for the explorer."""
    start = int(request.args.get("start", 0))
    count = int(request.args.get("count", 50))
    end = min(start + count, len(all_data))

    samples = []
    for i in range(start, end):
        samples.append({
            "index": i,
            "mse": float(all_mse[i]),
            "true_label": "Normal" if all_labels[i] == 1.0 else "Anomaly",
            "predicted_label": "Normal" if all_mse[i] <= THRESHOLD else "Anomaly",
        })
    return jsonify({"samples": samples, "total": len(all_data)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"\n🫀 ECG Anomaly Detection Dashboard")
    print(f"   Open http://localhost:{port} in your browser\n")
    app.run(host="0.0.0.0", port=port, debug=False)
