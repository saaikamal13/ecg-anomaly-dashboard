# 🫀 ECG Anomaly Detection Interactive Dashboard

An interactive web application for detecting anomalies in ECG (Electrocardiogram) signals using a Deep Neural Autoencoder trained with PyTorch.

## How It Works

The system uses a **Deep Neural Autoencoder** trained exclusively on **normal** ECG heartbeat signals (Label 1.0). The model learns to compress and reconstruct the characteristic morphology of normal heartbeats (P-wave, QRS-complex, ST-segment, T-wave).

When presented with an anomalous heartbeat (e.g. premature ventricular contraction, ventricular ectopic beats, or bundle branch blocks), the reconstruction error (Mean Squared Error) spikes significantly above the learned decision threshold, immediately triggering an anomaly alert with localized time-step error visualization.

### Model Architecture
```
Input (140 timesteps)
  │
  ▼ [Encoder]
  Linear(140 → 64) → BatchNorm1d → LeakyReLU(0.01) → Dropout(0.1)
  Linear(64 → 32)  → BatchNorm1d → LeakyReLU(0.01) → Dropout(0.1)
  Linear(32 → 16)  → BatchNorm1d → LeakyReLU(0.01)
  │
  ▼ [Bottleneck] 16 Latent Dimensions
  │
  ▼ [Decoder]
  Linear(16 → 32)  → BatchNorm1d → LeakyReLU(0.01) → Dropout(0.1)
  Linear(32 → 64)  → BatchNorm1d → LeakyReLU(0.01) → Dropout(0.1)
  Linear(64 → 140)
  │
  ▼
Reconstructed Signal (140 timesteps)
```

## Performance Metrics

- **Overall Accuracy**: **98.91%**
- **Precision**: **99.18%**
- **Recall**: **99.42%**
- **F1 Score**: **0.9930**
- **Normal Heartbeats Classified Correctly**: 567 / 584 (97.09%)
- **Anomalous Heartbeats Detected**: 2067 / 2079 (99.42%)
- **Decision Threshold ($\mu + 2\sigma$)**: ~0.001593 MSE

## Dataset

- **4,998 ECG samples** with 140 time-step features each
- **2,919 Normal** heartbeats (label = 1.0)
- **2,079 Anomalous** heartbeats (label = 0.0)

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train_model.py
```

### 3. Launch the dashboard
```bash
python app.py
```
Open **http://localhost:5001** in your browser.

## Dashboard Features

| Tab | Description |
|-----|-------------|
| 🔍 **Signal Explorer** | Browse all 4,998 ECG signals, navigate by index, or load random normal/anomaly samples. See original vs reconstructed waveforms and per-sample MSE. |
| 🧠 **Live Prediction** | Paste custom ECG data (140 values) for real-time anomaly detection. Load example normal or anomaly signals with one click. |
| 📊 **Model Performance** | View accuracy, precision, recall, F1 score, confusion matrix, reconstruction error distribution, and scatter plots. |
| 📈 **Training History** | Visualize training and validation loss curves across epochs alongside model architecture details. |

## Project Structure

```
ecg-anomaly-detector/
├── ecg.csv                  # Dataset (4,998 samples × 140 timesteps + 1 label)
├── train_model.py           # Model training script (PyTorch)
├── app.py                   # Flask web application & REST API
├── requirements.txt         # Python dependencies
├── templates/
│   └── index.html           # Single-page interactive dashboard UI (Plotly.js + Dark theme)
├── saved_model/
│   └── ecg_autoencoder.pt   # PyTorch model weights
├── scaler_params.json       # Normalization parameters (min, max, range)
├── threshold.json           # Anomaly detection threshold & performance metrics
└── training_history.json    # Loss curves per epoch
```
