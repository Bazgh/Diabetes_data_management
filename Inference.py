import pandas as pd
import numpy as np
import glob
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import joblib

data = torch.load("data/Ohio2018_processed/test_dataset.pt")
X, Y = data["X"], data["Y"]
print("Loaded dataset shapes:", X.shape, Y.shape)
x_nan_count = torch.isnan(X).sum().item()
y_nan_count = torch.isnan(Y).sum().item()

print(f"NaNs in X: {x_nan_count}")
print(f"NaNs in Y: {y_nan_count}")

import numpy as np
X_np = X.numpy()
Y_np = Y.numpy()
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# Load model
model1 = joblib.load("linear_regression_model.pkl")


# Evaluate
Y_pred = model1.predict(X_np)
import matplotlib.pyplot as plt


# pick one test sample
i = 4
y_true = Y_np[i]
y_pred = Y_pred[i]

plt.figure(figsize=(6,4))
plt.plot(range(1, len(y_true)+1), y_true, 'o-', label='Ground Truth')
plt.plot(range(1, len(y_pred)+1), y_pred, 's--', label='Prediction')
plt.xlabel("Prediction step (5-min intervals)")
plt.ylabel("Normalized CBG")
plt.title("Linear Regression — 6-step Prediction Example")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
rmse = np.sqrt(mean_squared_error(Y_np, Y_pred))
r2 = r2_score(Y_np, Y_pred)

print("✅ Linear Regression trained successfully!")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Load saved weights
import torch
import torch.nn as nn

class LSTMRegressor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, output_size=6):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)

# --------------------------
# Load model properly
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTMRegressor().to(device)            # ✅ instantiate model
state_dict = torch.load("lstm_regressor.pth", map_location=device)
model.load_state_dict(state_dict)             # ✅ load weights
model.eval()

# --------------------------
# Prepare input properly
# --------------------------

# If X_np is numpy array, convert to torch:
X_seq = torch.from_numpy(X_np).float()        # [N, 36]
X_seq = X_seq.unsqueeze(-1).to(device)        # [N, 36, 1]

# --------------------------
# Predict
# --------------------------
with torch.no_grad():
    Y_pred = model(X_seq).cpu().numpy()       # [N, 6]

# --------------------------
# Visualization
# --------------------------

i = 0
y_true = Y_np[i]
y_pred = Y_pred[i]

plt.figure(figsize=(6, 4))
plt.plot(range(1, 7), y_true, 'o-', label="Ground Truth")
plt.plot(range(1, 7), y_pred, 's--', label="LSTM Prediction")
plt.xlabel("Prediction Step (5-min intervals)")
plt.ylabel("Normalized CBG")
plt.title("LSTM — 6-step Forecast Example")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()