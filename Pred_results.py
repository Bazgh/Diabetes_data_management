import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cbg_max = 400
cbg_min = 40

files = {
    "LR Generalised": "lr-generalised-pred/lr-generalised/generalised_lr_recursive_predictions.csv",
    "TFT": "tft_generalised/tft_generalised/tft_predictions.csv",
    "TFT + HR": "tft_generalised_withHr/generalised/tft_cbg_hr_predictions.csv",
    "LSTM": "lstm_Y_true_and_predictions_len24.csv"
}

def extract_index(colname):
    digits = re.findall(r'\d+', colname)
    return int(digits[0]) if digits else -1

model_colors = {
    "LR Generalised": "blue",
    "TFT": "green",
    "TFT + HR": "orange",
    "LSTM": "red"
}

plt.figure(figsize=(8, 8))

for model_name, fpath in files.items():
    df = pd.read_csv(fpath)

    true_cols = sorted([c for c in df.columns if c.startswith("y_")], key=extract_index)
    pred_cols = sorted([c for c in df.columns if c.startswith("pred_")], key=extract_index)

    # Extract full arrays across ALL rows (flatten scatter)
    y_true = df[true_cols].values.astype(float)
    y_pred = df[pred_cols].values.astype(float)

    # Rescale from normalized to mg/dL
    y_true = (y_true * (cbg_max - cbg_min)) + cbg_min
    y_pred = (y_pred * (cbg_max - cbg_min)) + cbg_min

    # Flatten for scatter
    act = y_true.flatten()
    pred = y_pred.flatten()

    plt.scatter(
        act, pred,
        alpha=0.3,
        s=10,
        color=model_colors[model_name],
        label=model_name
    )

# Diagonal line
min_val = 0 # min(40, min(act.min(), pred.min()))
max_val = 440# max(400, max(act.max(), pred.max()))
plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2)
plt.xlim(min_val, max_val)

plt.xlabel("True Glucose (mg/dL)")
plt.ylabel("Predicted Glucose (mg/dL)")
plt.title("True vs Predicted Glucose — All Models")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
