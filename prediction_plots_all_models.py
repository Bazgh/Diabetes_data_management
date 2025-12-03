import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from error_grids import clarke_error_zone_detailed, zone_accuracy

cbg_max = 400
cbg_min = 40

# File paths
files = {
    "LR Generalised": "lr-generalised-pred/lr-generalised/generalised_lr_recursive_predictions.csv",
    "TFT": "tft_generalised/tft_generalised/tft_predictions.csv",
    "TFT + HR": "tft_generalised_withHr/generalised/tft_cbg_hr_predictions.csv",
    "LSTM": "lstm_Y_true_and_predictions_len24.csv"
}

# --- Clarke grid colors ---
zone_colors = {
    0: "green",  # A
    1: "blue", 2: "blue",  # B
    3: "orange", 4: "orange",  # C
    5: "red", 6: "red",        # D
    7: "purple", 8: "purple"   # E
}

# Prepare 2x2 plot layout
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, (model_name, fpath) in enumerate(files.items()):
    df = pd.read_csv(fpath)

    # ---- Extract true & predicted values ----
    true_cols = [c for c in df.columns if c.startswith("y")]
    pred_cols = [c for c in df.columns if c.startswith("pred")]

    Y_true = df[true_cols].values
    Y_pred = df[pred_cols].values

    # ---- Rescaling ----
    act = (Y_true * (cbg_max - cbg_min)) + cbg_min
    pred = (Y_pred * (cbg_max - cbg_min)) + cbg_min

    act = act.flatten()
    pred = pred.flatten()

    # ---- Compute Clarke zones ----
    zones = clarke_error_zone_detailed(act, pred)

    ax = axes[idx]
    ax.set_title(f"{model_name}")

    # ---- Scatter per zone ----
    for z, color in zone_colors.items():
        mask = (zones == z)
        if np.any(mask):
            ax.scatter(act[mask], pred[mask], s=5, alpha=0.4, color=color)

    # diagonal
    min_val = 0
    max_val = 440
    ax.plot([min_val, max_val], [min_val, max_val], "k--")

    ax.set_xlabel("Actual Glucose")
    ax.set_ylabel("Predicted Glucose")
    ax.grid(True)

    # ---- Add min/max annotation ----




# ---- Add single shared legend ----
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='green', label='Zone A'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label='Zone B'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', label='Zone C'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='Zone D'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', label='Zone E'),
]

fig.legend(handles=legend_elements, loc="lower center", ncol=5, fontsize=12)
fig.tight_layout(rect=[0, 0.05, 1, 1])

plt.show()

def zone_accuracy(act_arr, pred_arr, mode='clarke', detailed=False, diabetes_type=1):
    
#Calculates the average percentage of each zone based on Clarke or Parkes
#Error Grid analysis for an array of predictions and an array of actual values
   
    
    acc = np.zeros(9)
    if mode == 'clarke':
        res = clarke_error_zone_detailed(act_arr, pred_arr)

    else:
        raise Exception('Unsupported error grid mode')

    acc_bin = np.bincount(res)
    acc[:len(acc_bin)] = acc_bin

    if not detailed:
        acc[1] = acc[1] + acc[2]
        acc[2] = acc[3] + acc[4]
        acc[3] = acc[5] + acc[6]
        acc[4] = acc[7] + acc[8]
        acc = acc[:5]

    return acc / sum(acc)
zone_labels = ["A", "B", "C", "D", "E"]

for model_name, fpath in files.items():
    df = pd.read_csv(fpath)

    # columns: Y0..Y23 (true) and Y_pred0..Y_pred23 (pred)
    true_cols = [c for c in df.columns if c.startswith("y") and not c.startswith("Y_pred")]
    pred_cols = [c for c in df.columns if c.startswith("pred")]

    Y_true = df[true_cols].values
    Y_pred = df[pred_cols].values

    # rescale back to glucose units
    act = (Y_true * (cbg_max - cbg_min)) + cbg_min
    pred = (Y_pred * (cbg_max - cbg_min)) + cbg_min

    act_flat = act.flatten()
    pred_flat = pred.flatten()

    acc = zone_accuracy(act_flat, pred_flat, mode='clarke', detailed=False)  # A–E

    print(f"\nModel: {model_name}")
    for z_label, z_val in zip(zone_labels, acc):
        print(f"  Zone {z_label}: {z_val*100:.2f}%")
   

"""
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

# --- Extract index from columns like "y_0" or "pred_3" ---
def extract_index(colname):
    digits = re.findall(r'\d+', colname)
    return int(digits[0]) if digits else -1

row_idx = 4000    # which row to visualize
horizon = 24

all_preds = {}
y_true_row = None

for model_name, fpath in files.items():
    df = pd.read_csv(fpath)

    # Correct column selection
    true_cols = sorted(
        [c for c in df.columns if c.startswith("y_")],
        key=extract_index
    )
    pred_cols = sorted(
        [c for c in df.columns if c.startswith("pred_")],
        key=extract_index
    )

    # Extract the chosen row
    y_true = df.loc[row_idx, true_cols].values.astype(float)[:horizon]
    y_pred = df.loc[row_idx, pred_cols].values.astype(float)[:horizon]

    # Rescale back to glucose units
    y_true_rescaled = (y_true * (cbg_max - cbg_min)) + cbg_min
    y_pred_rescaled = (y_pred * (cbg_max - cbg_min)) + cbg_min

    if y_true_row is None:
        y_true_row = y_true_rescaled

    all_preds[model_name] = y_pred_rescaled

# ---- Plot ----
t = np.arange(horizon)

plt.figure(figsize=(12, 6))
plt.plot(t, y_true_row, label="Ground Truth", marker="o", linewidth=2)

for model_name, preds in all_preds.items():
    plt.plot(t, preds, label=model_name, marker="o", linestyle="--")

plt.xlabel("Prediction Step (0–23)")
plt.ylabel("Glucose (mg/dL)")
plt.title(f"24-step Predictions for Sample Row {row_idx}")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
"""