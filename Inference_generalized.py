import glob
import os
import pandas as pd
import numpy as np
import ast
import torch
import torch.nn as nn
import joblib
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from error_grids import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cbg_max=400
cbg_min=40

dir = "generalised/generalised/test.csv"
data = pd.read_csv(dir)
# Convert string to Python list
data["X"] = data["X"].apply(lambda x: ast.literal_eval(x))
data["y"] = data["y"].apply(lambda x: ast.literal_eval(x))

# Convert list column to NumPy array
X_test = np.array(data["X"].tolist())  # (n_test,36)
Y_test = np.array(data["y"].tolist())  # (n_test,24)

print(X_test.shape)
print(Y_test.shape)

# load Linear regression weights
Linear_Regression_model = joblib.load("linear_regression_model_best.pkl")

pred_length = [6, 12, 24]  # 30, 60, 120 minutes


def Test(model, X_test, Y_test, pred_length):
    # create containers with fixed length
    n_test = len(X_test)

    Y_true = np.zeros((n_test, pred_length))  # ground truth outputs      [n_test,pred_length]
    Y_pred = np.zeros((n_test, pred_length))  # predictions               [n_test,pred_length]

    for i in range(n_test):
        # predict from previous window
        Y_true[i] = Y_test[i][:pred_length]
        x = X_test[i].copy()

        for j in range(pred_length):
            Y_pred[i, j] = model.predict(x.reshape(1, -1))[0]
            # build next input window by shifting and appending prediction
            # assumes X windows are 1D: shape (window_len,)
            x = np.concatenate([x[1:], np.atleast_1d(Y_pred[i, j])])

    # compute MSE
    mse = mean_squared_error(Y_true, Y_pred)
    print("Test MSE:", mse)

    # ------- SCATTER PLOT: True vs Predicted -------
    plt.figure()
    plt.scatter(Y_true.flatten(), Y_pred.flatten(), alpha=0.5)  # <-- MANY POINTS
    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    plt.title(f"Predicted vs True (Test), pred_length={pred_length}")
    plt.grid(True)

    # optional: y = x reference line
    min_val = min(Y_true.min(), Y_pred.min())
    max_val = max(Y_true.max(), Y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val])

    plt.show()  #
    # or: plt.savefig(f"val_scatter_{pred_length}.png")

    idxs = [5, 10, 17, 50, 120, 400]

    # avoid IndexError if some idx are >= n_test
    valid_idxs = [idx for idx in idxs if idx < n_test]

    for i, idx in enumerate(valid_idxs):  # sequence index to visualize
        t = np.arange(pred_length)  # time steps 0..pred_length-1

        plt.figure()
        plt.plot(t, Y_true[idx], label="Ground Truth", marker='o')
        plt.plot(t, Y_pred[idx], label="Prediction", marker='o')

        plt.xlabel("Prediction Step")
        plt.ylabel("Value")
        plt.title(f"Prediction Curve for X_test[{idx}], pred_length={pred_length}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        # save figure
        save_name = f"prediction_curve_X_test_{model}_{idx}_predlen_{pred_length}.png"
        plt.savefig(save_name)

        plt.show()

        print("Saved plot to:", save_name)


#for i, length in enumerate(pred_length):
#    Test(Linear_Regression_model, X_test, Y_test, length)


class LSTMRegressor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, output_size=1):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: [batch, seq_len, features]
        lstm_out, _ = self.lstm(x)      # [batch, seq_len, hidden_size]
        last_output = lstm_out[:, -1, :]  # take last time step
        out = self.fc(last_output)        # [batch, output_size]
        return out


#load LSTM weights
lstm_model = LSTMRegressor(input_size=1, hidden_size=64, num_layers=1, output_size=1)
lstm_model.load_state_dict(torch.load("best_model_univariate_Generalised.npy", map_location=device))  # change name if needed
lstm_model.to(device)
lstm_model.eval()



def Test_LSTM(model, X_test, Y_test, pred_length):
    # create containers with fixed length
    n_test = len(X_test)

    Y_true = np.zeros((n_test, pred_length))
    Y_pred_lstm = np.zeros((n_test, pred_length))  # LSTM predictions          [n_test,pred_length]

    for i in range(n_test):
        # predict from previous window
        Y_true[i] = Y_test[i][:pred_length]
        x = X_test[i].copy()  # shape (window_len,)

        for j in range(pred_length):
            # LSTM expects: [batch, seq_len, features] = [1, window_len, 1]
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
            with torch.no_grad():
                y_hat = model(x_tensor)  # [1,1]
            Y_pred_lstm[i, j] = y_hat.item()

            # build next input window by shifting and appending prediction
            x = np.concatenate([x[1:], np.atleast_1d(Y_pred_lstm[i, j])])

    # compute MSE
    mse = mean_squared_error(Y_true, Y_pred_lstm)
    rmse= np.sqrt(mse)
    print(f"LSTM Test MSE (pred_length={pred_length}):", rmse)
    # === SAVE CSV WHEN pred_length == 24 ===
    if pred_length == 24:
        # column names for ground truth and predictions
        cols_true = [f"y_{i}" for i in range(pred_length)]  # Y0 ... Y23
        cols_pred = [f"pred_{i}" for i in range(pred_length)]  # Y_pred0 ... Y_pred23

        # concatenate true and predicted along axis 1: shape (n_test, 48)
        data_mat = np.hstack([Y_true, Y_pred_lstm])

        df = pd.DataFrame(data_mat, columns=cols_true + cols_pred)
        csv_name = "lstm_Y_true_and_predictions_len24.csv"
        df.to_csv(csv_name, index=False)
        print(f"Saved CSV to: {csv_name}")
    """
    # ---- Clarke Error Grid scatter plot (fast) ----
    Y_true = (Y_true * (cbg_max - cbg_min)) + cbg_min            # original scale
    Y_pred_lstm = (Y_pred_lstm * (cbg_max - cbg_min)) + cbg_min  # original scale

    act = Y_true.flatten()
    pred = Y_pred_lstm.flatten()
    zones = clarke_error_zone_detailed(act, pred)

    zone_colors = {
        0: "green",  # Zone A — clinically accurate
        1: "blue",  # Zone B — minor errors (lower)
        2: "blue",  # Zone B — minor errors (upper)
        3: "orange",  # Zone C
        4: "orange",
        5: "red",  # Zone D
        6: "red",
        7: "purple",  # Zone E
        8: "purple"
    }

    plt.figure(figsize=(6, 6))

    # Plot all points for each zone in a single scatter call
    for z, color in zone_colors.items():
        mask = (zones == z)
        if not np.any(mask):
            continue  # skip empty zones
        plt.scatter(
            act[mask],
            pred[mask],
            color=color,
            s=5,
            alpha=0.5,
        )

    # Diagonal reference line
    min_val = min(act.min(), pred.min())
    max_val = max(act.max(), pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')

    plt.xlabel("Actual Glucose")
    plt.ylabel("Predicted Glucose")
    plt.title(f"Clarke Error Grid — Predictions (pred_length={pred_length})")

    # Legend (manual, one entry per zone group)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', label='Zone A'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label='Zone B'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', label='Zone C'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='Zone D'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', label='Zone E')
    ]
    plt.legend(handles=legend_elements)

    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # ------- SCATTER PLOT: True vs Predicted -------
    plt.figure()
    plt.scatter(Y_true.flatten(), Y_pred_lstm.flatten(), alpha=0.5)  # <-- MANY POINTS
    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    plt.title(f"LSTM Predicted vs True (Test), pred_length={pred_length}")
    plt.grid(True)

    # optional: y = x reference line
    min_val = min(Y_true.min(), Y_pred_lstm.min())
    max_val = max(Y_true.max(), Y_pred_lstm.max())
    plt.plot([min_val, max_val], [min_val, max_val])

    plt.tight_layout()
    plt.show()
    # or: plt.savefig(f"val_scatter_LSTM_{pred_length}.png")

    # some example sequences to visualize
    idxs = [5, 10, 17, 50, 120, 400, 1000]  # keep only indices that exist
    valid_idxs = [idx for idx in idxs if idx < n_test]

    for i, idx in enumerate(valid_idxs):  # sequence index to visualize
        t = np.arange(pred_length)  # time steps 0..pred_length-1

        plt.figure()
        plt.plot(t, Y_true[idx], label="Ground Truth", marker='o')
        plt.plot(t, Y_pred_lstm[idx], label="LSTM Prediction", marker='o')

        plt.xlabel("Prediction Step")
        plt.ylabel("Value")
        plt.title(f"LSTM Prediction Curve for X_test[{idx}], pred_length={pred_length}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        # save figure
        save_name_lstm = f"prediction_curve_LSTM_X_test_{idx}_predlen_{pred_length}.png"
        plt.savefig(save_name_lstm)

        plt.show()

        print("Saved LSTM plot to:", save_name_lstm)
"""

for i, length in enumerate(pred_length):
    Test_LSTM(lstm_model, X_test, Y_test, length)


