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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dir = "generalised_with_hr/generalised_with_hr/test.csv"
data = pd.read_csv(dir)
# Convert string to Python list
data["X1"] = data["X1"].apply(lambda x: ast.literal_eval(x))
data["X2"] = data["X2"].apply(lambda x: ast.literal_eval(x))
data["y"] = data["y"].apply(lambda x: ast.literal_eval(x))

# Convert list columns to NumPy arrays
X1 = np.array(data["X1"].tolist())   # shape: [N, seq_len]
X2 = np.array(data["X2"].tolist())   # shape: [N, seq_len]

# Stack into a 3D array: [N, seq_len, 2]
X_test = np.stack([X1, X2], axis=-1)      # last dim = features (cbg, hr)

Y_test = np.array(data["y"].tolist())  # (n_test,24)

print(X_test.shape)
print(Y_test.shape)

class LSTMRegressor(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=1, output_size=1):
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
lstm_model = LSTMRegressor(input_size=2, hidden_size=64, num_layers=1, output_size=1)
lstm_model.load_state_dict(torch.load("best_model.npy", map_location=device))  # change name if needed
lstm_model.to(device)
lstm_model.eval()

def Test_LSTM(model, X_test, Y_test, pred_length):
    """
    X_test: (n_test, window_len, 2)  -> features = [cbg, hr]
    Y_test: (n_test, 24)            -> ground-truth future cbg
    pred_length: int                -> number of steps to predict
    """
    n_test = len(X_test)

    Y_true = np.zeros((n_test, pred_length))      # [n_test, pred_length]
    Y_pred_lstm = np.zeros((n_test, pred_length)) # [n_test, pred_length]

    for i in range(n_test):
        # ground truth for first pred_length steps
        Y_true[i] = Y_test[i][:pred_length]

        # current input window: shape (window_len, 2)
        x = X_test[i].copy()

        for j in range(pred_length):
            # LSTM expects: [batch, seq_len, features] = [1, window_len, 2]
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)  # (1, L, 2)
            with torch.no_grad():
                y_hat = model(x_tensor)  # (1, 1)
            y_hat_val = y_hat.item()
            Y_pred_lstm[i, j] = y_hat_val

            # ---- build next input window ----
            # split cbg and hr
            cbg_seq = x[:, 0]  # (window_len,)
            hr_seq  = x[:, 1]  # (window_len,)

            # shift and append
            cbg_seq = np.concatenate([cbg_seq[1:], np.array([y_hat_val])])
            # here we just repeat the last HR value; adjust if you know future HR
            hr_seq  = np.concatenate([hr_seq[1:], np.array([hr_seq[-1]])])

            # recombine into shape (window_len, 2)
            x = np.stack([cbg_seq, hr_seq], axis=-1)

    # compute MSE
    mse = mean_squared_error(Y_true, Y_pred_lstm)
    rmse = np.sqrt(mse)

    print(f"LSTM Test RMSE (pred_length={pred_length}): {rmse}")

    # ------- SCATTER PLOT: True vs Predicted -------
    plt.figure()
    plt.scatter(Y_true.flatten(), Y_pred_lstm.flatten(), alpha=0.5)
    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    plt.title(f"LSTM Predicted vs True (Test), pred_length={pred_length}")
    plt.grid(True)

    # y = x reference line
    min_val = min(Y_true.min(), Y_pred_lstm.min())
    max_val = max(Y_true.max(), Y_pred_lstm.max())
    plt.plot([min_val, max_val], [min_val, max_val])
    plt.tight_layout()
    plt.show()

    # ------- Example sequences to visualize -------
    idxs = [5, 120, 400, 1000]
    valid_idxs = [idx for idx in idxs if idx < n_test]

    for idx in valid_idxs:
        t = np.arange(pred_length)  # 0..pred_length-1

        plt.figure()
        plt.plot(t, Y_true[idx], label="Ground Truth", marker='o')
        plt.plot(t, Y_pred_lstm[idx], label="LSTM Prediction", marker='o')
        plt.xlabel("Prediction Step")
        plt.ylabel("Value")
        plt.title(f"LSTM Prediction_with_HR Curve for X_test[{idx}], pred_length={pred_length}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        save_name_lstm = f"prediction_curve_LSTM_with_HR_X_test_{idx}_predlen_{pred_length}.png"
        plt.savefig(save_name_lstm)
        plt.show()
        print("Saved LSTM plot to:", save_name_lstm)

pred_length=[6,12,24]
for length in pred_length:
    Test_LSTM(lstm_model, X_test, Y_test, length)
