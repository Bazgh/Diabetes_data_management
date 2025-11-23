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

dir="test/test"
files_list = []
X_test = []
Y_test = []

for file in glob.glob(dir + "/*.csv"):
    files_list.append(file)

print(files_list)

for file in files_list:
    df = pd.read_csv(file)

    X_parsed = [ast.literal_eval(item) for item in df["X"].values]
    X = np.array(X_parsed)
    X_test.extend(X)

    y_parsed = [ast.literal_eval(item) for item in df["y"].values]
    y = np.array(y_parsed)
    Y_test.extend(y)

X_test = np.array(X_test)  # shape: (n_test, 36)
Y_test = np.array(Y_test)  # shape: (n_test, 24)

print(X_test.shape)
print(Y_test.shape)

#Todo
#load Linear regression weights
Linear_Regression_model = joblib.load("linear_regression_model_best.pkl")

# create containers with fixed length
n_test = len(X_test)
pred_lenght=6              #30 minutes ahead

Y_true = np.zeros((n_test,pred_lenght))  # ground truth outputs      [n_test,pred_length]
Y_pred = np.zeros((n_test,pred_lenght))  # predictions               [n_test,pred_length]


for i in range(n_test):   # <-- use all test samples (not start from 1)
    # predict from previous window
    Y_true[i] = Y_test[i][:pred_lenght]
    x = X_test[i].copy()

    for j in range(pred_lenght):
        Y_pred[i,j] = Linear_Regression_model.predict(x.reshape(1, -1))[0]
        # build next input window by shifting and appending prediction
        # assumes X windows are 1D: shape (window_len,)
        x = np.concatenate([x[1:], np.atleast_1d(Y_pred[i,j])])


# compute MSE
mse = mean_squared_error(Y_true, Y_pred)
print("Test MSE:", mse)

# ------- SCATTER PLOT: True vs Predicted -------
plt.figure()
plt.scatter(Y_true.flatten(), Y_pred.flatten(), alpha=0.5)   # <-- MANY POINTS
plt.xlabel("True values")
plt.ylabel("Predicted values")
plt.title("Predicted vs True (Test)")
plt.grid(True)

# optional: y = x reference line
min_val = min(Y_true.min(), Y_pred.min())
max_val = max(Y_true.max(), Y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val])

plt.show()          #
# or: plt.savefig("val_scatter.png")
idx = 2                         # sequence index to visualize
t = np.arange(pred_lenght)      # time steps 0..pred_length-1

plt.figure()
plt.plot(t, Y_true[idx], label="Ground Truth", marker='o')
plt.plot(t, Y_pred[idx], label="Prediction", marker='o')

plt.xlabel("Prediction Step")
plt.ylabel("Value")
plt.title(f"Prediction Curve for X_test[{idx}]")
plt.legend()
plt.grid(True)
plt.tight_layout()
# save figure
save_name = f"prediction_curve_X_test_{idx}.png"
plt.savefig(save_name)

plt.show()

print("Saved plot to:", save_name)

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Todo
#load LSTM weights
lstm_model = LSTMRegressor(input_size=1, hidden_size=64, num_layers=1, output_size=1)
lstm_model.load_state_dict(torch.load("best_model.npy", map_location=device))  # change name if needed
lstm_model.to(device)
lstm_model.eval()
Y_pred_lstm = np.zeros((n_test, pred_lenght))  # LSTM predictions [n_test,pred_length]

for i in range(n_test):
    # predict from previous window
    # Y_true[i] is already set from your LR code
    x = X_test[i].copy()   # shape (window_len,)

    for j in range(pred_lenght):
        # LSTM expects: [batch, seq_len, features] = [1, window_len, 1]
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
        y_hat = lstm_model(x_tensor)              # [1,1]
        Y_pred_lstm[i,j] = y_hat.item()

        # build next input window by shifting and appending prediction
        x = np.concatenate([x[1:], np.atleast_1d(Y_pred_lstm[i,j])])
# ------ PLOT: LSTM prediction curve vs ground truth for X_test[0] -------

idx = 0                         # sequence index to visualize
t = np.arange(pred_lenght)      # time steps 0..pred_length-1

plt.figure()
plt.plot(t, Y_true[idx], label="Ground Truth", marker='o')
plt.plot(t, Y_pred_lstm[idx], label="LSTM Prediction", marker='o')

plt.xlabel("Prediction Step")
plt.ylabel("Value")
plt.title(f"LSTM Prediction Curve for X_test[{idx}]")
plt.legend()
plt.grid(True)
plt.tight_layout()

save_name_lstm = f"prediction_curve_LSTM_X_test_{idx}.png"
plt.savefig(save_name_lstm)
plt.show()

print("Saved LSTM plot to:", save_name_lstm)
