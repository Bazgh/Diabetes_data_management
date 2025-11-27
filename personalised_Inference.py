import glob
import os
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import joblib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# assumes LSTMRegressor and device are already defined, e.g.:
# class LSTMRegressor(nn.Module): ...
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_dir = "personalised/personalised/test"  # folder with per-case test CSVs

for file in glob.glob(os.path.join(test_dir, "*.csv")):
    base = os.path.splitext(os.path.basename(file))[0]
    print("\n==============================")
    print(f"Testing case: {base}")
    print("==============================")
    base = os.path.splitext(os.path.basename(file))[0]  # "test_2020_596 ..."
    base = base.replace("test", "train")  # "train_2020_596 ..."

    # ---- LOAD TEST DATA ----
    data = pd.read_csv(file)
    # Convert string to Python list
    data["X"] = data["X"].apply(lambda x: ast.literal_eval(x))

    # Convert list column to NumPy array
    X_test = np.array(data["X"].tolist())          # (n_test, seq_len)
    Y_test = np.array(data.iloc[:, -1].values)     # (n_test,)

    print("X_test shape:", X_test.shape)
    print("Y_test shape:", Y_test.shape)

    # ---- LOAD BEST LINEAR REGRESSION MODEL FOR THIS CASE ----
    lr_path = f"linear_regression_model_best_{base}.pkl"
    if not os.path.exists(lr_path):
        print(f"[WARN] Missing LR model for {base}: {lr_path}")
        continue

    Linear_Regression_model = joblib.load(lr_path)

    pred_length = [6, 12, 18, 24]  # 30, 60, 120 minutes


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

        idxs = [5, 10, 17, 50, 120, 400, 1000, 5600, 10783, 13000]

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
lstm_model.load_state_dict(torch.load("best_model.npy", map_location=device))  # change name if needed
lstm_model.to(device)
lstm_model.eval()



def Test_LSTM(model, X_test, Y_test, pred_length):
    # create containers with fixed length
    n_test = len(X_test)

    Y_true = np.zeros((n_test, pred_length))  # ground truth outputs      [n_test,pred_length]
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
    print(f"LSTM Test MSE (pred_length={pred_length}):", mse)

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

for i, length in enumerate(pred_length):
    Test_LSTM(lstm_model, X_test, Y_test, length)


#for i, length in enumerate(pred_length):
#    Test(Linear_Regression_model, X_test, Y_test, length)