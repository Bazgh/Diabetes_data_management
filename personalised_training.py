import pandas as pd
import numpy as np
import glob
import os
import torch
import torch.nn as nn
import random
from sklearn.model_selection import cross_val_score

from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import ast

# ----- LSTM MODEL + DEVICE -----
class LSTMRegressor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, output_size=1):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: [batch, seq_len, features]
        lstm_out, _ = self.lstm(x)           # [batch, seq_len, hidden_size]
        last_output = lstm_out[:, -1, :]     # take last time step
        out = self.fc(last_output)           # [batch, output_size]
        return out                           # [batch, 1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# --------------------------------

dir="personalised/personalised/train" #adjust this to your dir please
files_list=[]
for file in glob.glob(dir + "/*.csv"):
    files_list.append(file)
    data = pd.read_csv(file)
    # Convert string to Python list
    data["X"] = data["X"].apply(lambda x: ast.literal_eval(x))

    # Convert list column to NumPy array
    X = np.array(data["X"].tolist())        # (N, seq_len)
    Y = np.array(data.iloc[:, -1].values)   # (N,)

    # split them into train and val
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.33, random_state=42
    )

    print(X_train.shape, Y_train.shape, X_val.shape, Y_val.shape)

    # 5-fold cross-validated RMSE (LR)
    model_lr = LinearRegression()
    cv_scores = np.sqrt(-cross_val_score(
        model_lr,
        np.concatenate([X_train, X_val]),
        np.concatenate([Y_train, Y_val]),
        cv=5,
        scoring='neg_mean_squared_error'
    ))

    print(f"CV RMSE: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

    # ----- TRAIN LINEAR REGRESSION -----
    model_lr.fit(X_train, Y_train)

    base = os.path.splitext(os.path.basename(file))[0]

    # save a "best" LR model per person/file
    save_path_best_lr = f"linear_regression_model_best_{base}.pkl"
    joblib.dump(model_lr, save_path_best_lr)
    print(f"Best LR model for {file} saved to: {save_path_best_lr}")

    # validation predictions (LR)
    val_pred_lr = model_lr.predict(X_val)

    # ----- TRAIN LSTM PER CASE -----
    # prepare tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)  # [N, seq_len, 1]
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(-1)  # [N, 1]

    X_val_t   = torch.tensor(X_val,   dtype=torch.float32).unsqueeze(-1)
    Y_val_t   = torch.tensor(Y_val,   dtype=torch.float32).unsqueeze(-1)

    train_ds = TensorDataset(X_train_t, Y_train_t)
    val_ds   = TensorDataset(X_val_t,   Y_val_t)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False)

    lstm_model = LSTMRegressor(input_size=1, hidden_size=64, num_layers=1, output_size=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(lstm_model.parameters(), lr=1e-3)

    best_val_loss = float("inf")
    num_epochs = 20  # adjust if you like

    for epoch in range(num_epochs):
        # train
        lstm_model.train()
        running_train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            yhat = lstm_model(xb)
            loss = criterion(yhat, yb)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        running_train_loss /= len(train_loader)

        # val
        lstm_model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                yhat = lstm_model(xb)
                loss = criterion(yhat, yb)
                running_val_loss += loss.item()
        running_val_loss /= len(val_loader)

        # save best per case
        if running_val_loss < best_val_loss:
            best_val_loss = running_val_loss
            save_path_best_lstm = f"lstm_model_best_{base}.pth"
            torch.save(lstm_model.state_dict(), save_path_best_lstm)

        print(f"{base} | epoch {epoch+1}/{num_epochs} "
              f"train_loss={running_train_loss:.4f} val_loss={running_val_loss:.4f}")

    print(f"Best LSTM model for {file} saved to: {save_path_best_lstm}")

    # reload best LSTM (optional but clean)
    lstm_model.load_state_dict(torch.load(save_path_best_lstm, map_location=device))
    lstm_model.eval()

    # validation predictions (LSTM)
    with torch.no_grad():
        X_val_full = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1).to(device)
        val_pred_lstm = lstm_model(X_val_full).cpu().numpy().flatten()

    # ----- GLOBAL SCATTER: LR vs GT (you can add LSTM here too if you like) -----
    plt.figure()
    plt.scatter(Y_val, val_pred_lr, alpha=0.5, label="LR")
    plt.xlabel("Ground Truth")
    plt.ylabel("Prediction")
    plt.title(f"Linear Regression (personal) – {base}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"linear_regression_scatter_{base}.png")
    plt.show()

    # ---- PLOT: GT vs LR vs LSTM for some indices ----
    idxs = np.random.choice(Y_val.shape[0], size=20,replace=False)

    if len(idxs) > 0:
        plt.figure()
        plt.scatter(idxs, Y_val[idxs], label="Ground Truth", s=12)
        plt.scatter(idxs, val_pred_lr[idxs], label="LR", s=12)
        plt.scatter(idxs, val_pred_lstm[idxs], label="LSTM", s=12)

        plt.xlabel("Validation sample index")
        plt.ylabel("Value")
        plt.title(f"GT vs LR vs LSTM (sampled points) – {base}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"personal_points_LR_LSTM_{base}.png")
        plt.show()

    # optional: final LR model per case
    save_path_final = f"linear_regression_model_final_{base}.pkl"
    joblib.dump(model_lr, save_path_final)
    print(f"Linear regression model saved to: {save_path_final}")
