import pandas as pd
import numpy as np
import glob
import os
import torch
import torch.nn as nn
import random

from mpmath.libmp import to_float
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import joblib
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader

dir = "train"  # adjust this to your dir please


def create_train_val_pairs(dir, train_ratio):
    # load files
    files_list = []
    X_train = []
    Y_train = []
    X_val = []
    Y_val = []
    for file in glob.glob(dir + "/*.csv"):
        files_list.append(file)
    # shuffle them and split them into train and val
    random.seed(42)
    random.shuffle(files_list)
    N = len(files_list)
    train_len = int(N * train_ratio)
    train_files = files_list[:train_len]
    val_files = files_list[train_len:]

    # fill in the train and val arrays
    # TODO: Make me a func
    for file in train_files:
        df = pd.read_csv(file)
        X = [eval(item) for item in df["X"].values]
        X_train.extend(X)
        Y_train.extend(df["y"])
    X_train = np.array(X_train)  # shape: (total_rows, num_features)
    Y_train = np.array(Y_train)  # shape: (total_rows,)

    for file in val_files:
        df = pd.read_csv(file)
        X = [eval(item) for item in df["X"].values]
        X_val.extend(X)
        Y_val.extend(df["y"])

    X_val = np.array(X_val)  # shape: (total_rows, num_features)
    Y_val = np.array(Y_val)  # shape: (total_rows,)

    return X_train, Y_train, X_val, Y_val


X_train, Y_train, X_val, Y_val = create_train_val_pairs(dir, train_ratio=0.8)

print(X_train.shape, Y_train.shape, X_val.shape, Y_val.shape)


# trainloader & valloader

class SequenceDataset(Dataset):
    def __init__(self, X, Y):
        # Convert to tensors
        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.float32)

        # Ensure shape: [N, seq_len, features]
        if X.ndim == 2:
            # Assume shape [N, seq_len] → add feature dim
            X = X.unsqueeze(-1)   # [N, seq_len, 1]

        # Ensure targets are [N, 1]
        if Y.ndim == 1:
            Y = Y.unsqueeze(-1)

        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


train_dataset = SequenceDataset(X_train, Y_train)
val_dataset = SequenceDataset(X_val, Y_val)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)


# 5-fold cross-validated RMSE

model = LinearRegression()
cv_scores = np.sqrt(-cross_val_score(
    model,
    np.concatenate([X_train, X_val]),
    np.concatenate([Y_train, Y_val]),
    cv=5,
    scoring='neg_mean_squared_error'
))

print(f"CV RMSE: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

model.fit(X_train, Y_train)
val_pred = model.predict(X_val)
plt.figure()
plt.scatter(Y_val, val_pred)
# plt.scatter(list(range(len(val_pred))), Y_val, alpha=0.5)
plt.savefig("linear_regression.jpg")
plt.show()
# y_scores =
# model = LinearRegression()

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


def train_model(model: nn.Module, train_dataloader, val_dataloader):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    train_loss=0
    val_loss=0
    for epoch in range(10):
        train_loss = 0
        model.train()
        for x, y in train_dataloader:

            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            train_loss+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_train = train_loss / len(train_dataloader)
        train_losses.append(loss_train)

        with torch.no_grad():
            val_loss =0
            model.eval()
            for x, y in val_dataloader:
                x = x.to(device)
                y = y.to(device)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                val_loss+=loss.item()
            val_loss = val_loss / len(val_dataloader)

            val_losses.append(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # TODO: save model
                torch.save(model.state_dict(),"best_model.npy")
        print(f"epoch: {epoch}",f"train_loss: {loss_train}", f"val_loss: {val_loss}")
    epochs = range(1, 10 + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Progress")
    plt.legend()
    plt.tight_layout()
    plt.savefig(( "training_progress.jpg"))
    plt.show()

    print(f"Best model saved to: {"/best_model.npy"}")

model=LSTMRegressor(input_size=1, hidden_size=64, num_layers=1, output_size=1)
train_model(model, train_dataloader, val_dataloader)

model.load_state_dict(torch.load("best_model.npy"))
model.eval()
with torch.no_grad():
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1).to(device)
    val_pred_lstm = model(X_val_tensor).cpu().numpy().flatten()
plt.figure()
plt.scatter(Y_val, val_pred_lstm)
# plt.scatter(list(range(len(val_pred))), Y_val, alpha=0.5)
plt.savefig("lstm.jpg")
plt.show()