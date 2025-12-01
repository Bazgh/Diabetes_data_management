import pandas as pd
import numpy as np
import ast
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dir = "generalised_with_hr/generalised_with_hr/train.csv"
data = pd.read_csv(dir)


# Convert string columns to Python lists
data["X1"] = data["X1"].apply(lambda x: ast.literal_eval(x))
data["X2"] = data["X2"].apply(lambda x: ast.literal_eval(x))

# Convert list columns to NumPy arrays
X1 = np.array(data["X1"].tolist())   # shape: [N, seq_len]
X2 = np.array(data["X2"].tolist())   # shape: [N, seq_len]

# Stack into a 3D array: [N, seq_len, 2]
X = np.stack([X1, X2], axis=-1)      # last dim = features (cbg, hr)

# Target (assuming last column is Y)
Y = data.iloc[:, -1].values          # shape: [N]
Y = Y.astype(np.float32)

X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y, test_size=0.33, random_state=42
)

print(X_train.shape, Y_train.shape, X_val.shape, Y_val.shape)

# Convert to tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_val_t   = torch.tensor(X_val,   dtype=torch.float32)

Y_train_t = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(-1)  # [N, 1]
Y_val_t   = torch.tensor(Y_val,   dtype=torch.float32).unsqueeze(-1)

# Datasets & loaders
train_dataset = TensorDataset(X_train_t, Y_train_t)
val_dataset   = TensorDataset(X_val_t,   Y_val_t)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader   = DataLoader(val_dataset,   batch_size=64, shuffle=False)

class LSTMRegressor(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=1, output_size=1):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: [batch, seq_len, features]
        lstm_out, _ = self.lstm(x)            # [batch, seq_len, hidden_size]
        last_output = lstm_out[:, -1, :]      # take last time step
        out = self.fc(last_output)            # [batch, output_size]
        return out

def train_model(model: nn.Module, train_dataloader, val_dataloader):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    num_epochs = 20

    for epoch in range(num_epochs):
        train_loss = 0.0
        model.train()
        for x, y in train_dataloader:
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)
            loss = criterion(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        loss_train = train_loss / len(train_dataloader)
        train_losses.append(loss_train)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_dataloader:
                x = x.to(device)
                y = y.to(device)

                y_pred = model(x)
                loss = criterion(y_pred, y)
                val_loss += loss.item()

        val_loss = val_loss / len(val_dataloader)
        val_losses.append(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.npy")

        print(f"epoch: {epoch+1}  train_loss: {loss_train:.6f}  val_loss: {val_loss:.6f}")

    # Plot losses
    epochs = range(1, num_epochs + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Progress")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_progress.jpg")
    plt.show()

    print("Best model saved to: best_model.npy")

model2 = LSTMRegressor(input_size=2, hidden_size=64, num_layers=1, output_size=1)
train_model(model2, train_dataloader, val_dataloader)
