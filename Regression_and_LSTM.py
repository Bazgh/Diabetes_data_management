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
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

dir="C:/fall 2025/Data driven Diabetes/train_2018_559.csv" #adjust this to your dir please
data = pd.read_csv(dir)
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]
# --- Convert X from comma-separated strings to numeric matrix ---
X_list = data["X"].apply(lambda s: list(map(float, s.split(","))))
X = np.vstack(X_list.values)     # shape becomes (n_samples, n_features)

# Target
Y = data["y"].values.astype(float)
# Split into train/val (80/20)
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train linear regression
model1 = LinearRegression()
model1.fit(X_train, Y_train)
# Save model
save_path = "linear_regression_model_final.pkl"
joblib.dump(model1, save_path)

print(f"Linear regression model saved to: {save_path}")
# --------- EVALUATION (recursive) ---------

# create containers with fixed length
n_val = len(X_val)
x = [None] * n_val        # will store input windows
y_true = [None] * n_val   # ground truth outputs
y_pred = [None] * n_val   # predictions

# initialise first window and true value
x[0] = X_val[0].copy()
y_true[0] = Y_val[0]

for i in range(1, n_val):
    # predict from previous window
    y_pred[i-1] = model1.predict(x[i-1].reshape(1, -1))[0]

    # build next input window by shifting and appending prediction
    # assumes X windows are 1D: shape (window_len,)
    x[i] = np.concatenate([x[i-1][1:], np.atleast_1d(y_pred[i-1])])  # <--  concatenate

    # store true target for this step
    y_true[i] = Y_val[i]

# make prediction for the last window as well
y_pred[-1] = model1.predict(x[-1].reshape(1, -1))[0]

# convert to numpy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# compute MSE
mse = mean_squared_error(y_true, y_pred)  # <-- np.msre -> mean_squared_error
print("Validation MSE:", mse)

# ------- SCATTER PLOT: True vs Predicted -------

plt.figure()
plt.scatter(y_true, y_pred)
plt.xlabel("True values")
plt.ylabel("Predicted values")
plt.title("Predicted vs True (Validation)")
plt.grid(True)

# optional: y = x reference line
min_val = min(y_true.min(), y_pred.min())
max_val = max(y_true.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val])

plt.show()          # <-- THIS actually displays the plot
# or: plt.savefig("val_scatter.png")

#LSTM

#data = torch.load("data/Ohio2018_processed/train_dataset_2018.pt")
#X, Y = data["X"], data["Y"]
#X_np = X.numpy()
#Y_np = Y.numpy()
#print(X_np.shape)
#print(Y_np.shape)
#X = X.unsqueeze(-1)

# Split into train/val (80/20)
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

X_train_t = torch.from_numpy(X_train).float().unsqueeze(-1)   # (N_train, seq_len, 1)
X_val_t   = torch.from_numpy(X_val).float().unsqueeze(-1)     # (N_val,   seq_len, 1)

Y_train_t = torch.from_numpy(Y_train).float().unsqueeze(-1)   # (N_train, 1)
Y_val_t   = torch.from_numpy(Y_val).float().unsqueeze(-1)     # (N_val,   1)

train_ds = TensorDataset(X_train_t, Y_train_t)
test_ds  = TensorDataset(X_val_t, Y_val_t)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False)

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
model2 = LSTMRegressor().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model2.parameters(), lr=1e-3)

num_epochs = 30

for epoch in range(num_epochs):
    model2.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model2(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model2.eval()
    with torch.no_grad():
        val_loss = 0
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            val_loss += criterion(model2(xb), yb).item()

    print(f"Epoch {epoch+1:02d} | Train Loss: {total_loss/len(train_loader):.6f} | Val Loss: {val_loss/len(test_loader):.6f}")
import matplotlib.pyplot as plt
model2.eval()

with torch.no_grad():
    xb, yb = next(iter(test_loader))
    xb, yb = xb.to(device), yb.to(device)
    y_pred = model2(xb).cpu().numpy()
    y_true = yb.cpu().numpy()
torch.save(model2.state_dict(), "lstm_regressor.pth")
print("Saved LSTM weights to lstm_regressor.pth")
print("LSTM model saved to:", save_path)
plt.figure()
plt.scatter(y_true, y_pred)
plt.xlabel("True values")
plt.ylabel("Predicted values")
plt.title("Predicted vs True (Validation)")
plt.grid(True)

# optional: y = x reference line
min_val = min(y_true.min(), y_pred.min())
max_val = max(y_true.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val])

plt.show()