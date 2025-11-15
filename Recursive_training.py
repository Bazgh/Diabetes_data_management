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


data = torch.load("data/Ohio2018_processed/train_dataset_recursive.pt")
X, Y = data["X"], data["Y"]
print("Loaded dataset shapes:", X.shape, Y.shape)
x_nan_count = torch.isnan(X).sum().item()
y_nan_count = torch.isnan(Y).sum().item()

print(f"NaNs in X: {x_nan_count}")
print(f"NaNs in Y: {y_nan_count}")


X_np = X.numpy()
Y_np = Y.numpy()

# Split into train/val (80/20)
X_train, X_val, Y_train, Y_val = train_test_split(X_np, Y_np, test_size=0.2, random_state=42)

# Train linear regression
model1 = LinearRegression()
model1.fit(X_train, Y_train)
# Save model
save_path = "linear_regression_model.pkl"
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
    y_pred[i-1] = model1.predict(x[i-1].reshape(1, -1))[0]  # <-- reshape + [0]

    # build next input window by shifting and appending prediction
    # assumes X windows are 1D: shape (window_len,)
    x[i] = np.concatenate([x[i-1][1:], np.atleast_1d(y_pred[i-1])])  # <-- fixed concatenate

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



