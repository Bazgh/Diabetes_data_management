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
# Evaluate
Y_pred = model1.predict(X_val)
input_len=36
pred_len=1
x=[]
y_true=[]
y_pred=[]
x[0]=X_val[0]
y_true[0]=Y_val[0]

for i in range(1,len(X_val)):
    y_pred[i-1] = model1.predict(x[i-1])
    #pass the prediction to the input
    x[i]=np.concatenate(x[i:],y_pred[i-1])






import matplotlib.pyplot as plt


# pick one test sample
i = 0
y_true = Y_val[i]
y_pred = Y_pred[i]

plt.figure(figsize=(6,4))
plt.plot(range(1, len(y_true)+1), y_true, 'o-', label='Ground Truth')
plt.plot(range(1, len(y_pred)+1), y_pred, 's--', label='Prediction')
plt.xlabel("Prediction step (5-min intervals)")
plt.ylabel("Normalized CBG")
plt.title("Linear Regression — 6-step Prediction Example")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
rmse = np.sqrt(mean_squared_error(Y_val, Y_pred))
r2 = r2_score(Y_val, Y_pred)