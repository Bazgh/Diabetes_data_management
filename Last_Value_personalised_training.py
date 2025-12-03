#imports...
import pandas as pd
import matplotlib.pyplot as plt
import glob
import ast
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib  # for saving/loading the model
import os
import math

class LastValueBaseline:
    def __init__(self):
        """
        noise_scale will be learned from data in fit()
        """
        self.noise_scale = None

    def fit(self, X, y):
        # Learn noise scale from residuals between target and last value
        X = np.asarray(X)
        y = np.asarray(y)

        last_vals = X[:, -1]          # last value in each sequence
        residuals = y - last_vals     # how far off the pure baseline is

        # small positive number in case residuals are exactly zero
        self.noise_scale = max(np.mean(np.abs(residuals)), 1e-6)
        return self

    def predict(self, X):
        """
        X shape: (batch_size, sequence_length)
        Returns last value of each sequence + small learned epsilon.
        """
        X = np.asarray(X)

        # if predict is called before fit for some reason
        noise_scale = self.noise_scale if self.noise_scale is not None else 0.0

        if X.ndim == 1:
            # Single sample
            base = X[-1]
            epsilon = np.random.uniform(-noise_scale, noise_scale)
            return base + epsilon
        else:
            # Batch
            base = X[:, -1]
            epsilon = np.random.uniform(-noise_scale, noise_scale, size=base.shape)
            return base + epsilon



data_dir = "personalised/personalised/train"
save_root = "models"

os.makedirs(save_root, exist_ok=True)

for file in glob.glob(data_dir + "/*.csv"):

    print("\nProcessing:", file)

    # base name from CSV file
    base = os.path.splitext(os.path.basename(file))[0]  # e.g., "patient1"
    model_dir = os.path.join(save_root, base)
    os.makedirs(model_dir, exist_ok=True)

    print("Model will be saved into folder:", model_dir)

    # Load CSV
    data = pd.read_csv(file)
    data["X"] = data["X"].apply(lambda x: ast.literal_eval(x))
    X = np.array(data["X"].tolist())
    Y = np.array(data.iloc[:, -1].values)

    # Split data
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2)
    print("Shapes:", X_train.shape, X_val.shape, Y_train.shape, Y_val.shape)

    # Train model
    model = LastValueBaseline()
    model.fit(X_train, Y_train)

    # Validation MSE
    error = 0.0
    for x_val, y_val in zip(X_val, Y_val):
        pred = model.predict(x_val)
        error += (y_val - pred) ** 2
    MSE = error / len(X_val)

    print("Validation MSE:", MSE)
    print("Learned noise_scale:", model.noise_scale)

    # --- SAVE MODEL WITH BASE NAME ---
    model_path = os.path.join(model_dir, f"{base}.joblib")
    joblib.dump(model, model_path)
    print("Saved model to:", model_path)

# ---- Load TEST data ----
Test_data_path = "personalised/personalised/test/"
for file in glob.glob(Test_data_path + "*.csv"):
    print("Test file:", file)
    data = pd.read_csv(file)
    data["X"] = data["X"].apply(lambda x: ast.literal_eval(x))
    X_test = np.array(data["X"].tolist())
    data["Y"] = data["y"].apply(lambda x: ast.literal_eval(x))
    Y_test = np.array(data["Y"].tolist())

    # load model for each data
    # --- load MODEL WITH BASE NAME ---
    base = os.path.splitext(os.path.basename(file))[0]  # e.g. "user1"
    base=base.replace("test", "train")
    model_path = os.path.join("models", base, f"{base}.joblib")
    print("Loading model from:", model_path)
    model = joblib.load(model_path)

    idxs = [5, 120, 400]
    n_test = X_test.shape[0]
    len_preds = [6, 12, 24]

    for len_pred in len_preds:
        y_pred = np.zeros((n_test, len_pred))

        for i in range(n_test):
            y_true = Y_test[i, :len_pred]
            X_ = X_test[i]
            for j in range(len_pred):
                y_pred[i, j] = model.predict(X_)
                # roll the window and append prediction
                X_ = np.concatenate([X_[1:], np.atleast_1d(y_pred[i, j])])

        y_true_all = Y_test[:, :len_pred]


        rmse = np.sqrt(mean_squared_error(y_true_all, y_pred))
        print(f"Test RMSE (pred_len={len_pred}):", rmse)
        # === SAVE CSV WHEN pred_length == 24 ===
        if len_pred == 24:
            # column names for ground truth and predictions
            cols_true = [f"y_{i}" for i in range(len_pred)]  # Y0 ... Y23
            cols_pred = [f"pred_{i}" for i in range(len_pred)]  # Y_pred0 ... Y_pred23

            # concatenate true and predicted along axis 1: shape (n_test, 48)
            data_mat = np.hstack([y_true_all, y_pred])

            df = pd.DataFrame(data_mat, columns=cols_true + cols_pred)
            csv_name = f"last_value_Y_true_and_predictions_len24_{base}.csv"
            df.to_csv(csv_name, index=False)
            print(f"Saved CSV to: {csv_name}")
        valid_idxs = [idx for idx in idxs if idx < n_test]
        """
        for i, idx in enumerate(valid_idxs):  # sequence index to visualize
            t = np.arange(len_pred)  # time steps 0..pred_length-1

            plt.figure()
            plt.plot(t, y_true_all[idx], label="Ground Truth", marker='o')
            plt.plot(t, y_pred[idx], label="Prediction", marker='o')

            plt.xlabel("Prediction Step")
            plt.ylabel("Value")
            plt.title(f"last value_Prediction Curve for X_test[{idx}], pred_length={len_pred}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            # save figure
            model_name = type(model).__name__
            save_name = f"last value_prediction_curve_X_test_{model_name}_{idx}_predlen_{len_pred}.png"
            plt.savefig(save_name)

            plt.show()

            print("Saved plot to:", save_name)
        """