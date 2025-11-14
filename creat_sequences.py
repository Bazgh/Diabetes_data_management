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
from scipy.interpolate import CubicSpline
from torch.utils.data import ConcatDataset
import json

class TimeSeriesWindowDataset(Dataset):
    """
    Build a sliding-window dataset from CSV files.
    """
    def __init__(
        self,
        root_dir,
        split="train",
        input_len=24,
        pred_len=1,
        stride=5,
        max_missing_length=12,   # in steps (12*5min = 60min)
        normalize=True,
        ewma_span=5              # EWMA smoothing span (in steps)
    ):
        self.input_len = input_len
        self.pred_len = pred_len
        self.stride = stride
        self.max_missing_length = max_missing_length
        self.normalize = normalize
        self.ewma_span = ewma_span

        self.split_dir = os.path.join(root_dir, split)
        self.files = glob.glob(os.path.join(self.split_dir, "*.csv")) \
                     + glob.glob(os.path.join(self.split_dir, "*.xlsx"))

        X_all, Y_all = [], []

        for f in self.files:
            if f.endswith(".csv"):
                df = pd.read_csv(f)
            else:
                df = pd.read_excel(f)
            df.columns = df.columns.str.strip()

            # Ensure numeric for all but the first column (assumed timestamp)
            timestamps = df.iloc[:, 0].apply(pd.to_numeric, errors='coerce')
            numeric_cols = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
            data = pd.concat([timestamps, numeric_cols], axis=1)

            # Make sure these columns exist
            if 'cbg' not in data.columns:
                continue
            if '5minute_intervals_timestamp' not in data.columns:
                # fall back: assume first column is timestamp
                data.rename(columns={data.columns[0]: '5minute_intervals_timestamp'}, inplace=True)

            # fix missing cbg by fingerstick when marked
            if 'missing_cbg' in data.columns and 'finger' in data.columns:
                mask = (data['missing_cbg'] == 1) & (data['finger'].notna())
                data.loc[mask, 'cbg'] = data.loc[mask, 'finger']

            # clamp out-of-bounds to NaN
            invalid_mask = (data['cbg'] < 20) | (data['cbg'] > 400)
            data.loc[invalid_mask, 'cbg'] = np.nan

            # index by datetime
            #data['5minute_intervals_timestamp'] = pd.to_datetime(
             #   data['5minute_intervals_timestamp'], errors='coerce'
            #)
            #data = data.dropna(subset=['5minute_intervals_timestamp']).sort_values('5minute_intervals_timestamp')
            cbg_df = data.set_index('5minute_intervals_timestamp')[['cbg']]

            # optional normalization (per-file)
            #if self.normalize:
                #mean = cbg_df['cbg'].mean(skipna=True)
                #std = cbg_df['cbg'].std(skipna=True)

                #if pd.notna(std) and std > 0:
                    #cbg_df['cbg'] = (cbg_df['cbg'] - mean) / std
                #else:
                    #cbg_df['cbg'] = cbg_df['cbg'] - mean

            total_len = len(cbg_df)
            window_len = input_len + pred_len

            for i in range(0, total_len - window_len + 1, stride):
                # values for the whole window (past+future), 1-D float array
                vals = cbg_df['cbg'].iloc[i:i + window_len].to_numpy(dtype=float)

                # 1) skip if any contiguous NaN run exceeds threshold
                nan_mask = np.isnan(vals)
                max_run, run = 0, 0
                for is_nan in nan_mask:
                    run = run + 1 if is_nan else 0
                    if run > max_run:
                        max_run = run
                if max_run > self.max_missing_length:
                    continue

                # 2) impute short gaps via cubic spline (need >=4 valid points)
                t = np.arange(window_len, dtype=float) * 5.0  # minutes, 5-min steps
                valid = ~nan_mask
                if valid.sum() >= 4 and nan_mask.any():
                    try:
                        cs = CubicSpline(t[valid], vals[valid], bc_type='natural')
                        # fill ONLY inside the known range (no extrapolation!)
                        t_min, t_max = t[valid].min(), t[valid].max()
                        fill_idx = nan_mask & (t >= t_min) & (t <= t_max)
                        vals[fill_idx] = cs(t[fill_idx])
                        # leave edge NaNs (outside [t_min, t_max]) as NaN
                    except Exception:
                        continue
                # 3) Fill remaining edge NaNs with nearest valid values (short edges)
                s = pd.Series(vals).fillna(method="ffill").fillna(method="bfill")
                vals = s.to_numpy()

                # 3) EWMA smoothing
                vals = pd.Series(vals).ewm(span=self.ewma_span, adjust=False).mean().to_numpy()

                # split into X (input) and Y (future)
                x_window = vals[:input_len].astype(np.float32)
                y_window = vals[input_len:].astype(np.float32)

                X_all.append(x_window)  # shape (input_len,)
                Y_all.append(y_window)  # shape (pred_len,)

        if not X_all:
            raise ValueError(
                "No valid windows found. Check data columns ('5minute_intervals_timestamp', 'cbg'), "
                "value ranges, and max_missing_length."
            )

        self.X = torch.from_numpy(np.stack(X_all, axis=0))  # [N, input_len]
        self.Y = torch.from_numpy(np.stack(Y_all, axis=0))  # [N, pred_len]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


#instantiate the class and build the data

train_dataset1 = TimeSeriesWindowDataset(
    root_dir="data/Ohio2018_processed",        # path to your main folder
    split="train",          # or "val" or "test"
    input_len=24,           # past 100 minutes (20 * 5min)
    pred_len=1,             # predict next 30 minutes
    stride=1,               # slide by 25 minutes
    max_missing_length=12,  # skip if >60 minutes missing
    normalize=False,         # per-file normalization
    ewma_span=8             # smoothing parameter
)
train_dataset2 = TimeSeriesWindowDataset(
    root_dir="data/Ohio2018_processed",        # path to your main folder
    split="test",          # or "val" or "test"
    input_len=24,           # past 100 minutes (20 * 5min)
    pred_len=1,             # predict next 30 minutes
    stride=1,               # slide by 25 minutes
    max_missing_length=12,  # skip if >60 minutes missing
    normalize=False,         # per-file normalization
    ewma_span=8             # smoothing parameter
)
train_dataset = ConcatDataset([train_dataset1, train_dataset2])

# ---- STEP 2: compute global stats from *raw* training data ----
# Concatenate tensors from underlying datasets
X = np.concatenate(
    [train_dataset1.X.numpy(), train_dataset2.X.numpy()],
    axis=0
)
Y = np.concatenate(
    [train_dataset1.Y.numpy(), train_dataset2.Y.numpy()],
    axis=0
)

# STEP 2 — Compute global stats from raw training data
mean = X.mean()
std = X.std()

print("Training mean:", mean)
print("Training std:", std)

# STEP 3 — Save global stats
np.savez("global_train_stats.npz", mean=mean, std=std)
"""
# Extract tensors
X = test_dataset.X.numpy()
Y = test_dataset.Y.numpy()
#Apply global normalization
stats = np.load("global_train_stats.npz")
mean = stats["mean"]   # scalar
std  = stats["std"]    # scalar
"""
#Apply global normalization (same as training)
X_norm = (X - mean) / std
Y_norm = (Y - mean) / std
# STEP 5 — Save normalized dataset
save_path = "data/Ohio2018_processed/train_dataset_2018.pt"
torch.save({"X": torch.tensor(X_norm), "Y": torch.tensor(Y_norm)}, save_path)
print("Saved normalized test dataset:", save_path)
