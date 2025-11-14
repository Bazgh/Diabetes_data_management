import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def to_numpy_xy(dataset, cbg_col='cbg_zscore'):
    X_list, y_list = [], []

    # get feature column index
    df = dataset.sequences[0][1]
    cols = list(df.columns)
    cbg_idx = cols.index(cbg_col)

    for x_win, y_win, subject in dataset:
        x_np = x_win.numpy()
        y_np = y_win.numpy()

        # Skip if y window is empty
        if y_np.shape[0] == 0:
            continue

        # Use only cbg column
        x_cbg = x_np[:, cbg_idx]
        X_list.append(x_cbg.flatten())

        # Target: predict first future CBG
        y_cbg = y_np[0, cbg_idx]
        y_list.append(y_cbg)

    X = np.stack(X_list)
    y = np.array(y_list)
    return X, y
