import pandas as pd
import numpy as np
import glob
import os
import torch
import torch.nn as nn
import random

from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score



dir="train-data/train" #adjust this to your dir please

class create_train_val_pairs:
    def __init__(self,dir,train_ratio):
        self.dir = dir
        self.train_ratio = train_ratio

        self.files_list=[]
        self.train_files=[]
        self.val_files=[]
        self.X_train=[]
        self.Y_train=[]
        self.X_val=[]
        self.Y_val=[]

        self.X_train, self.Y_train, self.X_val, self.Y_val = self.run()
    def run(self):
        #load files
        for file in glob.glob(self.dir + "/*.csv"):
            self.files_list.append(file)
        #shuffle them and split them into train and val
        random.seed(42)
        random.shuffle(self.files_list)
        N = len(self.files_list)
        train_len = int(N * self.train_ratio)
        self.train_files = self.files_list[:train_len]
        self.val_files = self.files_list[train_len:]

        #fill in the train and val arrays
        for file in self.train_files:
            df = pd.read_csv(file)
            X = df.iloc[:, :-1]
            self.X_train.append(X)
            y = df.iloc[:, -1]
            self.Y_train.append(y)
        self.X_train = np.concatenate(self.X_train, axis=0)  # shape: (total_rows, num_features)
        self.Y_train = np.concatenate(self.Y_train, axis=0)  # shape: (total_rows,)

        for file in self.val_files:
            df = pd.read_csv(file)
            X = df.iloc[:, :-1]
            self.X_val.append(X)
            y = df.iloc[:, -1]
            self.Y_val.append(y)
        self.X_val = np.concatenate(self.X_val, axis=0)  # shape: (total_rows, num_features)
        self.Y_val = np.concatenate(self.Y_val, axis=0)  # shape: (total_rows,)

        return self.X_train, self.Y_train, self.X_val, self.Y_val

pairs = create_train_val_pairs(dir="train-data/train", train_ratio=0.8)

X_train = pairs.X_train
Y_train = pairs.Y_train
X_val   = pairs.X_val
Y_val   = pairs.Y_val

print(X_train.shape, Y_train.shape, X_val.shape, Y_val.shape)