import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

df = pd.read_excel("DataSets/Pumpkin_Seeds_Dataset(logisticRorC).xlsx")

one_hot_class = pd.get_dummies(df["Class"], drop_first=True)
df = df.drop(["Class"], axis=1)
df = df.join(one_hot_class)
df = df.rename(columns={"Ürgüp Sivrisi": "Class"})

X = df.drop(["Class"], axis=1).to_numpy()
X = (X - X.mean(axis=0)) / X.std(axis=0)
y = df["Class"].to_numpy()

def split_dataset(X, y, percentage):
    np.random.seed(42)
    idx = np.random.permutation(len(X))
    split = int(percentage/100 * len(X))
    X_1, X_2 = X[idx[:split]], X[idx[split:]]
    y_1, y_2 = y[idx[:split]], y[idx[split:]]
    return X_1, X_2, y_1, y_2

X_train, X_test, y_train, y_test = split_dataset(X, y, percentage=70)

def linear_function(X, y, w, b):
    pass

def compute_cost():
    pass



w_in = np.random.rand(len(X[0]))
b_in = random.random()
