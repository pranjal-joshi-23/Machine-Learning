import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

df = pd.read_csv("DataSets/Student_Performance(multipleLR).csv")

one_hot_extra_car = pd.get_dummies(df["Extracurricular Activities"], drop_first=True)
df = df.drop("Extracurricular Activities", axis=1)
df = df.join(one_hot_extra_car)

X = df.drop("Performance Index", axis=1).to_numpy()
X = (X - X.mean(axis=0)) / X.std(axis=0)
y = df["Performance Index"].to_numpy()

def split_dataset(X, y, percentage):
    # Normal Python Way
    # X_1, X_2, y_1, y_2 = [], [], [], []
    # split_count = int(percentage/100*len(X))
    # size = len(X)
    # random.seed(42)
    # index_in_1 = random.sample(range(size), split_count)
    # index_in_2 = [i for i in range(size) if i not in index_in_1]
    # for i in index_in_1:
    #     X_1.append(X[i])
    #     y_1.append(y[i])
    # for i in index_in_2:
    #     X_2.append(X[i])
    #     y_2.append(y[i])
    # X_1 = np.array(X_1)
    # X_2 = np.array(X_2)
    # y_1 = np.array(y_1)
    # y_2 = np.array(y_2)
    # return (X_1, X_2, y_1, y_2)
    
    # Numpy Way
    np.random.seed(42)
    idx = np.random.permutation(len(X))
    split = int(percentage/100 * len(X))
    X_1, X_2 = X[idx[:split]], X[idx[split:]]
    y_1, y_2 = y[idx[:split]], y[idx[split:]]
    return X_1, X_2, y_1, y_2


X_train, X_test, y_train, y_test = split_dataset(X, y, percentage=67)

def mae(y_pred, y):
    error = 0
    for i in range(len(y)):
        error += abs(y_pred[i] - y[i])
    total_error = error/len(y)
    
    return total_error

def mse(y_pred, y):
    error = 0
    for i in range(len(y)):
        error += (y_pred[i] - y[i])**2
    total_error = error/len(y)
    
    return total_error

def rmse(y_pred, y):
    error = 0
    for i in range(len(y)):
        error += (y_pred[i] - y[i])**2
    total_error = (error/len(y))**(1/2)
    
    return total_error

def r2_score(y_pred, y):
    y_mean = [sum(y)/len(y)] * len(y)
    
    mse_r = mse(y_pred, y)
    mse_m = mse(y, y_mean)
    
    score = 1 - mse_r/mse_m
    return score

def compute_cost(X, y, w, b):
    loss = 0
    for i in range(len(y)):
        loss += ((np.dot(X[i], w) + b) - y[i])**2
    cost = loss/len(y)
    
    return cost

def compute_gradient(X, y, w_in, b_in):
    j_wb_w = 0
    j_wb_b = 0
    for i in range(len(X)):
        j_wb = np.dot(X[i], w_in)+b_in
        j_wb_w += (j_wb - y[i]) * X[i]
        j_wb_b += j_wb - y[i]
    dj_w = 2*j_wb_w/len(X)
    dj_b = 2*j_wb_b/len(X)
        
    return (dj_w, dj_b)

def gradient_descent(X, y, w_in, b_in, alpha, epoches):
    w_fin = w_in
    b_fin = b_in
    w_history = []
    b_history = []
    cost_history = []
    for i in range(epoches):
        w_history.append(w_fin)
        b_history.append(b_fin)
        cost = compute_cost(X, y, w_fin, b_fin)
        cost_history.append(cost)
        
        df_w, df_b = compute_gradient(X, y, w_fin, b_fin)
        
        w_fin -= alpha * df_w
        b_fin -= alpha * df_b
        
    return (w_fin, b_fin, w_history, b_history, cost_history)

def predict(X, w, b):
    predictions = []
    for i in range(len(X)):
        predictions.append(np.dot(X[i], w) + b)
    
    return predictions

w_in = np.random.rand(len(X[0]))
b_in = random.random()
alpha = 0.01
epoches = 1000

w_fin, b_fin, w_history, b_history, cost_history = gradient_descent(X_train, y_train, w_in, b_in, alpha, epoches)
print(f"Weight= {w_fin}, Bias= {b_fin}")
np.set_printoptions(precision=3, suppress=False)

w_fin = np.round(w_fin, 2)
b_fin = np.round(b_fin, 2)

print(f"Weight= {w_fin}, Bias= {b_fin}")


y_pred_train = predict(X_train, w_fin, b_fin)
y_pred_test = predict(X_test, w_fin, b_fin)

print(f"MAE: training = {mae(y_pred_train, y_train)}, testing = {mae(y_pred_test, y_test)}")
print(f"MSE: training = {mse(y_pred_train, y_train)}, testing = {mse(y_pred_test, y_test)}")
print(f"RMSE: training = {rmse(y_pred_train, y_train)}, testing = {rmse(y_pred_test, y_test)}")
print(f"R2 Score: {r2_score(y_pred_test, y_test)}")
