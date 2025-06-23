# svrg_configured_runner.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression

# Central configuration
svrg_config = {
    "mode": "real",  # Options: "real", "synthetic"
    "dataset_path": "dataset/SPX_clean.csv",
    "features": ['MA_10', 'MA_20', 'STD_20', 'Bollinger_Width', 'Lagged_Return_1'],
    "target": "Z_Score",
    "train_test_split": {
        "test_size": 0.2,
        "random_state": 1
    },
    "svrg": {
        "lambda": 0.01,
        "learning_rate": 0.01,
        "epochs": 20,
        "inner_iterations": 60000
    }
}

def prepare_data(X, y, scale=True):
    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X
    return train_test_split(X_scaled, y,
        test_size=svrg_config["train_test_split"]["test_size"],
        random_state=svrg_config["train_test_split"]["random_state"]
    )

def compute_loss(X, y, w, lam):
    n = len(y)
    residuals = X @ w - y
    return (1 / n) * np.sum(residuals ** 2) + lam * np.sum(w ** 2)

def closed_form_computation(X, y, lam=0.01):
    n, d = X.shape
    I = np.eye(d)
    return np.linalg.solve((1 / n) * X.T @ X + lam * I, (1 / n) * X.T @ y)

def svrg_ridge_regression(X, y, lam, lr, epochs, m):
    n, d = X.shape
    w_tilde = np.zeros(d)
    history = []
    dist_history = []
    w_star = closed_form_computation(X, y, lam)

    for epoch in range(epochs):
        full_grad = (2 / n) * X.T @ (X @ w_tilde - y) + 2 * lam * w_tilde
        w = w_tilde.copy()
        inner_ws = []

        for _ in range(m):
            i = np.random.randint(0, n)
            x_i = X[i].reshape(1, -1)
            y_i = y[i]

            grad_i = 2 * x_i.T @ (x_i @ w - y_i) + 2 * lam * w
            grad_i_tilde = 2 * x_i.T @ (x_i @ w_tilde - y_i) + 2 * lam * w_tilde

            w -= lr * (grad_i - grad_i_tilde + full_grad)
            inner_ws.append(w.copy())

        w_tilde = sum(inner_ws) / len(inner_ws)
        loss = compute_loss(X, y, w, lam)
        dist = np.linalg.norm(w - w_star)
        history.append(loss)
        dist_history.append(dist)

    return w, history, dist_history, w_star

def svrg_with_analytical_solution_comparison(X_train, y_train, feature_names):
    cfg = svrg_config["svrg"]
    w, loss_history, dist_history, w_star = svrg_ridge_regression(
        X_train, y_train,
        lam=cfg["lambda"],
        lr=cfg["learning_rate"],
        epochs=cfg["epochs"],
        m=cfg["inner_iterations"]
    )

    print("SVRG Final Loss:", loss_history[-1])
    print("SVRG Weights:", w)
    print("Closed-Form Weights:", w_star)

    x = np.arange(len(feature_names))
    fig, axs = plt.subplots(3, 1, figsize=(12, 12))

    axs[0].plot(x, w_star, label="w* (Closed-form)", marker='o')
    axs[0].plot(x, w, label="w (SVRG)", marker='x')
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(feature_names, rotation=45)
    axs[0].set_ylabel("Weight Value")
    axs[0].set_title("w (SVRG) vs w* (Closed-form)")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(loss_history, marker='o')
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].set_title("SVRG Loss History")
    axs[1].grid(True)

    axs[2].plot(dist_history, marker='o')
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("||w - w*||")
    axs[2].set_title("Distance from SVRG to Closed-form")
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

def load_data():
    if svrg_config["mode"] == "real":
        df = pd.read_csv(svrg_config["dataset_path"])
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').dropna().reset_index(drop=True)
        X = df[svrg_config["features"]].values
        y = df[svrg_config["target"]].values
        feature_names = svrg_config["features"]

    elif svrg_config["mode"] == "synthetic":
        # Manually specify the path to your synthetic dataset CSV
        synthetic_path = "dataset/synthetic_data.csv"  # <-- Change this path as needed

        df = pd.read_csv(synthetic_path)

        # Specify which columns to use
        feature_names = ['f1', 'f2', 'f3', 'f4', 'f5']  # <-- Adjust based on your file
        target_column = 'y'  # <-- Adjust based on your file

        X = df[feature_names].values
        y = df[target_column].values
        return X, y, feature_names

    else:
        raise ValueError("Invalid mode. Choose 'real' or 'synthetic'.")

    return X, y, feature_names

def run():
    X, y, feature_names = load_data()
    X_train, X_test, y_train, y_test = prepare_data(X, y)
    svrg_with_analytical_solution_comparison(X_train, y_train, feature_names)

if __name__ == "__main__":
    run()
