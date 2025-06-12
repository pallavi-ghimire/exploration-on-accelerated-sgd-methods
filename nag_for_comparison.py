# Refactored ASG Ridge Regression code with centralized configuration

import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Centralized configuration
asg_config = {
    "dataset_path": "dataset/SPX_clean.csv",
    "features": ['MA_10', 'MA_20', 'STD_20', 'Bollinger_Width', 'Lagged_Return_1'],
    "target": "Z_Score",
    "train_test_split": {
        "test_size": 0.2,
        "random_state": 1
    },
    "ridge_regression": {
        "lambda": 0.01,
        "alpha": 0.01,
        "beta": 0.04,
        "iterations": 3000,
        "noise_std": 0.05,
    },
    "ridge_regression_minibatch": {
        "lambda": 0.01,
        "alpha": 0.01,
        "beta": 0.045,
        "iterations": 500,
        "noise_std": 0.05,
        "batch_size": 32
    },
    "spectral_analysis": {
        "lambda_eigen": 0.1,
        "eta": 0.01,
        "beta": 0.04
    },
    "plot": {
        "interval": 10
    }
}

# Load and preprocess data
df = pd.read_csv(asg_config["dataset_path"])
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').dropna().reset_index(drop=True)

X = df[asg_config["features"]].values
y = df[asg_config["target"]].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=asg_config["train_test_split"]["test_size"],
    random_state=asg_config["train_test_split"]["random_state"]
)

def construct_A(H, alpha, beta):
    d = H.shape[0]
    I = np.eye(d)
    top_left = I - alpha * (1 + beta) * H
    top_right = beta ** 2 * I
    bottom_left = -alpha * H
    bottom_right = beta * I
    return np.block([
        [top_left, top_right],
        [bottom_left, bottom_right]
    ])


def compute_loss(X, y, w, lam):
    n = len(y)
    residuals = X @ w - y
    return (1 / n) * np.sum(residuals ** 2) + lam * np.sum(w ** 2)


def closed_form_computation(X, y, lam):
    n, d = X.shape
    I = np.eye(d)
    return np.linalg.solve((1 / n) * X.T @ X + lam * I, (1 / n) * X.T @ y)


# Full-batch Nesterov Accelerated Gradient (deterministic)
def nag_ridge_regression_deterministic(X, y, lam, lr, beta, total_iterations, w_closed_form):
    n, d = X.shape
    weights = np.zeros(d)
    weights_prev = weights.copy()
    loss_history = []
    dist_history = []

    for t in range(total_iterations):
        # Lookahead step
        lookahead = weights + beta * (weights - weights_prev)

        # Full-batch gradient at lookahead
        grad = (2 / n) * X.T @ (X @ lookahead - y) + 2 * lam * lookahead

        # Weight update
        weights_next = lookahead - lr * grad

        # Update memory
        weights_prev = weights
        weights = weights_next

        # Logging
        if t % asg_config["plot"]["interval"] == 0:
            loss = compute_loss(X, y, weights, lam)
            loss_history.append(loss)
            dist = np.linalg.norm(weights - w_closed_form)
            dist_history.append(dist)

    return weights, loss_history, dist_history


def nag_deterministic_with_analytical_solution_comparison(w_closed_form):
    lam = asg_config["ridge_regression"]["lambda"]
    alpha = asg_config["ridge_regression"]["alpha"]
    beta = asg_config["ridge_regression"]["beta"]
    iterations = asg_config["ridge_regression"]["iterations"]

    w_nag, loss_history, dist_history = nag_ridge_regression_deterministic(
        X_train, y_train, lam, alpha, beta, iterations, w_closed_form
    )

    x = np.arange(len(asg_config["features"]))
    fig, axs = plt.subplots(3, 1, figsize=(12, 12))

    axs[0].plot(x, w_closed_form, label="w_* (Closed-form)", marker='o')
    axs[0].plot(x, w_nag, label="w (Deterministic NAG)", marker='x')
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(asg_config["features"], rotation=45)
    axs[0].set_ylabel("Weight Value")
    axs[0].set_title("Comparison of w (NAG) vs w_* (Closed-form)")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(range(1, len(loss_history) + 1), loss_history, marker='o')
    axs[1].set_xlabel("Iterations")
    axs[1].set_ylabel("Loss")
    axs[1].set_title("NAG Optimization History (Deterministic)")
    axs[1].grid(True)

    axs[2].plot(range(1, len(dist_history) + 1), dist_history, marker='o')
    axs[2].set_xlabel("Iterations")
    axs[2].set_ylabel("||w_k - x*||")
    axs[2].set_title("Distance from Optimal Solution (||w_k - x*||)")
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()


def deterministic_minibatch_nag_multi_batch(X, y, lam, lr, beta, total_epochs, w_closed_form, batch_size=1500):
    n, d = X.shape
    weights = np.zeros(d)
    weights_prev = np.zeros(d)
    loss_history = []
    dist_history = []

    # Precompute fixed batches deterministically
    num_batches = int(np.ceil(n / batch_size))
    batch_indices_list = [np.arange(i * batch_size, min((i + 1) * batch_size, n)) for i in range(num_batches)]

    for epoch in range(total_epochs):
        for batch_indices in batch_indices_list:
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            # Lookahead
            lookahead = weights + beta * (weights - weights_prev)

            # Gradient at lookahead
            grad = 2 * X_batch.T @ (X_batch @ lookahead - y_batch) / len(batch_indices) + 2 * lam * lookahead

            # Momentum update
            new_weights = lookahead - lr * grad
            weights_prev = weights
            weights = new_weights

        # Monitoring after each epoch
        if epoch % asg_config["plot"]["interval"] == 0:
            loss = compute_loss(X, y, weights, lam)
            loss_history.append(loss)
            dist = np.linalg.norm(weights - w_closed_form)
            dist_history.append(dist)

    print("optimized weights (deterministic multi-batch NAG)", weights)
    return weights, loss_history, dist_history


def deterministic_nag_minibatch_comparison(w_closed_form, batch_size=500):
    lam = asg_config["ridge_regression_minibatch"]["lambda"]
    alpha = asg_config["ridge_regression_minibatch"]["alpha"]
    beta = asg_config["ridge_regression_minibatch"]["beta"]
    iterations = asg_config["ridge_regression_minibatch"]["iterations"]

    w_nag_det_mb, loss_history, dist_history = deterministic_minibatch_nag_multi_batch(
        X_train, y_train, lam, alpha, beta, iterations, w_closed_form, batch_size=batch_size
    )

    x = np.arange(len(asg_config["features"]))
    fig, axs = plt.subplots(3, 1, figsize=(12, 12))

    axs[0].plot(x, w_closed_form, label="w_* (Closed-form)", marker='o')
    axs[0].plot(x, w_nag_det_mb, label="w (Deterministic NAG Minibatch)", marker='x')
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(asg_config["features"], rotation=45)
    axs[0].set_ylabel("Weight Value")
    axs[0].set_title("Deterministic NAG Minibatch vs Closed-form")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(range(1, len(loss_history) + 1), loss_history, marker='o')
    axs[1].set_xlabel("Iterations")
    axs[1].set_ylabel("Loss")
    axs[1].set_title("Deterministic NAG Minibatch Loss")
    axs[1].grid(True)

    axs[2].plot(range(1, len(dist_history) + 1), dist_history, marker='o')
    axs[2].set_xlabel("Iterations")
    axs[2].set_ylabel("||y_k - x*||")
    axs[2].set_title("Minibatch (Deterministic): Lookahead Distance to Optimal")
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()


closed_form_value = closed_form_computation(X_train, y_train, lam=asg_config["ridge_regression"]["lambda"])
# nag_deterministic_with_analytical_solution_comparison(closed_form_value)

deterministic_nag_minibatch_comparison(closed_form_value)
