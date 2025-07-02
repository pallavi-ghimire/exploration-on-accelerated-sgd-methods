import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --------------------------
# Centralized Configuration
# --------------------------
svrg_config = {
    "data": {
        "file_path": "dataset/n100000_d20_mu1_L3198.csv",
        "features": [f"x{i}" for i in range(10)],
        "target": "y",
    },
    "train_test_split": {
        "test_size": 0.2,
        "random_state": 42,
    },
    "ridge_regression": {
        "lambda": 0.01,
        "lr": 0.01,
        "epochs": 20,
        "m": 1000,
    },
    "spectral_analysis": {
        "lambda_eigen": 0.01,
    },
    "plot": {
        "interval": 1,
    }
}

# --------------------------
# Load Dataset
# --------------------------
df = pd.read_csv(svrg_config["data"]["file_path"])
X = df[svrg_config["data"]["features"]].values
y = df[svrg_config["data"]["target"]].values

# --------------------------
# Preprocessing
# --------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=svrg_config["train_test_split"]["test_size"],
    random_state=svrg_config["train_test_split"]["random_state"]
)

# --------------------------
# Functions
# --------------------------
def get_largest_eigenvalue(lam):
    n_train = X_train.shape[0]
    hessian = 2 * (X_train.T @ X_train) / n_train + 2 * lam * np.eye(X_train.shape[1])
    eigenvalues = np.linalg.eigvalsh(hessian)
    L = np.max(eigenvalues)
    print(f"Largest eigenvalue (L): {L}")
    return L


def compute_loss(X, y, w, lam):
    w_unscaled = w / scaler.scale_
    X_unscaled = X * scaler.scale_
    n = len(y)
    residuals = X_unscaled @ w_unscaled - y
    return (1 / n) * np.sum(residuals ** 2) + lam * np.sum(w_unscaled ** 2)


def closed_form_computation():
    sol = np.loadtxt("dataset/n100000_d20_mu1_L3198_w_star")
    return sol


def svrg_ridge_regression(X, y, lam, lr, epochs, m):
    n, d = X.shape
    w_tilde = np.zeros(d)
    loss_history = []
    dist_history = []

    # Load closed-form solution (in unscaled/original space)
    w_closed = closed_form_computation()

    for epoch in range(epochs):
        full_grad = (2 / n) * X.T @ (X @ w_tilde - y) + 2 * lam * w_tilde
        w = w_tilde.copy()

        inner_iterates_w = []

        for t in range(m):
            i = np.random.randint(0, n)
            x_i = X[i].reshape(1, -1)
            y_i = y[i]

            grad_i = 2 * x_i.T @ (x_i @ w - y_i) + 2 * lam * w
            grad_i_tilde = 2 * x_i.T @ (x_i @ w_tilde - y_i) + 2 * lam * w_tilde

            w -= lr * (grad_i - grad_i_tilde + full_grad)

            inner_iterates_w.append(w.copy())

        w_tilde = sum(inner_iterates_w) / len(inner_iterates_w)

        if epoch % svrg_config["plot"]["interval"] == 0:
            w_unscaled = w / scaler.scale_
            # Loss is computed in scaled space (matches training)
            loss = compute_loss(X, y, w, lam)

            # Distance is computed in unscaled/original space

            dist = np.linalg.norm(w_unscaled - w_closed)

            loss_history.append(loss)
            dist_history.append(dist)

    # Return weights in unscaled space
    w_unscaled = w / scaler.scale_
    w_closed_unscaled = w_closed  # Already in unscaled space

    return w_unscaled, loss_history, dist_history, w_closed_unscaled


def compute_sigma(X, y, w_star, lam):
    n = X.shape[0]
    return np.mean([
        np.linalg.norm(2 * (X[i] @ w_star - y[i]) * X[i] + 2 * lam * w_star)
        for i in range(n)
    ])


# --------------------------
# Main Function with Plotting
# --------------------------
def svrg_with_analytical_solution_comparison():
    lam = svrg_config["ridge_regression"]["lambda"]
    lr = svrg_config["ridge_regression"]["lr"]
    epochs = svrg_config["ridge_regression"]["epochs"]
    m = svrg_config["ridge_regression"]["m"]

    w, loss_history, dist_history, w_closed = svrg_ridge_regression(
        X_train, y_train, lam, lr, epochs, m
    )

    sigma = compute_sigma(X_train, y_train, w_closed, lam)

    x = np.arange(len(svrg_config["data"]["features"]))
    fig, axs = plt.subplots(3, 1, figsize=(12, 12))

    axs[0].plot(x, w_closed, label="w_* (Closed-form)", marker='o')
    axs[0].plot(x, w, label="w (SVRG)", marker='x')
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(svrg_config["data"]["features"], rotation=45)
    axs[0].set_ylabel("Weight Value")
    axs[0].set_title("Comparison of w (SVRG) vs w_* (Closed-form)")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(range(1, len(loss_history) + 1), loss_history, marker='o')
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss (Scaled Space)")
    axs[1].set_title("SVRG Optimization History (Loss)")
    axs[1].grid(True)

    axs[2].plot(range(1, len(dist_history) + 1), dist_history, marker='o')
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("||w - w*||")
    axs[2].set_title("Distance between SVRG Weight and Closed-form Solution")
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

    return sigma


# --------------------------
# Run
# --------------------------
if __name__ == "__main__":
    L = get_largest_eigenvalue(svrg_config["spectral_analysis"]["lambda_eigen"])
    sigma = svrg_with_analytical_solution_comparison()
    print("Sigma (gradient noise):", sigma)
