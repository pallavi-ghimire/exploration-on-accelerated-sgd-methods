import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --------------------------
# Centralized Configuration
# --------------------------
sgd_config = {
    "data": {
        "file_path": "dataset/n100000_d20_mu1_L3198.csv",
        "features": [f"x{i}" for i in range(50)],
        "target": "y",
    },
    "train_test_split": {
        "test_size": 0.2,
        "random_state": 42,
    },
    "ridge_regression": {
        "lambda": 0.01,
        "lr": 0.005,
        "iterations": 80000,
    },
    "plot": {
        "interval": 1000,
    }
}

# --------------------------
# Load Dataset
# --------------------------
df = pd.read_csv(sgd_config["data"]["file_path"])
X = df[sgd_config["data"]["features"]].values
y = df[sgd_config["data"]["target"]].values

# --------------------------
# Preprocessing
# --------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=sgd_config["train_test_split"]["test_size"],
    random_state=sgd_config["train_test_split"]["random_state"]
)

# --------------------------
# Functions
# --------------------------
def compute_loss(X, y, w, lam):
    w_unscaled = w / scaler.scale_
    X_unscaled = X * scaler.scale_
    residuals = X_unscaled @ w_unscaled - y
    return (1 / len(y)) * np.sum(residuals ** 2) + lam * np.sum(w_unscaled ** 2)

def closed_form_solution():
    return np.loadtxt("dataset/n100000_d20_mu1_L3198_w_star")

def sgd_ridge_regression(X, y, lam, lr, iterations):
    n, d = X.shape
    w = np.zeros(d)
    loss_history = []
    dist_history = []

    w_star = closed_form_solution()

    for k in range(iterations):
        i = np.random.randint(0, n)
        x_i = X[i].reshape(1, -1)
        y_i = y[i]

        if k == 0:
            print("w_0", w)
            w_0 = w

        grad_i = 2 * x_i.T @ (x_i @ w - y_i) + 2 * lam * w
        w -= lr * grad_i.flatten()

        if (k + 1) % sgd_config["plot"]["interval"] == 0:
            loss = compute_loss(X, y, w, lam)
            dist = np.linalg.norm((w / scaler.scale_) - w_star)
            loss_history.append(loss)
            dist_history.append(dist)

    return w, loss_history, dist_history, w_star, w_0

def estimate_flops_ridge_regression_no_logging(n, d, T):
    """
    Estimate the number of floating point operations (FLOPs) for:
    - Closed-form ridge regression
    - SGD ridge regression (excluding any logging)

    Parameters:
        n (int): Number of training samples
        d (int): Number of features
        T (int): Number of SGD iterations

    Returns:
        dict: Dictionary with estimated FLOP counts for each method (integers)
    """
    # Closed-form FLOPs:
    # 2nd^2 + 2nd + 2d^2 + (2/3)d^3
    flops_closed_form = 2 * n * d ** 2 + 2 * n * d + 2 * d ** 2 + (2 / 3) * d ** 3

    # SGD FLOPs:
    # 5d FLOPs per iteration (no logging)
    flops_sgd = T * 5 * d

    return {
        "closed_form_flops": int(flops_closed_form),
        "sgd_flops": int(flops_sgd)
    }

def get_largest_and_smallest_eigenvalue(lam, w_0, w_k, w_opt):
    n_train = X_train.shape[0]
    d = X_train.shape[1]

    # Hessian of the ridge regression loss
    hessian = 2 * (X_train.T @ X_train) / n_train + 2 * lam * np.eye(d)
    eigenvalues = np.linalg.eigvals(hessian)
    L = np.max(eigenvalues)    # Smoothness constant
    mu = np.min(eigenvalues)   # Strong convexity constant
    Q = L / mu                 # Condition number
    alpha = 1 / L              # Ideal learning rate

    eta = sgd_config["ridge_regression"]["lr"]
    m = sgd_config["ridge_regression"]["iterations"]

    # Convergence rate constant phi (as derived from SVRG analysis)
    phi = Q / (m * (1 - 2 * L * eta)) + (2 * L * eta) / (1 - 2 * L * eta)

    # Loss function
    def P(w):
        return compute_loss(X_train, y_train, w, lam)

    # Convergence bound for function value
    # expectation_bound = phi ** k * (P(w_0) - P(w_opt))

    expectation_with = P(w_k) - P(w_opt)
    expectation_error = eta * Q / 2 * mu

    M_by_theta = expectation_with * 2 * mu / (eta * Q)

    print(f"L = {L:.5f}")
    print(f"mu = {mu:.5f}")
    print(f"Q = {Q:.5f}")
    print(f"alpha (1/L) = {alpha:.5f}")
    # print(f"phi = {phi:.5f}")
    # print(f"phi^k * (P(w_0) - P(w_*)) {k} epochs: {expectation_bound:.25e}")
    print(f"E[P(w_k)] - E[P(w_*)]): {expectation_with:.25e}")
    print("M/theta (estimated) = ", M_by_theta)

    return alpha, Q, hessian



# --------------------------
# Main Execution
# --------------------------
def sgd_with_analytical_solution_comparison():
    lam = sgd_config["ridge_regression"]["lambda"]
    lr = sgd_config["ridge_regression"]["lr"]
    iterations = sgd_config["ridge_regression"]["iterations"]

    w, loss_history, dist_history, w_star, w_0 = sgd_ridge_regression(X_train, y_train, lam, lr, iterations)

    print("Final Loss:", loss_history[-1])

    al, q, hess = get_largest_and_smallest_eigenvalue(lam=lam, w_0=w_0, w_k=w, w_opt=w_star)

    x = np.arange(len(sgd_config["data"]["features"]))
    fig, axs = plt.subplots(3, 1, figsize=(12, 12))

    axs[0].plot(x, w_star, label="w_* (Closed-form)", marker='o')
    axs[0].plot(x, w / scaler.scale_, label="w (SGD)", marker='x')
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(sgd_config["data"]["features"], rotation=45)
    axs[0].set_ylabel("Weight Value")
    axs[0].set_title("Comparison of w (SGD) vs w_* (Closed-form)")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(np.arange(1, len(loss_history) + 1) * sgd_config["plot"]["interval"], loss_history, marker='o')
    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel("Loss")
    axs[1].set_title("SGD Optimization History")
    axs[1].grid(True)

    axs[2].plot(np.arange(1, len(dist_history) + 1) * sgd_config["plot"]["interval"], dist_history, marker='o')
    axs[2].set_xlabel("Iteration")
    axs[2].set_ylabel("||w - w*||")
    axs[2].set_title("Distance from Closed-form Solution")
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

# --------------------------
# Run
# --------------------------
if __name__ == "__main__":
    sgd_with_analytical_solution_comparison()

print(estimate_flops_ridge_regression_no_logging(n=80000, d=50, T=sgd_config["ridge_regression"]["iterations"]))
