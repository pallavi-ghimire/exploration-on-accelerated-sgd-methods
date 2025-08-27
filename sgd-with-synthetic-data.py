import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# --------------------------
# Centralized Configuration
# --------------------------
sgd_config = {
    "data": {
        "file_path": "dataset/synthetic/d10_mu1_L10.csv",
        "file_path_optimal": "dataset/synthetic/d10_mu1_L10_w_star",
        "features": [f"x{i}" for i in range(10)],
        "target": "y",
        "d": 10,
    },
    "train_test_split": {
        "test_size": 0.2,
        "random_state": 42,
    },
    "ridge_regression": {
        "lambda": 0.01,
        "lr": 0.01,
        "iterations": 6000,
    },
    "plot": {
        "interval": 100,
        "save_path": "results/sgd/synthetic/sgd_d_10_Q_10_final.svg"
    },
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
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
X_scaled = X

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=sgd_config["train_test_split"]["test_size"],
    random_state=sgd_config["train_test_split"]["random_state"]
)

# --------------------------
# Functions
# --------------------------
def compute_loss(X, y, w, lam):
    # w_unscaled = w / scaler.scale_
    w_unscaled = w
    X_unscaled = X
    # X_unscaled = X * scaler.scale_
    residuals = X_unscaled @ w_unscaled - y
    return (1 / len(y)) * np.sum(residuals ** 2) + lam * np.sum(w_unscaled ** 2)

def closed_form_solution():
    # return np.loadtxt(sgd_config["data"]["file_path_optimal"])
    n, d = X_train.shape
    lam = sgd_config["ridge_regression"]["lambda"]
    A = X_train.T @ X_train + lam * n * np.eye(d)
    b = X_train.T @ y_train
    return np.linalg.solve(A, b)

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
            # dist = np.linalg.norm((w / scaler.scale_) - w_star)
            dist = np.linalg.norm(w - w_star)
            loss_history.append(loss)
            dist_history.append(dist)

    return w, loss_history, dist_history, w_star, w_0

def compute_ridge_loss(X, y, w, lam):
    residual = X @ w - y
    n = X.shape[0]
    mse = (1 / n) * np.sum(residual ** 2)
    reg = lam * np.sum(w ** 2)
    return mse + reg

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

    w_closed = closed_form_solution()
    closed_form_loss = compute_ridge_loss(X_train, y_train, w_closed, lam)

    print("Final Loss:", loss_history[-1])
    print("Closed-Form Loss: ", closed_form_loss)

    al, q, hess = get_largest_and_smallest_eigenvalue(lam=lam, w_0=w_0, w_k=w, w_opt=w_star)

    x = np.arange(len(sgd_config["data"]["features"]))
    fig, axs = plt.subplots(2, 1, figsize=(8, 10))

    axs[0].plot(np.arange(1, len(loss_history) + 1) * sgd_config["plot"]["interval"],
                loss_history, marker='o', label='SGD Loss')

    axs[0].axhline(y=closed_form_loss, color='red', linestyle='--', label='Closed-form Loss')

    axs[0].set_xlabel("Iteration", fontsize=18)
    axs[0].set_ylabel("Loss", fontsize=18)
    axs[0].tick_params(axis='both', labelsize=16)
    axs[0].set_title("SGD Optimization History", fontsize=18)
    axs[0].grid(True)
    axs[0].legend(fontsize=15)

    axs[1].plot(np.arange(1, len(dist_history) + 1) * sgd_config["plot"]["interval"], dist_history, marker='o')
    axs[1].set_xlabel("Iteration", fontsize=18)
    axs[1].set_ylabel("||w - w*||", fontsize=18)
    axs[1].tick_params(axis='both', labelsize=16)
    axs[1].set_title("Distance from Closed-form Solution", fontsize=18)
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(sgd_config["plot"]["save_path"], format="svg")
    plt.show()

# --------------------------
# Run
# --------------------------
if __name__ == "__main__":
    sgd_with_analytical_solution_comparison()

print(estimate_flops_ridge_regression_no_logging(n=80000, d=sgd_config["data"]["d"], T=sgd_config["ridge_regression"]["iterations"]))
