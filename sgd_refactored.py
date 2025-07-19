import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

# Configuration
sgd_config = {
    "dataset_path": "dataset/SPX_clean.csv",
    "features": ['MA_10', 'MA_20', 'STD_20', 'Bollinger_Width', 'Lagged_Return_1'],
    "target": "Z_Score",
    "train_test_split": {
        "test_size": 0.2,
        "random_state": 1
    },
    "sgd": {
        "lambda": 1,
        "lr": 0.001,
        "iterations": 2000,
        "interval": 50
    },
    "plot": {
        "save_path": "results/sgd/sgd_plot_lambda_1_test.svg"
    }
}

# Load and preprocess data
df = pd.read_csv(sgd_config["dataset_path"])
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').dropna().reset_index(drop=True)

X = df[sgd_config["features"]].values
y = df[sgd_config["target"]].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=sgd_config["train_test_split"]["test_size"],
    random_state=sgd_config["train_test_split"]["random_state"]
)


def compute_loss(X, y, w, lam):
    n = len(y)
    return (1 / n) * np.sum((X @ w - y) ** 2) + lam * np.sum(w ** 2)


def closed_form_solution(X, y, lam):
    n, d = X.shape
    I = np.eye(d)
    return np.linalg.solve((1 / n) * X.T @ X + lam * I, (1 / n) * X.T @ y)


def sgd_ridge_regression(X, y, lam, lr, iterations, interval, w_star):
    n, d = X.shape
    w = np.zeros(d)
    loss_history = []
    dist_history = []

    for k in range(iterations):
        i = np.random.randint(0, n)
        x_i = X[i].reshape(1, -1)
        y_i = y[i]

        if k == 0:
            print("w_0", w)
            w_0 = w

        if k == iterations - 1:
            print("w_k+1", w)

        grad_i = 2 * x_i.T @ (x_i @ w - y_i) + 2 * lam * w
        w -= lr * grad_i.flatten()

        if (k + 1) % interval == 0:
            loss = compute_loss(X, y, w, lam)
            dist = np.linalg.norm(w - w_star)
            loss_history.append(loss)
            dist_history.append(dist)

    return w, loss_history, dist_history, w_0

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

    eta = sgd_config["sgd"]["lr"]
    m = sgd_config["sgd"]["iterations"]

    # Convergence rate constant phi (as derived from SVRG analysis)
    phi = Q / (m * (1 - 2 * L * eta)) + (2 * L * eta) / (1 - 2 * L * eta)

    # Loss function
    def P(w):
        return compute_loss(X_train, y_train, w, lam)

    # Convergence bound for function value
    # expectation_bound = phi ** k * (P(w_0) - P(w_opt))

    expectation_with = P(w_k) - P(w_opt)

    M_by_theta = expectation_with * 2 / (eta * Q)
    expectation_error = eta * Q * M_by_theta / 2

    print(f"L = {L:.5f}")
    print(f"mu = {mu:.5f}")
    print(f"Q = {Q:.5f}")
    print(f"alpha (1/L) = {alpha:.5f}")
    # print(f"phi = {phi:.5f}")
    # print(f"phi^k * (P(w_0) - P(w_*)) {k} epochs: {expectation_bound:.25e}")
    print(f"E[P(w_k)] - E[P(w_*)]): {expectation_with:.25e}")
    # print("expectation error: ", expectation_error)
    print("M/theta (estimated) = ", M_by_theta)

    return alpha, Q, hessian


def compute_ridge_loss(X, y, w, lam):
    residual = X @ w - y
    n = X.shape[0]
    mse = (1 / n) * np.sum(residual ** 2)
    reg = lam * np.sum(w ** 2)
    return mse + reg



def sgd_with_analytical_solution():
    lam = sgd_config["sgd"]["lambda"]
    lr = sgd_config["sgd"]["lr"]
    iterations = sgd_config["sgd"]["iterations"]
    interval = sgd_config["sgd"]["interval"]

    w_star = closed_form_solution(X_train, y_train, lam)
    w_sgd, loss_history, dist_history, w_0 = sgd_ridge_regression(
        X_train, y_train, lam, lr, iterations, interval, w_star
    )

    al, q, hess = get_largest_and_smallest_eigenvalue(lam=lam, w_0=w_0, w_k=w_sgd, w_opt=w_star)

    print("Closed-form w_*:", w_star)
    print("SGD weights w:", w_sgd)
    # w_closed = closed_form_solution()
    closed_form_loss = compute_ridge_loss(X_train, y_train, w_star, lam)

    print("Final Loss:", loss_history[-1])
    print("Closed-Form Loss: ", closed_form_loss)
    # print(f"RMSE on test set: {math.sqrt(mean_squared_error(y_test, X_test @ w_sgd)):.5f}")

    # Plotting
    x = np.arange(len(sgd_config["features"]))
    fig, axs = plt.subplots(3, 1, figsize=(12, 12))

    label_fontsize = 15
    tick_fontsize = 15
    title_fontsize = 16
    legend_fontsize = 15

    # Plot 1: Weight comparison
    axs[0].plot(x, w_star, label="w_* (Closed-form)", marker='o')
    axs[0].plot(x, w_sgd, label="w (SGD)", marker='x')
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(sgd_config["features"], rotation=45, fontsize=tick_fontsize)
    axs[0].set_title("Weight Comparison", fontsize=title_fontsize)
    axs[0].set_ylabel("Weight Value", fontsize=label_fontsize)
    axs[0].legend(fontsize=legend_fontsize)
    axs[0].tick_params(axis='y', labelsize=tick_fontsize)
    axs[0].grid(True)

    # Plot 2: Loss history
    axs[1].plot(range(1, len(loss_history) + 1), loss_history, marker='o', label='SGD Loss')
    axs[1].axhline(y=closed_form_loss, color='red', linestyle='--', label='Closed-form Loss')
    axs[1].set_xlabel("Interval", fontsize=label_fontsize)
    axs[1].set_ylabel("Loss", fontsize=label_fontsize)
    axs[1].set_title("SGD Loss History", fontsize=title_fontsize)
    axs[1].tick_params(axis='both', labelsize=tick_fontsize)
    axs[1].grid(True)
    axs[1].legend(fontsize=14)

    # Plot 3: Distance to optimal
    axs[2].plot(range(1, len(dist_history) + 1), dist_history, marker='o')
    axs[2].set_xlabel("Interval", fontsize=label_fontsize)
    axs[2].set_ylabel("||w - w*||", fontsize=label_fontsize)
    axs[2].set_title("Distance to Optimal Solution", fontsize=title_fontsize)
    axs[2].tick_params(axis='both', labelsize=tick_fontsize)
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig(sgd_config["plot"]["save_path"], format="svg")
    plt.show()


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


# === Run Experiment ===
sgd_with_analytical_solution()
print(estimate_flops_ridge_regression_no_logging(n=18658, d=5, T=sgd_config["sgd"]["iterations"]))


