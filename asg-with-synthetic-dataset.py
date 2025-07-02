import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --------------------------
# Centralized Configuration
# --------------------------
asg_config = {
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
        "alpha": 0.001,
        "beta": 0.04,
        "iterations": 1000,
    },
    "spectral_analysis": {
        "lambda_eigen": 0.01,
        "eta": 0.001,
        "beta": 0.04,
    },
    "plot": {
        "interval": 20,
    }
}

# --------------------------
# Load Dataset
# --------------------------
df = pd.read_csv(asg_config["data"]["file_path"])
X = df[asg_config["data"]["features"]].values
y = df[asg_config["data"]["target"]].values

# --------------------------
# Preprocessing
# --------------------------
scaler = StandardScaler()
X_scaled = X
# X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=asg_config["train_test_split"]["test_size"],
    random_state=asg_config["train_test_split"]["random_state"]
)

# --------------------------
# Functions
# --------------------------
def get_largest_and_smallest_eigenvalue(lam):
    n_train = X_train.shape[0]
    hessian = 2 * (X_train.T @ X_train) / n_train + 2 * lam * np.eye(X_train.shape[1])
    eigenvalues = np.linalg.eigvalsh(hessian)  # More stable for symmetric matrices
    L = np.max(eigenvalues)
    mu = np.min(eigenvalues)
    Q = L / mu
    print("Q", Q, "L", L, "mu", mu)
    alpha = 1 / L
    beta = (np.sqrt(Q) - 1) / (np.sqrt(Q) + 1)
    return alpha, beta, Q, hessian


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
    # n, d = X.shape
    # I = np.eye(d)
    # sol = np.linalg.solve((1 / n) * X.T @ X + lam * I, (1 / n) * X.T @ y)
    # np.savetxt("n100000_d20_mu1_L3198_sol", sol)
    sol = np.loadtxt("dataset/n100000_d20_mu1_L3198_w_star")
    return sol


def asg_ridge_regression(X, y, lam, lr, beta, total_iterations):
    n, d = X.shape
    weights = np.zeros(d)
    weights_prev = weights.copy()
    loss_history = []
    dist_history = []
    w_closed_form = closed_form_computation(X, y, lam)

    for t in range(total_iterations):
        i = np.random.randint(0, n)
        x_i = X[i].reshape(1, -1)
        y_i = y[i]

        lookahead = weights + beta * (weights - weights_prev)
        grad = 2 * x_i.T @ (x_i @ lookahead - y_i) + 2 * lam * lookahead

        weights_prev = weights.copy()
        weights = lookahead - lr * grad

        if t % asg_config["plot"]["interval"] == 0:
            loss = compute_loss(X, y, weights, lam)
            loss_history.append(loss)
            lookahead = weights + beta * (weights - weights_prev)
            dist = np.linalg.norm(lookahead - w_closed_form)
            dist_history.append(dist)

    return weights, loss_history, dist_history, w_closed_form


def compute_sigma(X, y, w_star, lam):
    n = X.shape[0]
    return np.mean([np.linalg.norm(2 * (X[i] @ w_star - y[i]) * X[i] + 2 * lam * w_star) for i in range(n)])


# --------------------------
# Main Function with Plotting
# --------------------------
def asg_with_analytical_solution_comparison():
    lam = asg_config["ridge_regression"]["lambda"]
    alpha = asg_config["ridge_regression"]["alpha"]
    beta = asg_config["ridge_regression"]["beta"]
    iterations = asg_config["ridge_regression"]["iterations"]

    w_asg, loss_history, dist_history, w_closed_form = asg_ridge_regression(
        X_train, y_train, lam, alpha, beta, iterations
    )
    np.savetxt("n100000_d20_mu1_L3198_sol", w_asg)

    sigma = compute_sigma(X_train, y_train, w_closed_form, lam)

    x = np.arange(len(asg_config["data"]["features"]))
    fig, axs = plt.subplots(3, 1, figsize=(12, 12))

    axs[0].plot(x, w_closed_form, label="w_* (Closed-form)", marker='o')
    axs[0].plot(x, w_asg, label="w (ASG)", marker='x')
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(asg_config["data"]["features"], rotation=45)
    axs[0].set_ylabel("Weight Value")
    axs[0].set_title("Comparison of w (ASG) vs w_* (Closed-form)")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(range(1, len(loss_history) + 1), loss_history, marker='o')
    axs[1].set_xlabel("Iterations")
    axs[1].set_ylabel("Loss")
    axs[1].set_title("ASG Optimization History")
    axs[1].grid(True)

    axs[2].plot(range(1, len(dist_history) + 1), dist_history, marker='o')
    axs[2].set_xlabel("Iterations")
    axs[2].set_ylabel("||y_k - x*||")
    axs[2].set_title("Difference between Lookahead and Optimal Solution (||y_k - x*||)")
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()
    return sigma


# --------------------------
# Spectral Radius and Run
# --------------------------
if __name__ == "__main__":
    eta, b, Q, hess = get_largest_and_smallest_eigenvalue(asg_config["spectral_analysis"]["lambda_eigen"])
    A = construct_A(hess, eta, b)

    lam = asg_config["spectral_analysis"]["lambda_eigen"]
    eta = asg_config["spectral_analysis"]["eta"]
    b = asg_config["spectral_analysis"]["beta"]

    C_lambda = (1 - eta * (1 + b) * lam) ** 2 + eta ** 2 * lam ** 2
    del_lambda = C_lambda ** 2 - 4 * b ** 2 * (b ** 2 + 1)
    R_lambda = 1 / (2 ** 0.5) * (C_lambda + del_lambda ** 0.5) ** 0.5

    print("C_lambda, del_lambda, R_lambda:", C_lambda, del_lambda, R_lambda)
    print("Valid spectral radius!" if abs(R_lambda) < 1 else "Spectral radius too large!")

    rho = max(abs(np.linalg.eigvals(A)))
    print("\nSpectral radius ρ(A):", rho)
    max_singular_value = max(np.linalg.svd(A, compute_uv=False))
    print("\nLargest singular value:", max_singular_value)

    sigma = asg_with_analytical_solution_comparison()
    noise_term = (eta * ((1 + b) ** 2 + 1) ** 0.5 * sigma) / (1 - R_lambda)
    print("\nThe noise neighborhood proportional to sigma:", noise_term)
