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
        "file_path": "dataset/test/d25_mu1_L50.csv",
        "file_path_optimal": "dataset/test/d25_mu1_L50_w_star",
        "features": [f"x{i}" for i in range(25)],
        "target": "y",
    },
    "train_test_split": {
        "test_size": 0.2,
        "random_state": 42,
    },
    "ridge_regression": {
        "lambda": 0.01,
        "lr": 0.01,
        "epochs": 12,
        "m": 5000,
    },
    "spectral_analysis": {
        "lambda_eigen": 0.01,
    },
    "plot": {
        "interval": 1,
        "save_path": 'results/svrg/synthetic/svrg_d_25_q_50_plot.svg'
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
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
X_scaled = X

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
    w_unscaled = w
    X_unscaled = X
    # X_unscaled = X * scaler.scale_
    residuals = X_unscaled @ w_unscaled - y
    return (1 / len(y)) * np.sum(residuals ** 2) + lam * np.sum(w_unscaled ** 2)


# def closed_form_computation():
#     sol = np.loadtxt("dataset/n100000_d20_mu1_L3198_w_star")
#     return sol

def closed_form_computation():
    n, d = X_train.shape
    lam = svrg_config["ridge_regression"]["lambda"]
    A = X_train.T @ X_train + lam * n * np.eye(d)
    b = X_train.T @ y_train
    return np.linalg.solve(A, b)

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

    eta = svrg_config["ridge_regression"]["lr"]
    m = svrg_config["ridge_regression"]["m"]
    k = svrg_config["ridge_regression"]["epochs"]

    # Convergence rate constant phi (as derived from SVRG analysis)
    phi = Q / (m * (1 - 2 * L * eta)) + (2 * L * eta) / (1 - 2 * L * eta)

    # Loss function
    def P(w):
        return compute_loss(X_train, y_train, w, lam)

    # Convergence bound for function value
    expectation_bound = phi ** k * (P(w_0) - P(w_opt))

    expectation_with = P(w_k) - P(w_opt)

    print(f"L = {L:.5f}")
    print(f"mu = {mu:.5f}")
    print(f"Q = {Q:.5f}")
    print(f"alpha (1/L) = {alpha:.5f}")
    print(f"phi = {phi:.5f}")
    print(f"phi^k * (P(w_0) - P(w_*)) {k} epochs: {expectation_bound:.25e}")
    print(f"E[P(w_k)] - E[P(w_*)]): {expectation_with:.25e}")

    return alpha, Q, hessian

def compute_ridge_loss(X, y, w, lam):
    residual = X @ w - y
    n = X.shape[0]
    mse = (1 / n) * np.sum(residual ** 2)
    reg = lam * np.sum(w ** 2)
    return mse + reg

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

        if epoch == 0:
            print("w_0", w)
            w_0 = w

        if epoch == epochs - 1:
            print("w_k+1", w)

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
            # w_unscaled = w / scaler.scale_
            w_unscaled = w
            # Loss is computed in scaled space (matches training)
            loss = compute_loss(X, y, w, lam)

            # Distance is computed in unscaled/original space

            dist = np.linalg.norm(w_unscaled - w_closed)

            loss_history.append(loss)
            dist_history.append(dist)

    # Return weights in unscaled space
    # w_unscaled = w / scaler.scale_
    w_unscaled = w
    w_closed_unscaled = w_closed  # Already in unscaled space

    return w_unscaled, loss_history, dist_history, w_closed_unscaled, w_0


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

    w, loss_history, dist_history, w_closed, w_0 = svrg_ridge_regression(
        X_train, y_train, lam, lr, epochs, m
    )
    get_largest_and_smallest_eigenvalue(lam=lam, w_0=w_0, w_k=w, w_opt=w_closed)

    sigma = compute_sigma(X_train, y_train, w_closed, lam)

    # w_closed = closed_form_solution()
    closed_form_loss = compute_ridge_loss(X_train, y_train, w_closed, lam)

    print("Final Loss:", loss_history[-1])
    print("Closed-Form Loss: ", closed_form_loss)

    x = np.arange(len(svrg_config["data"]["features"]))

    fig, axs = plt.subplots(2, 1, figsize=(8, 10))

    # Loss plot
    axs[0].plot(
        np.arange(1, len(loss_history) + 1) * svrg_config["plot"]["interval"],
        loss_history,
        marker='o', label='SVRG Loss'
    )
    axs[0].axhline(y=closed_form_loss, color='red', linestyle='--', label='Closed-form Loss')

    axs[0].set_xlabel("Iteration", fontsize=15)
    axs[0].set_ylabel("Loss", fontsize=15)
    axs[0].tick_params(axis='both', labelsize=15)
    axs[0].set_title("SVRG Optimization History", fontsize=15)
    axs[0].grid(True)
    axs[0].legend(fontsize=15)
    # axs[0].set_xlabel("Iteration", fontsize=13)
    # axs[0].set_ylabel("Loss", fontsize=13)
    # axs[0].tick_params(axis='both', labelsize=14)
    # axs[0].set_title("SVRG Optimization History", fontsize=15)
    # axs[0].grid(True)

    # Distance plot
    axs[1].plot(
        np.arange(1, len(dist_history) + 1) * svrg_config["plot"]["interval"],
        dist_history,
        marker='o'
    )
    axs[1].set_xlabel("Iteration", fontsize=15)
    axs[1].set_ylabel("||w - w*||", fontsize=15)
    axs[1].tick_params(axis='both', labelsize=15)
    axs[1].set_title("Distance from Closed-form Solution", fontsize=15)
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(svrg_config["plot"]["save_path"], format="svg")
    plt.show()

    return sigma


def estimate_flops_closed_form_and_svrg(n, d, T_svrg, m, lam=0.01):
    """
    Estimate FLOPs for:
    - Closed-form ridge regression
    - SVRG for ridge regression

    Parameters:
        n (int): Number of training samples
        d (int): Number of features
        T_svrg (int): Number of outer epochs in SVRG
        m (int): Number of inner steps per epoch
        lam (float): Regularization strength (used only for naming clarity)

    Returns:
        dict: Estimated FLOPs for both methods
    """
    # === Closed-form ===
    # FLOPs: 2nd^2 + 2nd + 2d^2 + (2/3)d^3
    flops_closed_form = 2 * n * d**2 + 2 * n * d + 2 * d**2 + (2 / 3) * d**3

    # === SVRG ===
    # For each epoch:
    #   Full gradient: X @ w_tilde → 2nd
    #   Then X.T @ (...) → 2nd
    #   Regularization term: 2d
    flops_full_gradient = (T_svrg - 1) * (2 * n * d + 2 * n * d + 2 * d)

    # Inner loop:
    # Each inner iteration (m steps per epoch):
    #   - grad_i: x_i.T @ (x_i @ w - y_i) → 2d
    #   - grad_i_tilde: same → 2d
    #   - regularization terms: 2d
    #   - final update (subtract scaled gradient): 2d
    flops_per_inner_iter = 2 * d + 2 * d + 2 * d + 2 * d  # = 8d
    flops_inner_loop = (T_svrg - 1) * m * flops_per_inner_iter

    # Total SVRG FLOPs
    flops_svrg = flops_full_gradient + flops_inner_loop

    return {
        "closed_form_flops": int(flops_closed_form),
        "svrg_flops": int(flops_svrg)
    }

print(estimate_flops_closed_form_and_svrg(n=80000, d=50, T_svrg=svrg_config["ridge_regression"]["epochs"], m=svrg_config["ridge_regression"]["m"], lam = svrg_config["ridge_regression"]["lambda"]))


# --------------------------
# Run
# --------------------------
if __name__ == "__main__":
    L = get_largest_eigenvalue(svrg_config["spectral_analysis"]["lambda_eigen"])
    sigma = svrg_with_analytical_solution_comparison()
    # print("Sigma (gradient noise):", sigma)
