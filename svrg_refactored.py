import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

# Configuration dictionary
svrg_config = {
    "dataset_path": "dataset/SPX_clean.csv",
    "features": ['MA_10', 'MA_20', 'STD_20', 'Bollinger_Width', 'Lagged_Return_1'],
    "target": "Z_Score",
    "train_test_split": {
        "test_size": 0.2,
        "random_state": 1
    },
    "svrg": {
        "lambda": 1,
        "lr": 0.01,
        "epochs": 5,
        "m": 200,
    },
    "plot": {
        "save_path": "results/svrg/name_of_file.svg"
    }
}

# Load and preprocess data
df = pd.read_csv(svrg_config["dataset_path"])
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').dropna().reset_index(drop=True)

X = df[svrg_config["features"]].values
y = df[svrg_config["target"]].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=svrg_config["train_test_split"]["test_size"],
    random_state=svrg_config["train_test_split"]["random_state"]
)


def compute_loss(X, y, w, lam):
    n = len(y)
    return (1 / n) * np.sum((X @ w - y) ** 2) + lam * np.sum(w ** 2)


def compute_ridge_loss(X, y, w, lam):
    residual = X @ w - y
    n = X.shape[0]
    mse = (1 / n) * np.sum(residual ** 2)
    reg = lam * np.sum(w ** 2)
    return mse + reg

def closed_form_solution(X, y, lam):
    n, d = X.shape
    I = np.eye(d)
    return np.linalg.solve((1 / n) * X.T @ X + lam * I, (1 / n) * X.T @ y)


def svrg_ridge_regression(X, y, lam, lr, epochs, m, w_star):
    n, d = X.shape
    w_tilde = np.zeros(d)
    loss_history = []
    dist_history = []

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

        w_tilde = np.mean(inner_iterates_w, axis=0)
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

    eta = svrg_config["svrg"]["lr"]
    m = svrg_config["svrg"]["m"]
    k = svrg_config["svrg"]["epochs"]

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
    print(f"alpha (1/L) = {eta:.5f}")
    print(f"phi = {phi:.5f}")
    print(f"phi^k * (P(w_0) - P(w_*)) {k} epochs: {expectation_bound:.25e}")
    print(f"E[P(w_k)] - E[P(w_*)]): {expectation_with:.25e}")

    return eta, Q, hessian


def svrg_with_analytical_solution():
    lam = svrg_config["svrg"]["lambda"]
    lr = svrg_config["svrg"]["lr"]
    epochs = svrg_config["svrg"]["epochs"]
    m = svrg_config["svrg"]["m"]

    w_star = closed_form_solution(X_train, y_train, lam)
    w_svrg, loss_history, dist_history, w_0 = svrg_ridge_regression(X_train, y_train, lam, lr, epochs, m, w_star)
    get_largest_and_smallest_eigenvalue(lam=lam, w_0=w_0, w_k=w_svrg, w_opt=w_star)
    print("Closed-form w_*:", w_star)
    print("SVRG weights w:", w_svrg)

    closed_form_loss = compute_ridge_loss(X_train, y_train, w_star, lam)

    print("Final Loss:", loss_history[-1])
    print("Closed-Form Loss: ", closed_form_loss)

    x = np.arange(len(svrg_config["features"]))
    fig, axs = plt.subplots(3, 1, figsize=(12, 12))

    # Set common font sizes
    label_fontsize = 15
    tick_fontsize = 15
    title_fontsize = 16
    legend_fontsize = 15

    # Plot 1: Weight comparison
    axs[0].plot(x, w_star, label="w_* (Closed-form)", marker='o')
    axs[0].plot(x, w_svrg, label="w (SVRG)", marker='x')
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(svrg_config["features"], rotation=45, fontsize=tick_fontsize)
    axs[0].set_title("Weight Comparison", fontsize=title_fontsize)
    axs[0].set_ylabel("Weight Value", fontsize=label_fontsize)
    axs[0].legend(fontsize=legend_fontsize)
    axs[0].tick_params(axis='y', labelsize=tick_fontsize)
    axs[0].grid(True)

    # Plot 2: Loss history
    axs[1].plot(range(1, len(loss_history) + 1), loss_history, marker='o', label='SVRG Loss')
    axs[1].axhline(y=closed_form_loss, color='red', linestyle='--', label='Closed-form Loss')
    axs[1].set_xlabel("Epoch", fontsize=label_fontsize)
    axs[1].set_ylabel("Loss", fontsize=label_fontsize)
    axs[1].set_title("SVRG Loss History", fontsize=title_fontsize)
    axs[1].tick_params(axis='both', labelsize=tick_fontsize)
    axs[1].grid(True)
    axs[1].legend(fontsize=15)

    # Plot 3: Distance to optimal
    axs[2].plot(range(1, len(dist_history) + 1), dist_history, marker='o')
    axs[2].set_xlabel("Epoch", fontsize=label_fontsize)
    axs[2].set_ylabel("||w_s - w*||", fontsize=label_fontsize)
    axs[2].set_title("SVRG Distance to Optimal Solution", fontsize=title_fontsize)
    axs[2].tick_params(axis='both', labelsize=tick_fontsize)
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig(svrg_config["plot"]["save_path"], format="svg")
    plt.show()


# === RUN HERE ===
# Uncomment below lines to run experiments

# best_lam = tune_lambda_for_svrg()
# svrg_config["svrg"]["lambda"] = best_lam

svrg_with_analytical_solution()

def estimate_flops_closed_form_and_svrg(n, d, T_svrg, m, lam=0.01):
    """
    Estimate FLOPs for:
    - Closed-form ridge regression
    - SVRG for ridge regression
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


# --------------------------
# Run
# --------------------------
if __name__ == "__main__":
    svrg_with_analytical_solution()
    print(estimate_flops_closed_form_and_svrg(n=18658, d=5, T_svrg=svrg_config["svrg"]["epochs"],
                                              m=svrg_config["svrg"]["m"], lam=svrg_config["svrg"]["lambda"]))
