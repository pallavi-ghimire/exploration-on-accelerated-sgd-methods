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
        "file_path": "dataset/test/d50_mu1_L50.csv",
        "file_path_optimal": "dataset/test/d50_mu1_L50_w_star",
        "features": [f"x{i}" for i in range(50)],
        "target": "y",
    },
    "train_test_split": {
        "test_size": 0.2,
        "random_state": 42,
    },
    "ridge_regression_minibatch": {
        "lambda": 0.01,
        "alpha": 0.01,
        "beta": 0.48,
        "iterations": 2000,
        "noise_std": 0.05,
        "batch_size": 150
    },
    "spectral_analysis": {
        "lambda_eigen": 0.01,
        "eta": 0.001,
        "beta": 0.04,
    },
    "plot": {
        "interval": 20,
        "save_path": "results/asg/synthetic/asg_syn_d_50_Q_50.svg",
        "save_path_pareto": "results/asg/synthetic/asg_syn_d_50_Q_50_pareto.svg"
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


# def closed_form_computation(X, y, lam):
#     # n, d = X.shape
#     # I = np.eye(d)
#     # sol = np.linalg.solve((1 / n) * X.T @ X + lam * I, (1 / n) * X.T @ y)
#     # np.savetxt("n100000_d20_mu1_L3198_sol", sol)
#     sol = np.loadtxt("dataset/n100000_d20_mu1_L3198_w_star")
#     return sol

def closed_form_computation():
    n, d = X_train.shape
    lam = asg_config["ridge_regression_minibatch"]["lambda"]
    A = X_train.T @ X_train + lam * n * np.eye(d)
    b = X_train.T @ y_train
    return np.linalg.solve(A, b)

# def get_asg_convergence_properties(lam, w_0, w_k, w_opt, alpha, beta):
#     """
#     Compute ASG-specific spectral properties and convergence estimates.
#
#     Parameters:
#         lam (float): Ridge regularization parameter
#         w_0 (np.ndarray): Initial weights
#         w_k (np.ndarray): Final weights from ASG
#         w_opt (np.ndarray): Closed-form optimal solution
#         alpha (float): Step size (learning rate)
#         beta (float): Momentum parameter
#
#     Returns:
#         tuple: (rho, alpha, beta, C_lambda, R_lambda)
#     """
#     n_train = X_train.shape[0]
#     d = X_train.shape[1]
#
#     # A = construct_A(hess, alpha, b)
#     # lam = max(np.linalg.eigvals(A))
#
#     # Hessian of the ridge regression loss
#     H = 2 * (X_train.T @ X_train) / n_train + 2 * lam * np.eye(d)
#
#     # Construct system matrix A based on Nesterov-style ASG updates
#     A = construct_A(H, alpha, beta)
#
#     # Compute L, mu, Q
#     eigenvalues = np.linalg.eigvals(H)
#     L = np.max(eigenvalues)
#     mu = np.min(eigenvalues)
#     Q = L / mu
#
#     # compute values for R_lambda
#     eta = asg_config["spectral_analysis"]["eta"]
#     b = asg_config["spectral_analysis"]["beta"]
#     _, _, Q, hess = get_largest_and_smallest_eigenvalue(asg_config["spectral_analysis"]["lambda_eigen"])
#     A = construct_A(hess, eta, b)
#     l_for_r = max(np.linalg.eigvals(A))
#
#     # Spectral radius and singular value
#     rho = max(abs(np.linalg.eigvals(A)))
#     max_singular = max(np.linalg.svd(A, compute_uv=False))
#
#     # C_lambda and R_lambda (for bounding noise neighborhood)
#     C_lambda = (1 - alpha * (1 + beta) * l_for_r) ** 2 + alpha ** 2 * l_for_r ** 2
#     delta_lambda = C_lambda ** 2 - 4 * beta ** 2 * ((1 - alpha * l_for_r) ** 2)
#
#     if delta_lambda < 0:
#         R_lambda = np.nan
#     else:
#         R_lambda = (1 / np.sqrt(2)) * np.sqrt(C_lambda + np.sqrt(delta_lambda))
#
#     # Loss function
#     def P(w): return compute_loss(X_train, y_train, w, lam)
#
#     sigma = compute_sigma(X_train, y_train, w_opt, asg_config["ridge_regression_minibatch"]["lambda"], b, R_lambda)
#     print("sigma =", sigma)
#
#     LHS = P(w_k) - P(w_opt)
#     k = asg_config["ridge_regression_minibatch"]["iterations"]
#     RHS = L * ((alpha ** 2) * ((((1 + beta) ** 2) + 1) ** 2) * (sigma ** 2)) / 2 * (1 - (R_lambda ** 2))
#     # Print diagnostic information
#     print(f"L = {L:.5f}, mu = {mu:.5f}, Q = {Q:.5f}")
#     print(f"alpha = {alpha:.5f}, beta = {beta:.5f}")
#     print(f"rho (spectral radius of A) = {rho:.5f}")
#     print(f"max singular value of A = {max_singular:.5f}")
#     print(f"C_lambda = {C_lambda:.5f}")
#     print(f"delta_lambda = {delta_lambda:.5f}")
#     print(f"R_lambda = {R_lambda:.5f}")
#     print("E[P(v_k+1)] - E[P(w_*)] = ", LHS)
#     print("noise neighborhood proportional to sigma^2 = ", RHS)
#
#     return alpha, beta, Q, R_lambda, rho, C_lambda

def get_asg_convergence_properties(lam, w_0, w_k, w_opt, alpha, beta):
    """
    Compute ASG-specific spectral properties and convergence estimates.

    Parameters:
        lam (float): Ridge regularization parameter
        w_0 (np.ndarray): Initial weights
        w_k (np.ndarray): Final weights from ASG
        w_opt (np.ndarray): Closed-form optimal solution
        alpha (float): Step size (learning rate)
        beta (float): Momentum parameter

    Returns:
        tuple: (rho, alpha, beta, C_lambda, R_lambda)
    """
    n_train = X_train.shape[0]
    d = X_train.shape[1]

    # A = construct_A(hess, alpha, b)
    # lam = max(np.linalg.eigvals(A))

    # Hessian of the ridge regression loss
    H = 2 * (X_train.T @ X_train) / n_train + 2 * lam * np.eye(d)

    # Construct system matrix A based on Nesterov-style ASG updates
    A = construct_A(H, alpha, beta)

    # Compute L, mu, Q
    eigenvalues = np.linalg.eigvals(H)
    L = np.max(eigenvalues)
    mu = np.min(eigenvalues)
    Q = L / mu

    # compute values for R_lambda
    eta = asg_config["spectral_analysis"]["eta"]
    b = asg_config["spectral_analysis"]["beta"]
    _, _, Q, hess = get_largest_and_smallest_eigenvalue(asg_config["spectral_analysis"]["lambda_eigen"])
    A = construct_A(hess, eta, b)
    l_for_r = max(np.linalg.eigvals(A))

    # Spectral radius and singular value
    rho = max(abs(np.linalg.eigvals(A)))
    max_singular = max(np.linalg.svd(A, compute_uv=False))

    # C_lambda and R_lambda (for bounding noise neighborhood)
    C_lambda = (1 - alpha * (1 + beta) * l_for_r) ** 2 + alpha ** 2 * l_for_r ** 2
    delta_lambda = C_lambda ** 2 - 4 * beta ** 2 * ((1 - alpha * l_for_r) ** 2)

    if delta_lambda < 0:
        R_lambda = np.nan
    else:
        R_lambda = (1 / np.sqrt(2)) * np.sqrt(C_lambda + np.sqrt(delta_lambda))

    # Loss function
    def P(w): return compute_loss(X_train, y_train, w, lam)

    sigma = compute_sigma(X_train, y_train, w_opt, asg_config["ridge_regression_minibatch"]["lambda"], b, R_lambda)
    print("sigma =", sigma)

    LHS = P(w_k) - P(w_opt)
    k = asg_config["ridge_regression_minibatch"]["iterations"]
    RHS = L * ((alpha ** 2) * ((((1 + beta) ** 2) + 1) ** 2) * (sigma ** 2)) / 2 * (1 - (R_lambda ** 2))
    # Print diagnostic information
    print(f"L = {L:.5f}, mu = {mu:.5f}, Q = {Q:.5f}")
    print(f"alpha = {alpha:.5f}, beta = {beta:.5f}")
    print(f"rho (spectral radius of A) = {rho:.5f}")
    print(f"max singular value of A = {max_singular:.5f}")
    print(f"C_lambda = {C_lambda:.5f}")
    print(f"delta_lambda = {delta_lambda:.5f}")
    print(f"R_lambda = {R_lambda:.5f}")
    print("E[P(v_k+1)] - E[P(w_*)] = ", LHS)
    print("noise neighborhood proportional to sigma^2 = ", RHS)

    return alpha, beta, Q, R_lambda, rho, C_lambda




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

def compute_noise_term(sigma, eta, beta, R_lambda):
    noise_term = (eta * ((1 + beta) ** 2 + 1) ** 0.5 * sigma) / (1 - R_lambda)
    print("\nthe neighborhood proportional to sigma", noise_term)
    return noise_term

def compute_sigma(X, y, w_star, ridge_strength, beta, R_lambda):
    n = X.shape[0]
    eta = asg_config["ridge_regression_minibatch"]["alpha"]
    sigma = np.mean([np.linalg.norm(2 * (X[i] @ w_star - y[i]) * X[i] + 2 * ridge_strength * w_star) for i in range(n)])
    compute_noise_term(sigma, eta, beta, R_lambda)
    return sigma

def compute_ridge_loss(X, y, w, lam):
    residual = X @ w - y
    n = X.shape[0]
    mse = (1 / n) * np.sum(residual ** 2)
    reg = lam * np.sum(w ** 2)
    return mse + reg

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

    closed_form_loss = compute_ridge_loss(X_train, y_train, w_closed_form, lam)

    print("Final Loss:", loss_history[-1])
    print("Closed-Form Loss: ", closed_form_loss)

    x = np.arange(len(asg_config["data"]["features"]))
    # fig, axs = plt.subplots(3, 1, figsize=(12, 12))
    #
    # axs[0].plot(x, w_closed_form, label="w_* (Closed-form)", marker='o')
    # axs[0].plot(x, w_asg, label="w (ASG)", marker='x')
    # axs[0].set_xticks(x)
    # axs[0].set_xticklabels(asg_config["data"]["features"], rotation=45)
    # axs[0].set_ylabel("Weight Value")
    # axs[0].set_title("Comparison of w (ASG) vs w_* (Closed-form)")
    # axs[0].legend()
    # axs[0].grid(True)
    #
    # axs[1].plot(range(1, len(loss_history) + 1), loss_history, marker='o')
    # axs[1].set_xlabel("Iterations")
    # axs[1].set_ylabel("Loss")
    # axs[1].set_title("ASG Optimization History")
    # axs[1].grid(True)
    #
    # axs[2].plot(range(1, len(dist_history) + 1), dist_history, marker='o')
    # axs[2].set_xlabel("Iterations")
    # axs[2].set_ylabel("||y_k - x*||")
    # axs[2].set_title("Difference between Lookahead and Optimal Solution (||y_k - x*||)")
    # axs[2].grid(True)
    #
    # plt.tight_layout()
    # plt.show()

    fig, axs = plt.subplots(2, 1, figsize=(6, 6))

    # Loss plot
    axs[0].plot(
        np.arange(1, len(loss_history) + 1) * asg_config["plot"]["interval"],
        loss_history,
        marker='o', label='ASG Loss'
    )
    axs[0].axhline(y=closed_form_loss, color='red', linestyle='--', label='Closed-form Loss')

    axs[0].set_xlabel("Iteration", fontsize=13)
    axs[0].set_ylabel("Loss", fontsize=13)
    axs[0].tick_params(axis='both', labelsize=14)
    axs[0].set_title("SVRG Optimization History", fontsize=15)
    axs[0].grid(True)
    axs[0].legend(fontsize=12)

    # Distance plot
    axs[1].plot(
        np.arange(1, len(dist_history) + 1) * asg_config["plot"]["interval"],
        dist_history,
        marker='o'
    )
    axs[1].set_xlabel("Iteration", fontsize=13)
    axs[1].set_ylabel("||w - w*||", fontsize=13)
    axs[1].tick_params(axis='both', labelsize=14)
    axs[1].set_title("Distance from Closed-form Solution", fontsize=15)
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(asg_config["plot"]["save_path"], format="svg")
    plt.show()
    return sigma


def asg_ridge_regression_minibatch(X, y, lam, lr, beta, total_iterations, w_closed_form, batch_size=32):
    n, d = X.shape
    weights = np.zeros(d)
    weights_prev = np.zeros(d)
    loss_history = []
    dist_history = []
    print("alpha, beta", lr, beta)

    for k in range(total_iterations):
        # pick random choices for batch
        batch_indices = np.random.choice(n, size=batch_size, replace=False)
        X_batch = X[batch_indices]
        y_batch = y[batch_indices]

        # Lookahead point: y_k+1 = x_k + beta(x_k - x_k-1)
        lookahead = weights + beta * (weights - weights_prev)

        if k == 0:
            print("y_1", lookahead)
            y_1 = lookahead

        if k == total_iterations - 1:
            print("y_k+1", lookahead)

        # Stochastic gradient at lookahead: g_k+1 = 1/m * sum(gradient f_i(y_k+1))
        # g_k+1 = 2*X_batch^T*(X_batch*y_k - y_batch) / batch_size + 2*lambda*lookahead
        grad = 2 * X_batch.T @ (X_batch @ lookahead - y_batch) / batch_size + 2 * lam * lookahead

        # Momentum update
        new_weights = lookahead - lr * grad
        weights_prev = weights
        weights = new_weights

        # Monitoring
        if k % asg_config["plot"]["interval"] == 0:
            loss = compute_loss(X, y, weights, lam)
            loss_history.append(loss)
            dist = np.linalg.norm(weights - w_closed_form)
            dist_history.append(dist)

    print("optimized weights (minibatch)", weights)
    return weights, loss_history, dist_history, y_1


def generate_valid_pareto_front_new(
    asg_config,
    X_train,
    y_train,
    w_star,
    num_points=50
):
    d = X_train.shape[1]
    lam = asg_config["ridge_regression_minibatch"]["lambda"]
    batch_size = asg_config["ridge_regression_minibatch"]["batch_size"]
    # iterations = asg_config["ridge_regression_minibatch"]["iterations"]
    iterations = 100

    # Compute Hessian
    H = (2 / len(X_train)) * X_train.T @ X_train + 2 * lam * np.eye(d)

    alphas = np.linspace(0.001, 0.9, num_points)
    betas = np.linspace(0.001, 0.9, num_points)

    results = []  # stores (alpha, beta, final_dist, R_lambda)
    valid_params = 0

    for alpha in alphas:
        for beta in betas:
            try:
                A = construct_A(H, alpha, beta)
                rho = max(np.linalg.eigvals(A))

                C_lambda = (1 - alpha * (1 + beta) * rho) ** 2 + alpha ** 2 * rho ** 2
                del_lambda = C_lambda ** 2 - 4 * (beta ** 2) * ((1 - alpha * lam) ** 2)

                if del_lambda < 0:
                    continue
                else:
                    sqrt_del_lambda = np.sqrt(del_lambda)
                    if del_lambda >= 0 and np.isrealobj(sqrt_del_lambda) and (C_lambda + sqrt_del_lambda) >= 0:
                        R_lambda = (1 / np.sqrt(2)) * np.sqrt(C_lambda + sqrt_del_lambda)

                        if np.isrealobj(R_lambda) and not np.isnan(R_lambda) and R_lambda < 1:
                            print("R_lambda is", R_lambda, "for valid alpha-beta pair: ", alpha, "and", beta)
                            weights, loss_history, dist_history, _ = asg_ridge_regression_minibatch(
                                X_train, y_train,
                                lam=lam,
                                lr=alpha,
                                beta=beta,
                                total_iterations=iterations,
                                w_closed_form=w_star,
                                batch_size=batch_size
                            )
                            final_dist = np.linalg.norm(weights - w_star)
                            if not (np.any(np.isnan(final_dist)) or np.any(np.abs(final_dist) > 4)):
                                final_loss = compute_loss(X_train, y_train, weights, lam)
                                results.append((alpha, beta, final_dist, R_lambda))
                                valid_params += 1
                            else:
                                print("Exploding gradients occur in this case, so we continue...")
                        else:
                            print(f"Skipping alpha={alpha}, beta={beta} due to invalid R_lambda: {R_lambda}")
                            continue
                    else:
                        print(f"R_lambda is NaN or complex due to del_lambda: {del_lambda}, skipping...")
                        continue

            except Exception as e:
                print(f"Error at alpha={alpha}, beta={beta}: {e}")
                continue

    if not results:
        print("No valid (alpha, beta) combinations found.")
        return None

    results = np.array(results)

    alphas = results[:, 0]
    betas = results[:, 1]
    distances = results[:, 2]
    R_lambdas = results[:, 3]

    log_distances = np.log1p(distances)  # log(1 + distance) for better contrast

    # Plotting: log distance, quantiles, and R_lambda
    plt.figure(figsize=(8, 10))

    # Subplot 1: Pareto front with log distance coloring
    plt.subplot(2, 1, 1)
    scatter1 = plt.scatter(alphas, betas, c=log_distances, cmap='hsv_r', s=60, edgecolor='k')
    plt.colorbar(scatter1, label='log(1 + Distance to Optimum)')
    plt.xlabel("Alpha (Learning Rate)")
    plt.ylabel("Beta (Momentum)")
    plt.title("Pareto Front: log(1 + Distance) Coloring")
    plt.grid(True)

    # Subplot 2: Quantile plot of final distances
    # plt.subplot(3, 1, 2)
    # sorted_distances = np.sort(distances)
    # quantiles = np.linspace(0, 1, len(sorted_distances))
    # plt.plot(quantiles, sorted_distances, marker='o')
    # plt.xlabel("Quantile")
    # plt.ylabel("Distance to Optimum")
    # plt.title("Quantile Plot of Final Distance")
    # plt.grid(True)

    # Subplot 3: Pareto front with R_lambda coloring
    plt.subplot(2, 1, 2)
    scatter2 = plt.scatter(alphas, betas, c=R_lambdas, cmap='hsv_r', s=60, edgecolor='k')
    plt.colorbar(scatter2, label='R_lambda')
    plt.xlabel("Alpha (Learning Rate)")
    plt.ylabel("Beta (Momentum)")
    plt.title("Pareto Front: R_lambda Coloring")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(asg_config["plot"]["save_path_pareto"], format="svg")
    plt.show()
    print(f"Number of valid parameter combinations: {valid_params}")
    print(f"Lowest R_lambda: {np.min(R_lambdas)} at (alpha, beta) = {results[np.argmin(R_lambdas), 0:2]}")
    print(f"Highest R_lambda: {np.max(R_lambdas)} at (alpha, beta) = {results[np.argmax(R_lambdas), 0:2]}")

def estimate_flops_closed_form_and_asg_minibatch(n, d, T_asg, batch_size):
    """
    Estimate FLOPs for:
    - Closed-form ridge regression
    - ASG ridge regression with minibatching

    Parameters:
        n (int): Number of training samples
        d (int): Number of features
        T_asg (int): Number of ASG iterations
        batch_size (int): Mini-batch size used per iteration

    Returns:
        dict: Estimated FLOPs for closed-form and ASG minibatch
    """
    # === Closed-form ===
    # 2nd^2 + 2nd + 2d^2 + (2/3)d^3
    flops_closed_form = 2 * n * d**2 + 2 * n * d + 2 * d**2 + (2 / 3) * d**3

    # === ASG Minibatch ===
    # Each iteration:
    # - lookahead: weights + beta * (weights - weights_prev) → 2d
    # - X_batch @ lookahead → 2 * batch_size * d
    # - X_batch.T @ (X_batch @ lookahead - y_batch) → 2 * batch_size * d
    # - regularization: 2d
    # - update: weights = lookahead - lr * grad → 2d
    #
    # Total per iteration:
    # ≈ 4 * batch_size * d (matrix ops) + 6d (lookahead + reg + update)
    flops_per_iter = 4 * batch_size * d + 6 * d
    flops_asg_minibatch = T_asg * flops_per_iter

    return {
        "closed_form_flops": int(flops_closed_form),
        "asg_minibatch_flops": int(flops_asg_minibatch)
    }


print(estimate_flops_closed_form_and_asg_minibatch(n=80000, d=50, T_asg=asg_config["ridge_regression_minibatch"]["iterations"], batch_size=asg_config["ridge_regression_minibatch"]["batch_size"]))


def asg_minibatch_comparison(w_closed_form, batch_size=32):
    lam = asg_config["ridge_regression_minibatch"]["lambda"]
    alpha = asg_config["ridge_regression_minibatch"]["alpha"]
    beta = asg_config["ridge_regression_minibatch"]["beta"]
    iterations = asg_config["ridge_regression_minibatch"]["iterations"]

    w_asg_mb, loss_history, dist_history, y_1 = asg_ridge_regression_minibatch(
        X_train, y_train, lam, alpha, beta, iterations, w_closed_form, batch_size=batch_size
    )

    closed_form_loss = compute_ridge_loss(X_train, y_train, w_closed_form, lam)

    print("Final Loss:", loss_history[-1])
    print("Closed-Form Loss: ", closed_form_loss)

    get_asg_convergence_properties(lam=lam, w_0=y_1, w_k=w_asg_mb, w_opt=w_closed_form, alpha=alpha, beta=beta)
    x = np.arange(len(asg_config["data"]["features"]))
    # fig, axs = plt.subplots(3, 1, figsize=(12, 12))
    #
    # print(f"Final Loss: {loss_history[-1]:.15f}")
    #
    # axs[0].plot(x, w_closed_form, label="w_* (Closed-form)", marker='o')
    # axs[0].plot(x, w_asg_mb, label="w (ASG Minibatch)", marker='x')
    # axs[0].set_xticks(x)
    # axs[0].set_xticklabels(asg_config["data"]["features"], rotation=45)
    # axs[0].set_ylabel("Weight Value")
    # axs[0].set_title("ASG Minibatch vs Closed-form")
    # axs[0].legend()
    # axs[0].grid(True)
    #
    # axs[1].plot(range(1, len(loss_history) + 1), loss_history, marker='o')
    # axs[1].set_xlabel("Iterations")
    # axs[1].set_ylabel("Loss")
    # axs[1].set_title("ASG Minibatch Loss")
    # axs[1].grid(True)
    #
    # axs[2].plot(range(1, len(dist_history) + 1), dist_history, marker='o')
    # axs[2].set_xlabel("Iterations")
    # axs[2].set_ylabel("||y_k - x*||")
    # axs[2].set_title("Minibatch: Lookahead Distance to Optimal")
    # axs[2].grid(True)
    #
    # plt.tight_layout()
    # plt.savefig(asg_config["plot"]["save_path"], format="svg")
    # plt.show()
    fig, axs = plt.subplots(2, 1, figsize=(8, 10))

    # Loss plot
    axs[0].plot(
        np.arange(1, len(loss_history) + 1) * asg_config["plot"]["interval"],
        loss_history,
        marker='o', label='ASG Loss'
    )
    axs[0].axhline(y=closed_form_loss, color='red', linestyle='--', label='Closed-form Loss')

    axs[0].set_xlabel("Iteration", fontsize=15)
    axs[0].set_ylabel("Loss", fontsize=15)
    axs[0].tick_params(axis='both', labelsize=15)
    axs[0].set_title("ASG Optimization History", fontsize=15)
    axs[0].grid(True)
    axs[0].legend(fontsize=15)

    # Distance plot
    axs[1].plot(
        np.arange(1, len(dist_history) + 1) * asg_config["plot"]["interval"],
        dist_history,
        marker='o'
    )
    axs[1].set_xlabel("Iteration", fontsize=15)
    axs[1].set_ylabel("||w - w*||", fontsize=15)
    axs[1].tick_params(axis='both', labelsize=15)
    axs[1].set_title("Distance from Closed-form Solution", fontsize=15)
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(asg_config["plot"]["save_path"], format="svg")
    plt.show()

    generate_valid_pareto_front_new(
        asg_config,
        X_train,
        y_train,
        w_closed_form,
        num_points=100
    )


asg_minibatch_comparison(
    # w_closed_form=closed_form_computation(X_train, y_train, asg_config["ridge_regression_minibatch"]["lambda"]),
    w_closed_form=closed_form_computation(),
    batch_size=asg_config["ridge_regression_minibatch"]["batch_size"])




# --------------------------
# Spectral Radius and Run
# --------------------------
# if __name__ == "__main__":
#     eta, b, Q, hess = get_largest_and_smallest_eigenvalue(asg_config["spectral_analysis"]["lambda_eigen"])
#     A = construct_A(hess, eta, b)
#
#     lam = asg_config["spectral_analysis"]["lambda_eigen"]
#     eta = asg_config["spectral_analysis"]["eta"]
#     b = asg_config["spectral_analysis"]["beta"]
#
#     C_lambda = (1 - eta * (1 + b) * lam) ** 2 + eta ** 2 * lam ** 2
#     del_lambda = C_lambda ** 2 - 4 * b ** 2 * (b ** 2 + 1)
#     R_lambda = 1 / (2 ** 0.5) * (C_lambda + del_lambda ** 0.5) ** 0.5
#
#     print("C_lambda, del_lambda, R_lambda:", C_lambda, del_lambda, R_lambda)
#     print("Valid spectral radius!" if abs(R_lambda) < 1 else "Spectral radius too large!")
#
#     rho = max(abs(np.linalg.eigvals(A)))
#     print("\nSpectral radius ρ(A):", rho)
#     max_singular_value = max(np.linalg.svd(A, compute_uv=False))
#     print("\nLargest singular value:", max_singular_value)
#
#     sigma = asg_with_analytical_solution_comparison()
#     noise_term = (eta * ((1 + b) ** 2 + 1) ** 0.5 * sigma) / (1 - R_lambda)
#     print("\nThe noise neighborhood proportional to sigma:", noise_term)
