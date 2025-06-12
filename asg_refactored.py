# Refactored ASG Ridge Regression code with centralized configuration

import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

import matplotlib.ticker as ticker

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
        "beta": 0.45,
        "iterations": 9000,
        "noise_std": 0.05,
    },
    "ridge_regression_minibatch": {
        "lambda": 0.01,
        "alpha": 0.01,
        "beta": 0.48,
        "iterations": 100,
        "noise_std": 0.05,
        "batch_size": 150
    },
    "spectral_analysis": {
        "lambda_eigen": 0.01,
        "eta": 0.01,
        "beta": 0.48
    },
    "plot": {
        "interval": 50
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


def get_largest_and_smallest_eigenvalue(lam):
    n_train = X_train.shape[0]
    hessian = 2 * (X_train.T @ X_train) / n_train + 2 * lam * np.eye(X_train.shape[1])
    eigenvalues = np.linalg.eigvals(hessian)
    L = np.max(eigenvalues)
    print("L is", L)
    mu = np.min(eigenvalues)
    Q = L / mu
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
    n, d = X.shape
    I = np.eye(d)
    closed_form_solution = np.linalg.solve((1 / n) * X.T @ X + lam * I, (1 / n) * X.T @ y)
    print("closed form solution", closed_form_solution)
    return closed_form_solution


def asg_ridge_regression(X, y, lam, lr, beta, total_iterations, w_closed_form):
    n, d = X.shape
    weights = np.zeros(d)
    weights_prev = weights.copy()
    loss_history = []
    dist_history = []
    # w_closed_form = closed_form_computation(X, y, lam)

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
    print("optimized weights", weights)
    return weights, loss_history, dist_history


def compute_sigma(X, y, w_star, ridge_strength, beta):
    n = X.shape[0]
    sigma = np.mean([np.linalg.norm(2 * (X[i] @ w_star - y[i]) * X[i] + 2 * ridge_strength * w_star) for i in range(n)])
    compute_noise_term(sigma, eta, beta, R_lambda)
    return sigma


def compute_noise_term(sigma, eta, beta, R_lambda):
    noise_term = (eta * ((1 + beta) ** 2 + 1) ** 0.5 * sigma) / (1 - R_lambda)
    print("\nthe neighborhood proportional to sigma", noise_term)
    return noise_term


def asg_with_analytical_solution_comparison(w_closed_form):
    lam = asg_config["ridge_regression"]["lambda"]
    alpha = asg_config["ridge_regression"]["alpha"]
    beta = asg_config["ridge_regression"]["beta"]
    iterations = asg_config["ridge_regression"]["iterations"]

    w_asg, loss_history, dist_history = asg_ridge_regression(
        X_train, y_train, lam, alpha, beta, iterations, w_closed_form
    )

    x = np.arange(len(asg_config["features"]))
    fig, axs = plt.subplots(3, 1, figsize=(12, 12))

    axs[0].plot(x, w_closed_form, label="w_* (Closed-form)", marker='o')
    axs[0].plot(x, w_asg, label="w (ASG)", marker='x')
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(asg_config["features"], rotation=45)
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


# def asg_ridge_regression_minibatch(X, y, lam, lr, beta, total_iterations, w_closed_form, batch_size=32):
#     n, d = X.shape
#     weights = np.zeros(d)
#     weights_prev = weights.copy()
#     loss_history = []
#     dist_history = []
#
#     for t in range(100):
#         batch_indices = np.random.choice(n, size=batch_size, replace=False)
#         X_batch = X[batch_indices]
#         y_batch = y[batch_indices]
#
#         lookahead = weights + beta * (weights - weights_prev)
#
#         grad = 2 * X_batch.T @ (X_batch @ lookahead - y_batch) / batch_size + 2 * lam * lookahead
#
#         weights_prev = weights.copy()
#         weights = lookahead - lr * grad
#
#         if t % asg_config["plot"]["interval"] == 0:
#             loss = compute_loss(X, y, weights, lam)
#             loss_history.append(loss)
#             # Use the same lookahead logic as above for correct distance calculation
#             lookahead_now = weights + beta * (weights - weights_prev)
#             dist = np.linalg.norm(lookahead_now - w_closed_form)
#             dist_history.append(dist)
#
#     print("optimized weights (minibatch)", weights)
#     return weights, loss_history, dist_history

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
    return weights, loss_history, dist_history


def asg_minibatch_comparison(w_closed_form, batch_size=32):
    lam = asg_config["ridge_regression_minibatch"]["lambda"]
    alpha = asg_config["ridge_regression_minibatch"]["alpha"]
    beta = asg_config["ridge_regression_minibatch"]["beta"]
    iterations = asg_config["ridge_regression_minibatch"]["iterations"]

    w_asg_mb, loss_history, dist_history = asg_ridge_regression_minibatch(
        X_train, y_train, lam, alpha, beta, iterations, w_closed_form, batch_size=batch_size
    )

    x = np.arange(len(asg_config["features"]))
    fig, axs = plt.subplots(3, 1, figsize=(12, 12))

    axs[0].plot(x, w_closed_form, label="w_* (Closed-form)", marker='o')
    axs[0].plot(x, w_asg_mb, label="w (ASG Minibatch)", marker='x')
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(asg_config["features"], rotation=45)
    axs[0].set_ylabel("Weight Value")
    axs[0].set_title("ASG Minibatch vs Closed-form")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(range(1, len(loss_history) + 1), loss_history, marker='o')
    axs[1].set_xlabel("Iterations")
    axs[1].set_ylabel("Loss")
    axs[1].set_title("ASG Minibatch Loss")
    axs[1].grid(True)

    axs[2].plot(range(1, len(dist_history) + 1), dist_history, marker='o')
    axs[2].set_xlabel("Iterations")
    axs[2].set_ylabel("||y_k - x*||")
    axs[2].set_title("Minibatch: Lookahead Distance to Optimal")
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()


# asg_minibatch_comparison(
#     w_closed_form=closed_form_computation(X_train, y_train, asg_config["ridge_regression"]["lambda"]),
#     batch_size=asg_config["ridge_regression_minibatch"]["batch_size"])

"""run experiments here!"""
# eta, b, Q, hess = get_largest_and_smallest_eigenvalue(asg_config["spectral_analysis"]["lambda_eigen"])
# A = construct_A(hess, eta, b)
#
# lam = asg_config["spectral_analysis"]["lambda_eigen"]
# eta = asg_config["spectral_analysis"]["eta"]
# b = asg_config["spectral_analysis"]["beta"]
#
# C_lambda = (1 - eta * (1 + b) * lam) ** 2 + eta ** 2 * lam ** 2
# del_lambda = C_lambda ** 2 - 4 * b ** 2 * (b ** 2 + 1)
# R_lambda = 1 / (2 ** 0.5) * (C_lambda + del_lambda ** 0.5) ** 0.5
#
# print("C_lambda, del_lambda, R_lambda", C_lambda, del_lambda, R_lambda)
# print("valid value!" if abs(R_lambda) < 1 else "value greater!")
#
# rho = max(abs(np.linalg.eigvals(A)))
# print("\nSpectral radius ρ(A):", rho)
# max_singular_value = max(np.linalg.svd(A, compute_uv=False))
# print("\nlargest singular value", max_singular_value)
#
# closed_form_value = closed_form_computation(X_train, y_train, asg_config["ridge_regression"]["lambda"])
#
# compute_sigma(X_train, y_train, closed_form_value, asg_config["ridge_regression"]["lambda"], b)
#
# asg_with_analytical_solution_comparison(closed_form_value)

"""
the lambda is obtained as the maximum eigenvalue of the Hessian
the eta and beta are obtained at random, such that the following condition is satisfied:
R_lambda < 1
where,
R_lambda = (1/2^n) * sqrt((C_lambda + sqrt(del_lambda))),
C_lambda = 1 - eta * (1+b) * lambda
del_lambda = C_lambda^2 - 4*(b^2) + (b^2 + 1)
"""

# minibatch
eta = asg_config["spectral_analysis"]["eta"]
b = asg_config["spectral_analysis"]["beta"]
_, _, Q, hess = get_largest_and_smallest_eigenvalue(asg_config["spectral_analysis"]["lambda_eigen"])
A = construct_A(hess, eta, b)
lam = max(np.linalg.eigvals(A))



C_lambda = (1 - eta * (1 + b) * lam) ** 2 + eta ** 2 * lam ** 2
del_lambda = C_lambda ** 2 - 4 * (b ** 2) * ((1 - eta * lam) ** 2)
R_lambda = 1 / (2 ** 0.5) * (C_lambda + del_lambda ** 0.5) ** 0.5

print("C_lambda, del_lambda, R_lambda", C_lambda, del_lambda, R_lambda)
if del_lambda >= 0 and abs(R_lambda) < 1:
    print("valid value!")
else:
    print("value greater or invalid!")

rho = max(abs(np.linalg.eigvals(A)))
print("\nSpectral radius ρ(A):", rho)
max_singular_value = max(np.linalg.svd(A, compute_uv=False))
print("\nlargest singular value", max_singular_value)

closed_form_value = closed_form_computation(X_train, y_train, asg_config["ridge_regression_minibatch"]["lambda"])
sigma = compute_sigma(X_train, y_train, closed_form_value, asg_config["ridge_regression_minibatch"]["lambda"], b)

print("sigma =", sigma)

asg_minibatch_comparison(
    w_closed_form=closed_form_computation(X_train, y_train, asg_config["ridge_regression_minibatch"]["lambda"]),
    batch_size=asg_config["ridge_regression_minibatch"]["batch_size"])


# def generate_valid_pareto_front(
#         asg_config,
#         X_train,
#         y_train,
#         w_star,
#         num_points=100
# ):
#     d = X_train.shape[1]
#     lam = asg_config["ridge_regression_minibatch"]["lambda"]
#     batch_size = asg_config["ridge_regression_minibatch"]["batch_size"]
#     iterations = asg_config["ridge_regression_minibatch"]["iterations"]
#
#     # Compute Hessian
#     H = (2 / len(X_train)) * X_train.T @ X_train + 2 * lam * np.eye(d)
#
#     alphas = np.linspace(0.001, 0.9, num_points)
#     betas = np.linspace(0.001, 0.9, num_points)
#
#     # print(alphas, betas)
#
#     results = []
#     valid_params = 0
#
#     """the actual code is commented here"""
#     for alpha in alphas:
#         for beta in betas:
#             print("trying for alpha and beta as: ", alpha, beta)
#             try:
#                 # Construct matrix A and compute spectral radius
#                 A = construct_A(H, alpha, beta)
#                 rho = max(np.linalg.eigvals(A))
#                 # who gives L: the worst-case eigenvalue, which is seleched whenever the total n is used
#                 # print("rho", rho)
#
#                 C_lambda = (1 - alpha * (1 + beta) * rho) ** 2 + alpha ** 2 * rho ** 2
#                 del_lambda = C_lambda ** 2 - 4 * (b ** 2) * ((1 - eta * lam) ** 2)
#                 # R_lambda = 1 / (2 ** 0.5) * (C_lambda + del_lambda ** 0.5) ** 0.5
#                 if del_lambda < 0:
#                     R_lambda = np.nan
#                 else:
#                     R_lambda = (1 / np.sqrt(2)) * np.sqrt(C_lambda + np.sqrt(del_lambda))
#
#                 if not np.isnan(R_lambda) and np.isreal(R_lambda) and R_lambda < 1 and del_lambda >= 0:
#                     print("R_lambda is valid, with R_lambda:", R_lambda, ", rho is", rho, ", alpha is", alpha, ", beta is", beta)
#                     # print("R_lambda =", R_lambda)
#                 # Check the convergence condition
#                 # if R_lambda < 1:
#                     # Run ASG minibatch
#                     weights, loss_history, dist_history = asg_ridge_regression_minibatch(
#                         X_train, y_train,
#                         lam=lam,
#                         lr=alpha,
#                         beta=beta,
#                         total_iterations=iterations,
#                         w_closed_form=w_star,
#                         batch_size=batch_size
#                     )
#                     final_loss = compute_loss(X_train, y_train, weights, lam)
#                     final_dist = np.linalg.norm(weights - w_star)
#
#                     results.append((final_loss, final_dist, alpha, beta))
#                     valid_params += 1
#
#             except Exception as e:
#                 print(f"Skipping alpha={alpha:.3f}, beta={beta:.3f} due to error: {e}")
#
#     if not results:
#         print("No valid (alpha, beta) combinations found with spectral radius < 1.")
#         return None
#
#     results = np.array(results)
#     print(len(results))
#
#     # Plotting Pareto front
#     plt.figure(figsize=(10, 6))
#     scatter = plt.scatter(results[:, 0], results[:, 1], c=results[:, 2], cmap='viridis', s=60, edgecolor='k')
#     plt.colorbar(scatter, label='Alpha values')
#     plt.xlabel("Final Loss")
#     plt.ylabel("Distance to Optimal Solution ||w - w*||")
#     plt.title(f"Pareto Front for Minibatch ASG (Valid alpha-beta pairs: {valid_params})")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
#
#     return results

def generate_valid_pareto_front(
    asg_config,
    X_train,
    y_train,
    w_star,
    num_points=50
):
    d = X_train.shape[1]
    lam = asg_config["ridge_regression_minibatch"]["lambda"]
    batch_size = asg_config["ridge_regression_minibatch"]["batch_size"]
    iterations = asg_config["ridge_regression_minibatch"]["iterations"]

    # Compute Hessian
    H = (2 / len(X_train)) * X_train.T @ X_train + 2 * lam * np.eye(d)

    alphas = np.linspace(0.001, 0.9, num_points)
    betas = np.linspace(0.001, 0.9, num_points)

    results = []
    valid_params = 0

    for alpha in alphas:
        for beta in betas:
            try:
                A = construct_A(H, alpha, beta)
                rho = max(np.linalg.eigvals(A))

                C_lambda = (1 - alpha * (1 + beta) * rho) ** 2 + alpha ** 2 * rho ** 2
                del_lambda = C_lambda ** 2 - 4 * (beta ** 2) * ((1 - alpha * lam) ** 2)

                if(del_lambda < 0):
                    continue
                else:
                    sqrt_del_lambda = np.sqrt(del_lambda)
                    if del_lambda >= 0 and np.isrealobj(sqrt_del_lambda) and (C_lambda + sqrt_del_lambda) >= 0:
                        R_lambda = (1 / np.sqrt(2)) * np.sqrt(C_lambda + sqrt_del_lambda)

                        if np.isrealobj(R_lambda) and not np.isnan(R_lambda) and R_lambda < 1:
                            print("R_lambda is", R_lambda, "for valid alpha-beta pair: ", alpha, "and", beta)
                            weights, loss_history, dist_history = asg_ridge_regression_minibatch(
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
                                results.append((alpha, beta, final_dist))
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
    print(f"Number of valid parameter combinations: {valid_params}")

    # Plotting: alpha vs beta with color as distance to optimum
    # plt.figure(figsize=(10, 6))
    # scatter = plt.scatter(results[:, 0], results[:, 1], c=results[:, 2], cmap='viridis', s=60, edgecolor='k')
    # plt.colorbar(scatter, label='Distance to Optimal Solution ||w - w*||')
    # plt.xlabel("Alpha (Learning Rate)")
    # plt.ylabel("Beta (Momentum)")
    # plt.title("Pareto Front: Alpha vs Beta colored by Distance to Optimal Solution")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    alphas = results[:, 0]
    betas = results[:, 1]
    distances = results[:, 2]

    # Scatter plot
    plt.figure(figsize=(12, 6))

    # Subplot 1: Scatter plot (Pareto front)
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(alphas, betas, c=distances, cmap='viridis', s=60, edgecolor='k')
    plt.colorbar(scatter, label='Distance to Optimal ||w - w*||')
    plt.xlabel("Alpha (Learning Rate)")
    plt.ylabel("Beta (Momentum)")
    plt.title("Pareto Front (Distance Coloring)")
    plt.grid(True)

    # Subplot 2: Quantile plot
    plt.subplot(1, 2, 2)
    sorted_distances = np.sort(distances)
    quantiles = np.linspace(0, 1, len(sorted_distances))
    plt.plot(quantiles, sorted_distances, marker='o')
    plt.xlabel("Quantile")
    plt.ylabel("Distance to Optimum")
    plt.title("Quantile Plot of Final Distance ||w - w*||")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return results


closed_form_value = closed_form_computation(X_train, y_train, asg_config["ridge_regression_minibatch"]["lambda"])
# pareto_results = generate_valid_pareto_front(
#     asg_config,
#     X_train,
#     y_train,
#     closed_form_value,
#     num_points=100
# )
