import cmath

import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# read cleaned data
df = pd.read_csv('dataset/SPX_clean.csv')
# print(df.shape)

# Ensure datetime and sort, then drop NaN values
df['Date'] = pd.to_datetime(df['Date'])
spx = df.sort_values('Date').dropna().reset_index(drop=True)

# Prepare features and target
features = ['MA_10', 'MA_20', 'STD_20', 'Bollinger_Width', 'Lagged_Return_1']
target = 'Z_Score'  # this means R^1

# set features for input and target
X = spx[features].values
y = spx[target].values

# scale values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# X_scaled = X.copy()
# print(spx.head())
# print(X_scaled[:5])
# print("X_scaled max/min", X_scaled.max(), X_scaled.min())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=1)


# print(X_train.shape)
# print("scaled\n", np.max(X_scaled, axis=0))
# print("unscaled\n", np.max(X, axis=0))


def get_largest_and_smallest_eigenvalue(lam=0.1):
    """
    taking lambda = 1, for stochastic approximation. This is done to reduce the condition number Q, to 4.5
    The paper does not have the value of Q more than 32 for their charts and comparisons
    """

    n_train = X_train.shape[0]
    hessian = 2 * (X_train.T @ X_train) / n_train + 2 * lam * np.eye(5)
    # print("hessian (5x5 matrix):\n", hessian)

    eigenvalues = np.linalg.eigvals(hessian)
    L = np.max(eigenvalues)
    print("\nlargest eigenvalue", L)
    mu = np.min(eigenvalues)
    print("\nsmallest eigenvalue", mu)

    Q = L / mu
    alpha = 1 / L
    beta = (np.sqrt(Q) - 1) / (np.sqrt(Q) + 1)
    print("\nalpha, beta", alpha, beta)
    return alpha, beta, Q, hessian


def construct_A(H, alpha, beta):
    """
    Calculate matrix A.

    Matrix A is of the form:
    | I - alpha(1 + beta) * H       beta^2 * I |
    | -alpha * H                    beta * I   |
    """
    d = H.shape[0]
    I = np.eye(d)
    top_left = I - alpha * (1 + beta) * H
    top_right = beta ** 2 * I
    bottom_left = -alpha * H
    bottom_right = beta * I

    """
    Although the matrix may appear as a 2x2 one, it is actually a 10x10 matrix
    This is because H and I are 5x5, and they are being placed to the 2x2 matrix
    """
    A = np.block([
        [top_left, top_right],
        [bottom_left, bottom_right]
    ])
    # print("\nmatrix A:\n", A)
    return A


def compute_loss(X, y, w, lambda_hyperparameter):
    """Compute ridge regression loss."""
    n = len(y)
    residuals = X @ w - y  # @ is the matrix multiplication operator
    # np.sum() is being used to compute that summation over all data points
    return (1 / n) * np.sum(residuals ** 2) + lambda_hyperparameter * np.sum(w ** 2)


def closed_form_computation(X=X_train, y=y_train, lam=0.01):
    n, d = X.shape
    I = np.eye(d)
    w_closed_form = np.linalg.solve((1 / n) * X.T @ X + lam * I, (1 / n) * X.T @ y)
    return w_closed_form


def asg_ridge_regression(X, y, lambda_hyperparameter=0.01, lr=0.01, beta=0.9, total_iterations=60000):
    """
    Accelerated Stochastic Gradient (S-NAG ASG) implementation for Ridge Regression
    """
    n, d = X.shape
    weights = np.zeros(d)
    weights_prev = weights.copy()
    loss_history = []
    dist_history = []

    w_closed_form = closed_form_computation(X=X_train, y=y_train, lam=lambda_hyperparameter)

    for t in range(total_iterations):
        # Sample one random data point
        i = np.random.randint(0, n)
        x_i = X[i].reshape(1, -1)
        y_i = y[i]

        # Compute lookahead weights
        lookahead = weights + beta * (weights - weights_prev)  # y_k = x_k + β(x_k - x_k-1)

        # Compute stochastic gradient at the lookahead point
        grad = 2 * x_i.T @ (x_i @ lookahead - y_i) + 2 * lambda_hyperparameter * lookahead

        # Store current weights for momentum
        weights_prev = weights.copy()

        # Gradient update
        weights = lookahead - lr * grad

        # w_closed_form = closed_form_computation(X=X_train, y=y_train, lam=lambda_hyperparameter)

        # Optionally track loss every 1000 iterations
        if t % 1000 == 0:
            loss = compute_loss(X, y, weights, lambda_hyperparameter)
            loss_history.append(loss)
            lookahead = weights + beta * (weights - weights_prev)
            dist = np.linalg.norm(lookahead - w_closed_form)
            dist_history.append(dist)

    return weights, loss_history, dist_history, w_closed_form


def asg_ridge_regression_sa(
        X, y,
        lambda_hyperparameter=0.001,
        lr=0.01,
        beta=0.9,
        total_iterations=40000,
        noise_std=0.05
):
    """
    Accelerated Stochastic Gradient (ASG) method for ridge regression
    in the stochastic approximation setting with additive noise
    on the full gradient (as assumed in Theorem 1 of the paper).
    """
    n, d = X.shape
    weights = np.zeros(d)
    weights_prev = np.zeros(d)
    loss_history = []

    sigma_squared = d * (noise_std ** 2)
    print(f"Using gradient noise variance σ² ≈ {sigma_squared:.6f}")

    for t in range(total_iterations):
        # Compute lookahead point: y_k = x_k + β(x_k - x_k-1)
        lookahead = weights + beta * (weights - weights_prev)

        # Compute full gradient at lookahead point
        pred = X @ lookahead
        grad_true = (2 / n) * X.T @ (pred - y) + 2 * lambda_hyperparameter * lookahead

        # Add zero-mean Gaussian noise to simulate stochastic approximation
        noise = np.random.normal(0, noise_std, size=grad_true.shape)
        grad = grad_true + noise

        # Momentum update
        weights_prev = weights.copy()
        weights = lookahead - lr * grad

        # Track loss every 1000 iterations
        if t % 1000 == 0:
            loss = compute_loss(X, y, weights, lambda_hyperparameter)
            loss_history.append(loss)
            print(f"Iter {t}: Loss = {loss:.6f}, Grad Norm = {np.linalg.norm(grad):.4f}")

    return weights, loss_history


def asg_with_analytical_solution_comparison():
    """
    Finite Sum Setting params. This is implemented for random data points at the moment, and not minibatches
    When alpha and beta are placed as normal (alpha=1/L or 2/L), the values diverge.
    So, smaller values are picked for alpha and beta.
    """
    alpha = 0.001
    beta = 0.5
    lam = 0.01
    iterations = 60000

    """
    Stochastic Approximation Setting params.
    The parameters are a bit different than for the finite sum setting, since the condition
    number Q needed to be in range for the 
    """
    # alpha = 0.09
    # beta = 0.24
    # lam = 2
    # iterations = 60000

    """
    for Closed-Form Solution
    """
    d = X_train.shape[1]
    n = X_train.shape[0]
    I = np.eye(d)

    """
        Training ASG. There are two approaches:
        1. finite sum setting
            1.1 for randomized values
            1.2 for minibatches
        2. stochastic approximation setting
        
        
    1. Finite Sum Setting
        1.1 Randomized Values.
        This section has parameter values similar to the ones for SVRG algorithm, with
        difference in the learning rate (alpha=0.001) instead of 0.01, and the absence of the outer loop. 
        The number of iterations = 60000, and lambda = 0.01 which is the same as SVRG.
        Since NAG has a momentum parameter instead (beta), the value is set at 0.9.
    """
    w_asg, loss_history, dist_history, w_closed_form = asg_ridge_regression(X_train, y_train,
                                                             lambda_hyperparameter=lam,
                                                             lr=alpha,
                                                             beta=beta,
                                                             total_iterations=iterations)

    print("ASG Weights:", w_asg)
    print("Closed-Form Weights", w_closed_form)
    print(dist_history[0])
    print(w_closed_form)

    """
        1.2 Mini Batches
    """

    """
    2. Stochastic Approximation Setting
    """
    # w_asg, loss_history = asg_ridge_regression_sa(X_train, y_train,
    #                                               lambda_hyperparameter=lam,
    #                                               lr=alpha,
    #                                               beta=beta,
    #                                               total_iterations=iterations)

    # w_closed_form = np.linalg.solve((1 / n) * X_train.T @ X_train + lam * I, (1 / n) * X_train.T @ y_train)
    # print("ASG Weights:", w_asg)
    # print("Closed-form Weights:", w_closed_form)

    # Feature comparison
    x = np.arange(len(features))

    fig, axs = plt.subplots(3, 1, figsize=(12, 12))

    # Plot 1: weight comparison
    axs[0].plot(x, w_closed_form, label="w_* (Closed-form)", marker='o')
    axs[0].plot(x, w_asg, label="w (ASG)", marker='x')
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(features, rotation=45)
    axs[0].set_ylabel("Weight Value")
    axs[0].set_title("Comparison of w (ASG) vs w_* (Closed-form)")
    axs[0].legend()
    axs[0].grid(True)

    # Plot 2: loss history
    axs[1].plot(range(1, len(loss_history) + 1), loss_history, marker='o')
    axs[1].set_xlabel("Iterations")
    axs[1].set_ylabel("Loss")
    axs[1].set_title("ASG Optimization History")
    axs[1].grid(True)

    """
    Finite sum setting: also plot ||y_k - x*||
    """
    # Plot 3: difference between lookahead and optimal
    axs[2].plot(range(1, len(dist_history) + 1), dist_history, marker='o')
    axs[1].set_xlabel("Iterations")
    axs[2].set_ylabel("||y_k - x*||")
    axs[2].set_title("Difference between Lookahead and Optimal Solution (||y_k - x*||)")
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()


# Run experiments here!

"""
1. Get L (largest eigenvalue) and mu (smallest eigenvalue) of the problem
2. Calculate the matrix A
3. Get spectral radius rho(alpha, beta)
4. Get largest singular value ||A||
"""

lam = 0.01
eta, b, Q, hess = get_largest_and_smallest_eigenvalue(lam=lam)
A = construct_A(hess, eta, b)

"""
Calculate:
C_lambda = (1 - alpha * (1 + beta) * lambda) ** 2 + alpha ** 2 * lambda ** 2
del_lambda = C_lambda ** 2 - 4 * beta ** 2 * (beta ** 2 + 1)  
R_lambda = 1/(2 ** 0.5) * (C_lambda + del_lambda ** 0.5) ** 0.5
assume eta = 0.001
"""
e = 0.001
b = 0.5
C_lambda = (1 - e * (1 + b) * lam) ** 2 + e ** 2 * lam ** 2
del_lambda = C_lambda ** 2 - 4 * b ** 2 * (b ** 2 + 1)
R_lambda = 1/(2 ** 0.5) * (C_lambda + del_lambda ** 0.5) ** 0.5

print("C_lambda, del_lambda, R_lambda", C_lambda, del_lambda, R_lambda)

if abs(R_lambda) < 1:
    print("valid value!")
else:
    print("value greater!")

# Now compute the spectral radius
eigs = np.linalg.eigvals(A)
# print("\neigenvalues of A", eigs)
rho = max(abs(eigs))
print("\nSpectral radius ρ(A):", rho)
singular_vals = np.linalg.svd(A, compute_uv=False)
max_singular_value = max(singular_vals)
print("\nlargest singular value", max_singular_value)



# stochastic approximation
# noise_term = (eta ** 2 * ((1 + b) ** 2 + 1) * 0.0125) / (1 - rho ** 2)
# C_epsilon = 1 + (1 - rho ** 2) * (max_singular_value ** 2 - rho ** 2)
# print(f"\nnoise term: {noise_term:.6f}")
# print(f"\nC_epsilon: {C_epsilon:.6f}")
# end of stochastic approximation

# get_largest_and_smallest_eigenvalue()
# sigma = estimate_gradient_noise_variance(X_train, y_train)
# print(sigma)
asg_with_analytical_solution_comparison()
