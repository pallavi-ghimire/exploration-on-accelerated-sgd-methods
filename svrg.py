import math
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
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

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=1)
# print(X_train.shape)
# print("scaled\n", np.max(X_scaled, axis=0))
# print("unscaled\n", np.max(X, axis=0))


def get_largest_eigenvalue():
    """
    we get value for L = 7.005059610760887
    Paper recommends going with eta value as eta = 0.1/L
    Roughly, eta needs to be < 0.014, so selecting eta as 0.01 satisfies the given inequality
    """
    n_train = X_train.shape[0]
    A = 2 * (X_train.T @ X_train) / n_train + 2 * 0.01 * np.eye(5)
    print(A)
    eigenvalues = np.linalg.eigvals(A)
    print("\nlargest eigenvalue")
    print(np.max(eigenvalues))

# print(X_train.shape, X_test.shape)
# get_largest_eigenvalue()


def compute_loss(X, y, w, lambda_hyperparameter):
    """Compute ridge regression loss."""
    n = len(y)
    residuals = X @ w - y  # @ is the matrix multiplication operator
    # np.sum() is being used to compute that summation over all data points
    # print()
    return (1 / n) * np.sum(residuals ** 2) + lambda_hyperparameter * np.sum(w ** 2)


def closed_form_computation(X=X_train, y=y_train, lam=0.01):
    n, d = X.shape
    I = np.eye(d)
    w_closed_form = np.linalg.solve((1 / n) * X.T @ X + lam * I, (1 / n) * X.T @ y)
    return w_closed_form


def svrg_ridge_regression(X, y, lambda_hyperparameter=0.001, lr=0.01, epochs=10, m=40000):
    """
    SVRG implementation for Ridge Regression.
    lambda_hyperparameter: regularization strength
    lr: learning rate
    epochs: number of outer loops
    m: number of inner iterations
    """
    n, d = X.shape
    w_tilde = np.zeros(d)  # initialize w~_0 as a 0's vector with dimension 1x5
    history = []  # set an array to record all the losses
    dist_history = []

    w_closed_form = closed_form_computation(X=X_train, y=y_train, lam=lambda_hyperparameter)

    for epoch in range(epochs):  # s loop iteration
        # numpy gives transpose of an array with .T
        full_grad = (2 / n) * X.T @ (X @ w_tilde - y) + 2 * lambda_hyperparameter * w_tilde  # mu: full gradient
        w = w_tilde.copy()  # w_0 initialization as 0's in the form R^(5x1)

        # To store all inner iterates value for w
        inner_iterates_w = []

        for t in range(m):
            # taking one random index i
            i = np.random.randint(0, n)

            # taking inputs of ith index, then reshaping it to be a row vector, of shape 1xd, d=5
            x_i = X[i].reshape(1, -1)

            # taking y of ith index
            y_i = y[i]

            # x^T.x.w - x^T.y + lambda.w at ith index
            grad_i = 2 * x_i.T @ (x_i @ w - y_i) + 2 * lambda_hyperparameter * w

            # x^T.x.w~ - x^T.y + lambda.w~ at ith index
            grad_i_tilde = 2 * x_i.T @ (x_i @ w_tilde - y_i) + 2 * lambda_hyperparameter * w_tilde

            # update weights, which is a 1d array with 5 values
            w -= lr * (grad_i - grad_i_tilde + full_grad)

            # maintaining inner_iterates array to randomly assign w value to w~ later
            inner_iterates_w.append(w.copy())

        """ Option
        II — choose
        random
        inner
        iterate as next
        snapshot """
        # assign random w_t to w~
        # rand_index = np.random.randint(0, m)
        # w_tilde = inner_iterates_w[rand_index]

        """option I — calculate average of it"""
        w_tilde = sum(inner_iterates_w) / len(inner_iterates_w)

        # compute loss
        loss = compute_loss(X, y, w, lambda_hyperparameter)
        dist = np.linalg.norm(w - w_closed_form)
        dist_history.append(dist)
        history.append(loss)

    return w, history, dist_history


def find_lambda_then_run_svrg():
    """
    Finding the best lambda value for a given learning rate of 0.01. These values are interdependent on each other,
    since the value for learning rate depends on the highest eigenvalue of the Hessian, which can be calculated as a
    Hessian = sum of 1/n(X^TX) + lambda * I
    """
    lambda_values = [0.0001, 0.001, 0.01, 0.1, 1, 10]
    eta = 0.01
    results = []

    for l in lambda_values:
        w, loss, _ = svrg_ridge_regression(X_train, y_train, lambda_hyperparameter=l, lr=eta, epochs=10, m=60000)
        y_pred = X_test @ w
        mse = mean_squared_error(y_test, y_pred)
        r_mse = math.sqrt(mse)
        results.append((l, r_mse))

    best_lambda, best_r_mse = min(results, key=lambda x: x[1])
    print(f"\nBest lambda: {best_lambda}, with R_MSE: {best_r_mse:.5f}")
    # w_svrg, loss_history = svrg_ridge_regression(X_train, y_train, lambda_hyperparameter=best_lambda, lr=eta,
    #                                              epochs=10,
    #                                              m=60000)
    #
    # minimized_value = sum(loss_history) / len(loss_history)
    # print('The minimized value for the loss function is', minimized_value)
    # print('The value for w is', w_svrg)

    # plot the chart
    # plt.figure(figsize=(8, 5))
    # plt.plot(loss_history, marker='o')
    # plt.title("SVRG Loss History Over Epochs")
    # plt.xlabel("Epoch")
    # plt.ylabel("Ridge Regression Loss")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
# find_lambda_then_run_svrg()

def svrg_with_analytical_solution_comparison():
    eta = 0.01
    lam = 0.01
    # lam = 0

    # Train SVRG with best lambda

    w_svrg, loss_history, dist_history = svrg_ridge_regression(X_train, y_train, lambda_hyperparameter=lam, lr=eta,
                                                 epochs=20, m=5000)

    minimized_value = sum(loss_history) / len(loss_history)

    print('The minimized value for the loss function is', minimized_value)
    print('The value for w (SVRG) is', w_svrg)

    # Closed-form solution w*
    d = X_train.shape[1]
    n = X_train.shape[0]
    I = np.eye(d)
    # the closed form solution is of the form (X^T . X + lambda * I)^-1 . X^T . y
    # this is placed as a linear algebra equation as ((1/n) * X^T . X + lambda * I) * w_star = (1/n) X^T . y
    w_optimal = np.linalg.solve((1/n) * X_train.T @ X_train + lam * I, (1/n) * X_train.T @ y_train)

    print("Closed-form solution w_*:", w_optimal)

    # lot comparison of w vs w*
    feature_names = ['MA_10', 'MA_20', 'STD_20', 'Bollinger_Width', 'Lagged_Return_1']
    x = np.arange(len(feature_names))

    # Plot both comparison and optimization history
    fig, axs = plt.subplots(3, 1, figsize=(12, 12))  # 3 rows, 1 column

    # Plot 1: Weight comparison
    axs[0].plot(x, w_optimal, label="w_* (Closed-form)", marker='o')
    axs[0].plot(x, w_svrg, label="w (SVRG)", marker='x')
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(feature_names, rotation=45)
    axs[0].set_ylabel("Weight Value")
    axs[0].set_title("Comparison of w (SVRG) vs w_* (Closed-form)")
    axs[0].legend()
    axs[0].grid(True)

    # Plot 2: Optimization history
    axs[1].plot(range(1, len(loss_history) + 1), loss_history, marker='o')
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].set_title("SVRG Optimization History")
    axs[1].grid(True)

    """
    Also plot ||w_t - w*||
    """
    # Plot 3: difference between lookahead and optimal
    axs[2].plot(range(1, len(dist_history) + 1), dist_history, marker='o')
    axs[1].set_xlabel("Iterations")
    axs[2].set_ylabel("||w_s - w*||")
    axs[2].set_title("Difference between weight at epoch s and Optimal Solution (||w_s - x*||)")
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()


""" run the functions here! """
# find_lambda_then_run_svrg()
# get_largest_eigenvalue()
svrg_with_analytical_solution_comparison()

# checking data, and the value ranges (max and min), to determine whether scaling needs to be performed
# print(spx.head())
# print(spx.shape)
# print(np.max(X_scaled[:, 1]), np.min(X_scaled[:, 1]))
# print(spx['Z_Score'].max(), spx['Z_Score'].min())
# print(spx['MA_10'].max(), spx['MA_10'].min())
# print(spx['MA_20'].max(), spx['MA_20'].min())
# print(spx['STD_20'].max(), spx['STD_20'].min())
# print(spx['Bollinger_Width'].max(), spx['Bollinger_Width'].min())
# print(spx['Lagged_Return_1'].max(), spx['Lagged_Return_1'].min())

