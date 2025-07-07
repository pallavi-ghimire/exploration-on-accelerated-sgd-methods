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


def sgd_ridge_regression(X, y, lambda_hyperparameter=0.01, lr=0.005, iterations=20000):
    """
    - lambda_hyperparameter: regularization strength
    - lr: learning rate
    - iterations: total number of SGD updates (i.e., steps)
    """
    n, d = X.shape
    w = np.zeros(d)  # initialize weights
    history = []     # store loss values
    dist_history = []

    w_closed_form = closed_form_computation(X=X_train, y=y_train, lam=lambda_hyperparameter)

    for k in range(iterations):
        i = np.random.randint(0, n)
        x_i = X[i].reshape(1, -1)
        y_i = y[i]

        grad_i = 2 * x_i.T @ (x_i @ w - y_i) + 2 * lambda_hyperparameter * w
        w -= lr * grad_i.flatten()

        # Optionally store history every 1000 iterations
        if (k + 1) % 1000 == 0:
            loss = compute_loss(X, y, w, lambda_hyperparameter)
            dist = np.linalg.norm(w - w_closed_form)
            history.append(loss)
            dist_history.append(dist)

    return w, history, dist_history




def sgd_with_analytical_solution_comparison():
    eta = 0.001
    lam = 0.01

    w_sgd, loss_history, dist_history = sgd_ridge_regression(X_train, y_train,
                                                              lambda_hyperparameter=lam,
                                                              lr=eta, iterations=80000)

    minimized_value = sum(loss_history) / len(loss_history)
    print('The minimized value for the loss function is', minimized_value)
    print('The value for w (SGD) is', w_sgd)

    # Closed-form solution
    w_optimal = closed_form_computation(X_train, y_train, lam)

    print("Closed-form solution w_*:", w_optimal)

    feature_names = ['MA_10', 'MA_20', 'STD_20', 'Bollinger_Width', 'Lagged_Return_1']
    x = np.arange(len(feature_names))

    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(12, 12))

    # Plot 1: Weight comparison
    axs[0].plot(x, w_optimal, label="w_* (Closed-form)", marker='o')
    axs[0].plot(x, w_sgd, label="w (SGD)", marker='x')
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(feature_names, rotation=45)
    axs[0].set_ylabel("Weight Value")
    axs[0].set_title("Comparison of w (SGD) vs w_* (Closed-form)")
    axs[0].legend()
    axs[0].grid(True)

    # Plot 2: Loss over epochs
    axs[1].plot(range(1, len(loss_history) + 1), loss_history, marker='o')
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].set_title("SGD Optimization History")
    axs[1].grid(True)

    # Plot 3: Distance to closed-form weights
    axs[2].plot(range(1, len(dist_history) + 1), dist_history, marker='o')
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("||w - w*||")
    axs[2].set_title("Distance from Optimal Solution (||w - w*||)")
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()


""" run the functions here! """
# find_lambda_then_run_svrg()
# get_largest_eigenvalue()
sgd_with_analytical_solution_comparison()




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

