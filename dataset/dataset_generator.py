import numpy as np
import pandas as pd


def generate_synthetic_data_with_known_optimum(n_samples=100000, n_features=5, Q=32, noise_std=0.1, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # 1. Random optimal weights
    w_star = np.random.randn(n_features)
    print('w_star is:', w_star)
    np.savetxt("n100000_d20_mu1_L3198_w_star", w_star)


    # 2. Generate eigenvalues with given condition number
    # generate evenly spaced numbers in a given interval
    eigvals = np.linspace(0.98, Q, n_features)
    Sigma = np.diag(eigvals)
    np.savetxt("n100000_d20_mu1_L3198_eigvals", Sigma)

    # 3. Random orthogonal matrix for transformation. UTU is I
    U, _ = np.linalg.qr(np.random.randn(n_features, n_features))
    A = U @ np.sqrt(Sigma)

    # 4. Generate X with desired covariance
    X = np.random.randn(n_samples, n_features) @ A.T

    # 5. Generate noisy targets
    y = X @ w_star + noise_std * np.random.randn(n_samples)

    return X, y, w_star, eigvals


X, y, w_star, eigvals = generate_synthetic_data_with_known_optimum(
    n_samples=100000,
    n_features=10,
    Q=49.98,
    noise_std=2
)

print(w_star, eigvals)

# Combine X and y into a DataFrame
df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
df["y"] = y

# Save to CSV
df.to_csv("n100000_d20_mu1_L3198.csv", index=False)
