import pandas as pd

import numpy as np


def generate_X_with_condition(d, n, lam, Q_desired, noise_std=0.1, seed=None):
    """
    Generate synthetic dataset (X, y, w_star) for ridge regression such that
    the Hessian H = 2/n * X^T X + 2 * lam * I has a condition number ≈ Q_desired.

    Parameters:
        d (int): Number of features.
        n (int): Number of samples.
        lam (float): Ridge regularization parameter.
        Q_desired (float): Desired condition number of the Hessian.
        noise_std (float): Standard deviation of Gaussian noise added to y.
        seed (int or None): Random seed for reproducibility.

    Returns:
        X (ndarray): (n, d) design matrix.
        y (ndarray): (n,) response vector.
        w_star (ndarray): (d,) ground truth weights.
        H (ndarray): (d, d) Hessian matrix.
        eigvals_H (ndarray): Eigenvalues of H.
    """
    if seed is not None:
        np.random.seed(seed)

    # Target eigenvalues of (1/n) X^T X
    mu = 1.0
    L = Q_desired * mu
    eig_min = (mu - 2 * lam) / 2
    eig_max = (L - 2 * lam) / 2
    eigenvalues = np.geomspace(eig_max, eig_min, d)
    Sigma_sqrt = np.diag(np.sqrt(eigenvalues))

    # Orthonormal matrices U (dxd) and V (nxd)
    U, _ = np.linalg.qr(np.random.randn(d, d))
    V, _ = np.linalg.qr(np.random.randn(n, d))

    # Construct X
    X = np.sqrt(n) * V @ Sigma_sqrt @ U.T
    X_max = np.abs(X).max()
    X = 4 * X / X_max  # scales to [-4, 4]

    # Generate true weights and labels
    w_star = np.random.randn(d)
    y = X @ w_star + noise_std * np.random.randn(n)

    # Compute Hessian and its condition number
    H = 2 * (X.T @ X) / n + 2 * lam * np.eye(d)
    eigvals_H = np.linalg.eigvalsh(H)
    Q_empirical = eigvals_H[-1] / eigvals_H[0]

    print(f"Target Q: {Q_desired:.2f}, Empirical Q: {Q_empirical:.2f}, λ = {lam}")

    return X, y, w_star, H, eigvals_H


X, y, w_star, H, eigvals_H = generate_X_with_condition(
    d=10, n=100000, lam=0.01, Q_desired=10.5, noise_std=0.05, seed=42
)

np.savetxt("d10_mu1_L10_w_star", w_star)
np.savetxt("d10_mu1_L10_eigvals", eigvals_H)

df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
df["y"] = y

# Save to CSV
df.to_csv("d10_mu1_L10.csv", index=False)


