import numpy as np
import pandas as pd


def generate_synthetic_data_with_known_optimum(n_samples=100000, n_features=5, Q=32, noise_std=0.1, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # 1. Random optimal weights
    w_star = np.random.randn(n_features)
    print('w_star is:', w_star)
    np.savetxt("d50_mu1_L50_w_star1", w_star)


    # 2. Generate eigenvalues with given condition number
    # generate evenly spaced numbers in a given interval
    eigvals = np.linspace(1, Q, n_features)
    Sigma = np.diag(eigvals)
    np.savetxt("d50_mu1_L50_eigvals1", Sigma)

    # 3. Random orthogonal matrix for transformation. UTU is I
    U, _ = np.linalg.qr(np.random.randn(n_features, n_features))
    A = U @ np.sqrt(Sigma)

    # 4. Generate X with desired covariance
    X = np.random.randn(n_samples, n_features) @ A.T

    # 5. Generate noisy targets
    y = X @ w_star + noise_std * np.random.randn(n_samples)

    return X, y, w_star, eigvals


# X, y, w_star, eigvals = generate_synthetic_data_with_known_optimum(
#     n_samples=100000,
#     n_features=50,
#     Q=50,
#     noise_std=0
# )

def generate_synthetic_data_with_target_mu_L(n_samples=100000, n_features=5,
                                             mu_target=1, L_target=50,
                                             lam=0.5, noise_std=0.1, seed=None):
    """
    Generate synthetic dataset for ridge regression such that
    the Hessian has desired mu and L, for a given lambda (regularization strength).
    """
    import numpy as np

    if seed is not None:
        np.random.seed(seed)

    # Solve for desired min and max eigenvalues of X^T X / n
    sigma_min = mu_target / 2 - lam
    sigma_max = L_target / 2 - lam

    if sigma_min <= 0:
        raise ValueError(f"Chosen lambda={lam} is too large for mu_target={mu_target}. "
                         f"Need sigma_min > 0 ⇒ lambda < {mu_target/2:.4f}")

    eigvals = np.linspace(sigma_min, sigma_max, n_features)
    Sigma = np.diag(eigvals)

    # Create random orthogonal matrix
    U, _ = np.linalg.qr(np.random.randn(n_features, n_features))
    A = U @ np.sqrt(Sigma)

    # Generate synthetic dataset
    X = np.random.randn(n_samples, n_features) @ A.T
    w_star = np.random.randn(n_features)
    y = X @ w_star + noise_std * np.random.randn(n_samples)

    # Save for reproducibility (optional)
    try:
        np.savetxt("sgd/d50_mu1_L50_w_star", w_star)
        np.savetxt("sgd/d50_mu1_L50_eigvals", eigvals)
    except Exception as e:
        print("Error saving file:", e)

    print(f"Generated data with λ = {lam}, mu ≈ {mu_target}, L ≈ {L_target}, Q ≈ {L_target / mu_target}")
    return X, y, w_star, eigvals

# X, y, w_star, eigvals = generate_synthetic_data_with_target_mu_L(
#     n_samples=100000,
#     n_features=50,
#     mu_target=1,
#     L_target=50,
#     noise_std=0,
#     lam=0.1,            # one of your allowed lambda values
#     seed=42
# )


import numpy as np

# def generate_dataset_for_condition_number(n=100000, d=20,
#                                           target_condition=50,
#                                           lam=0.01,
#                                           noise_std=0.1,
#                                           seed=5):
#     """
#     Generate synthetic dataset for ridge regression such that
#     the Hessian has condition number ≈ target_condition for given lambda.
#     """
#     if seed is not None:
#         np.random.seed(seed)
#
#     # Step 1: Pick sigma_min
#     sigma_min = 1.0  # Can be tuned, must be > 0
#     sigma_max = target_condition * (sigma_min + lam) - lam
#
#     # Step 2: Construct eigvals of (1/n) X^T X
#     eigvals = np.linspace(sigma_min, sigma_max, d)
#     Sigma = np.diag(eigvals)
#
#     # Step 3: Random orthogonal matrix
#     U, _ = np.linalg.qr(np.random.randn(d, d))
#     A = U @ np.sqrt(Sigma)
#
#     # Step 4: Generate X
#     X = np.random.randn(n, d) @ A.T
#     w_star = np.random.randn(d)
#     y = X @ w_star + noise_std * np.random.randn(n)
#
#     # Step 5: Compute condition number of Hessian
#     H = (2 / n) * X.T @ X + 2 * lam * np.eye(d)
#     h_eigs = np.linalg.eigvalsh(H)
#     Q_empirical = h_eigs[-1] / h_eigs[0]
#
#     print(f"Target Q: {target_condition:.2f}, Empirical Q: {Q_empirical:.2f}, λ = {lam}")
#     return X, y, w_star, eigvals, H

import numpy as np

def generate_dataset_for_condition_number(n=100000, d=20,
                                          target_condition=50,
                                          lam=0.01,
                                          noise_std=0.1,
                                          seed=5):
    """
    Generate synthetic dataset for ridge regression such that
    the Hessian has condition number ≈ target_condition for given lambda.
    """
    if seed is not None:
        np.random.seed(seed)

    # Step 1: Pick sigma_min and compute sigma_max
    sigma_min = 1.0  # Can be adjusted
    sigma_max = target_condition * (sigma_min + lam) - lam
    eigvals = np.linspace(sigma_max, sigma_min, d)  # decreasing order

    # Step 2: Construct diagonal matrix and orthogonal matrices
    Sigma = np.diag(eigvals)
    U, _ = np.linalg.qr(np.random.randn(d, d))     # d x d
    V, _ = np.linalg.qr(np.random.randn(n, d))     # n x d

    # Step 3: Construct X with exact spectrum
    X = np.sqrt(n) * V @ np.sqrt(Sigma) @ U.T

    # Step 4: Generate target vector y
    w_star = np.random.randn(d)
    y = X @ w_star + noise_std * np.random.randn(n)

    # Step 5: Compute empirical Hessian condition number
    H = 2 * (X.T @ X) / n + 2 * lam * np.eye(d)
    h_eigs = np.linalg.eigvalsh(H)
    Q_empirical = h_eigs[-1] / h_eigs[0]

    print(f"Target Q: {target_condition:.2f}, Empirical Q: {Q_empirical:.2f}, λ = {lam}")
    return X, y, w_star, eigvals, H


X, y, w_star, eigvals, H = generate_dataset_for_condition_number(
    n=100000, d=50, target_condition=50, lam=0.01, noise_std=0.1, seed=42)# X2, y2, w2, _, _ = generate_dataset_for_condition_number(target_condition=25, lam=0.1)
# X3, y3, w3, _, _ = generate_dataset_for_condition_number(target_condition=10, lam=1.0)

np.savetxt("sgd/d50_mu1_L50_w_star", w_star)
np.savetxt("sgd/d50_mu1_L50_eigvals", eigvals)

print(w_star, eigvals)

# Combine X and y into a DataFrame
df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
df["y"] = y

# Save to CSV
df.to_csv("sgd/d50_mu1_L50.csv", index=False)
