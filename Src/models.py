import numpy as np

def bias_baseline(R):
    """Overall mean plus one user effect and one movie effect."""
    known = ~np.isnan(R)
    mu = np.nanmean(R)
    b_user = np.nanmean(R - mu, axis=1)
    residual = R - mu - b_user[:, None]
    n_item = known.sum(axis=0)
    b_item = np.zeros(R.shape[1])
    has_rating = n_item > 0
    b_item[has_rating] = (
        np.nansum(residual, axis=0)[has_rating] / n_item[has_rating])
    return mu, b_user, b_item


def baseline_matrix(R):
    # Get the overall average, user effects, and movie effects
    mu, b_user, b_item = bias_baseline(R)

    # Use those values to predict ratings for every user and movie
    return mu + b_user[:, None] + b_item[None, :]


def rank_k(U, s, Vt, k):
    # Keep the first k parts of the SVD to make a smaller approximation
    return (U[:, :k] * s[:k]) @ Vt[:k, :]


def iterative_svd_predict(R, k, n_iter=15, after_round=None):
    # Find which ratings are already in the matrix
    known = ~np.isnan(R)

    # Get the baseline predictions
    B = baseline_matrix(R)

    # Subtract the baseline and fill missing values with zero
    residual = R - B
    X = np.where(known, residual, 0.0)

    # Repeat the SVD process to improve the missing ratings
    for _ in range(n_iter):
        # Break the current matrix down using SVD
        U, s, Vt = np.linalg.svd(X, full_matrices=False)

        # Rebuild the matrix using only the first k components
        L = rank_k(U, s, Vt, k)

        # Keep the actual ratings and update the missing ones
        X = np.where(known, residual, L)

        # The notebook can use this to check the error after each round
        if after_round is not None:
            after_round(L + B)

    # Add the baseline back to get the final predictions
    return L + B

