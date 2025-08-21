import numpy as np

# Part 3: Bayesian unfolding
def bayesian_unfold(measured, response_matrix, iterations=4):
    n_bins = len(measured)
    prior = np.ones(n_bins) / n_bins
    for _ in range(iterations):
        expected = response_matrix.T @ prior
        correction = measured / (expected + 1e-12)
        posterior = prior * (response_matrix @ correction)
        posterior /= posterior.sum()
        prior = posterior
    return prior
