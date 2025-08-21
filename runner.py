import numpy as np
import matplotlib.pyplot as plt

# Import from other modules
from monte_carlo import simulate_scattering, mott_distribution
from SQL_Pipeline import build_sql_pipeline
from bayesian_unfolding import bayesian_unfold

# Main runner
if __name__ == "__main__":
    # Simulate scattering
    sampled = simulate_scattering(100000)

    # Store + preprocess
    df = build_sql_pipeline(sampled)

    # Bin data
    counts, bins = np.histogram(df["theta"], bins=50, range=(0, np.pi))
    measured = counts / counts.sum()

    # Dummy response matrix (identity = perfect detector)
    response = np.eye(len(measured))

    # Bayesian unfolding
    unfolded = bayesian_unfold(measured, response)

    # Compare with theory
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    true_dist = mott_distribution(bin_centers)
    true_dist /= true_dist.sum()

    fidelity = 1 - np.mean(np.abs(unfolded - true_dist))
    print(f"Retrieved angular distributions with ±{(1-fidelity)*100:.1f}% fidelity.")

    # Plot
    plt.plot(bin_centers, true_dist, label="Mott curve (theory)")
    plt.step(bin_centers, unfolded, where="mid", label="Unfolded distribution")
    plt.legend()
    plt.show()
