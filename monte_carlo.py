import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt

# Part 1: Monte Carlo simulation
def mott_distribution(theta):
    """Mott scattering angular distribution ~ 1/sin^4(theta/2)."""
    return 1.0 / (np.sin(theta / 2) ** 4)

def simulate_scattering(n_events=100000):
    thetas = np.random.uniform(0.01, np.pi-0.01, n_events)  # avoid singularities
    weights = mott_distribution(thetas)
    probs = weights / weights.sum()
    sampled = np.random.choice(thetas, size=n_events, p=probs)
    return sampled
