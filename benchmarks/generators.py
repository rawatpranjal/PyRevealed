"""Synthetic data generators for scaling benchmarks."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray


def generate_benchmark_data(
    n_observations: int,
    n_goods: int,
    data_type: Literal["rational", "random", "mixed"] = "rational",
    seed: int = 42,
    noise_level: float = 0.2,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Generate benchmark data at specified scale.

    Args:
        n_observations: Number of observations T (up to 10000)
        n_goods: Number of goods N (up to 50)
        data_type: Type of data to generate
        seed: Random seed for reproducibility
        noise_level: For mixed type, fraction of noise to inject

    Returns:
        Tuple of (prices, quantities) matrices
    """
    rng = np.random.default_rng(seed)

    if data_type == "rational":
        return _generate_cobb_douglas_data(n_observations, n_goods, rng)
    elif data_type == "random":
        return _generate_random_data(n_observations, n_goods, rng)
    elif data_type == "mixed":
        return _generate_mixed_data(n_observations, n_goods, rng, noise_level)
    else:
        raise ValueError(f"Unknown data_type: {data_type}")


def _generate_cobb_douglas_data(
    T: int, N: int, rng: np.random.Generator
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate utility-maximizing data using Cobb-Douglas preferences.

    This data is guaranteed to be GARP-consistent.
    """
    alpha = rng.dirichlet(np.ones(N))
    prices = rng.uniform(0.5, 2.0, size=(T, N))
    budgets = rng.uniform(10.0, 100.0, size=T)

    # Analytical solution: x_i = (alpha_i * m) / p_i
    quantities = np.zeros((T, N))
    for t in range(T):
        quantities[t] = (alpha * budgets[t]) / prices[t]

    return prices, quantities


def _generate_random_data(
    T: int, N: int, rng: np.random.Generator
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate random data (likely to violate GARP)."""
    prices = rng.uniform(0.5, 2.0, size=(T, N))
    budgets = rng.uniform(10.0, 100.0, size=T)

    quantities = np.zeros((T, N))
    for t in range(T):
        shares = rng.dirichlet(np.ones(N))
        quantities[t] = (shares * budgets[t]) / prices[t]

    return prices, quantities


def _generate_mixed_data(
    T: int, N: int, rng: np.random.Generator, noise: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate rational data with noise injection."""
    prices, quantities = _generate_cobb_douglas_data(T, N, rng)

    # Inject noise: randomly perturb quantities
    perturbation = rng.uniform(1.0 - noise, 1.0 + noise, size=quantities.shape)
    quantities = np.maximum(quantities * perturbation, 0.01)

    return prices, quantities
