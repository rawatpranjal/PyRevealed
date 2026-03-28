"""Synthetic demo dataset for PrefGraph.

Generates a deterministic dataset of budget-constrained consumers with
varying rationality levels - no downloads, no setup.

Usage:
    from prefgraph.datasets import load_demo
    from prefgraph.engine import Engine

    users = load_demo()
    results = Engine(metrics=["garp", "ccei", "mpi"]).analyze_arrays(users)
"""

from __future__ import annotations

import numpy as np


def load_demo(
    n_users: int = 100,
    n_obs: int = 15,
    n_goods: int = 5,
    seed: int = 42,
    return_panel: bool = False,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Load a synthetic demo dataset (offline, zero setup).

    Generates deterministic budget data with a mix of rational and
    irrational consumers for testing and demos.

    The dataset contains three types of consumers:
    - ~40% perfectly rational (Cobb-Douglas utility maximization)
    - ~40% noisy rational (perturbations from optimal)
    - ~20% irrational (random choices)

    This creates an interesting CCEI distribution for demonstrations.

    Args:
        n_users: Number of synthetic users (default 100).
        n_obs: Observations per user (default 15).
        n_goods: Number of goods (default 5).
        seed: Random seed for reproducibility (default 42).
        return_panel: If True, return a BehaviorPanel instead.

    Returns:
        List of (prices T*K, quantities T*K) tuples ready for
        Engine.analyze_arrays(). If return_panel=True, returns
        a BehaviorPanel.
    """
    rng = np.random.RandomState(seed)

    n_rational = int(n_users * 0.4)
    n_noisy = int(n_users * 0.4)
    n_irrational = n_users - n_rational - n_noisy

    users: list[tuple[np.ndarray, np.ndarray]] = []

    for i in range(n_users):
        # Random prices and budgets per observation
        prices = rng.uniform(0.5, 5.0, size=(n_obs, n_goods))
        budgets = rng.uniform(10.0, 100.0, size=n_obs)

        # Cobb-Douglas utility weights for this user
        alpha = rng.dirichlet(np.ones(n_goods))

        if i < n_rational:
            # Perfectly rational: exact Cobb-Douglas demand
            quantities = (alpha[np.newaxis, :] * budgets[:, np.newaxis]) / prices
        elif i < n_rational + n_noisy:
            # Noisy rational: perturbed demand
            sigma = rng.uniform(0.1, 0.5)
            optimal = (alpha[np.newaxis, :] * budgets[:, np.newaxis]) / prices
            noise = np.exp(rng.normal(0, sigma, size=(n_obs, n_goods)))
            quantities = optimal * noise
        else:
            # Irrational: random quantities within budget
            max_q = budgets[:, np.newaxis] / prices
            quantities = rng.uniform(0, 1, size=(n_obs, n_goods)) * max_q

        # Ensure positive quantities
        quantities = np.maximum(quantities, 1e-6)
        users.append((prices, quantities))

    if return_panel:
        from prefgraph.core.session import BehaviorLog
        from prefgraph.core.panel import BehaviorPanel

        logs = [
            BehaviorLog(
                cost_vectors=p,
                action_vectors=q,
                user_id=f"demo_{i:03d}",
            )
            for i, (p, q) in enumerate(users)
        ]
        return BehaviorPanel.from_logs(logs)

    return users
