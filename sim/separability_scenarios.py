"""Separability scenario generators for superapp analysis.

Generate synthetic data for testing separability of product groups.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import sys
sys.path.insert(0, '/Users/pranjal/Code/revealed/src')

from pyrevealed.core.session import ConsumerSession


def generate_separable_superapp(
    n_observations: int = 100,
    seed: int | None = None,
) -> tuple[ConsumerSession, list[int], list[int]]:
    """
    Generate a superapp with SEPARABLE product lines.

    Products: [Rides_Economy, Rides_Premium, Eats_Fast, Eats_Fancy]
    Rides (0,1) and Eats (2,3) are independent - changing price of
    one group doesn't affect demand for the other.

    Args:
        n_observations: Number of consumption observations
        seed: Random seed

    Returns:
        Tuple of (ConsumerSession, rides_indices, eats_indices)
    """
    rng = np.random.default_rng(seed)

    n_goods = 4
    rides_idx = [0, 1]
    eats_idx = [2, 3]

    # Generate prices with variation
    prices = np.zeros((n_observations, n_goods))
    quantities = np.zeros((n_observations, n_goods))

    # User preferences (separable utility)
    # U(x) = u_rides(x_0, x_1) + u_eats(x_2, x_3)
    # where u_rides and u_eats are Cobb-Douglas sub-utilities

    alpha_rides = rng.dirichlet([2, 1])  # Prefer economy rides
    alpha_eats = rng.dirichlet([1, 2])   # Prefer fancy eats

    # Budget allocation: fixed split between rides and eats
    rides_budget_share = 0.4
    total_budget = rng.uniform(50, 150, n_observations)

    for t in range(n_observations):
        # Random prices with some correlation within groups
        base_rides = rng.uniform(0.8, 1.2)
        base_eats = rng.uniform(0.8, 1.2)

        prices[t, 0] = base_rides * rng.uniform(0.8, 1.2)  # Economy
        prices[t, 1] = base_rides * rng.uniform(1.5, 2.5)  # Premium
        prices[t, 2] = base_eats * rng.uniform(0.8, 1.2)   # Fast food
        prices[t, 3] = base_eats * rng.uniform(1.5, 2.5)   # Fancy

        # SEPARABLE demand: each group optimizes independently
        rides_budget = total_budget[t] * rides_budget_share
        eats_budget = total_budget[t] * (1 - rides_budget_share)

        # Cobb-Douglas demand within each group
        quantities[t, 0] = alpha_rides[0] * rides_budget / prices[t, 0]
        quantities[t, 1] = alpha_rides[1] * rides_budget / prices[t, 1]
        quantities[t, 2] = alpha_eats[0] * eats_budget / prices[t, 2]
        quantities[t, 3] = alpha_eats[1] * eats_budget / prices[t, 3]

        # Add small multiplicative noise (preserves separability structure)
        quantities[t] *= (1 + rng.normal(0, 0.05, n_goods))
        quantities[t] = np.maximum(quantities[t], 0.01)

    session = ConsumerSession(
        prices=prices,
        quantities=quantities,
        session_id="separable_superapp",
        metadata={
            "product_names": ["Rides_Economy", "Rides_Premium", "Eats_Fast", "Eats_Fancy"],
            "true_separable": True,
        },
    )

    return session, rides_idx, eats_idx


def generate_cannibalized_superapp(
    n_observations: int = 100,
    cannibalization_strength: float = 0.5,
    seed: int | None = None,
) -> tuple[ConsumerSession, list[int], list[int]]:
    """
    Generate a superapp with CANNIBALIZATION between product lines.

    Products: [Rides_Economy, Rides_Premium, Eats_Fast, Eats_Fancy]
    When Eats prices drop, users order more Eats and take fewer Rides
    (they eat at home instead of going out).

    Args:
        n_observations: Number of consumption observations
        cannibalization_strength: How strongly groups affect each other (0-1)
        seed: Random seed

    Returns:
        Tuple of (ConsumerSession, rides_indices, eats_indices)
    """
    rng = np.random.default_rng(seed)

    n_goods = 4
    rides_idx = [0, 1]
    eats_idx = [2, 3]

    prices = np.zeros((n_observations, n_goods))
    quantities = np.zeros((n_observations, n_goods))

    total_budget = rng.uniform(50, 150, n_observations)

    for t in range(n_observations):
        # Random prices
        prices[t, 0] = rng.uniform(1.0, 2.0)
        prices[t, 1] = rng.uniform(2.0, 4.0)
        prices[t, 2] = rng.uniform(0.8, 1.5)
        prices[t, 3] = rng.uniform(1.5, 3.0)

        # CANNIBALIZATION: budget share depends on relative prices
        # If eats is cheap, spend more on eats (and less on rides)
        avg_rides_price = np.mean(prices[t, rides_idx])
        avg_eats_price = np.mean(prices[t, eats_idx])

        price_ratio = avg_eats_price / (avg_rides_price + avg_eats_price)

        # Base allocation
        base_rides_share = 0.4

        # Cannibalization effect: cheap eats -> more eats, less rides
        rides_share = base_rides_share + cannibalization_strength * (price_ratio - 0.5)
        rides_share = np.clip(rides_share, 0.1, 0.9)

        rides_budget = total_budget[t] * rides_share
        eats_budget = total_budget[t] * (1 - rides_share)

        # Allocate within groups (Cobb-Douglas style)
        quantities[t, 0] = 0.6 * rides_budget / prices[t, 0]
        quantities[t, 1] = 0.4 * rides_budget / prices[t, 1]
        quantities[t, 2] = 0.5 * eats_budget / prices[t, 2]
        quantities[t, 3] = 0.5 * eats_budget / prices[t, 3]

        # Add noise
        quantities[t] += rng.normal(0, 0.1, n_goods)
        quantities[t] = np.maximum(quantities[t], 0.01)

    session = ConsumerSession(
        prices=prices,
        quantities=quantities,
        session_id="cannibalized_superapp",
        metadata={
            "product_names": ["Rides_Economy", "Rides_Premium", "Eats_Fast", "Eats_Fancy"],
            "true_separable": False,
            "cannibalization_strength": cannibalization_strength,
        },
    )

    return session, rides_idx, eats_idx


def generate_mixed_superapp(
    n_observations: int = 100,
    complementary_fraction: float = 0.3,
    seed: int | None = None,
) -> tuple[ConsumerSession, list[int], list[int]]:
    """
    Generate a superapp with MIXED effects.

    Some users treat products as separable, others show cannibalization,
    and some show complementarity (use rides TO get eats).

    Args:
        n_observations: Number of consumption observations
        complementary_fraction: Fraction of users who use both together
        seed: Random seed

    Returns:
        Tuple of (ConsumerSession, rides_indices, eats_indices)
    """
    rng = np.random.default_rng(seed)

    n_goods = 4
    rides_idx = [0, 1]
    eats_idx = [2, 3]

    prices = np.zeros((n_observations, n_goods))
    quantities = np.zeros((n_observations, n_goods))

    total_budget = rng.uniform(50, 150, n_observations)

    for t in range(n_observations):
        # Random prices
        prices[t, 0] = rng.uniform(1.0, 2.0)
        prices[t, 1] = rng.uniform(2.0, 4.0)
        prices[t, 2] = rng.uniform(0.8, 1.5)
        prices[t, 3] = rng.uniform(1.5, 3.0)

        # Determine user type for this observation
        user_type = rng.random()

        if user_type < complementary_fraction:
            # Complementary: rides and eats together
            # High correlation: both high or both low
            intensity = rng.uniform(0.5, 1.5)
            quantities[t, 0] = intensity * 2.0
            quantities[t, 1] = intensity * 0.5
            quantities[t, 2] = intensity * 1.5
            quantities[t, 3] = intensity * 1.0
        elif user_type < complementary_fraction + 0.4:
            # Cannibalization: one or the other
            if rng.random() < 0.5:
                # Heavy rides user
                quantities[t, 0] = rng.uniform(3, 6)
                quantities[t, 1] = rng.uniform(0.5, 2)
                quantities[t, 2] = rng.uniform(0.1, 0.5)
                quantities[t, 3] = rng.uniform(0.1, 0.3)
            else:
                # Heavy eats user
                quantities[t, 0] = rng.uniform(0.1, 0.5)
                quantities[t, 1] = rng.uniform(0.1, 0.3)
                quantities[t, 2] = rng.uniform(2, 4)
                quantities[t, 3] = rng.uniform(1, 3)
        else:
            # Separable: independent allocation
            rides_budget = total_budget[t] * 0.4
            eats_budget = total_budget[t] * 0.6

            quantities[t, 0] = 0.6 * rides_budget / prices[t, 0]
            quantities[t, 1] = 0.4 * rides_budget / prices[t, 1]
            quantities[t, 2] = 0.5 * eats_budget / prices[t, 2]
            quantities[t, 3] = 0.5 * eats_budget / prices[t, 3]

        # Add noise
        quantities[t] += rng.normal(0, 0.05, n_goods)
        quantities[t] = np.maximum(quantities[t], 0.01)

    session = ConsumerSession(
        prices=prices,
        quantities=quantities,
        session_id="mixed_superapp",
        metadata={
            "product_names": ["Rides_Economy", "Rides_Premium", "Eats_Fast", "Eats_Fancy"],
            "true_separable": False,
            "complementary_fraction": complementary_fraction,
        },
    )

    return session, rides_idx, eats_idx


def generate_amazon_scenario(
    n_observations: int = 100,
    seed: int | None = None,
) -> tuple[ConsumerSession, list[int], list[int], list[int]]:
    """
    Generate an Amazon-style scenario with 3 product groups.

    Groups: Books, Electronics, Groceries
    Tests whether these can be priced independently.

    Args:
        n_observations: Number of observations
        seed: Random seed

    Returns:
        Tuple of (session, books_idx, electronics_idx, groceries_idx)
    """
    rng = np.random.default_rng(seed)

    n_goods = 6  # 2 per category
    books_idx = [0, 1]
    electronics_idx = [2, 3]
    groceries_idx = [4, 5]

    prices = np.zeros((n_observations, n_goods))
    quantities = np.zeros((n_observations, n_goods))

    for t in range(n_observations):
        # Prices
        prices[t, 0] = rng.uniform(10, 30)    # Paperback
        prices[t, 1] = rng.uniform(20, 50)    # Hardcover
        prices[t, 2] = rng.uniform(100, 500)  # Electronics small
        prices[t, 3] = rng.uniform(500, 2000) # Electronics big
        prices[t, 4] = rng.uniform(3, 10)     # Groceries basic
        prices[t, 5] = rng.uniform(10, 30)    # Groceries premium

        # Mostly independent consumption with some correlation
        # (people who buy electronics might buy fewer groceries that month)
        budget = rng.uniform(100, 1000)

        # Base allocations
        book_spend = rng.exponential(30)
        electronics_spend = rng.exponential(100) if rng.random() < 0.3 else 0
        grocery_spend = budget - book_spend - electronics_spend
        grocery_spend = max(grocery_spend, 10)

        # Quantities
        if book_spend > 0:
            quantities[t, 0] = rng.binomial(3, 0.7) * (book_spend / prices[t, 0])
            quantities[t, 1] = rng.binomial(1, 0.3) * (book_spend / prices[t, 1])

        if electronics_spend > 0:
            if electronics_spend > prices[t, 3]:
                quantities[t, 3] = 1
                quantities[t, 2] = (electronics_spend - prices[t, 3]) / prices[t, 2]
            else:
                quantities[t, 2] = electronics_spend / prices[t, 2]

        quantities[t, 4] = grocery_spend * 0.7 / prices[t, 4]
        quantities[t, 5] = grocery_spend * 0.3 / prices[t, 5]

        quantities[t] = np.maximum(quantities[t], 0)

    session = ConsumerSession(
        prices=prices,
        quantities=quantities,
        session_id="amazon_scenario",
        metadata={
            "product_names": [
                "Book_Paper", "Book_Hard",
                "Elec_Small", "Elec_Big",
                "Grocery_Basic", "Grocery_Premium"
            ],
        },
    )

    return session, books_idx, electronics_idx, groceries_idx
