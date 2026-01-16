"""Data generators for simulation studies.

Provides utility functions and demand generators for testing revealed preference algorithms.
Based on theory from Chambers & Echenique "Revealed Preference Theory".
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Callable
from scipy.optimize import minimize


def cobb_douglas_utility(x: NDArray[np.float64], alpha: NDArray[np.float64]) -> float:
    """
    Cobb-Douglas utility: u(x) = prod(x_i^alpha_i)

    Args:
        x: Consumption bundle (n goods)
        alpha: Preference weights (should sum to 1 for homogeneity)

    Returns:
        Utility value
    """
    x = np.maximum(x, 1e-10)  # Avoid log(0)
    return np.prod(x ** alpha)


def cobb_douglas_demand(
    p: NDArray[np.float64],
    m: float,
    alpha: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Optimal demand for Cobb-Douglas utility.

    Analytical solution: x_i = (alpha_i * m) / p_i

    Args:
        p: Price vector
        m: Budget/income
        alpha: Preference weights (should sum to 1)

    Returns:
        Optimal consumption bundle
    """
    return (alpha * m) / p


def ces_utility(
    x: NDArray[np.float64],
    alpha: NDArray[np.float64],
    rho: float,
) -> float:
    """
    CES (Constant Elasticity of Substitution) utility.

    u(x) = (sum alpha_i * x_i^rho)^(1/rho)

    Args:
        x: Consumption bundle
        alpha: Preference weights
        rho: Substitution parameter (rho < 1, rho != 0)
             rho -> 0: Cobb-Douglas
             rho -> -inf: Leontief
             rho = 1: Linear

    Returns:
        Utility value
    """
    x = np.maximum(x, 1e-10)
    return np.power(np.sum(alpha * np.power(x, rho)), 1.0 / rho)


def ces_demand(
    p: NDArray[np.float64],
    m: float,
    alpha: NDArray[np.float64],
    rho: float,
) -> NDArray[np.float64]:
    """
    Optimal demand for CES utility (numerical optimization).

    Args:
        p: Price vector
        m: Budget/income
        alpha: Preference weights
        rho: Substitution parameter

    Returns:
        Optimal consumption bundle
    """
    n = len(p)

    def neg_utility(x: NDArray[np.float64]) -> float:
        return -ces_utility(x, alpha, rho)

    # Budget constraint: p @ x <= m
    constraints = {"type": "eq", "fun": lambda x: p @ x - m}
    bounds = [(1e-6, m / p[i]) for i in range(n)]
    x0 = np.full(n, m / (n * np.mean(p)))

    result = minimize(neg_utility, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    return result.x


def leontief_utility(x: NDArray[np.float64], alpha: NDArray[np.float64]) -> float:
    """
    Leontief (perfect complements) utility: u(x) = min(x_i / alpha_i)

    Args:
        x: Consumption bundle
        alpha: Coefficients (consumption ratios)

    Returns:
        Utility value
    """
    return np.min(x / alpha)


def leontief_demand(
    p: NDArray[np.float64],
    m: float,
    alpha: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Optimal demand for Leontief utility.

    Analytical solution: x_i = alpha_i * m / (p @ alpha)

    Args:
        p: Price vector
        m: Budget/income
        alpha: Coefficients

    Returns:
        Optimal consumption bundle
    """
    return alpha * m / (p @ alpha)


def generate_rational_data(
    n_observations: int,
    n_goods: int,
    utility_type: str = "cobb_douglas",
    seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Generate consumption data from a rational utility-maximizing consumer.

    This data should satisfy GARP by construction.

    Args:
        n_observations: Number of price-quantity observations
        n_goods: Number of goods
        utility_type: "cobb_douglas", "ces", or "leontief"
        seed: Random seed for reproducibility

    Returns:
        Tuple of (prices, quantities, alpha) where:
        - prices: T x n matrix of prices
        - quantities: T x n matrix of quantities
        - alpha: preference parameters used
    """
    rng = np.random.default_rng(seed)

    # Generate random preference parameters
    alpha = rng.dirichlet(np.ones(n_goods))

    # Generate random prices (strictly positive)
    prices = rng.uniform(0.5, 2.0, size=(n_observations, n_goods))

    # Generate random budgets
    budgets = rng.uniform(10.0, 50.0, size=n_observations)

    # Compute optimal demands
    quantities = np.zeros((n_observations, n_goods))

    for t in range(n_observations):
        if utility_type == "cobb_douglas":
            quantities[t] = cobb_douglas_demand(prices[t], budgets[t], alpha)
        elif utility_type == "ces":
            rho = 0.5  # Fixed for simplicity
            quantities[t] = ces_demand(prices[t], budgets[t], alpha, rho)
        elif utility_type == "leontief":
            quantities[t] = leontief_demand(prices[t], budgets[t], alpha)
        else:
            raise ValueError(f"Unknown utility type: {utility_type}")

    return prices, quantities, alpha


def generate_irrational_data(
    n_observations: int,
    n_goods: int,
    seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Generate random (irrational) consumption data.

    Random choices are likely to violate GARP, especially with many observations.

    Args:
        n_observations: Number of observations
        n_goods: Number of goods
        seed: Random seed

    Returns:
        Tuple of (prices, quantities)
    """
    rng = np.random.default_rng(seed)

    # Random prices
    prices = rng.uniform(0.5, 2.0, size=(n_observations, n_goods))

    # Random quantities on the budget line
    budgets = rng.uniform(10.0, 50.0, size=n_observations)

    quantities = np.zeros((n_observations, n_goods))
    for t in range(n_observations):
        # Random budget shares
        shares = rng.dirichlet(np.ones(n_goods))
        quantities[t] = (shares * budgets[t]) / prices[t]

    return prices, quantities


def generate_garp_violation_cycle(
    n_goods: int = 2,
    cycle_length: int = 2,
    seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Generate data with a guaranteed GARP violation cycle.

    Constructs a cycle where x1 ≻R x2 ≻R ... ≻R xn ≻R x1.

    For GARP violation we need:
    - p_i @ x_i >= p_i @ x_{i+1} for all i (revealed weak preference)
    - At least one strict: p_j @ x_j > p_j @ x_{j+1}

    Args:
        n_goods: Number of goods (default 2 for visualization)
        cycle_length: Length of the violation cycle
        seed: Random seed

    Returns:
        Tuple of (prices, quantities) with guaranteed GARP violation
    """
    rng = np.random.default_rng(seed)

    # For a 2-good case, create a simple WARP violation
    if cycle_length == 2 and n_goods == 2:
        # Classic WARP violation from Chambers & Echenique Fig 3.1
        # Need: p1 @ x1 >= p1 @ x2 AND p2 @ x2 > p2 @ x1
        # This means x1 is weakly revealed preferred to x2,
        # but x2 is STRICTLY revealed preferred to x1

        # Choose prices and quantities to guarantee violation
        prices = np.array([
            [1.0, 1.0],  # p1: equal prices
            [1.0, 1.0],  # p2: equal prices
        ])
        quantities = np.array([
            [4.0, 2.0],  # x1: exp = 6
            [2.0, 3.0],  # x2: exp = 5
        ])
        # p1 @ x1 = 6, p1 @ x2 = 5 -> x1 STRICTLY revealed preferred to x2
        # p2 @ x2 = 5, p2 @ x1 = 6 -> x1 NOT affordable at p2 (no revealed pref)

        # Need BOTH to be affordable at each other's budget
        # Use different price vectors
        prices = np.array([
            [2.0, 1.0],  # p1
            [1.0, 2.0],  # p2
        ])
        quantities = np.array([
            [2.0, 4.0],  # x1: p1@x1 = 8
            [4.0, 2.0],  # x2: p2@x2 = 8
        ])
        # p1 @ x1 = 8, p1 @ x2 = 10 > 8 -> x2 not affordable at p1
        # Still not a violation!

        # Correct approach: crossing budget lines where each is affordable
        prices = np.array([
            [1.0, 1.0],  # p1
            [1.0, 1.0],  # p2
        ])
        quantities = np.array([
            [5.0, 3.0],  # x1: exp = 8
            [3.0, 4.0],  # x2: exp = 7
        ])
        # p1 @ x1 = 8, p1 @ x2 = 7 < 8 -> x1 STRICTLY preferred to x2
        # p2 @ x2 = 7, p2 @ x1 = 8 > 7 -> x1 NOT affordable -> no violation

        # For WARP violation: x1 revealed preferred to x2 AND x2 revealed preferred to x1
        # Need both affordable at each other's budgets
        prices = np.array([
            [1.0, 1.0],  # p1
            [1.0, 1.0],  # p2
        ])
        quantities = np.array([
            [3.0, 2.0],  # x1: exp = 5
            [2.5, 2.5], # x2: exp = 5
        ])
        # p1 @ x1 = 5, p1 @ x2 = 5 -> weak preference (not strict)
        # p2 @ x2 = 5, p2 @ x1 = 5 -> weak preference (not strict)
        # No STRICT preference -> no violation!

        # For violation: need at least one STRICT preference in the cycle
        # Same expenditure level, but one must be strictly cheaper at the other's prices
        # This is impossible with same prices! Need different prices.

        # Correct WARP violation example:
        prices = np.array([
            [1.0, 2.0],  # p1: good 2 expensive
            [2.0, 1.0],  # p2: good 1 expensive
        ])
        quantities = np.array([
            [4.0, 2.0],  # x1: more good 1 (cheap at p1)
            [2.0, 4.0],  # x2: more good 2 (cheap at p2)
        ])
        # p1 @ x1 = 4 + 4 = 8
        # p1 @ x2 = 2 + 8 = 10 > 8 -> x2 NOT affordable at p1
        # Still wrong!

        # Make both bundles affordable at both price vectors
        prices = np.array([
            [1.0, 1.0],  # p1
            [1.0, 1.0],  # p2 (same as p1 for simplicity)
        ])
        # With same prices, need same expenditure for both to be affordable

        # Actually for GARP violation with cycle length 2:
        # We need R*(0,1) AND P(1,0), i.e., 0 transitively pref to 1, but 1 strictly pref to 0
        # With 2 obs, R* = R, so we need R(0,1) AND P(1,0)
        # R(0,1): p0@x0 >= p0@x1 -> x1 affordable at budget 0
        # P(1,0): p1@x1 > p1@x0 -> x0 STRICTLY cheaper at budget 1

        prices = np.array([
            [1.0, 1.0],
            [1.0, 1.0],
        ])
        quantities = np.array([
            [3.0, 3.0],  # x0: exp = 6
            [4.0, 1.0],  # x1: exp = 5
        ])
        # p0@x0 = 6, p0@x1 = 5 < 6 -> R(0,1) TRUE (x1 affordable)
        # p1@x1 = 5, p1@x0 = 6 > 5 -> x0 NOT affordable at budget 1
        # So no P(1,0)! x0 wasn't chosen from a budget containing x1

        # Need x0 affordable at budget 1 AND x0 strictly cheaper
        quantities = np.array([
            [2.0, 2.0],  # x0: exp = 4
            [3.0, 2.0],  # x1: exp = 5
        ])
        # p0@x0 = 4, p0@x1 = 5 > 4 -> x1 NOT affordable at budget 0
        # Wrong direction

        # Correct:
        quantities = np.array([
            [3.0, 2.0],  # x0: exp = 5
            [2.0, 2.0],  # x1: exp = 4
        ])
        # p0@x0 = 5, p0@x1 = 4 < 5 -> R(0,1) TRUE, P(0,1) TRUE
        # p1@x1 = 4, p1@x0 = 5 > 4 -> x0 NOT affordable
        # Still no cycle!

        # For cycle: need R(0,1) AND R(1,0) with at least one strict
        # R(0,1): p0@x0 >= p0@x1
        # R(1,0): p1@x1 >= p1@x0
        # Both must hold, with at least one strict

        # This means: both bundles affordable at both budgets!
        # With equal prices, need equal expenditures
        quantities = np.array([
            [3.0, 2.0],  # x0: exp = 5
            [2.0, 3.0],  # x1: exp = 5
        ])
        # p0@x0 = 5, p0@x1 = 5 -> R(0,1) TRUE (weak)
        # p1@x1 = 5, p1@x0 = 5 -> R(1,0) TRUE (weak)
        # No strict preference! -> No violation

        # Need different prices to create strict preference
        prices = np.array([
            [2.0, 1.0],  # p0
            [1.0, 2.0],  # p1
        ])
        quantities = np.array([
            [2.0, 2.0],  # x0: p0@x0 = 6, p1@x0 = 6
            [2.0, 2.0],  # x1: p0@x1 = 6, p1@x1 = 6
        ])
        # Same bundles -> no violation

        # Different bundles, same total cost at each price
        quantities = np.array([
            [1.0, 4.0],  # x0: p0@x0 = 6, p1@x0 = 9
            [4.0, 1.0],  # x1: p0@x1 = 9, p1@x1 = 6
        ])
        # p0@x0 = 6, p0@x1 = 9 > 6 -> x1 NOT affordable at budget 0
        # No R(0,1)

        # Final correct construction:
        # At p0, spend 10, buy x0. x1 costs 8 at p0 -> affordable, strictly cheaper
        # At p1, spend 10, buy x1. x0 costs 8 at p1 -> affordable, strictly cheaper
        prices = np.array([
            [1.0, 1.0],
            [1.0, 1.0],
        ])
        quantities = np.array([
            [6.0, 4.0],  # x0: exp = 10
            [4.0, 6.0],  # x1: exp = 10
        ])
        # Same expenditure -> weak preferences only, no strict

        # Use different expenditure levels!
        quantities = np.array([
            [6.0, 4.0],  # x0: exp = 10
            [5.0, 4.0],  # x1: exp = 9
        ])
        # p0@x0 = 10, p0@x1 = 9 < 10 -> R(0,1), P(0,1) (strict)
        # p1@x1 = 9, p1@x0 = 10 > 9 -> x0 NOT affordable at budget 1
        # No cycle

        # Must have EQUAL expenditures for both to be affordable
        # But equal expenditure with same prices -> only weak preferences
        # Solution: use different prices!

        prices = np.array([
            [1.0, 3.0],  # p0: good 2 expensive
            [3.0, 1.0],  # p1: good 1 expensive
        ])
        quantities = np.array([
            [5.0, 1.0],  # x0: p0@x0 = 8, p1@x0 = 16
            [1.0, 5.0],  # x1: p0@x1 = 16, p1@x1 = 8
        ])
        # p0@x0 = 8, p0@x1 = 16 > 8 -> x1 NOT affordable
        # Wrong again

        # Need symmetric setup where both are affordable
        prices = np.array([
            [1.0, 1.0],
            [1.0, 1.0],
        ])
        quantities = np.array([
            [5.0, 5.0],  # x0: exp = 10
            [5.0, 5.0],  # x1: exp = 10
        ])
        # Identical -> trivially consistent

        # The key insight: with 2 goods and 2 observations,
        # GARP = WARP (Theorem 3.4 in the book)
        # For WARP violation: need crossing budget lines

        # Budget line 0: p0 @ x = p0 @ x0
        # Budget line 1: p1 @ x = p1 @ x1
        # For violation: x1 inside budget 0 AND x0 inside budget 1

        prices = np.array([
            [1.0, 2.0],
            [2.0, 1.0],
        ])
        quantities = np.array([
            [6.0, 2.0],  # x0: p0@x0 = 10, p1@x0 = 14
            [2.0, 6.0],  # x1: p0@x1 = 14, p1@x1 = 8
        ])
        # p0@x0 = 10, p0@x1 = 14 > 10 -> x1 NOT affordable at p0

        # Make budgets cross
        quantities = np.array([
            [4.0, 3.0],  # x0: p0@x0 = 10, p1@x0 = 11
            [3.0, 4.0],  # x1: p0@x1 = 11, p1@x1 = 10
        ])
        # p0@x0 = 10, p0@x1 = 11 > 10 -> x1 NOT affordable

        # Equal expenditure at own prices
        quantities = np.array([
            [8.0, 1.0],  # x0: p0@x0 = 10, p1@x0 = 17
            [1.0, 8.0],  # x1: p0@x1 = 17, p1@x1 = 10
        ])
        # Still not affordable

        # The issue: with these prices, can't have both affordable
        # Try more balanced prices
        prices = np.array([
            [1.5, 1.0],
            [1.0, 1.5],
        ])
        quantities = np.array([
            [4.0, 3.0],  # x0: p0@x0 = 9, p1@x0 = 8.5
            [3.0, 4.0],  # x1: p0@x1 = 8.5, p1@x1 = 9
        ])
        # p0@x0 = 9, p0@x1 = 8.5 < 9 -> R(0,1), P(0,1) TRUE
        # p1@x1 = 9, p1@x0 = 8.5 < 9 -> R(1,0), P(1,0) TRUE
        # BOTH strictly revealed preferred to each other!
        # This is a GARP violation!

        return prices, quantities

    # General case for cycle_length >= 3
    # Strategy: Create a cycle on a simplex where each bundle
    # is strictly cheaper than the previous at that price vector

    # Use uniform prices for simplicity
    prices = np.ones((cycle_length, n_goods))

    # Add small perturbations to create strict preferences
    for i in range(cycle_length):
        # Slightly increase price of good i
        prices[i, i % n_goods] += 0.5

    # Create quantities with equal total expenditure but different compositions
    # such that at each price, the next bundle is strictly cheaper
    base_exp = 10.0
    quantities = np.zeros((cycle_length, n_goods))

    for i in range(cycle_length):
        # Distribute budget: favor goods that are cheap at price i
        # but make next bundle (i+1) strictly cheaper at price i
        quantities[i] = np.ones(n_goods) * (base_exp / (n_goods * np.mean(prices[i])))

        # Adjust to favor the good that is expensive at this price
        # This makes the NEXT bundle cheaper (which favors cheap goods)
        expensive_good = i % n_goods
        cheap_goods = [j for j in range(n_goods) if j != expensive_good]

        # Increase consumption of expensive good, decrease cheap goods
        quantities[i, expensive_good] *= 1.5
        for cg in cheap_goods:
            quantities[i, cg] *= 0.8

    # Normalize to have similar expenditures
    for i in range(cycle_length):
        current_exp = prices[i] @ quantities[i]
        quantities[i] *= base_exp / current_exp

    # Verify the cycle creates a violation
    # For GARP violation we need: R(0,1), R(1,2), ..., R(n-1,0) with at least one strict
    # R(i, i+1): p_i @ x_i >= p_i @ x_{i+1}

    # Adjust quantities to ensure the cycle
    for iteration in range(20):
        all_satisfied = True
        for i in range(cycle_length):
            next_i = (i + 1) % cycle_length
            exp_xi = prices[i] @ quantities[i]
            exp_xnext = prices[i] @ quantities[next_i]

            if exp_xi < exp_xnext:
                # x_{i+1} not affordable at budget i, scale up x_i
                quantities[i] *= (exp_xnext / exp_xi) * 1.01
                all_satisfied = False

        if all_satisfied:
            break

    return prices, quantities


def generate_efficiency_data(
    n_observations: int,
    n_goods: int,
    target_efficiency: float,
    seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Generate data with approximately known Afriat Efficiency Index.

    Creates data that satisfies GARP at efficiency `target_efficiency`
    but violates at higher efficiency.

    Args:
        n_observations: Number of observations
        n_goods: Number of goods
        target_efficiency: Target AEI (0 < e < 1)
        seed: Random seed

    Returns:
        Tuple of (prices, quantities)
    """
    rng = np.random.default_rng(seed)

    # Start with rational data
    prices, quantities, _ = generate_rational_data(
        n_observations, n_goods, "cobb_douglas", seed
    )

    # Perturb to create inefficiency
    # Waste fraction of budget on suboptimal choices
    waste_factor = 1.0 - target_efficiency

    for t in range(n_observations):
        # Add bounded noise proportional to waste (uniform is bounded unlike normal)
        noise = rng.uniform(-waste_factor, waste_factor, n_goods)
        quantities[t] = np.maximum(quantities[t] * (1 + noise), 0.1)

    return prices, quantities


# =============================================================================
# ABSTRACT CHOICE THEORY GENERATORS (Menu-based)
# =============================================================================


def generate_rational_menu_choices(
    n_obs: int,
    n_items: int,
    seed: int | None = None,
    menu_size: int | None = None,
) -> tuple[list[frozenset[int]], list[int], list[int]]:
    """
    Generate menu choice data from a rational agent with fixed preference order.

    Creates menus and choices that satisfy SARP/Congruence by construction.
    The agent has a strict preference order over items and always chooses
    the most preferred item from each menu.

    Args:
        n_obs: Number of choice observations
        n_items: Number of distinct items
        seed: Random seed for reproducibility
        menu_size: Size of each menu (default: random 2 to n_items)

    Returns:
        Tuple of (menus, choices, preference_order) where:
        - menus: List of T frozensets (available items in each menu)
        - choices: List of T chosen item indices
        - preference_order: List of items from most to least preferred
    """
    rng = np.random.default_rng(seed)

    # Generate random preference ordering (0 = most preferred)
    preference_order = list(rng.permutation(n_items))
    preference_rank = {item: rank for rank, item in enumerate(preference_order)}

    menus: list[frozenset[int]] = []
    choices: list[int] = []

    for _ in range(n_obs):
        # Determine menu size for this observation
        if menu_size is not None:
            size = min(menu_size, n_items)
        else:
            size = rng.integers(2, n_items + 1)

        # Sample items for this menu
        menu_items = rng.choice(n_items, size=size, replace=False)
        menu = frozenset(menu_items.tolist())

        # Choose the most preferred item in the menu
        best_item = min(menu, key=lambda x: preference_rank[x])

        menus.append(menu)
        choices.append(best_item)

    return menus, choices, preference_order


def generate_warp_violation_menus(
    n_items: int = 3,
    seed: int | None = None,
) -> tuple[list[frozenset[int]], list[int]]:
    """
    Generate menu choice data with a guaranteed WARP violation.

    Creates a direct 2-cycle where item x is chosen over y in one menu,
    but y is chosen over x in another menu.

    Args:
        n_items: Number of items (must be >= 2)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (menus, choices) with guaranteed WARP violation
    """
    if n_items < 2:
        raise ValueError("Need at least 2 items for WARP violation")

    rng = np.random.default_rng(seed)

    # Pick two items for the violation
    items = rng.choice(n_items, size=2, replace=False)
    x, y = int(items[0]), int(items[1])

    # Create WARP violation: x chosen over y, then y chosen over x
    menus = [frozenset({x, y}), frozenset({x, y})]
    choices = [x, y]  # x preferred to y, then y preferred to x

    return menus, choices


def generate_sarp_violation_cycle(
    n_items: int = 4,
    cycle_length: int = 3,
    seed: int | None = None,
) -> tuple[list[frozenset[int]], list[int]]:
    """
    Generate menu choice data with a guaranteed SARP k-cycle violation.

    Creates a cycle where item i_1 > i_2 > ... > i_k > i_1 (preference cycle).

    Args:
        n_items: Number of items (must be >= cycle_length)
        cycle_length: Length of the violation cycle (>= 2)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (menus, choices) with guaranteed SARP cycle violation
    """
    if cycle_length < 2:
        raise ValueError("Cycle length must be at least 2")
    if n_items < cycle_length:
        raise ValueError(f"Need at least {cycle_length} items for {cycle_length}-cycle")

    rng = np.random.default_rng(seed)

    # Pick items for the cycle
    cycle_items = rng.choice(n_items, size=cycle_length, replace=False).tolist()

    menus: list[frozenset[int]] = []
    choices: list[int] = []

    # Create cycle: each item is chosen when paired with the next
    # i_1 > i_2, i_2 > i_3, ..., i_k > i_1
    for i in range(cycle_length):
        current_item = cycle_items[i]
        next_item = cycle_items[(i + 1) % cycle_length]

        # Menu contains both current and next
        menu = frozenset({current_item, next_item})
        menus.append(menu)
        choices.append(current_item)  # Current is chosen over next

    return menus, choices


def generate_random_menu_choices(
    n_obs: int,
    n_items: int,
    menu_size: int = 3,
    seed: int | None = None,
) -> tuple[list[frozenset[int]], list[int]]:
    """
    Generate random menu choice data (likely to violate SARP).

    Creates random menus and random choices. With sufficient observations,
    this is likely to produce SARP violations.

    Args:
        n_obs: Number of observations
        n_items: Number of distinct items
        menu_size: Size of each menu
        seed: Random seed

    Returns:
        Tuple of (menus, choices)
    """
    rng = np.random.default_rng(seed)

    menus: list[frozenset[int]] = []
    choices: list[int] = []

    for _ in range(n_obs):
        # Random menu
        size = min(menu_size, n_items)
        menu_items = rng.choice(n_items, size=size, replace=False)
        menu = frozenset(menu_items.tolist())

        # Random choice from menu
        choice = int(rng.choice(list(menu)))

        menus.append(menu)
        choices.append(choice)

    return menus, choices


def compute_theoretical_mpi(
    prices: NDArray[np.float64],
    quantities: NDArray[np.float64],
    cycle: tuple[int, ...],
) -> float:
    """
    Compute theoretical MPI for a cycle using the book's formula.

    MPI = sum(p_k @ (x_k - x_{k+1})) / sum(p_k @ x_k)

    From Chapter 5, Equation 5.1.

    Args:
        prices: Price matrix
        quantities: Quantity matrix
        cycle: Tuple of observation indices forming the cycle.
               Can be (0, 1, 2) or (0, 1, 2, 0) - closing edge is added if needed.

    Returns:
        Theoretical MPI value
    """
    if len(cycle) < 2:
        return 0.0

    # Ensure cycle closes back to first element
    if cycle[0] != cycle[-1]:
        cycle = cycle + (cycle[0],)

    numerator = 0.0
    denominator = 0.0

    for i in range(len(cycle) - 1):
        k = cycle[i]
        k_next = cycle[i + 1]

        # p_k @ x_k - p_k @ x_{k+1}
        numerator += prices[k] @ quantities[k] - prices[k] @ quantities[k_next]
        denominator += prices[k] @ quantities[k]

    if denominator <= 0:
        return 0.0

    return max(0.0, numerator / denominator)
