"""Production theory and firm behavior analysis.

Tests profit maximization and cost minimization consistency for firm data.
Based on Chapter 15 of Chambers & Echenique (2016) "Revealed Preference Theory".

The production analogue of GARP tests whether observed input-output choices
are consistent with profit maximization.

Tech-Friendly Names (Primary):
    - test_profit_maximization(): Test production GARP
    - check_cost_minimization(): Test cost minimization consistency
    - estimate_returns_to_scale(): Estimate returns to scale

Economics Names (Legacy Aliases):
    - check_production_garp() -> test_profit_maximization()
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from pyrevealed.core.result import ProductionGARPResult
from pyrevealed.core.types import Cycle
from pyrevealed.graph.transitive_closure import floyd_warshall_transitive_closure

if TYPE_CHECKING:
    from pyrevealed.core.session import ProductionLog


def test_profit_maximization(
    log: "ProductionLog",
    tolerance: float = 1e-6,
) -> ProductionGARPResult:
    """
    Test if firm behavior is consistent with profit maximization.

    Production GARP: For observations i and j, if firm at i could have
    achieved the output of j at the prices of i, and the profit would
    have been at least as high, then the reverse should not also hold.

    The test constructs a revealed preferred-at-least-as-profitable
    relation and checks for cycles.

    Args:
        log: ProductionLog with input/output prices and quantities
        tolerance: Numerical tolerance

    Returns:
        ProductionGARPResult with profit maximization analysis

    Example:
        >>> from pyrevealed import ProductionLog, test_profit_maximization
        >>> result = test_profit_maximization(firm_data)
        >>> if result.is_profit_maximizing:
        ...     print("Firm behavior is profit-maximizing consistent")
        >>> else:
        ...     print(f"Found {result.num_violations} violations")

    References:
        Chambers & Echenique (2016), Chapter 15
        Varian, H.R. (1984). "The Nonparametric Approach to Production Analysis"
    """
    start_time = time.perf_counter()

    T = log.num_observations

    # Construct revealed profitability relation
    # R[i,j] = True if choosing (x_j, y_j) at prices (w_i, p_i) would give
    # at least as much profit as (x_i, y_i), AND i's actual profit >= j's counterfactual
    R = np.zeros((T, T), dtype=np.bool_)

    # Vectorized computation of actual profits
    actual_profits = log.total_revenue - log.total_cost  # Shape: (T,)

    # Vectorized computation of counterfactual profits
    # counterfactual_revenue[i,j] = output_prices[i] @ output_quantities[j]
    # counterfactual_cost[i,j] = input_prices[i] @ input_quantities[j]
    counterfactual_revenue = log.output_prices @ log.output_quantities.T  # Shape: (T, T)
    counterfactual_cost = log.input_prices @ log.input_quantities.T  # Shape: (T, T)
    counterfactual_profit = counterfactual_revenue - counterfactual_cost

    # R[i,j] = True if actual_profit[i] >= counterfactual_profit[i,j]
    # Using broadcasting: actual_profits[:, np.newaxis] has shape (T, 1)
    R = actual_profits[:, np.newaxis] >= counterfactual_profit - tolerance

    # Exclude self-comparisons
    np.fill_diagonal(R, False)

    # Compute transitive closure
    R_star = floyd_warshall_transitive_closure(R)

    # Find violations: cycles in R_star
    violations: list[Cycle] = []
    visited_pairs: set[frozenset[int]] = set()

    for i in range(T):
        for j in range(i + 1, T):
            if R_star[i, j] and R_star[j, i]:
                pair = frozenset({i, j})
                if pair not in visited_pairs:
                    visited_pairs.add(pair)
                    violations.append((i, j))

    is_profit_maximizing = len(violations) == 0

    # Compute cost efficiency score
    cost_efficiency = _compute_cost_efficiency(log, tolerance)

    # Estimate returns to scale
    returns_to_scale = estimate_returns_to_scale(log)

    # Compute overall profit efficiency
    profit_efficiency = _compute_profit_efficiency(log, tolerance)

    # Per-input and per-output efficiency
    input_efficiency = _compute_input_efficiency(log, tolerance)
    output_efficiency = _compute_output_efficiency(log, tolerance)

    # Technical efficiency
    technical_efficiency = _compute_technical_efficiency(log)

    computation_time = (time.perf_counter() - start_time) * 1000

    return ProductionGARPResult(
        is_profit_maximizing=is_profit_maximizing,
        violations=violations,
        cost_efficiency_score=cost_efficiency,
        returns_to_scale=returns_to_scale,
        profit_efficiency=profit_efficiency,
        input_efficiency_vector=input_efficiency,
        output_efficiency_vector=output_efficiency,
        technical_efficiency=technical_efficiency,
        computation_time_ms=computation_time,
    )


def check_cost_minimization(
    log: "ProductionLog",
    tolerance: float = 1e-6,
) -> dict:
    """
    Test if firm behavior is consistent with cost minimization.

    Cost minimization is the dual of profit maximization. The firm
    should choose inputs that minimize cost for a given output level.

    Args:
        log: ProductionLog with input/output prices and quantities
        tolerance: Numerical tolerance

    Returns:
        Dictionary with cost minimization analysis
    """
    T = log.num_observations

    # Construct revealed cost relation
    # R[i,j] = True if using j's inputs at i's input prices costs at least
    # as much as i's actual cost, AND outputs are comparable
    violations = []

    for i in range(T):
        for j in range(T):
            if i == j:
                continue

            # Actual cost at i
            cost_i = log.total_cost[i]

            # Counterfactual: using j's inputs at i's prices
            counterfactual_cost = np.dot(log.input_prices[i], log.input_quantities[j])

            # Check if outputs are comparable (j produces at least as much for ALL outputs)
            # Critical fix: use element-wise comparison, not sum-based
            # For multi-output firms, we need j to produce at least as much of EACH output
            output_comparable = np.all(
                log.output_quantities[j] >= log.output_quantities[i] - tolerance
            )

            if output_comparable:
                # j's inputs could produce at least as much output
                # If counterfactual cost < actual cost, violation
                if counterfactual_cost < cost_i - tolerance:
                    violations.append((i, j))

    return {
        "is_cost_minimizing": len(violations) == 0,
        "violations": violations,
        "num_violations": len(violations),
    }


def estimate_returns_to_scale(
    log: "ProductionLog",
) -> str:
    """
    Estimate returns to scale from production data.

    Returns to scale indicates how output changes when all inputs
    are scaled proportionally:
    - Increasing: doubling inputs more than doubles output
    - Constant: doubling inputs exactly doubles output
    - Decreasing: doubling inputs less than doubles output

    Args:
        log: ProductionLog with input/output quantities

    Returns:
        One of "increasing", "constant", "decreasing", or "variable"
    """
    T = log.num_observations

    if T < 3:
        return "variable"

    # Compute input and output indices
    input_indices = np.sum(log.input_quantities, axis=1)
    output_indices = np.sum(log.output_quantities, axis=1)

    if np.std(input_indices) < 1e-10:
        return "variable"

    # Estimate output elasticity with respect to inputs
    # Using log-log regression
    log_inputs = np.log(input_indices + 1e-10)
    log_outputs = np.log(output_indices + 1e-10)

    # Simple linear regression
    cov = np.cov(log_inputs, log_outputs)
    if cov[0, 0] > 1e-10:
        elasticity = cov[0, 1] / cov[0, 0]
    else:
        elasticity = 1.0

    # Classify returns to scale
    if elasticity > 1.1:
        return "increasing"
    elif elasticity < 0.9:
        return "decreasing"
    else:
        return "constant"


def compute_technical_efficiency(
    log: "ProductionLog",
    method: str = "output_oriented",
) -> NDArray[np.float64]:
    """
    Compute technical efficiency for each observation.

    Technical efficiency measures how close the firm operates to the
    production frontier. A score of 1.0 means fully efficient.

    Args:
        log: ProductionLog with input/output data
        method: "output_oriented" or "input_oriented"

    Returns:
        Array of efficiency scores (one per observation)
    """
    T = log.num_observations

    efficiencies = np.ones(T)

    for i in range(T):
        max_efficiency = 1.0

        for j in range(T):
            if i == j:
                continue

            if method == "output_oriented":
                # Can j produce more output with same or fewer inputs?
                input_ratio = np.sum(log.input_quantities[j]) / max(np.sum(log.input_quantities[i]), 1e-10)
                output_ratio = np.sum(log.output_quantities[j]) / max(np.sum(log.output_quantities[i]), 1e-10)

                if input_ratio <= 1.0 and output_ratio > 1.0:
                    # j is more efficient
                    efficiency = 1.0 / output_ratio
                    max_efficiency = min(max_efficiency, efficiency)

            else:  # input_oriented
                # Can j produce same output with fewer inputs?
                input_ratio = np.sum(log.input_quantities[j]) / max(np.sum(log.input_quantities[i]), 1e-10)
                output_ratio = np.sum(log.output_quantities[j]) / max(np.sum(log.output_quantities[i]), 1e-10)

                if output_ratio >= 1.0 and input_ratio < 1.0:
                    # j uses fewer inputs for same output
                    max_efficiency = min(max_efficiency, input_ratio)

        efficiencies[i] = max_efficiency

    return efficiencies


def _compute_cost_efficiency(
    log: "ProductionLog",
    tolerance: float,
) -> float:
    """Compute overall cost efficiency score."""
    T = log.num_observations

    efficient_count = 0
    for i in range(T):
        is_efficient = True
        for j in range(T):
            if i == j:
                continue

            # Can j produce same output at lower cost?
            output_comparable = np.all(log.output_quantities[j] >= log.output_quantities[i] - tolerance)
            counterfactual_cost = np.dot(log.input_prices[i], log.input_quantities[j])

            if output_comparable and counterfactual_cost < log.total_cost[i] - tolerance:
                is_efficient = False
                break

        if is_efficient:
            efficient_count += 1

    return efficient_count / T if T > 0 else 1.0


def _compute_profit_efficiency(
    log: "ProductionLog",
    tolerance: float,
) -> float:
    """Compute overall profit efficiency score."""
    profits = log.profit

    if len(profits) == 0:
        return 1.0

    max_profit = np.max(profits)
    if max_profit <= 0:
        return 1.0

    mean_profit = np.mean(profits)
    return max(0.0, min(1.0, mean_profit / max_profit))


def _compute_input_efficiency(
    log: "ProductionLog",
    tolerance: float,
) -> NDArray[np.float64]:
    """Compute per-input efficiency scores."""
    N_inputs = log.num_inputs

    efficiencies = np.ones(N_inputs)

    for k in range(N_inputs):
        # Compare input k usage relative to output
        input_k = log.input_quantities[:, k]
        total_output = np.sum(log.output_quantities, axis=1)

        if np.std(input_k) < tolerance:
            continue

        # Efficiency = how well input usage correlates with output
        corr = np.corrcoef(input_k, total_output)[0, 1]
        efficiencies[k] = max(0.0, corr) if not np.isnan(corr) else 1.0

    return efficiencies


def _compute_output_efficiency(
    log: "ProductionLog",
    tolerance: float,
) -> NDArray[np.float64]:
    """Compute per-output efficiency scores."""
    N_outputs = log.num_outputs

    efficiencies = np.ones(N_outputs)

    for k in range(N_outputs):
        # Compare output k relative to input usage
        output_k = log.output_quantities[:, k]
        total_input = np.sum(log.input_quantities, axis=1)

        if np.std(total_input) < tolerance:
            continue

        # Efficiency = how well output correlates with input
        corr = np.corrcoef(output_k, total_input)[0, 1]
        efficiencies[k] = max(0.0, corr) if not np.isnan(corr) else 1.0

    return efficiencies


def _compute_technical_efficiency(
    log: "ProductionLog",
) -> float:
    """Compute overall technical efficiency (mean of observations)."""
    efficiencies = compute_technical_efficiency(log)
    return float(np.mean(efficiencies))


# =============================================================================
# LEGACY ALIASES
# =============================================================================

check_production_garp = test_profit_maximization
"""Legacy alias: use test_profit_maximization instead."""

test_cost_minimization = check_cost_minimization
"""Legacy alias: use check_cost_minimization instead."""
