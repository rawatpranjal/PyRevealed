"""Afriat Efficiency Index (AEI/CCEI) computation.

Supports two methods:
- "discrete" (default): Binary search over the T^2 critical efficiency ratios.
  Finds the EXACT analytical CCEI with zero floating-point approximation error.
- "continuous": Legacy binary search over [0,1] interval with tolerance.
"""

from __future__ import annotations

import time

import numpy as np

from pyrevealed.core.session import ConsumerSession
from pyrevealed.core.result import AEIResult, GARPResult
from pyrevealed.core.types import Cycle
from pyrevealed.graph.transitive_closure import floyd_warshall_transitive_closure


def compute_aei(
    session: ConsumerSession,
    tolerance: float = 1e-6,
    max_iterations: int = 50,
    method: str = "discrete",
) -> AEIResult:
    """
    Compute Afriat Efficiency Index (CCEI).

    The AEI measures how close consumer behavior is to perfect rationality:

        AEI = sup{e in [0,1] : data satisfies GARP with efficiency e}

    where GARP with efficiency e deflates budgets by factor e:
        R_e[i,j] = True iff e * (p_i @ x_i) >= p_i @ x_j

    The critical value e* is guaranteed to equal one of the T^2 efficiency
    ratios E[i,j] / own_exp[i]. The "discrete" method exploits this by
    binary searching over these exact ratios, giving the analytical CCEI
    with zero floating-point error in ~2*log2(T) GARP checks.

    Interpretation:
    - AEI = 1.0: Perfectly consistent (satisfies GARP)
    - AEI = 0.5: Consumer wastes ~50% of budget on inconsistent choices
    - AEI = 0.0: Completely irrational behavior

    Args:
        session: ConsumerSession with prices and quantities
        tolerance: Convergence tolerance (used by "continuous" method)
        max_iterations: Max iterations (used by "continuous" method)
        method: "discrete" (exact, default) or "continuous" (legacy)

    Returns:
        AEIResult with efficiency index and supporting data

    Example:
        >>> import numpy as np
        >>> from pyrevealed import ConsumerSession, compute_aei
        >>> prices = np.array([[1.0, 2.0], [2.0, 1.0]])
        >>> quantities = np.array([[4.0, 1.0], [1.0, 4.0]])
        >>> session = ConsumerSession(prices=prices, quantities=quantities)
        >>> result = compute_aei(session)
        >>> print(f"AEI: {result.efficiency_index:.4f}")
    """
    start_time = time.perf_counter()

    # Try Rust backend for CCEI (binary search over T² ratios in Rust)
    from pyrevealed._rust_backend import HAS_RUST, _rust_analyze_batch
    if HAS_RUST and method == "discrete":
        try:
            import numpy as np
            p = np.ascontiguousarray(session.prices, dtype=np.float64)
            q = np.ascontiguousarray(session.quantities, dtype=np.float64)
            results = _rust_analyze_batch([p], [q], True, False, False, False, False, False, False, tolerance)
            ccei = results[0]["ccei"]
            is_consistent = results[0]["is_garp"]

            from pyrevealed.algorithms.garp import check_garp
            garp_result = check_garp(session, tolerance)

            computation_time = (time.perf_counter() - start_time) * 1000
            return AEIResult(
                efficiency_index=1.0 if is_consistent else ccei,
                is_perfectly_consistent=is_consistent,
                garp_result_at_threshold=garp_result,
                binary_search_iterations=0,
                tolerance=tolerance,
                computation_time_ms=computation_time,
            )
        except Exception:
            pass  # Fall through to Python

    # Python fallback
    from pyrevealed.algorithms.garp import check_garp

    garp_result = check_garp(session)

    if garp_result.is_consistent:
        computation_time = (time.perf_counter() - start_time) * 1000
        return AEIResult(
            efficiency_index=1.0,
            is_perfectly_consistent=True,
            garp_result_at_threshold=garp_result,
            binary_search_iterations=0,
            tolerance=tolerance,
            computation_time_ms=computation_time,
        )

    if method == "discrete":
        aei, iterations, last_result = _discrete_binary_search(session)
    else:
        aei, iterations, last_result = _continuous_binary_search(
            session, tolerance, max_iterations
        )

    if last_result is None:
        _, last_result = _check_garp_at_efficiency(session, 0.0, tolerance=1e-10)

    computation_time = (time.perf_counter() - start_time) * 1000

    return AEIResult(
        efficiency_index=aei,
        is_perfectly_consistent=False,
        garp_result_at_threshold=last_result,
        binary_search_iterations=iterations,
        tolerance=tolerance,
        computation_time_ms=computation_time,
    )


def _discrete_binary_search(
    session: ConsumerSession,
) -> tuple[float, int, GARPResult | None]:
    """
    Find exact CCEI by binary search over discrete efficiency ratios.

    The critical e* that breaks/restores GARP must equal one of the T^2
    ratios E[i,j] / own_exp[i]. Binary searching over this sorted array
    gives the exact answer in ~2*log2(T) iterations.
    """
    E = session.expenditure_matrix
    own_exp = session.own_expenditures

    # Compute all T^2 efficiency ratios: e at which R_e[i,j] flips
    # R_e[i,j] = True iff e * own_exp[i] >= E[i,j]
    # Critical e for (i,j): e = E[i,j] / own_exp[i]
    ratios = E / own_exp[:, np.newaxis]

    # Filter to (0, 1) range, flatten, deduplicate, sort descending
    flat = ratios.ravel()
    mask = (flat > 0) & (flat < 1.0)
    candidates = np.unique(flat[mask])
    candidates = np.sort(candidates)[::-1]  # Descending: try high e first

    if len(candidates) == 0:
        return 0.0, 0, None

    # Binary search over sorted candidates
    lo, hi = 0, len(candidates) - 1
    iterations = 0
    best_e = 0.0
    best_result: GARPResult | None = None

    while lo <= hi:
        mid = (lo + hi) // 2
        e = float(candidates[mid])

        is_consistent, garp_at_e = _check_garp_at_efficiency(
            session, e, tolerance=1e-10
        )
        iterations += 1

        if is_consistent:
            best_e = e
            best_result = garp_at_e
            # Try higher e (lower index in descending array)
            hi = mid - 1
        else:
            # Need lower e (higher index in descending array)
            lo = mid + 1

    return best_e, iterations, best_result


def _continuous_binary_search(
    session: ConsumerSession,
    tolerance: float,
    max_iterations: int,
) -> tuple[float, int, GARPResult | None]:
    """Legacy continuous binary search over [0, 1] interval."""
    e_low = 0.0
    e_high = 1.0
    iterations = 0
    last_consistent_e = 0.0
    last_consistent_result: GARPResult | None = None

    while (e_high - e_low > tolerance) and (iterations < max_iterations):
        e_mid = (e_low + e_high) / 2

        is_consistent, garp_at_e = _check_garp_at_efficiency(
            session, e_mid, tolerance=1e-10
        )

        if is_consistent:
            e_low = e_mid
            last_consistent_e = e_mid
            last_consistent_result = garp_at_e
        else:
            e_high = e_mid

        iterations += 1

    return last_consistent_e, iterations, last_consistent_result


def _check_garp_at_efficiency(
    session: ConsumerSession,
    efficiency: float,
    tolerance: float = 1e-10,
) -> tuple[bool, GARPResult]:
    """
    Check GARP at a given efficiency level e.

    Modified revealed preference relation:
        R_e[i,j] = True iff e * (p_i @ x_i) >= p_i @ x_j

    Args:
        session: ConsumerSession
        efficiency: Efficiency parameter e in [0, 1]
        tolerance: Numerical tolerance

    Returns:
        Tuple of (is_consistent, GARPResult)
    """
    E = session.expenditure_matrix
    own_exp = session.own_expenditures

    # Modified revealed preference with efficiency deflation
    # R_e[i,j] = (e * p_i @ x_i >= p_i @ x_j)
    R_e = (efficiency * own_exp[:, np.newaxis]) >= E - tolerance

    # P_e[i,j] = (e * p_i @ x_i > p_i @ x_j)
    P_e = (efficiency * own_exp[:, np.newaxis]) > E + tolerance
    np.fill_diagonal(P_e, False)

    # Transitive closure
    R_e_star = floyd_warshall_transitive_closure(R_e)

    # GARP violation check
    violation_matrix = R_e_star & P_e.T
    is_consistent = not np.any(violation_matrix)

    # Find violations if any (simplified for efficiency)
    violations: list[Cycle] = []
    if not is_consistent:
        # Just find the first violation pair for the result
        violation_pairs = np.argwhere(violation_matrix)
        if len(violation_pairs) > 0:
            i, j = int(violation_pairs[0, 0]), int(violation_pairs[0, 1])
            violations = [(i, j, i)]  # Simplified cycle representation

    result = GARPResult(
        is_consistent=is_consistent,
        violations=violations,
        direct_revealed_preference=R_e,
        transitive_closure=R_e_star,
        strict_revealed_preference=P_e,
        computation_time_ms=0.0,  # Not tracked for internal calls
    )

    return is_consistent, result


def compute_varian_index(
    session: ConsumerSession,
    tolerance: float = 1e-6,
    max_iterations: int = 50,
) -> float:
    """
    Compute Varian's index of efficiency (alternative to Afriat's).

    Varian's index finds the smallest uniform efficiency e such that
    all observations can be rationalized. It is equivalent to AEI for
    most practical purposes.

    Args:
        session: ConsumerSession
        tolerance: Convergence tolerance
        max_iterations: Maximum binary search iterations

    Returns:
        Efficiency index in [0, 1]

    Note:
        This is functionally equivalent to compute_aei but included for
        completeness as referenced in the literature.
    """
    return compute_aei(session, tolerance, max_iterations).efficiency_index


# =============================================================================
# TECH-FRIENDLY ALIASES
# =============================================================================

# compute_integrity_score: Tech-friendly name for compute_aei
compute_integrity_score = compute_aei
"""
Compute the behavioral integrity score (0-1).

This is the tech-friendly alias for compute_aei (Afriat Efficiency Index).

The integrity score measures consistency with utility maximization:
- 1.0 = Perfectly consistent behavior
- 0.9+ = Minor deviations from rationality
- 0.7-0.9 = Moderate inconsistencies
- <0.7 = Notable violations of rationality

Example:
    >>> from pyrevealed import BehaviorLog, compute_integrity_score
    >>> result = compute_integrity_score(user_log)
    >>> print(f"Integrity: {result.efficiency_index:.2f}")

Returns:
    IntegrityResult with efficiency_index in [0, 1]
"""
