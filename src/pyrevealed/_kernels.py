"""Numba JIT-compiled kernels for PyRevealed algorithms.

This module contains performance-critical functions compiled with Numba
for maximum speed. All functions use `@njit(cache=True)` to cache
compiled code to disk, avoiding recompilation overhead.

Typical speedups: 10-50x over pure Python for O(T^3) algorithms.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


# =============================================================================
# FLOYD-WARSHALL TRANSITIVE CLOSURE
# =============================================================================


@njit(cache=True, parallel=True)
def floyd_warshall_tc_numba(adjacency: np.ndarray) -> np.ndarray:
    """
    Compute transitive closure using parallel Floyd-Warshall algorithm.

    This is the core O(T^3) bottleneck used by GARP, AEI, and many other
    algorithms. Uses parallel Numba for massive speedup on multi-core.

    Scales to 100K+ observations on modern hardware.

    Args:
        adjacency: T x T boolean adjacency matrix (must be np.bool_)

    Returns:
        T x T boolean transitive closure matrix
    """
    T = adjacency.shape[0]
    closure = adjacency.copy()

    # Ensure reflexivity
    for i in range(T):
        closure[i, i] = True

    # Parallel Floyd-Warshall
    # For each intermediate vertex k, parallelize over rows i
    for k in range(T):
        # Get column k and row k for this iteration
        col_k = closure[:, k].copy()  # Which rows can reach k
        row_k = closure[k, :].copy()  # Which cols k can reach

        # Parallel update: if i->k and k->j, then i->j
        for i in prange(T):
            if col_k[i]:  # i can reach k
                for j in range(T):
                    if row_k[j]:  # k can reach j
                        closure[i, j] = True

    return closure


@njit(cache=True)
def floyd_warshall_tc_serial(adjacency: np.ndarray) -> np.ndarray:
    """
    Serial Floyd-Warshall for small matrices (T < 500).

    Faster than parallel version for small inputs due to no threading overhead.
    """
    T = adjacency.shape[0]
    closure = adjacency.copy()

    for i in range(T):
        closure[i, i] = True

    for k in range(T):
        for i in range(T):
            if closure[i, k]:
                for j in range(T):
                    if closure[k, j]:
                        closure[i, j] = True

    return closure


@njit(cache=True)
def floyd_warshall_max_log_numba(
    log_ratios: np.ndarray,
    adjacency: np.ndarray,
) -> np.ndarray:
    """
    Modified Floyd-Warshall for maximum log-product paths (HARP).

    Instead of boolean reachability, tracks the maximum sum of log-ratios
    on any path from i to j. Used for homothetic preference testing.

    Args:
        log_ratios: T x T matrix of log(expenditure ratios)
        adjacency: T x T boolean matrix of direct edges

    Returns:
        T x T matrix where result[i,j] = max sum of log_ratios on path i->j
        -inf if no path exists
    """
    T = log_ratios.shape[0]
    max_log = np.empty((T, T), dtype=np.float64)

    # Initialize: direct edges have their log_ratio, no edge = -inf
    for i in range(T):
        for j in range(T):
            if adjacency[i, j]:
                max_log[i, j] = log_ratios[i, j]
            else:
                max_log[i, j] = -np.inf

    # Diagonal: start at 0 (will accumulate cycle costs)
    for i in range(T):
        max_log[i, i] = 0.0

    # Floyd-Warshall: maximize log-sum through intermediate vertices
    for k in range(T):
        for i in range(T):
            for j in range(T):
                via_k = max_log[i, k] + max_log[k, j]
                if via_k > max_log[i, j]:
                    max_log[i, j] = via_k

    return max_log


# =============================================================================
# BFS PATH RECONSTRUCTION
# =============================================================================


@njit(cache=True)
def bfs_find_path_numba(
    adjacency: np.ndarray,
    start: np.int64,
    end: np.int64,
) -> np.ndarray:
    """
    BFS to find shortest path from start to end.

    Returns path array with -1 as sentinel for "no path found".
    The path ends with `start` again to complete the cycle.

    Args:
        adjacency: T x T boolean adjacency matrix
        start: Starting node
        end: Ending node

    Returns:
        Path array ending with start (to complete cycle).
        First element is -1 if no path found.
    """
    T = adjacency.shape[0]
    max_path_len = T + 2

    # Arrays for BFS
    queue = np.empty(T * T, dtype=np.int64)
    parent = np.full(T, -1, dtype=np.int64)
    visited = np.zeros(T, dtype=np.bool_)

    # Initialize with start
    queue[0] = start
    visited[start] = True
    head = 0
    tail = 1

    found = False

    while head < tail and not found:
        current = queue[head]
        head += 1

        if current == end and parent[end] != -1:
            found = True
            break

        for next_node in range(T):
            if adjacency[current, next_node] and not visited[next_node]:
                visited[next_node] = True
                parent[next_node] = current
                queue[tail] = next_node
                tail += 1

    if not found:
        # No path found
        result = np.empty(1, dtype=np.int64)
        result[0] = -1
        return result

    # Reconstruct path
    path = np.empty(max_path_len, dtype=np.int64)
    path_len = 0

    # Trace back from end to start
    node = end
    while node != start:
        path[path_len] = node
        path_len += 1
        node = parent[node]
        if path_len > T:
            # Safety: shouldn't happen, but prevent infinite loop
            result = np.empty(1, dtype=np.int64)
            result[0] = -1
            return result

    path[path_len] = start
    path_len += 1

    # Reverse path (we built it backwards)
    reversed_path = np.empty(path_len + 1, dtype=np.int64)
    for i in range(path_len):
        reversed_path[i] = path[path_len - 1 - i]
    # Add start at end to complete cycle
    reversed_path[path_len] = start

    return reversed_path


@njit(cache=True)
def bfs_find_cycle_numba(
    adjacency: np.ndarray,
    start: np.int64,
) -> np.ndarray:
    """
    BFS to find shortest cycle through the given node.

    Args:
        adjacency: T x T boolean adjacency matrix
        start: Node to find cycle through

    Returns:
        Cycle array ending with start. First element is -1 if no cycle.
    """
    T = adjacency.shape[0]
    max_path_len = T + 2

    # Arrays for BFS
    queue = np.empty(T * T, dtype=np.int64)
    parent = np.full(T, -1, dtype=np.int64)
    visited = np.zeros(T, dtype=np.bool_)

    # Initialize with start's neighbors
    head = 0
    tail = 0

    for next_node in range(T):
        if adjacency[start, next_node]:
            if next_node == start:
                # Self-loop
                result = np.array([start, start], dtype=np.int64)
                return result
            visited[next_node] = True
            parent[next_node] = start
            queue[tail] = next_node
            tail += 1

    found = False
    end_node = -1

    while head < tail and not found:
        current = queue[head]
        head += 1

        for next_node in range(T):
            if adjacency[current, next_node]:
                if next_node == start:
                    # Found cycle back to start
                    found = True
                    end_node = current
                    break
                if not visited[next_node]:
                    visited[next_node] = True
                    parent[next_node] = current
                    queue[tail] = next_node
                    tail += 1

    if not found:
        result = np.empty(1, dtype=np.int64)
        result[0] = -1
        return result

    # Reconstruct path from end_node back to start
    path = np.empty(max_path_len, dtype=np.int64)
    path_len = 0

    node = end_node
    while node != start:
        path[path_len] = node
        path_len += 1
        node = parent[node]
        if path_len > T:
            result = np.empty(1, dtype=np.int64)
            result[0] = -1
            return result

    path[path_len] = start
    path_len += 1

    # Reverse and add start at end
    reversed_path = np.empty(path_len + 1, dtype=np.int64)
    for i in range(path_len):
        reversed_path[i] = path[path_len - 1 - i]
    reversed_path[path_len] = start

    return reversed_path


# =============================================================================
# GARP VIOLATION DETECTION
# =============================================================================


@njit(cache=True)
def find_violation_pairs_numba(
    R_star: np.ndarray,
    P: np.ndarray,
) -> np.ndarray:
    """
    Find all (i, j) pairs where R*[i,j] AND P[j,i].

    Args:
        R_star: Transitive closure of R
        P: Strict preference matrix

    Returns:
        2D array of shape (N, 2) with violation pairs.
        Returns empty array if no violations.
    """
    T = R_star.shape[0]

    # First pass: count violations
    count = 0
    for i in range(T):
        for j in range(T):
            if R_star[i, j] and P[j, i]:
                count += 1

    if count == 0:
        return np.empty((0, 2), dtype=np.int64)

    # Second pass: collect violations
    result = np.empty((count, 2), dtype=np.int64)
    idx = 0
    for i in range(T):
        for j in range(T):
            if R_star[i, j] and P[j, i]:
                result[idx, 0] = i
                result[idx, 1] = j
                idx += 1

    return result


# =============================================================================
# QUASILINEARITY CYCLE SUMS
# =============================================================================


@njit(cache=True)
def compute_cycle2_sums_numba(
    S: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.int64]:
    """
    Compute sums for all 2-cycles and find violations.

    A 2-cycle i->j->i has sum S[i,j] + S[j,i].
    Violation if sum < -tolerance.

    Args:
        S: Surplus matrix S[i,j] = E[i,j] - E[i,i]
        tolerance: Tolerance for violation detection

    Returns:
        Tuple of (violation_sums, violation_pairs, count)
    """
    T = S.shape[0]
    max_violations = T * T

    violation_sums = np.empty(max_violations, dtype=np.float64)
    violation_pairs = np.empty((max_violations, 2), dtype=np.int64)
    count = 0

    for i in range(T):
        for j in range(i + 1, T):
            cycle_sum = S[i, j] + S[j, i]
            if cycle_sum < -tolerance:
                violation_sums[count] = cycle_sum
                violation_pairs[count, 0] = i
                violation_pairs[count, 1] = j
                count += 1

    return violation_sums[:count], violation_pairs[:count], count


@njit(cache=True)
def compute_cycle3_sums_numba(
    S: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.int64]:
    """
    Compute sums for all 3-cycles and find violations.

    A 3-cycle i->j->k->i has sum S[i,j] + S[j,k] + S[k,i].
    Violation if sum < -tolerance.

    Args:
        S: Surplus matrix S[i,j] = E[i,j] - E[i,i]
        tolerance: Tolerance for violation detection

    Returns:
        Tuple of (violation_sums, violation_triples, count)
    """
    T = S.shape[0]
    max_violations = T * T * T

    violation_sums = np.empty(max_violations, dtype=np.float64)
    violation_triples = np.empty((max_violations, 3), dtype=np.int64)
    count = 0

    for i in range(T):
        for j in range(T):
            if j == i:
                continue
            for k in range(T):
                if k == i or k == j:
                    continue
                cycle_sum = S[i, j] + S[j, k] + S[k, i]
                if cycle_sum < -tolerance:
                    violation_sums[count] = cycle_sum
                    violation_triples[count, 0] = i
                    violation_triples[count, 1] = j
                    violation_triples[count, 2] = k
                    count += 1

    return violation_sums[:count], violation_triples[:count], count


# =============================================================================
# GROSS SUBSTITUTES PAIRWISE CHECKS
# =============================================================================


@njit(cache=True)
def check_gross_substitutes_numba(
    prices: np.ndarray,
    quantities: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.int64]:
    """
    Check gross substitutes property for all observation pairs.

    For goods i, j: if price of i increases while j stays same,
    quantity of j should not decrease (substitutes replace each other).

    Args:
        prices: T x N price matrix
        quantities: T x N quantity matrix
        tolerance: Tolerance for comparisons

    Returns:
        Tuple of (violation_pairs, violation_goods, count)
        violation_pairs[k] = (t1, t2) - the observation pair
        violation_goods[k] = (i, j) - the goods that violate GS
    """
    T, N = prices.shape
    max_violations = T * T * N * N

    violation_obs = np.empty((max_violations, 2), dtype=np.int64)
    violation_goods = np.empty((max_violations, 2), dtype=np.int64)
    count = 0

    for t1 in range(T):
        for t2 in range(T):
            if t1 == t2:
                continue

            # Check each pair of goods
            for i in range(N):
                # Check if price of good i increased
                if prices[t2, i] > prices[t1, i] + tolerance:
                    for j in range(N):
                        if i == j:
                            continue
                        # Check if price of good j stayed same or decreased
                        if prices[t2, j] <= prices[t1, j] + tolerance:
                            # Quantity of j should not decrease (gross substitutes)
                            if quantities[t2, j] < quantities[t1, j] - tolerance:
                                violation_obs[count, 0] = t1
                                violation_obs[count, 1] = t2
                                violation_goods[count, 0] = i
                                violation_goods[count, 1] = j
                                count += 1

    return violation_obs[:count], violation_goods[:count], count


# =============================================================================
# BRONARS RANDOM BUNDLE GENERATION (PARALLEL)
# =============================================================================


@njit(cache=True, parallel=True)
def compute_random_expenditures_batch_numba(
    prices: np.ndarray,
    random_quantities: np.ndarray,
) -> np.ndarray:
    """
    Compute expenditure matrices for batch of random simulations.

    Args:
        prices: T x N price matrix
        random_quantities: (n_sim, T, N) random quantity matrices

    Returns:
        (n_sim, T, T) expenditure matrices
    """
    n_sim, T, N = random_quantities.shape
    result = np.empty((n_sim, T, T), dtype=np.float64)

    for sim in prange(n_sim):
        for i in range(T):
            for j in range(T):
                # E[i,j] = p_i @ x_j
                total = 0.0
                for k in range(N):
                    total += prices[i, k] * random_quantities[sim, j, k]
                result[sim, i, j] = total

    return result


@njit(cache=True)
def check_garp_fast_numba(
    expenditure_matrix: np.ndarray,
    tolerance: float,
) -> np.bool_:
    """
    Fast GARP consistency check without cycle reconstruction.

    Just returns True/False, no violation details.

    Args:
        expenditure_matrix: T x T expenditure matrix
        tolerance: Numerical tolerance

    Returns:
        True if consistent, False if violated
    """
    T = expenditure_matrix.shape[0]

    # Own expenditures (diagonal)
    own_exp = np.empty(T, dtype=np.float64)
    for i in range(T):
        own_exp[i] = expenditure_matrix[i, i]

    # Build R and P
    R = np.empty((T, T), dtype=np.bool_)
    P = np.empty((T, T), dtype=np.bool_)

    for i in range(T):
        for j in range(T):
            R[i, j] = own_exp[i] >= expenditure_matrix[i, j] - tolerance
            P[i, j] = own_exp[i] > expenditure_matrix[i, j] + tolerance

    # Remove self-loops from P
    for i in range(T):
        P[i, i] = False

    # Transitive closure of R
    R_star = floyd_warshall_tc_numba(R)

    # Check for violation: R*[i,j] AND P[j,i]
    for i in range(T):
        for j in range(T):
            if R_star[i, j] and P[j, i]:
                return False

    return True


# =============================================================================
# UTILITY CONSTRAINT BUILDING
# =============================================================================


@njit(cache=True)
def build_afriat_constraints_numba(
    expenditure_matrix: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build Afriat inequality constraint matrices for utility recovery.

    For each pair (i,j) where i is revealed preferred to j, we need:
    u_i >= u_j + lambda_j * (E[j,i] - E[j,j])

    Args:
        expenditure_matrix: T x T expenditure matrix
        tolerance: Numerical tolerance

    Returns:
        Tuple of (A_ub, b_ub) for scipy.optimize.linprog
    """
    T = expenditure_matrix.shape[0]

    # Own expenditures
    own_exp = np.empty(T, dtype=np.float64)
    for i in range(T):
        own_exp[i] = expenditure_matrix[i, i]

    # Count revealed preferences
    count = 0
    for i in range(T):
        for j in range(T):
            if i != j and own_exp[i] >= expenditure_matrix[i, j] - tolerance:
                count += 1

    # Build constraint matrix
    # Variables: [u_0, ..., u_{T-1}, lambda_0, ..., lambda_{T-1}]
    # Constraint: u_j - u_i + lambda_j * (E[j,i] - E[j,j]) <= 0
    # Which is: -u_i + u_j + lambda_j * (E[j,i] - e_j) <= 0

    A_ub = np.zeros((count, 2 * T), dtype=np.float64)
    b_ub = np.zeros(count, dtype=np.float64)

    idx = 0
    for i in range(T):
        for j in range(T):
            if i != j and own_exp[i] >= expenditure_matrix[i, j] - tolerance:
                # i R j: constraint is u_j - u_i + lambda_j * (E[j,i] - e_j) <= 0
                A_ub[idx, i] = -1.0  # -u_i
                A_ub[idx, j] = 1.0  # +u_j
                A_ub[idx, T + j] = expenditure_matrix[j, i] - own_exp[j]  # lambda_j coeff
                b_ub[idx] = 0.0
                idx += 1

    return A_ub, b_ub


# =============================================================================
# VEI (PER-OBSERVATION EFFICIENCY) HELPERS
# =============================================================================


@njit(cache=True)
def compute_efficiency_bounds_numba(
    expenditure_matrix: np.ndarray,
    R_star: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    """
    Compute per-observation efficiency bounds.

    For each observation i, find the minimum efficiency e_i such that
    deflating expenditure at i by e_i removes all violations involving i.

    Args:
        expenditure_matrix: T x T expenditure matrix
        R_star: Transitive closure of revealed preference
        tolerance: Numerical tolerance

    Returns:
        Array of efficiency bounds for each observation
    """
    T = expenditure_matrix.shape[0]

    own_exp = np.empty(T, dtype=np.float64)
    for i in range(T):
        own_exp[i] = expenditure_matrix[i, i]

    # Strict preference
    P = np.empty((T, T), dtype=np.bool_)
    for i in range(T):
        for j in range(T):
            P[i, j] = own_exp[i] > expenditure_matrix[i, j] + tolerance
    for i in range(T):
        P[i, i] = False

    # For each observation, compute efficiency bound
    bounds = np.ones(T, dtype=np.float64)

    for i in range(T):
        for j in range(T):
            # If R*[i,j] and P[j,i], we have a violation
            if R_star[i, j] and P[j, i]:
                # Need to deflate either at i or j
                # Efficiency at j: e_j such that e_j * E[j,j] <= E[j,i]
                if own_exp[j] > tolerance:
                    e_j = expenditure_matrix[j, i] / own_exp[j]
                    if e_j < bounds[j]:
                        bounds[j] = e_j

    return bounds


# =============================================================================
# SEPARABILITY CORRELATION HELPERS
# =============================================================================


@njit(cache=True)
def compute_correlation_matrix_numba(
    quantities: np.ndarray,
) -> np.ndarray:
    """
    Compute pairwise correlation matrix for quantities.

    Args:
        quantities: T x N quantity matrix

    Returns:
        N x N correlation matrix
    """
    T, N = quantities.shape

    # Compute means
    means = np.empty(N, dtype=np.float64)
    for j in range(N):
        total = 0.0
        for t in range(T):
            total += quantities[t, j]
        means[j] = total / T

    # Compute standard deviations
    stds = np.empty(N, dtype=np.float64)
    for j in range(N):
        total = 0.0
        for t in range(T):
            diff = quantities[t, j] - means[j]
            total += diff * diff
        stds[j] = np.sqrt(total / T)

    # Compute correlation matrix
    corr = np.empty((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            if i == j:
                corr[i, j] = 1.0
            elif stds[i] < 1e-10 or stds[j] < 1e-10:
                corr[i, j] = 0.0
            else:
                cov = 0.0
                for t in range(T):
                    cov += (quantities[t, i] - means[i]) * (quantities[t, j] - means[j])
                cov /= T
                corr[i, j] = cov / (stds[i] * stds[j])

    return corr


# =============================================================================
# RISK PREFERENCE UTILITIES
# =============================================================================


@njit(cache=True)
def crra_utility_batch_numba(
    consumption: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """
    Compute CRRA (Constant Relative Risk Aversion) utility.

    u(c) = c^(1-gamma) / (1-gamma) if gamma != 1
    u(c) = log(c) if gamma == 1

    Args:
        consumption: Array of consumption values
        gamma: Risk aversion parameter

    Returns:
        Array of utility values
    """
    n = consumption.shape[0]
    result = np.empty(n, dtype=np.float64)

    if abs(gamma - 1.0) < 1e-10:
        # Log utility
        for i in range(n):
            if consumption[i] > 0:
                result[i] = np.log(consumption[i])
            else:
                result[i] = -np.inf
    else:
        # Power utility
        exp = 1.0 - gamma
        for i in range(n):
            if consumption[i] > 0:
                result[i] = (consumption[i] ** exp) / exp
            else:
                result[i] = -np.inf

    return result


# =============================================================================
# SPATIAL/IDEAL POINT DISTANCE COMPUTATION
# =============================================================================


@njit(cache=True)
def compute_distances_batch_numba(
    ideal_point: np.ndarray,
    alternatives: np.ndarray,
) -> np.ndarray:
    """
    Compute Euclidean distances from ideal point to alternatives.

    Args:
        ideal_point: 1D array of length D (dimension)
        alternatives: 2D array of shape (M, D)

    Returns:
        1D array of M distances
    """
    M, D = alternatives.shape
    result = np.empty(M, dtype=np.float64)

    for i in range(M):
        dist_sq = 0.0
        for d in range(D):
            diff = ideal_point[d] - alternatives[i, d]
            dist_sq += diff * diff
        result[i] = np.sqrt(dist_sq)

    return result
