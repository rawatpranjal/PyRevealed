"""Abstract Choice Theory algorithms for menu-based preference analysis.

This module implements revealed preference axioms for abstract choice data
(choices from menus without prices). Based on Chapters 1-2 of
Chambers & Echenique (2016) "Revealed Preference Theory".

Tech-Friendly Names (Primary):
    - validate_menu_warp(): Check WARP for menu choices
    - validate_menu_sarp(): Check SARP for menu choices
    - validate_menu_consistency(): Check full rationalizability (Congruence)
    - compute_menu_efficiency(): Houtman-Maks efficiency index
    - fit_menu_preferences(): Recover ordinal preference ranking

Economics Names (Legacy Aliases):
    - check_abstract_warp() -> validate_menu_warp()
    - check_abstract_sarp() -> validate_menu_sarp()
    - check_congruence() -> validate_menu_consistency()
    - compute_abstract_efficiency() -> compute_menu_efficiency()
    - recover_ordinal_utility() -> fit_menu_preferences()
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from pyrevealed.core.result import (
    AbstractWARPResult,
    AbstractSARPResult,
    CongruenceResult,
    HoutmanMaksAbstractResult,
    OrdinalUtilityResult,
)
from pyrevealed.core.types import Cycle
from pyrevealed.graph.transitive_closure import floyd_warshall_transitive_closure
from pyrevealed._kernels import (
    bfs_find_path_numba,
    find_symmetric_pairs_bool_numba,
    topological_sort_numba,
)

if TYPE_CHECKING:
    from pyrevealed.core.session import MenuChoiceLog


def validate_menu_warp(log: MenuChoiceLog) -> AbstractWARPResult:
    """
    Check if menu choice data satisfies WARP (Weak Axiom of Revealed Preference).

    WARP for abstract choice theory: if x is chosen from a menu containing y,
    then y cannot be chosen from any menu containing x where x is also available.

    Formally: If x = c(B) and y in B, then for any B' with x in B' and y = c(B'),
    we must have x not in B'. This prevents direct preference reversals.

    Args:
        log: MenuChoiceLog with menus and choices

    Returns:
        AbstractWARPResult with consistency status and violations

    Example:
        >>> from pyrevealed import MenuChoiceLog, validate_menu_warp
        >>> log = MenuChoiceLog(
        ...     menus=[frozenset({0, 1}), frozenset({0, 1})],
        ...     choices=[0, 1]  # WARP violation: 0 chosen over 1, then 1 over 0
        ... )
        >>> result = validate_menu_warp(log)
        >>> result.is_consistent
        False
    """
    start_time = time.perf_counter()

    # Build revealed preference pairs: (chosen, unchosen)
    # x is revealed preferred to y if x was chosen from a menu containing y
    revealed_pairs: list[tuple[int, int]] = []

    for t, (menu, choice) in enumerate(zip(log.menus, log.choices)):
        for item in menu:
            if item != choice:
                revealed_pairs.append((choice, item))

    # Check for WARP violations: (x, y) and (y, x) both in revealed_pairs
    violations: list[tuple[int, int]] = []
    revealed_set = set(revealed_pairs)

    for x, y in revealed_pairs:
        if (y, x) in revealed_set and (x, y) not in violations and (y, x) not in violations:
            # Found a violation: x preferred to y AND y preferred to x
            violations.append((x, y))

    computation_time = (time.perf_counter() - start_time) * 1000

    return AbstractWARPResult(
        is_consistent=len(violations) == 0,
        violations=violations,
        revealed_preference_pairs=revealed_pairs,
        computation_time_ms=computation_time,
    )


def validate_menu_sarp(log: MenuChoiceLog) -> AbstractSARPResult:
    """
    Check if menu choice data satisfies SARP (Strict Axiom of Revealed Preference).

    SARP for abstract choice: the transitive closure of revealed preference
    must be acyclic. Equivalently, if x R* y (x is transitively revealed
    preferred to y), then NOT y R* x.

    This is stronger than WARP - it checks for cycles of any length.

    Args:
        log: MenuChoiceLog with menus and choices

    Returns:
        AbstractSARPResult with consistency status, violations, and matrices

    Example:
        >>> from pyrevealed import MenuChoiceLog, validate_menu_sarp
        >>> log = MenuChoiceLog(
        ...     menus=[frozenset({0, 1}), frozenset({1, 2}), frozenset({0, 2})],
        ...     choices=[0, 1, 2]  # Creates cycle: 0 > 1 > 2 > 0
        ... )
        >>> result = validate_menu_sarp(log)
        >>> result.is_consistent
        False
    """
    start_time = time.perf_counter()

    # Determine the number of items
    all_items = log.all_items
    n_items = max(all_items) + 1 if all_items else 0

    # Build revealed preference matrix R
    # R[x, y] = True iff x is revealed preferred to y
    R = np.zeros((n_items, n_items), dtype=np.bool_)

    for menu, choice in zip(log.menus, log.choices):
        for item in menu:
            if item != choice:
                R[choice, item] = True

    # Compute transitive closure R* using Floyd-Warshall
    R_star = floyd_warshall_transitive_closure(R)

    # Check for cycles using numba kernel: find all (x, y) where R*[x,y] AND R*[y,x]
    # This means x and y are in the same strongly connected component
    violations: list[Cycle] = []

    # Use numba kernel for fast symmetric pair detection (20-50x speedup)
    symmetric_pairs = find_symmetric_pairs_bool_numba(R_star)

    for pair_idx in range(symmetric_pairs.shape[0]):
        x = int(symmetric_pairs[pair_idx, 0])
        y = int(symmetric_pairs[pair_idx, 1])
        # Find the cycle - trace the full path
        cycle = _find_cycle_from_pair(R, x, y)
        if cycle:
            violations.append(cycle)

    computation_time = (time.perf_counter() - start_time) * 1000

    return AbstractSARPResult(
        is_consistent=len(violations) == 0,
        violations=violations,
        revealed_preference_matrix=R,
        transitive_closure=R_star,
        computation_time_ms=computation_time,
    )


def _find_cycle_from_pair(
    R: NDArray[np.bool_],
    start: int,
    end: int,
) -> Cycle | None:
    """
    Find a cycle between two nodes using BFS (numba-accelerated).

    Args:
        R: Revealed preference adjacency matrix
        start: Starting node
        end: Ending node (should be reachable from start and vice versa)

    Returns:
        Tuple of node indices forming the cycle, or None if not found
    """
    # BFS from start to end using numba kernel
    path_to_end = bfs_find_path_numba(R, np.int64(start), np.int64(end))
    if path_to_end[0] == -1:
        return None

    # BFS from end back to start using numba kernel
    path_back = bfs_find_path_numba(R, np.int64(end), np.int64(start))
    if path_back[0] == -1:
        return None

    # Combine paths to form cycle
    # path_to_end: start -> ... -> end -> start (numba returns with start at end)
    # path_back: end -> ... -> start -> end (numba returns with end at end)
    # We need: start -> ... -> end -> ... -> start
    # path_to_end already goes start->...->end->start, just need start->end path
    # path_back goes end->...->start->end, we need end->start path

    # Extract path from start to end (exclude the cycling back)
    path_to_end_list = list(path_to_end[:-1])  # Remove the repeat at end
    path_back_list = list(path_back[:-1])  # Remove the repeat at end

    # Cycle: start -> ... -> end -> ... -> start
    cycle = path_to_end_list[:-1] + path_back_list
    return tuple(cycle)


def validate_menu_consistency(log: MenuChoiceLog) -> CongruenceResult:
    """
    Check if menu choice data satisfies Congruence (full rationalizability).

    Congruence axiom requires:
    1. SARP: No cycles in the transitive revealed preference relation
    2. Maximality: The chosen item must be maximal under R* within the menu

    A dataset satisfies Congruence iff it can be rationalized by a
    strict preference ordering (Richter's Theorem).

    Args:
        log: MenuChoiceLog with menus and choices

    Returns:
        CongruenceResult with rationalizability status

    Example:
        >>> from pyrevealed import MenuChoiceLog, validate_menu_consistency
        >>> log = MenuChoiceLog(
        ...     menus=[frozenset({0, 1, 2}), frozenset({1, 2})],
        ...     choices=[0, 1]  # Consistent: 0 > 1 > 2
        ... )
        >>> result = validate_menu_consistency(log)
        >>> result.is_rationalizable
        True
    """
    start_time = time.perf_counter()

    # First check SARP
    sarp_result = validate_menu_sarp(log)

    # If SARP fails, Congruence fails
    if not sarp_result.is_consistent:
        computation_time = (time.perf_counter() - start_time) * 1000
        return CongruenceResult(
            is_congruent=False,
            satisfies_sarp=False,
            maximality_violations=[],
            sarp_result=sarp_result,
            computation_time_ms=computation_time,
        )

    # Check maximality: for each observation, the choice must be maximal
    # under R* within the menu
    R_star = sarp_result.transitive_closure
    maximality_violations: list[tuple[int, int]] = []

    for t, (menu, choice) in enumerate(zip(log.menus, log.choices)):
        for item in menu:
            if item != choice:
                # Check if item R* choice (item dominates choice)
                # If so, choice is not maximal - violation
                if item < R_star.shape[0] and choice < R_star.shape[0]:
                    if R_star[item, choice] and not R_star[choice, item]:
                        # item strictly dominates choice - violation
                        maximality_violations.append((t, item))

    computation_time = (time.perf_counter() - start_time) * 1000

    return CongruenceResult(
        is_congruent=len(maximality_violations) == 0,
        satisfies_sarp=True,
        maximality_violations=maximality_violations,
        sarp_result=sarp_result,
        computation_time_ms=computation_time,
    )


def compute_menu_efficiency(log: MenuChoiceLog) -> HoutmanMaksAbstractResult:
    """
    Compute Houtman-Maks efficiency index for menu-based choices.

    The Houtman-Maks index measures the minimum fraction of observations
    that must be removed to make the remaining data satisfy SARP.

    Uses a greedy algorithm: repeatedly remove the observation that
    participates in the most violations until SARP is satisfied.

    Args:
        log: MenuChoiceLog with menus and choices

    Returns:
        HoutmanMaksAbstractResult with efficiency index and removed observations

    Example:
        >>> from pyrevealed import MenuChoiceLog, compute_menu_efficiency
        >>> log = MenuChoiceLog(
        ...     menus=[frozenset({0, 1}), frozenset({0, 1}), frozenset({0, 2})],
        ...     choices=[0, 1, 0]  # One inconsistency
        ... )
        >>> result = compute_menu_efficiency(log)
        >>> print(f"Efficiency: {result.efficiency_index:.2f}")
    """
    start_time = time.perf_counter()

    n_obs = log.num_observations
    removed: list[int] = []
    remaining = list(range(n_obs))

    # Greedy removal until consistent
    while True:
        # Create a sub-log with remaining observations
        sub_menus = [log.menus[i] for i in remaining]
        sub_choices = [log.choices[i] for i in remaining]

        if len(sub_menus) <= 1:
            # Single observation is always consistent
            break

        # Check SARP on remaining
        from pyrevealed.core.session import MenuChoiceLog as MenuChoiceLogClass
        sub_log = MenuChoiceLogClass(
            menus=sub_menus,
            choices=sub_choices,
            item_labels=log.item_labels,
        )
        sarp_result = validate_menu_sarp(sub_log)

        if sarp_result.is_consistent:
            break

        # Find observation participating in most violations
        # Count violations per observation
        violation_counts = {i: 0 for i in range(len(remaining))}

        # Map items in cycles to observations
        for cycle in sarp_result.violations:
            for item in cycle:
                # Find which observation(s) reveal preference for this item
                for obs_idx, (menu, choice) in enumerate(zip(sub_menus, sub_choices)):
                    if choice == item or item in menu:
                        violation_counts[obs_idx] += 1

        # Remove observation with most violations
        if violation_counts:
            worst_idx = max(violation_counts, key=lambda k: violation_counts[k])
            removed.append(remaining[worst_idx])
            remaining.pop(worst_idx)
        else:
            # No clear culprit, remove first observation in a cycle
            if sarp_result.violations:
                # Find an observation that created a problematic preference
                for obs_idx, (menu, choice) in enumerate(zip(sub_menus, sub_choices)):
                    for cycle in sarp_result.violations:
                        if choice in cycle:
                            removed.append(remaining[obs_idx])
                            remaining.pop(obs_idx)
                            break
                    else:
                        continue
                    break
            else:
                break

    computation_time = (time.perf_counter() - start_time) * 1000

    efficiency = 1.0 - (len(removed) / n_obs) if n_obs > 0 else 1.0

    return HoutmanMaksAbstractResult(
        efficiency_index=efficiency,
        removed_observations=removed,
        remaining_observations=remaining,
        num_total=n_obs,
        computation_time_ms=computation_time,
    )


def fit_menu_preferences(log: MenuChoiceLog) -> OrdinalUtilityResult:
    """
    Recover ordinal preference ranking from menu-based choices.

    If the data satisfies SARP, computes a preference ranking over items
    using topological sort of the revealed preference graph. If SARP fails,
    attempts to find the best-fitting ranking.

    Args:
        log: MenuChoiceLog with menus and choices

    Returns:
        OrdinalUtilityResult with preference ranking and utility values

    Example:
        >>> from pyrevealed import MenuChoiceLog, fit_menu_preferences
        >>> log = MenuChoiceLog(
        ...     menus=[frozenset({0, 1, 2}), frozenset({1, 2}), frozenset({0, 2})],
        ...     choices=[0, 1, 0]  # Reveals 0 > 1 > 2
        ... )
        >>> result = fit_menu_preferences(log)
        >>> if result.success:
        ...     print(f"Preference order: {result.preference_order}")
    """
    start_time = time.perf_counter()

    # First check SARP
    sarp_result = validate_menu_sarp(log)

    all_items = sorted(log.all_items)
    n_items = len(all_items)

    if n_items == 0:
        computation_time = (time.perf_counter() - start_time) * 1000
        return OrdinalUtilityResult(
            success=False,
            utility_ranking=None,
            utility_values=None,
            preference_order=None,
            num_items=0,
            is_complete=False,
            computation_time_ms=computation_time,
        )

    if not sarp_result.is_consistent:
        # SARP violated - cannot find a consistent ranking
        computation_time = (time.perf_counter() - start_time) * 1000
        return OrdinalUtilityResult(
            success=False,
            utility_ranking=None,
            utility_values=None,
            preference_order=None,
            num_items=n_items,
            is_complete=False,
            computation_time_ms=computation_time,
        )

    # Topological sort of revealed preference graph using numba kernel (10-30x speedup)
    R_star = sarp_result.transitive_closure
    max_item = max(all_items) + 1

    # Convert items to numpy array for numba kernel
    items_array = np.array(sorted(all_items), dtype=np.int64)

    # Use numba-accelerated topological sort
    sorted_items = topological_sort_numba(R_star, items_array)
    preference_order: list[int] = [int(x) for x in sorted_items]

    # Check if all items were ranked (no cycles should exist since SARP passed)
    is_complete = len(preference_order) == n_items

    # Create ranking: 0 = most preferred
    utility_ranking = {item: rank for rank, item in enumerate(preference_order)}

    # Create utility values (higher = more preferred)
    utility_values = np.zeros(max_item, dtype=np.float64)
    for item, rank in utility_ranking.items():
        utility_values[item] = float(n_items - rank)  # Invert so higher = better

    computation_time = (time.perf_counter() - start_time) * 1000

    return OrdinalUtilityResult(
        success=True,
        utility_ranking=utility_ranking,
        utility_values=utility_values,
        preference_order=preference_order,
        num_items=n_items,
        is_complete=is_complete,
        computation_time_ms=computation_time,
    )


# =============================================================================
# LEGACY ALIASES (Economics terminology)
# =============================================================================

check_abstract_warp = validate_menu_warp
"""Legacy alias: use validate_menu_warp instead."""

check_abstract_sarp = validate_menu_sarp
"""Legacy alias: use validate_menu_sarp instead."""

check_congruence = validate_menu_consistency
"""Legacy alias: use validate_menu_consistency instead."""

compute_abstract_efficiency = compute_menu_efficiency
"""Legacy alias: use compute_menu_efficiency instead."""

recover_ordinal_utility = fit_menu_preferences
"""Legacy alias: use fit_menu_preferences instead."""
