"""Limited attention models for choice analysis.

Implements consideration set models where consumers don't see all options.
Based on Chapter 14 of Chambers & Echenique (2016) "Revealed Preference Theory".

Key insight: apparent irrationality may be due to limited attention rather
than inconsistent preferences. A choice is "attention-rational" if it's
optimal among the items actually considered.

Tech-Friendly Names (Primary):
    - estimate_consideration_sets(): Estimate which items are considered
    - test_attention_rationality(): Test rationalizability with limited attention
    - compute_salience_weights(): Estimate feature-based attention weights

Economics Names (Legacy Aliases):
    - identify_attention() -> estimate_consideration_sets()
    - check_attention_rationality() -> test_attention_rationality()
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from pyrevealed.core.result import AttentionResult

if TYPE_CHECKING:
    from pyrevealed.core.session import MenuChoiceLog


def test_attention_rationality(
    log: "MenuChoiceLog",
    max_consideration_size: int | None = None,
) -> AttentionResult:
    """
    Test if choices are rationalizable with limited attention.

    A choice is attention-rational if there exists:
    1. A preference ordering over items
    2. A consideration set function (what items are noticed)
    Such that each choice is optimal among considered items.

    This is a weaker notion than standard rationality - it allows
    apparent violations due to limited attention.

    Args:
        log: MenuChoiceLog with menus and choices
        max_consideration_size: Maximum consideration set size (None = no limit)

    Returns:
        AttentionResult with consideration sets and attention analysis

    Example:
        >>> from pyrevealed import MenuChoiceLog, test_attention_rationality
        >>> result = test_attention_rationality(choice_log)
        >>> if result.is_attention_rational:
        ...     print("Choices are rationalizable with limited attention")
        ...     print(f"Avg consideration set size: {result.mean_consideration_size:.1f}")

    Note:
        **Complexity**: This function calls validate_menu_sarp internally, which uses
        Floyd-Warshall with O(I³) complexity where I is the number of unique items.
        For large item sets, this can be slow.

    References:
        Chambers & Echenique (2016), Chapter 14
        Manzini, P. & Mariotti, M. (2014). "Stochastic Choice and Consideration Sets"
    """
    start_time = time.perf_counter()

    n_obs = log.num_observations

    # First check standard SARP consistency
    from pyrevealed.algorithms.abstract_choice import validate_menu_sarp
    sarp_result = validate_menu_sarp(log)

    if sarp_result.is_consistent:
        # Already rational without limited attention
        # Consideration sets are full menus
        consideration_sets = [set(menu) for menu in log.menus]

        computation_time = (time.perf_counter() - start_time) * 1000

        return AttentionResult(
            consideration_sets=consideration_sets,
            attention_parameter=1.0,
            is_attention_rational=True,
            salience_weights=np.ones(max(log.all_items) + 1),
            default_option=None,
            inattention_rate=0.0,
            rationalizable_observations=list(range(n_obs)),
            computation_time_ms=computation_time,
        )

    # Try to find consideration sets that rationalize the data
    consideration_sets, is_rational, rationalizable_obs = _find_consideration_sets(
        log, max_consideration_size
    )

    # Compute attention parameter (average consideration set size / menu size)
    total_considered = sum(len(cs) for cs in consideration_sets)
    total_available = sum(len(menu) for menu in log.menus)
    attention_parameter = total_considered / max(total_available, 1)

    # Estimate salience weights
    salience_weights = compute_salience_weights(log, consideration_sets)

    # Identify default option (if any)
    default_option = _identify_default_option(log, consideration_sets)

    # Inattention rate
    inattention_obs = [
        t for t in range(n_obs)
        if len(consideration_sets[t]) < len(log.menus[t])
    ]
    inattention_rate = len(inattention_obs) / n_obs if n_obs > 0 else 0.0

    computation_time = (time.perf_counter() - start_time) * 1000

    return AttentionResult(
        consideration_sets=consideration_sets,
        attention_parameter=attention_parameter,
        is_attention_rational=is_rational,
        salience_weights=salience_weights,
        default_option=default_option,
        inattention_rate=inattention_rate,
        rationalizable_observations=rationalizable_obs,
        computation_time_ms=computation_time,
    )


def estimate_consideration_sets(
    log: "MenuChoiceLog",
    method: str = "greedy",
) -> list[set[int]]:
    """
    Estimate consideration sets for each observation.

    The consideration set is the subset of menu items that the
    consumer actually notices/considers before making a choice.

    Args:
        log: MenuChoiceLog with menus and choices
        method: Estimation method ("greedy", "optimal", "salience")

    Returns:
        List of consideration sets, one per observation

    Note:
        The chosen item is always in the consideration set.
    """
    if method == "greedy":
        consideration_sets, _, _ = _find_consideration_sets(log, None)
    elif method == "salience":
        consideration_sets = _estimate_salience_based_consideration(log)
    else:
        consideration_sets, _, _ = _find_consideration_sets(log, None)

    return consideration_sets


def compute_salience_weights(
    log: "MenuChoiceLog",
    consideration_sets: list[set[int]] | None = None,
) -> NDArray[np.float64]:
    """
    Compute salience weights for each item.

    Higher weight = more likely to be noticed/considered.
    Estimated from frequency of appearing in consideration sets.

    Args:
        log: MenuChoiceLog with menus and choices
        consideration_sets: Optional pre-computed consideration sets

    Returns:
        Array of salience weights (one per item)
    """
    if consideration_sets is None:
        consideration_sets = estimate_consideration_sets(log)

    max_item = max(log.all_items)
    weights = np.zeros(max_item + 1)
    counts = np.zeros(max_item + 1)

    for t, (menu, cs) in enumerate(zip(log.menus, consideration_sets)):
        for item in menu:
            counts[item] += 1
            if item in cs:
                weights[item] += 1

    # Normalize to get probability of consideration
    for i in range(len(weights)):
        if counts[i] > 0:
            weights[i] = weights[i] / counts[i]

    return weights


def _find_consideration_sets(
    log: "MenuChoiceLog",
    max_size: int | None,
) -> tuple[list[set[int]], bool, list[int]]:
    """
    Find consideration sets that rationalize the data.

    Uses a greedy heuristic algorithm:
    1. Start with consideration = {chosen item}
    2. Add items needed to maintain preference consistency

    **Algorithmic Limitation**: This is a greedy heuristic, not an optimal algorithm.
    The problem of finding minimal consideration sets is NP-hard in general.
    This implementation may:
    - Produce larger-than-necessary consideration sets
    - Fail to find a rationalizing set even when one exists
    - Not guarantee the globally optimal solution

    For optimal results on small instances, consider integer programming formulations.

    Returns:
        Tuple of (consideration_sets, is_rational, rationalizable_observations)
    """
    n_obs = log.num_observations

    # Build revealed preference from choices
    # choice[t] is preferred to all unchosen items in menu[t]
    revealed_pref: dict[int, set[int]] = {}  # item -> items it's preferred to

    for t, (menu, choice) in enumerate(zip(log.menus, log.choices)):
        if choice not in revealed_pref:
            revealed_pref[choice] = set()
        for item in menu:
            if item != choice:
                revealed_pref[choice].add(item)

    # For each observation, find minimal consideration set
    consideration_sets = []
    rationalizable = []

    for t, (menu, choice) in enumerate(zip(log.menus, log.choices)):
        # Consideration set must contain chosen item
        consideration = {choice}

        # Add items that must be considered for consistency
        # An item x must be considered if:
        # 1. x is in menu
        # 2. x is revealed preferred to the choice (would cause violation)

        for item in menu:
            if item != choice:
                # Check if item is revealed preferred to choice
                if item in revealed_pref and choice in revealed_pref[item]:
                    # item > choice in revealed preference
                    # Must not consider item, or choice would be irrational
                    pass
                else:
                    # Safe to not consider this item
                    pass

        # Simple approach: consideration = items not strictly preferred to choice
        for item in menu:
            if item == choice:
                continue

            # Check if choosing 'choice' over 'item' is consistent
            # with some preference ordering

            # If 'item' is strictly revealed preferred to 'choice' elsewhere,
            # we should not consider 'item' (to avoid violation)
            item_preferred = item in revealed_pref and choice in revealed_pref[item]

            if not item_preferred:
                # Can safely consider this item
                if max_size is None or len(consideration) < max_size:
                    consideration.add(item)

        consideration_sets.append(consideration)

        # Check if this observation is rationalizable
        # Choice must be maximal in consideration set
        is_rational_obs = True
        for item in consideration:
            if item != choice:
                if item in revealed_pref and choice in revealed_pref[item]:
                    is_rational_obs = False
                    break

        if is_rational_obs:
            rationalizable.append(t)

    is_fully_rational = len(rationalizable) == n_obs

    return consideration_sets, is_fully_rational, rationalizable


def _estimate_salience_based_consideration(
    log: "MenuChoiceLog",
    salience_threshold: float = 0.1,
) -> list[set[int]]:
    """
    Estimate consideration sets based on item salience.

    Assumes more frequently chosen items are more salient.

    Args:
        log: MenuChoiceLog with menus and choices
        salience_threshold: Minimum choice frequency to be considered salient (default 0.1)
    """
    # Compute choice frequencies
    choice_counts: dict[int, int] = {}
    for choice in log.choices:
        choice_counts[choice] = choice_counts.get(choice, 0) + 1

    total_choices = len(log.choices)

    consideration_sets = []

    for t, (menu, choice) in enumerate(zip(log.menus, log.choices)):
        # Always include chosen item
        consideration = {choice}

        # Include items with high choice frequency
        for item in menu:
            if item != choice:
                freq = choice_counts.get(item, 0) / total_choices
                if freq > salience_threshold:
                    consideration.add(item)

        consideration_sets.append(consideration)

    return consideration_sets


def _identify_default_option(
    log: "MenuChoiceLog",
    consideration_sets: list[set[int]],
) -> int | None:
    """
    Identify if there's a default option (always considered).
    """
    if not consideration_sets:
        return None

    # Find items that appear in all consideration sets
    common_items = set(consideration_sets[0])
    for cs in consideration_sets[1:]:
        common_items &= cs

    if len(common_items) == 1:
        return list(common_items)[0]

    return None


def test_attention_filter(
    log: "MenuChoiceLog",
    filter_function: callable,
) -> dict:
    """
    Test if choices are rational given a specific attention filter.

    An attention filter specifies which items are considered at each
    observation. This tests if choices are optimal within filtered menus.

    Args:
        log: MenuChoiceLog with menus and choices
        filter_function: Function(menu, t) -> consideration_set

    Returns:
        Dictionary with test results
    """
    violations = []

    # Build revealed preference
    revealed_pref: dict[int, set[int]] = {}
    for t, (menu, choice) in enumerate(zip(log.menus, log.choices)):
        consideration = filter_function(menu, t)

        if choice not in consideration:
            violations.append(t)
            continue

        if choice not in revealed_pref:
            revealed_pref[choice] = set()
        for item in consideration:
            if item != choice:
                revealed_pref[choice].add(item)

    # Check for cycles in revealed preference
    has_cycle = _has_preference_cycle(revealed_pref)

    return {
        "is_rational": len(violations) == 0 and not has_cycle,
        "violations": violations,
        "has_preference_cycle": has_cycle,
    }


def _has_preference_cycle(revealed_pref: dict[int, set[int]]) -> bool:
    """
    Check if revealed preference relation has a cycle using DFS.
    """
    visited = set()
    rec_stack = set()

    def dfs(node: int) -> bool:
        visited.add(node)
        rec_stack.add(node)

        for neighbor in revealed_pref.get(node, set()):
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True

        rec_stack.remove(node)
        return False

    for node in revealed_pref:
        if node not in visited:
            if dfs(node):
                return True

    return False


# =============================================================================
# LEGACY ALIASES
# =============================================================================

identify_attention = estimate_consideration_sets
"""Legacy alias: use estimate_consideration_sets instead."""

check_attention_rationality = test_attention_rationality
"""Legacy alias: use test_attention_rationality instead."""
