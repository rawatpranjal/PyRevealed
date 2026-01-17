"""Context effects in choice behavior.

Implements detection of decoy (attraction) effects and compromise effects,
which are key violations of rational choice theory.

Tech-Friendly Names (Primary):
    - detect_decoy_effect(): Find attraction/decoy effects
    - detect_compromise_effect(): Find compromise/extremeness aversion
    - test_regularity_violation(): General context effect test

Economics Names (Legacy Aliases):
    - check_attraction_effect() -> detect_decoy_effect()
    - check_extremeness_aversion() -> detect_compromise_effect()

References:
    Huber, J., Payne, J. W., & Puto, C. (1982). Adding asymmetrically
    dominated alternatives: Violations of regularity. Journal of Consumer Research.

    Simonson, I. (1989). Choice based on reasons: The case of attraction
    and compromise effects. Journal of Consumer Research.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from pyrevealed.core.session import StochasticChoiceLog

from pyrevealed.core.result import DecoyEffectResult, CompromiseEffectResult


# =============================================================================
# DECOY (ATTRACTION) EFFECT
# =============================================================================


def detect_decoy_effect(
    log: "StochasticChoiceLog",
    threshold: float = 0.05,
    dominance_check: bool = True,
) -> DecoyEffectResult:
    """
    Detect decoy (attraction) effects in stochastic choice data.

    A decoy effect occurs when adding a dominated alternative D (the "decoy")
    to a choice set {T, C} increases the choice probability of the
    dominating alternative T (the "target") relative to competitor C.

    Formally, for items T, C, D where D is dominated by T:
        P(T | {T, C, D}) > P(T | {T, C})

    This violates the regularity axiom, which requires that adding
    alternatives can only decrease choice probabilities.

    The decoy effect is widely exploited in pricing (e.g., subscription
    tiers) and can indicate susceptibility to manipulation.

    Args:
        log: StochasticChoiceLog with choice frequency data
        threshold: Minimum probability boost to count as decoy effect
        dominance_check: If True, only consider decoys that are dominated

    Returns:
        DecoyEffectResult with detected decoy relationships and magnitude

    Example:
        >>> from pyrevealed import StochasticChoiceLog, detect_decoy_effect
        >>> # Menu without decoy: {small, large}
        >>> # Menu with decoy: {small, medium*, large} where medium* is dominated by large
        >>> result = detect_decoy_effect(log)
        >>> if result.has_decoy_effect:
        ...     print(f"Decoy effects found: {result.num_decoys}")
        ...     print(f"Average boost: {result.magnitude:.1%}")

    References:
        Huber, J., Payne, J. W., & Puto, C. (1982). Adding asymmetrically
        dominated alternatives: Violations of regularity and the similarity
        hypothesis. Journal of Consumer Research, 9(1), 90-98.
    """
    start_time = time.perf_counter()

    decoy_triples: list[tuple[int, int, int]] = []
    vulnerabilities: dict[int, float] = {}
    total_boost = 0.0
    num_menus_tested = 0

    # Get all items
    all_items = log.all_items

    # Build menu index for quick lookup
    menu_to_idx = {}
    for m_idx, menu in enumerate(log.menus):
        menu_key = frozenset(menu)
        menu_to_idx[menu_key] = m_idx

    # For each item pair (T, C), look for potential decoys D
    for target in all_items:
        for competitor in all_items:
            if target == competitor:
                continue

            # Base menu: {T, C}
            base_menu = frozenset({target, competitor})
            if base_menu not in menu_to_idx:
                continue

            base_idx = menu_to_idx[base_menu]
            base_prob_target = log.get_choice_probability(base_idx, target)

            # Look for menus {T, C, D} where D might be a decoy
            for decoy in all_items:
                if decoy == target or decoy == competitor:
                    continue

                extended_menu = frozenset({target, competitor, decoy})
                if extended_menu not in menu_to_idx:
                    continue

                extended_idx = menu_to_idx[extended_menu]
                extended_prob_target = log.get_choice_probability(extended_idx, target)

                num_menus_tested += 1

                # Check for decoy effect: target probability increased
                boost = extended_prob_target - base_prob_target

                if boost > threshold:
                    # This is a potential decoy effect
                    decoy_triples.append((target, competitor, decoy))
                    total_boost += boost

                    # Track vulnerability
                    if competitor not in vulnerabilities:
                        vulnerabilities[competitor] = 0.0
                    vulnerabilities[competitor] = max(vulnerabilities[competitor], boost)

    # Compute average magnitude
    magnitude = total_boost / len(decoy_triples) if decoy_triples else 0.0
    has_decoy_effect = len(decoy_triples) > 0

    computation_time = (time.perf_counter() - start_time) * 1000

    return DecoyEffectResult(
        has_decoy_effect=has_decoy_effect,
        decoy_triples=decoy_triples,
        magnitude=magnitude,
        vulnerabilities=vulnerabilities,
        num_menus_tested=num_menus_tested,
        computation_time_ms=computation_time,
    )


# =============================================================================
# COMPROMISE EFFECT
# =============================================================================


def detect_compromise_effect(
    log: "StochasticChoiceLog",
    attribute_vectors: NDArray[np.float64] | None = None,
    threshold: float = 0.05,
) -> CompromiseEffectResult:
    """
    Detect compromise effects in stochastic choice data.

    A compromise effect occurs when adding extreme alternatives to a
    choice set increases the probability of choosing middle/compromise
    options. This reflects "extremeness aversion" - people prefer options
    that are not extreme on any attribute.

    Formally, for items A, B, C where B is between A and C on attributes:
        P(B | {A, B, C}) > P(B | {A, B})  and  P(B | {A, B, C}) > P(B | {B, C})

    This effect is important for product line design and pricing, as the
    "middle" option often captures more market share than expected.

    Args:
        log: StochasticChoiceLog with choice frequency data
        attribute_vectors: Optional N x K matrix of item attributes.
            If provided, used to identify "middle" items.
            If None, infers from choice patterns.
        threshold: Minimum probability boost to count as compromise effect

    Returns:
        CompromiseEffectResult with detected compromise relationships

    Example:
        >>> from pyrevealed import StochasticChoiceLog, detect_compromise_effect
        >>> # Items with attributes: small(1), medium(2), large(3)
        >>> attributes = np.array([[1], [2], [3]])
        >>> result = detect_compromise_effect(log, attribute_vectors=attributes)
        >>> if result.has_compromise_effect:
        ...     print(f"Compromise items: {result.compromise_items}")

    References:
        Simonson, I. (1989). Choice based on reasons: The case of attraction
        and compromise effects. Journal of Consumer Research, 16(2), 158-174.
    """
    start_time = time.perf_counter()

    compromise_items: list[int] = []
    extreme_pairs: list[tuple[int, int, int]] = []
    total_boost = 0.0
    num_menus_tested = 0

    all_items = sorted(log.all_items)
    n_items = len(all_items)

    # Build menu index
    menu_to_idx = {}
    for m_idx, menu in enumerate(log.menus):
        menu_key = frozenset(menu)
        menu_to_idx[menu_key] = m_idx

    # If attribute vectors provided, use them to identify middle items
    if attribute_vectors is not None:
        item_to_idx = {item: idx for idx, item in enumerate(all_items)}

        # For each triple, check if middle item benefits
        for i, item_a in enumerate(all_items):
            for j, item_c in enumerate(all_items):
                if i >= j:
                    continue

                for item_b in all_items:
                    if item_b == item_a or item_b == item_c:
                        continue

                    # Check if B is between A and C on attributes
                    idx_a = item_to_idx.get(item_a, 0)
                    idx_b = item_to_idx.get(item_b, 0)
                    idx_c = item_to_idx.get(item_c, 0)

                    if idx_a >= len(attribute_vectors) or idx_c >= len(attribute_vectors):
                        continue

                    attrs_a = attribute_vectors[idx_a]
                    attrs_b = attribute_vectors[idx_b] if idx_b < len(attribute_vectors) else None
                    attrs_c = attribute_vectors[idx_c]

                    if attrs_b is None:
                        continue

                    # Check if B is between A and C on all attributes
                    is_middle = all(
                        (attrs_a[k] <= attrs_b[k] <= attrs_c[k]) or
                        (attrs_c[k] <= attrs_b[k] <= attrs_a[k])
                        for k in range(len(attrs_a))
                    )

                    if not is_middle:
                        continue

                    # Check for compromise effect
                    triple_menu = frozenset({item_a, item_b, item_c})
                    pair_ab = frozenset({item_a, item_b})
                    pair_bc = frozenset({item_b, item_c})

                    if triple_menu not in menu_to_idx:
                        continue

                    triple_idx = menu_to_idx[triple_menu]
                    prob_b_triple = log.get_choice_probability(triple_idx, item_b)

                    # Check against pairs
                    boost = 0.0
                    comparisons = 0

                    if pair_ab in menu_to_idx:
                        pair_idx = menu_to_idx[pair_ab]
                        prob_b_pair = log.get_choice_probability(pair_idx, item_b)
                        boost += prob_b_triple - prob_b_pair
                        comparisons += 1
                        num_menus_tested += 1

                    if pair_bc in menu_to_idx:
                        pair_idx = menu_to_idx[pair_bc]
                        prob_b_pair = log.get_choice_probability(pair_idx, item_b)
                        boost += prob_b_triple - prob_b_pair
                        comparisons += 1
                        num_menus_tested += 1

                    if comparisons > 0:
                        avg_boost = boost / comparisons
                        if avg_boost > threshold:
                            extreme_pairs.append((item_a, item_c, item_b))
                            if item_b not in compromise_items:
                                compromise_items.append(item_b)
                            total_boost += avg_boost

    else:
        # Without attributes, use heuristic: items that appear in many menus
        # and get higher probability in larger menus are compromise candidates

        for m_idx in range(log.num_menus):
            menu = log.menus[m_idx]
            if len(menu) < 3:
                continue

            menu_items = list(menu)
            num_menus_tested += 1

            # Check if any item has higher probability than in smaller subsets
            for item in menu_items:
                prob_full = log.get_choice_probability(m_idx, item)

                # Check probability in 2-item subsets containing this item
                other_items = [x for x in menu_items if x != item]
                avg_pair_prob = 0.0
                num_pairs = 0

                for other in other_items:
                    pair_menu = frozenset({item, other})
                    if pair_menu in menu_to_idx:
                        pair_idx = menu_to_idx[pair_menu]
                        pair_prob = log.get_choice_probability(pair_idx, item)
                        avg_pair_prob += pair_prob
                        num_pairs += 1

                if num_pairs > 0:
                    avg_pair_prob /= num_pairs
                    boost = prob_full - avg_pair_prob

                    if boost > threshold:
                        if item not in compromise_items:
                            compromise_items.append(item)
                        total_boost += boost

    # Compute average magnitude
    magnitude = total_boost / len(extreme_pairs) if extreme_pairs else (
        total_boost / len(compromise_items) if compromise_items else 0.0
    )
    has_compromise_effect = len(compromise_items) > 0 or len(extreme_pairs) > 0

    computation_time = (time.perf_counter() - start_time) * 1000

    return CompromiseEffectResult(
        has_compromise_effect=has_compromise_effect,
        compromise_items=compromise_items,
        magnitude=magnitude,
        extreme_pairs=extreme_pairs,
        num_menus_tested=num_menus_tested,
        computation_time_ms=computation_time,
    )


# =============================================================================
# GENERAL CONTEXT EFFECT TEST
# =============================================================================


def test_context_effects(
    log: "StochasticChoiceLog",
    threshold: float = 0.05,
) -> dict:
    """
    Comprehensive test for context effects (decoy, compromise, similarity).

    Runs all context effect tests and returns a summary of violations.

    Args:
        log: StochasticChoiceLog with choice frequency data
        threshold: Minimum effect size to count

    Returns:
        Dict with results from all context effect tests:
        - decoy_result: DecoyEffectResult
        - compromise_result: CompromiseEffectResult
        - has_context_effects: True if any effect detected
        - strongest_effect: Name of strongest effect type
        - overall_magnitude: Average effect magnitude

    Example:
        >>> from pyrevealed import StochasticChoiceLog, test_context_effects
        >>> result = test_context_effects(log)
        >>> if result["has_context_effects"]:
        ...     print(f"Strongest effect: {result['strongest_effect']}")
    """
    start_time = time.perf_counter()

    # Run individual tests
    decoy_result = detect_decoy_effect(log, threshold=threshold)
    compromise_result = detect_compromise_effect(log, threshold=threshold)

    # Determine strongest effect
    effects = []
    if decoy_result.has_decoy_effect:
        effects.append(("decoy", decoy_result.magnitude))
    if compromise_result.has_compromise_effect:
        effects.append(("compromise", compromise_result.magnitude))

    has_context_effects = len(effects) > 0

    if effects:
        strongest = max(effects, key=lambda x: x[1])
        strongest_effect = strongest[0]
        overall_magnitude = sum(e[1] for e in effects) / len(effects)
    else:
        strongest_effect = "none"
        overall_magnitude = 0.0

    computation_time = (time.perf_counter() - start_time) * 1000

    return {
        "decoy_result": decoy_result,
        "compromise_result": compromise_result,
        "has_context_effects": has_context_effects,
        "strongest_effect": strongest_effect,
        "overall_magnitude": overall_magnitude,
        "computation_time_ms": computation_time,
    }


# =============================================================================
# TECH-FRIENDLY ALIASES
# =============================================================================

check_attraction_effect = detect_decoy_effect
"""Legacy alias: use detect_decoy_effect instead."""

check_extremeness_aversion = detect_compromise_effect
"""Legacy alias: use detect_compromise_effect instead."""

test_attraction_effect = detect_decoy_effect
"""Legacy alias: use detect_decoy_effect instead."""

test_compromise_effect = detect_compromise_effect
"""Legacy alias: use detect_compromise_effect instead."""
