"""Ranking and pairwise comparison models.

Implements Bradley-Terry model for pairwise comparisons and ranking metrics
for comparing orderings. Essential for RLHF preference learning and
recommendation systems.

Tech-Friendly Names (Primary):
    - fit_bradley_terry(): Fit Bradley-Terry model to pairwise comparisons
    - compute_kendall_tau(): Kendall tau correlation between rankings
    - compute_spearman_footrule(): Spearman footrule distance
    - compute_rank_biased_overlap(): RBO for top-weighted comparison
    - predict_pairwise_probability(): Predict P(i beats j)
    - aggregate_rankings(): Combine multiple rankings

Economics Names (Legacy Aliases):
    - fit_bt_model() -> fit_bradley_terry()
    - kendall_correlation() -> compute_kendall_tau()

References:
    Bradley, R. A., & Terry, M. E. (1952). Rank analysis of incomplete
    block designs: I. The method of paired comparisons. Biometrika.

    Webber, W., Moffat, A., & Zobel, J. (2010). A similarity measure
    for indefinite rankings. ACM TOIS.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from pyrevealed.core.result import BradleyTerryResult, RankingComparisonResult
from pyrevealed.core.mixins import ResultSummaryMixin
from pyrevealed.core.display import ResultDisplayMixin, ResultPlotMixin
from pyrevealed.core.exceptions import ComputationalLimitError


# =============================================================================
# BRADLEY-TERRY MODEL
# =============================================================================


def fit_bradley_terry(
    comparisons: list[tuple[int, int, int]],
    method: str = "mle",
    max_iterations: int = 1000,
    tolerance: float = 1e-8,
) -> BradleyTerryResult:
    """
    Fit Bradley-Terry model to pairwise comparison data.

    The Bradley-Terry model represents choice probabilities as:
        P(i beats j) = pi_i / (pi_i + pi_j)

    where pi_i > 0 is the "strength" parameter for item i. This is
    equivalent to logistic regression on paired differences.

    The model is widely used in:
    - RLHF (Reinforcement Learning from Human Feedback)
    - Sports rankings (Elo is a special case)
    - Recommendation systems
    - A/B testing analysis

    Args:
        comparisons: List of (winner, loser, count) tuples.
            - winner: Index of winning item
            - loser: Index of losing item
            - count: Number of times this outcome was observed
        method: Fitting method:
            - "mle": Maximum likelihood estimation (default)
            - "mm": Minorization-maximization (faster for large data)
        max_iterations: Maximum optimization iterations
        tolerance: Convergence tolerance

    Returns:
        BradleyTerryResult with strength parameters, log-likelihood, and ranking

    Example:
        >>> comparisons = [
        ...     (0, 1, 5),  # Item 0 beat item 1 five times
        ...     (1, 0, 3),  # Item 1 beat item 0 three times
        ...     (0, 2, 7),  # Item 0 beat item 2 seven times
        ...     (2, 0, 2),  # etc.
        ... ]
        >>> result = fit_bradley_terry(comparisons)
        >>> print(f"Ranking: {result.ranking}")
        >>> print(f"P(0 beats 1): {result.predict_probability(0, 1):.3f}")

    References:
        Bradley, R. A., & Terry, M. E. (1952). Rank analysis of incomplete
        block designs: I. The method of paired comparisons. Biometrika, 39, 324-345.

        Hunter, D. R. (2004). MM algorithms for generalized Bradley-Terry models.
        Annals of Statistics, 32(1), 384-406.
    """
    start_time = time.perf_counter()

    if not comparisons:
        computation_time = (time.perf_counter() - start_time) * 1000
        return BradleyTerryResult(
            scores={},
            log_likelihood=0.0,
            converged=True,
            ranking=[],
            num_comparisons=0,
            num_items=0,
            computation_time_ms=computation_time,
        )

    # Extract all unique items
    all_items = set()
    for winner, loser, _ in comparisons:
        all_items.add(winner)
        all_items.add(loser)
    items = sorted(all_items)
    n_items = len(items)
    item_to_idx = {item: idx for idx, item in enumerate(items)}

    # Build comparison matrix
    wins = np.zeros((n_items, n_items))
    for winner, loser, count in comparisons:
        i, j = item_to_idx[winner], item_to_idx[loser]
        wins[i, j] += count

    total_comparisons = int(np.sum(wins))

    if method == "mm":
        scores_arr, log_likelihood, converged = _fit_bt_mm(
            wins, max_iterations, tolerance
        )
    else:  # MLE
        scores_arr, log_likelihood, converged = _fit_bt_mle(
            wins, max_iterations, tolerance
        )

    # Convert to dict and normalize (min score = 0)
    scores_arr = scores_arr - np.min(scores_arr)
    scores = {items[i]: float(scores_arr[i]) for i in range(n_items)}

    # Create ranking (highest score first)
    ranking = sorted(items, key=lambda x: -scores[x])

    computation_time = (time.perf_counter() - start_time) * 1000

    return BradleyTerryResult(
        scores=scores,
        log_likelihood=log_likelihood,
        converged=converged,
        ranking=ranking,
        num_comparisons=total_comparisons,
        num_items=n_items,
        computation_time_ms=computation_time,
    )


def _fit_bt_mle(
    wins: NDArray[np.float64],
    max_iterations: int,
    tolerance: float,
) -> tuple[NDArray[np.float64], float, bool]:
    """Fit Bradley-Terry using MLE (L-BFGS-B optimization)."""
    n = wins.shape[0]

    # Initial scores (log-space)
    log_scores_init = np.zeros(n)

    def neg_log_likelihood(log_scores: NDArray[np.float64]) -> float:
        """Compute negative log-likelihood."""
        scores = np.exp(log_scores)
        ll = 0.0
        for i in range(n):
            for j in range(n):
                if wins[i, j] > 0:
                    # P(i beats j) = scores[i] / (scores[i] + scores[j])
                    p_ij = scores[i] / (scores[i] + scores[j])
                    ll += wins[i, j] * np.log(max(p_ij, 1e-15))
        return -ll

    def gradient(log_scores: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute gradient of negative log-likelihood."""
        scores = np.exp(log_scores)
        grad = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    total = wins[i, j] + wins[j, i]
                    if total > 0:
                        p_ij = scores[i] / (scores[i] + scores[j])
                        grad[i] += wins[i, j] - total * p_ij
        return -grad  # Negative for minimization

    result = minimize(
        neg_log_likelihood,
        log_scores_init,
        method="L-BFGS-B",
        jac=gradient,
        options={"maxiter": max_iterations, "gtol": tolerance},
    )

    scores = np.exp(result.x)
    log_likelihood = -result.fun
    converged = result.success

    return scores, log_likelihood, converged


def _fit_bt_mm(
    wins: NDArray[np.float64],
    max_iterations: int,
    tolerance: float,
) -> tuple[NDArray[np.float64], float, bool]:
    """Fit Bradley-Terry using Minorization-Maximization (Hunter 2004)."""
    n = wins.shape[0]

    # Total wins for each item
    w = np.sum(wins, axis=1)

    # Total comparisons involving each pair
    n_ij = wins + wins.T

    # Initialize scores uniformly
    pi = np.ones(n)

    converged = False
    for iteration in range(max_iterations):
        pi_old = pi.copy()

        # MM update
        for i in range(n):
            if w[i] > 0:
                denom = 0.0
                for j in range(n):
                    if i != j and n_ij[i, j] > 0:
                        denom += n_ij[i, j] / (pi_old[i] + pi_old[j])
                if denom > 0:
                    pi[i] = w[i] / denom

        # Normalize to sum to n
        pi = pi * n / np.sum(pi)

        # Check convergence
        if np.max(np.abs(pi - pi_old)) < tolerance:
            converged = True
            break

    # Compute log-likelihood
    ll = 0.0
    for i in range(n):
        for j in range(n):
            if wins[i, j] > 0:
                p_ij = pi[i] / (pi[i] + pi[j])
                ll += wins[i, j] * np.log(max(p_ij, 1e-15))

    # Convert to log-scores for consistency with MLE
    scores = np.log(pi + 1e-15)

    return scores, ll, converged


def predict_pairwise_probability(
    result: BradleyTerryResult,
    item_i: int,
    item_j: int,
) -> float:
    """
    Predict probability that item i beats item j.

    Uses the Bradley-Terry formula:
        P(i beats j) = exp(s_i) / (exp(s_i) + exp(s_j))

    where s_i and s_j are the fitted scores.

    Args:
        result: Fitted BradleyTerryResult
        item_i: First item index
        item_j: Second item index

    Returns:
        Probability in [0, 1]

    Example:
        >>> result = fit_bradley_terry(comparisons)
        >>> p = predict_pairwise_probability(result, 0, 1)
        >>> print(f"P(item 0 beats item 1) = {p:.3f}")
    """
    if item_i not in result.scores or item_j not in result.scores:
        return 0.5  # Default to 50-50 for unknown items

    s_i = result.scores[item_i]
    s_j = result.scores[item_j]

    # Using softmax formulation for numerical stability
    exp_i = np.exp(s_i - max(s_i, s_j))
    exp_j = np.exp(s_j - max(s_i, s_j))

    return exp_i / (exp_i + exp_j)


def aggregate_rankings(
    rankings: list[list[int]],
    method: str = "borda",
) -> list[int]:
    """
    Aggregate multiple rankings into a consensus ranking.

    Args:
        rankings: List of rankings (each ranking is a list of items, best first)
        method: Aggregation method:
            - "borda": Borda count (sum of ranks)
            - "kemeny": Kemeny optimal (minimizes disagreements)

    Returns:
        Consensus ranking (best item first)

    Example:
        >>> rankings = [
        ...     [0, 1, 2],  # Judge 1 ranking
        ...     [0, 2, 1],  # Judge 2 ranking
        ...     [1, 0, 2],  # Judge 3 ranking
        ... ]
        >>> consensus = aggregate_rankings(rankings)
        >>> print(f"Consensus: {consensus}")
    """
    if not rankings:
        return []

    # Get all items
    all_items = set()
    for ranking in rankings:
        all_items.update(ranking)
    items = sorted(all_items)
    n = len(items)

    if method == "borda":
        # Borda count: sum ranks (lower is better)
        scores = {item: 0 for item in items}
        for ranking in rankings:
            for rank, item in enumerate(ranking):
                scores[item] += rank
        return sorted(items, key=lambda x: scores[x])

    elif method == "kemeny":
        # Kemeny: minimize total Kendall tau distance
        # This is NP-hard, exact solution requires factorial enumeration
        if n > 10:
            raise ComputationalLimitError(
                f"Kemeny aggregation is NP-hard and requires factorial enumeration. "
                f"For n={n} items, this is computationally infeasible (10! = 3.6M, n! grows rapidly). "
                f"Use method='borda' for a polynomial-time approximation."
            )

        # Brute force for small n (factorial complexity)
        from itertools import permutations

        min_distance = float("inf")
        best_ranking = list(items)

        for perm in permutations(items):
            total_distance = sum(
                _kendall_tau_distance(list(perm), ranking)
                for ranking in rankings
            )
            if total_distance < min_distance:
                min_distance = total_distance
                best_ranking = list(perm)

        return best_ranking

    else:
        raise ValueError(f"Unknown method: {method}. Use 'borda' or 'kemeny'.")


# =============================================================================
# RANKING COMPARISON METRICS
# =============================================================================


def compute_kendall_tau(
    ranking1: list[int],
    ranking2: list[int],
) -> float:
    """
    Compute Kendall tau correlation between two rankings.

    Kendall tau measures the ordinal association between two rankings.
    It counts the number of concordant minus discordant pairs, normalized
    to [-1, 1].

    tau = (concordant - discordant) / n_pairs

    where n_pairs = n*(n-1)/2

    Args:
        ranking1: First ranking (list of items, best first)
        ranking2: Second ranking (list of items, best first)

    Returns:
        Kendall tau correlation in [-1, 1]:
        - 1: Perfect agreement
        - 0: No correlation
        - -1: Perfect disagreement (reversed)

    Example:
        >>> r1 = [0, 1, 2, 3]  # Original ranking
        >>> r2 = [0, 2, 1, 3]  # One swap
        >>> tau = compute_kendall_tau(r1, r2)
        >>> print(f"Kendall tau: {tau:.3f}")  # 0.667

    References:
        Kendall, M. G. (1938). A new measure of rank correlation. Biometrika.
    """
    if len(ranking1) != len(ranking2):
        # Handle rankings of different lengths by using common items
        common = set(ranking1) & set(ranking2)
        ranking1 = [x for x in ranking1 if x in common]
        ranking2 = [x for x in ranking2 if x in common]

    if len(ranking1) < 2:
        return 1.0  # Trivially correlated

    n = len(ranking1)
    n_pairs = n * (n - 1) // 2

    # Build position lookup for ranking2
    pos2 = {item: i for i, item in enumerate(ranking2)}

    # Count concordant and discordant pairs
    concordant = 0
    discordant = 0

    for i in range(n):
        for j in range(i + 1, n):
            item_i = ranking1[i]
            item_j = ranking1[j]

            # In ranking1: item_i < item_j (i.e., item_i ranked higher)
            # Check if same in ranking2
            if pos2[item_i] < pos2[item_j]:
                concordant += 1
            else:
                discordant += 1

    tau = (concordant - discordant) / n_pairs
    return tau


def _kendall_tau_distance(
    ranking1: list[int],
    ranking2: list[int],
) -> int:
    """Compute Kendall tau distance (number of discordant pairs)."""
    common = set(ranking1) & set(ranking2)
    r1 = [x for x in ranking1 if x in common]
    r2 = [x for x in ranking2 if x in common]

    if len(r1) < 2:
        return 0

    pos2 = {item: i for i, item in enumerate(r2)}
    distance = 0

    for i in range(len(r1)):
        for j in range(i + 1, len(r1)):
            if pos2[r1[i]] > pos2[r1[j]]:
                distance += 1

    return distance


def compute_spearman_footrule(
    ranking1: list[int],
    ranking2: list[int],
    normalize: bool = True,
) -> float:
    """
    Compute Spearman footrule distance between two rankings.

    The footrule measures the sum of absolute rank differences:
        F(r1, r2) = sum_i |rank1(i) - rank2(i)|

    This is faster to compute than Kendall tau and has nice statistical
    properties.

    Args:
        ranking1: First ranking (list of items, best first)
        ranking2: Second ranking (list of items, best first)
        normalize: If True, normalize to [0, 1] range

    Returns:
        Footrule distance. If normalized:
        - 0: Identical rankings
        - 1: Maximally different

    Example:
        >>> r1 = [0, 1, 2, 3]
        >>> r2 = [3, 2, 1, 0]  # Reversed
        >>> d = compute_spearman_footrule(r1, r2)
        >>> print(f"Footrule distance: {d:.3f}")

    References:
        Diaconis, P., & Graham, R. L. (1977). Spearman's footrule as a
        measure of disarray. Journal of the Royal Statistical Society B.
    """
    # Find common items
    common = set(ranking1) & set(ranking2)
    r1 = [x for x in ranking1 if x in common]
    r2 = [x for x in ranking2 if x in common]

    if len(r1) == 0:
        return 0.0

    n = len(r1)

    # Build position lookups
    pos1 = {item: i for i, item in enumerate(r1)}
    pos2 = {item: i for i, item in enumerate(r2)}

    # Sum absolute differences
    footrule = sum(abs(pos1[item] - pos2[item]) for item in common)

    if normalize:
        # Maximum footrule for n items is floor(n^2 / 2)
        max_footrule = n * n // 2
        return footrule / max_footrule if max_footrule > 0 else 0.0

    return float(footrule)


def compute_rank_biased_overlap(
    ranking1: list[int],
    ranking2: list[int],
    p: float = 0.9,
) -> float:
    """
    Compute Rank-Biased Overlap (RBO) for top-weighted comparison.

    RBO emphasizes agreement at the top of the rankings, which is often
    more important in practice (e.g., search results, recommendations).

    The parameter p controls the weight given to top positions:
    - p close to 1: Equal weight to all positions (approaches Jaccard)
    - p close to 0: Only top positions matter

    Args:
        ranking1: First ranking (list of items, best first)
        ranking2: Second ranking (list of items, best first)
        p: Persistence parameter in (0, 1). Higher = more weight to deep positions.
            Default 0.9 weights top 10 positions heavily.

    Returns:
        RBO score in [0, 1]:
        - 1: Identical rankings
        - 0: Completely disjoint (no overlap)

    Example:
        >>> r1 = [0, 1, 2, 3, 4]
        >>> r2 = [0, 1, 3, 2, 5]  # Same top-2, different rest
        >>> rbo = compute_rank_biased_overlap(r1, r2, p=0.9)
        >>> print(f"RBO: {rbo:.3f}")

    References:
        Webber, W., Moffat, A., & Zobel, J. (2010). A similarity measure
        for indefinite rankings. ACM Transactions on Information Systems.
    """
    if not 0 < p < 1:
        raise ValueError("p must be in (0, 1)")

    if not ranking1 or not ranking2:
        return 0.0

    # Compute RBO using the extrapolated formula
    # RBO = (1-p) * sum_{d=1}^{k} p^{d-1} * A_d
    # where A_d is the overlap ratio at depth d

    k = min(len(ranking1), len(ranking2))

    # Sets seen at each depth
    set1 = set()
    set2 = set()

    rbo = 0.0
    agreement_sum = 0.0

    for d in range(1, k + 1):
        set1.add(ranking1[d - 1])
        set2.add(ranking2[d - 1])

        # Overlap at depth d
        overlap = len(set1 & set2)
        agreement = overlap / d

        # Weight by p^(d-1)
        weight = p ** (d - 1)
        agreement_sum += agreement * weight

    rbo = (1 - p) * agreement_sum

    # Add extrapolation for infinite tail
    # This accounts for the remaining probability mass
    overlap_at_k = len(set1 & set2)
    if k > 0:
        rbo += (p ** k) * (overlap_at_k / k)

    return rbo


def compare_rankings(
    ranking1: list[int],
    ranking2: list[int],
    p: float = 0.9,
) -> RankingComparisonResult:
    """
    Comprehensive comparison of two rankings using multiple metrics.

    Computes Kendall tau, Spearman footrule, and RBO to give a complete
    picture of ranking similarity.

    Args:
        ranking1: First ranking (list of items, best first)
        ranking2: Second ranking (list of items, best first)
        p: RBO persistence parameter

    Returns:
        RankingComparisonResult with all metrics

    Example:
        >>> r1 = [0, 1, 2, 3, 4]
        >>> r2 = [0, 2, 1, 3, 5]
        >>> result = compare_rankings(r1, r2)
        >>> print(result.summary())
    """
    start_time = time.perf_counter()

    kendall_tau = compute_kendall_tau(ranking1, ranking2)
    footrule = compute_spearman_footrule(ranking1, ranking2)
    rbo = compute_rank_biased_overlap(ranking1, ranking2, p)

    # Count common items
    common = set(ranking1) & set(ranking2)

    computation_time = (time.perf_counter() - start_time) * 1000

    return RankingComparisonResult(
        kendall_tau=kendall_tau,
        spearman_footrule=footrule,
        rank_biased_overlap=rbo,
        rbo_parameter=p,
        num_common_items=len(common),
        len_ranking1=len(ranking1),
        len_ranking2=len(ranking2),
        computation_time_ms=computation_time,
    )


# =============================================================================
# TECH-FRIENDLY ALIASES
# =============================================================================

fit_bt_model = fit_bradley_terry
"""Legacy alias: use fit_bradley_terry instead."""

kendall_correlation = compute_kendall_tau
"""Legacy alias: use compute_kendall_tau instead."""

spearman_footrule = compute_spearman_footrule
"""Legacy alias: use compute_spearman_footrule instead."""

rbo = compute_rank_biased_overlap
"""Legacy alias: use compute_rank_biased_overlap instead."""
