from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pyrevealed.core.types import Cycle
from pyrevealed.core.mixins import ResultSummaryMixin
from pyrevealed.core.display import ResultDisplayMixin, ResultPlotMixin

__all__ = [
    "RegularityViolation",
    "RegularityResult",
    "AttentionOverloadResult",
    "SwapsIndexResult",
    "ObservationContributionResult",
    "StatusQuoBiasResult",
    "BradleyTerryResult",
    "RankingComparisonResult",
    "StochasticTransitivityResult",
    "MinimumCostIndexResult",
    "DecoyEffectResult",
    "CompromiseEffectResult",
    "BootstrapCIResult",
    "PredictiveSuccessResult",
]


@dataclass(frozen=True)
class RegularityViolation(ResultDisplayMixin):
    """
    Details of a single regularity violation.

    Regularity states that P(x|A) >= P(x|B) when A is a subset of B.
    A violation means adding options INCREASED choice probability.

    Attributes:
        item: The item whose probability increased when options were added
        subset_menu_idx: Index of the smaller menu A
        superset_menu_idx: Index of the larger menu B (A ⊂ B)
        prob_in_subset: P(x|A) - probability in smaller menu
        prob_in_superset: P(x|B) - probability in larger menu
        magnitude: P(x|B) - P(x|A), positive means violation
    """

    item: int
    subset_menu_idx: int
    superset_menu_idx: int
    prob_in_subset: float
    prob_in_superset: float
    magnitude: float


@dataclass(frozen=True)
class RegularityResult(ResultDisplayMixin, ResultPlotMixin):
    """
    Result of standalone regularity test for stochastic choice data.

    Regularity (Luce axiom) states that adding options should never
    INCREASE choice probability. Violations indicate decoy effects,
    context-dependent preferences, or consideration set changes.

    Attributes:
        satisfies_regularity: True if no violations found
        violations: List of RegularityViolation details
        worst_violation: The violation with largest magnitude
        violation_rate: Fraction of testable pairs that violate
        num_testable_pairs: Total number of subset pairs tested
        computation_time_ms: Time taken in milliseconds
    """

    satisfies_regularity: bool
    violations: list[RegularityViolation]
    worst_violation: RegularityViolation | None
    violation_rate: float
    num_testable_pairs: int
    computation_time_ms: float

    @property
    def num_violations(self) -> int:
        """Number of regularity violations found."""
        return len(self.violations)

    @property
    def is_consistent(self) -> bool:
        """Alias for satisfies_regularity."""
        return self.satisfies_regularity

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns 1 - violation_rate.
        """
        return 1.0 - self.violation_rate

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("REGULARITY TEST REPORT")]

        # Status
        status = m._format_status(
            self.satisfies_regularity,
            "REGULARITY SATISFIED",
            "REGULARITY VIOLATED"
        )
        lines.append(f"\nStatus: {status}")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("Satisfies Regularity", self.satisfies_regularity))
        lines.append(m._format_metric("Violations", self.num_violations))
        lines.append(m._format_metric("Testable Pairs", self.num_testable_pairs))
        lines.append(m._format_metric("Violation Rate", f"{self.violation_rate:.2%}"))

        # Show worst violation
        if self.worst_violation:
            lines.append(m._format_section("Worst Violation"))
            wv = self.worst_violation
            lines.append(f"  Item {wv.item}: P(x|small) = {wv.prob_in_subset:.3f}")
            lines.append(f"             P(x|large) = {wv.prob_in_superset:.3f}")
            lines.append(f"  Magnitude: {wv.magnitude:.4f}")

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        if self.satisfies_regularity:
            lines.append("  Choice probabilities decrease with menu expansion.")
            lines.append("  Consistent with standard random utility models.")
        else:
            lines.append("  Adding options sometimes INCREASES choice probability.")
            lines.append("  Suggests decoy effects or consideration set changes.")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        violations_list = []
        for v in self.violations:
            violations_list.append({
                "item": v.item,
                "subset_menu_idx": v.subset_menu_idx,
                "superset_menu_idx": v.superset_menu_idx,
                "prob_in_subset": v.prob_in_subset,
                "prob_in_superset": v.prob_in_superset,
                "magnitude": v.magnitude,
            })
        return {
            "satisfies_regularity": self.satisfies_regularity,
            "num_violations": self.num_violations,
            "violation_rate": self.violation_rate,
            "num_testable_pairs": self.num_testable_pairs,
            "violations": violations_list,
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        status = "satisfied" if self.satisfies_regularity else f"{self.num_violations} violations"
        return f"RegularityResult({status}, rate={self.violation_rate:.2%})"


@dataclass(frozen=True)
class AttentionOverloadResult(ResultDisplayMixin, ResultPlotMixin):
    """
    Result of attention overload test (Lleras et al. 2017 "When More is Less").

    Tests whether choice quality degrades as menu size increases,
    indicating cognitive overload.

    Attributes:
        has_overload: True if significant quality decline detected
        critical_menu_size: Menu size where overload begins (None if no overload)
        overload_severity: 0-1 measure of how severe the overload is
        menu_size_quality: Dict mapping menu size to average quality score
        regression_slope: Slope of quality ~ log(menu_size) regression
        p_value: Statistical significance of the slope
        num_observations: Total observations analyzed
        computation_time_ms: Time taken in milliseconds
    """

    has_overload: bool
    critical_menu_size: int | None
    overload_severity: float
    menu_size_quality: dict[int, float]
    regression_slope: float
    p_value: float
    num_observations: int
    computation_time_ms: float

    @property
    def is_overloaded(self) -> bool:
        """Alias for has_overload."""
        return self.has_overload

    @property
    def max_quality_size(self) -> int | None:
        """Menu size with highest quality (optimal menu size)."""
        if not self.menu_size_quality:
            return None
        return max(self.menu_size_quality.keys(), key=lambda k: self.menu_size_quality[k])

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns 1 - overload_severity.
        """
        return 1.0 - self.overload_severity

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("ATTENTION OVERLOAD TEST REPORT")]

        # Status
        status = "OVERLOAD DETECTED" if self.has_overload else "NO OVERLOAD"
        lines.append(f"\nStatus: {status}")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("Has Overload", self.has_overload))
        lines.append(m._format_metric("Overload Severity", f"{self.overload_severity:.2%}"))
        if self.critical_menu_size is not None:
            lines.append(m._format_metric("Critical Menu Size", self.critical_menu_size))
        lines.append(m._format_metric("Regression Slope", self.regression_slope))
        lines.append(m._format_metric("P-value", self.p_value))
        lines.append(m._format_metric("Observations", self.num_observations))

        # Quality by menu size
        if self.menu_size_quality:
            lines.append(m._format_section("Quality by Menu Size"))
            for size in sorted(self.menu_size_quality.keys())[:6]:
                quality = self.menu_size_quality[size]
                lines.append(f"  Size {size}: {quality:.3f}")

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        if self.has_overload:
            lines.append("  Choice quality declines with larger menus.")
            if self.critical_menu_size:
                lines.append(f"  Recommend menus smaller than {self.critical_menu_size} items.")
        else:
            lines.append("  No significant quality decline with menu size.")
            lines.append("  Choice quality remains stable across menu sizes.")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "has_overload": self.has_overload,
            "critical_menu_size": self.critical_menu_size,
            "overload_severity": self.overload_severity,
            "menu_size_quality": self.menu_size_quality,
            "regression_slope": self.regression_slope,
            "p_value": self.p_value,
            "num_observations": self.num_observations,
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        if self.has_overload:
            return f"AttentionOverloadResult(overload, critical={self.critical_menu_size})"
        return f"AttentionOverloadResult(no overload)"


@dataclass(frozen=True)
class SwapsIndexResult(ResultDisplayMixin, ResultPlotMixin):
    """
    Result of swaps index computation (Apesteguia & Ballester 2015 JPE).

    The swaps index counts the minimum number of preference swaps
    needed to make the data consistent. More interpretable than AEI.

    Attributes:
        swaps_count: Minimum swaps for consistency
        swaps_normalized: 0 = consistent, 1 = maximally inconsistent
        max_possible_swaps: Maximum possible swaps for this dataset
        swap_pairs: List of (obs_i, obs_j) pairs to flip
        is_consistent: True if no swaps needed
        method: Algorithm used ("greedy" or "optimal")
        computation_time_ms: Time taken in milliseconds
    """

    swaps_count: int
    swaps_normalized: float
    max_possible_swaps: int
    swap_pairs: list[tuple[int, int]]
    is_consistent: bool
    method: str
    computation_time_ms: float

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns 1 - swaps_normalized.
        """
        return 1.0 - self.swaps_normalized

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("SWAPS INDEX REPORT")]

        # Status
        status = "CONSISTENT" if self.is_consistent else f"{self.swaps_count} SWAPS NEEDED"
        lines.append(f"\nStatus: {status}")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("Swaps Count", self.swaps_count))
        lines.append(m._format_metric("Swaps Normalized", f"{self.swaps_normalized:.2%}"))
        lines.append(m._format_metric("Max Possible Swaps", self.max_possible_swaps))
        lines.append(m._format_metric("Is Consistent", self.is_consistent))
        lines.append(m._format_metric("Method", self.method))

        # Show swap pairs
        if self.swap_pairs:
            lines.append(m._format_section("Swap Pairs"))
            for i, (obs_i, obs_j) in enumerate(self.swap_pairs[:5]):
                lines.append(f"  Swap {i+1}: obs {obs_i} <-> obs {obs_j}")
            if len(self.swap_pairs) > 5:
                lines.append(f"  ... and {len(self.swap_pairs) - 5} more")

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        if self.is_consistent:
            lines.append("  Data is consistent - no preference swaps needed.")
        else:
            lines.append(f"  Need {self.swaps_count} preference flip(s) for consistency.")
            lines.append(f"  This represents {self.swaps_normalized:.1%} of max possible.")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "swaps_count": self.swaps_count,
            "swaps_normalized": self.swaps_normalized,
            "max_possible_swaps": self.max_possible_swaps,
            "swap_pairs": self.swap_pairs,
            "is_consistent": self.is_consistent,
            "method": self.method,
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        return f"SwapsIndexResult(swaps={self.swaps_count}, normalized={self.swaps_normalized:.2%})"


@dataclass(frozen=True)
class ObservationContributionResult(ResultDisplayMixin, ResultPlotMixin):
    """
    Result of per-observation contribution analysis (Varian 1990).

    Identifies which observations contribute most to inconsistency.

    Attributes:
        contributions: Per-observation contribution score (0-1)
        worst_observations: List of (obs_idx, contribution) sorted by contribution
        removal_impact: Dict mapping obs_idx to AEI improvement if removed
        cycle_participation: Dict mapping obs_idx to number of cycles involved
        base_aei: AEI of the full dataset
        method: Method used ("removal" or "cycle_count")
        computation_time_ms: Time taken in milliseconds
    """

    contributions: NDArray[np.float64]
    worst_observations: list[tuple[int, float]]
    removal_impact: dict[int, float]
    cycle_participation: dict[int, int]
    base_aei: float
    method: str
    computation_time_ms: float

    @property
    def num_observations(self) -> int:
        """Number of observations analyzed."""
        return len(self.contributions)

    @property
    def num_problematic(self) -> int:
        """Number of observations with non-zero contribution."""
        return sum(1 for c in self.contributions if c > 0)

    @property
    def max_contribution(self) -> float:
        """Maximum contribution score."""
        return float(np.max(self.contributions)) if len(self.contributions) > 0 else 0.0

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns the base AEI.
        """
        return self.base_aei

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("OBSERVATION CONTRIBUTION REPORT")]

        # Status
        if self.num_problematic == 0:
            status = "ALL OBSERVATIONS CONSISTENT"
        else:
            status = f"{self.num_problematic} PROBLEMATIC OBSERVATIONS"
        lines.append(f"\nStatus: {status}")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("Base AEI", self.base_aei))
        lines.append(m._format_metric("Observations", self.num_observations))
        lines.append(m._format_metric("Problematic", self.num_problematic))
        lines.append(m._format_metric("Max Contribution", self.max_contribution))
        lines.append(m._format_metric("Method", self.method))

        # Worst observations
        if self.worst_observations:
            lines.append(m._format_section("Worst Observations"))
            for obs_idx, contrib in self.worst_observations[:5]:
                impact = self.removal_impact.get(obs_idx, 0.0)
                cycles = self.cycle_participation.get(obs_idx, 0)
                lines.append(f"  Obs {obs_idx}: contribution={contrib:.3f}, "
                           f"cycles={cycles}, removal_impact={impact:.3f}")

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        if self.num_problematic == 0:
            lines.append("  All observations are consistent.")
        else:
            worst_idx, worst_contrib = self.worst_observations[0]
            lines.append(f"  Observation {worst_idx} is responsible for "
                       f"{worst_contrib:.1%} of inconsistency.")
            impact = self.removal_impact.get(worst_idx, 0.0)
            if impact > 0:
                lines.append(f"  Removing it would improve AEI by {impact:.3f}.")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "contributions": self.contributions.tolist(),
            "worst_observations": self.worst_observations,
            "removal_impact": self.removal_impact,
            "cycle_participation": self.cycle_participation,
            "base_aei": self.base_aei,
            "num_observations": self.num_observations,
            "num_problematic": self.num_problematic,
            "max_contribution": self.max_contribution,
            "method": self.method,
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        return f"ObservationContributionResult(n={self.num_observations}, problematic={self.num_problematic})"


@dataclass(frozen=True)
class StatusQuoBiasResult(ResultDisplayMixin, ResultPlotMixin):
    """
    Result of status quo bias test (Masatlioglu & Ok 2005).

    Tests whether default options are chosen more than rational
    preference would predict.

    Attributes:
        has_status_quo_bias: True if significant bias detected
        default_advantage: Average probability boost for defaults (0-1)
        bias_by_item: Dict mapping item to its bias measure
        p_value: Statistical significance
        num_defaults: Number of menus with identified defaults
        num_observations: Total observations analyzed
        computation_time_ms: Time taken in milliseconds
    """

    has_status_quo_bias: bool
    default_advantage: float
    bias_by_item: dict[int, float]
    p_value: float
    num_defaults: int
    num_observations: int
    computation_time_ms: float

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns 1 - default_advantage (less bias = better).
        """
        return 1.0 - min(1.0, self.default_advantage)

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("STATUS QUO BIAS TEST REPORT")]

        # Status
        status = "BIAS DETECTED" if self.has_status_quo_bias else "NO BIAS"
        lines.append(f"\nStatus: {status}")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("Has Status Quo Bias", self.has_status_quo_bias))
        lines.append(m._format_metric("Default Advantage", f"{self.default_advantage:.2%}"))
        lines.append(m._format_metric("P-value", self.p_value))
        lines.append(m._format_metric("Menus with Defaults", self.num_defaults))
        lines.append(m._format_metric("Observations", self.num_observations))

        # Bias by item
        if self.bias_by_item:
            lines.append(m._format_section("Bias by Item"))
            sorted_items = sorted(self.bias_by_item.items(), key=lambda x: -x[1])
            for item, bias in sorted_items[:5]:
                lines.append(f"  Item {item}: {bias:.2%} advantage")

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        if self.has_status_quo_bias:
            lines.append("  Default options are chosen more than preference predicts.")
            lines.append(f"  Defaults have ~{self.default_advantage:.0%} probability boost.")
        else:
            lines.append("  No significant status quo bias detected.")
            lines.append("  Choices appear preference-driven, not default-driven.")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "has_status_quo_bias": self.has_status_quo_bias,
            "default_advantage": self.default_advantage,
            "bias_by_item": self.bias_by_item,
            "p_value": self.p_value,
            "num_defaults": self.num_defaults,
            "num_observations": self.num_observations,
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        if self.has_status_quo_bias:
            return f"StatusQuoBiasResult(bias={self.default_advantage:.2%})"
        return "StatusQuoBiasResult(no bias)"


# =============================================================================
# PHASE 3 ADDITIONS - RANKING AND PAIRWISE COMPARISON
# =============================================================================


@dataclass(frozen=True)
class BradleyTerryResult(ResultDisplayMixin, ResultPlotMixin):
    """
    Result of Bradley-Terry model fitting.

    The Bradley-Terry model represents choice probabilities as:
        P(i beats j) = pi_i / (pi_i + pi_j)

    where pi_i > 0 is the "strength" parameter for item i.

    This is the core model for RLHF preference learning and can be used
    to convert pairwise comparisons into a total ordering.

    Attributes:
        scores: Dict mapping item index to strength score
        log_likelihood: Log-likelihood of the fitted model
        converged: Whether optimization converged
        ranking: Items ordered by strength (highest first)
        num_comparisons: Total number of pairwise comparisons
        num_items: Number of unique items
        computation_time_ms: Time taken in milliseconds

    References:
        Bradley, R. A., & Terry, M. E. (1952). Rank analysis of incomplete
        block designs: I. The method of paired comparisons. Biometrika.
    """

    scores: dict[int, float]
    log_likelihood: float
    converged: bool
    ranking: list[int]
    num_comparisons: int
    num_items: int
    computation_time_ms: float

    def predict_probability(self, item_i: int, item_j: int) -> float:
        """Predict P(item_i beats item_j)."""
        if item_i not in self.scores or item_j not in self.scores:
            return 0.5
        s_i = self.scores[item_i]
        s_j = self.scores[item_j]
        exp_i = np.exp(s_i - max(s_i, s_j))
        exp_j = np.exp(s_j - max(s_i, s_j))
        return exp_i / (exp_i + exp_j)

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns 1.0 if converged, 0.0 otherwise.
        """
        return 1.0 if self.converged else 0.0

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("BRADLEY-TERRY MODEL REPORT")]

        # Status
        status = "CONVERGED" if self.converged else "NOT CONVERGED"
        lines.append(f"\nStatus: {status}")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("Log-Likelihood", f"{self.log_likelihood:.4f}"))
        lines.append(m._format_metric("Converged", self.converged))
        lines.append(m._format_metric("Number of Items", self.num_items))
        lines.append(m._format_metric("Number of Comparisons", self.num_comparisons))

        # Top-ranked items
        if self.ranking:
            lines.append(m._format_section("Top Rankings"))
            for rank, item in enumerate(self.ranking[:5], 1):
                score = self.scores.get(item, 0)
                lines.append(f"  {rank}. Item {item} (score: {score:.4f})")
            if len(self.ranking) > 5:
                lines.append(f"  ... and {len(self.ranking) - 5} more items")

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        lines.append("  Higher scores indicate stronger items.")
        lines.append("  P(i beats j) = exp(s_i) / (exp(s_i) + exp(s_j))")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "scores": self.scores,
            "log_likelihood": self.log_likelihood,
            "converged": self.converged,
            "ranking": self.ranking,
            "num_comparisons": self.num_comparisons,
            "num_items": self.num_items,
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        indicator = "[+]" if self.converged else "[-]"
        return f"BradleyTerryResult: {indicator} {self.num_items} items, LL={self.log_likelihood:.2f}"


@dataclass(frozen=True)
class RankingComparisonResult(ResultDisplayMixin, ResultPlotMixin):
    """
    Result of comparing two rankings using multiple metrics.

    Provides Kendall tau, Spearman footrule, and Rank-Biased Overlap
    to give a complete picture of ranking similarity.

    Attributes:
        kendall_tau: Kendall tau correlation in [-1, 1]
        spearman_footrule: Normalized footrule distance in [0, 1]
        rank_biased_overlap: RBO score in [0, 1]
        rbo_parameter: The p parameter used for RBO
        num_common_items: Number of items in both rankings
        len_ranking1: Length of first ranking
        len_ranking2: Length of second ranking
        computation_time_ms: Time taken in milliseconds
    """

    kendall_tau: float
    spearman_footrule: float
    rank_biased_overlap: float
    rbo_parameter: float
    num_common_items: int
    len_ranking1: int
    len_ranking2: int
    computation_time_ms: float

    @property
    def are_similar(self) -> bool:
        """True if rankings are similar (tau > 0.5 and RBO > 0.5)."""
        return self.kendall_tau > 0.5 and self.rank_biased_overlap > 0.5

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns average of normalized metrics.
        """
        # Kendall tau is in [-1, 1], normalize to [0, 1]
        tau_norm = (self.kendall_tau + 1) / 2
        # Footrule is distance, invert
        footrule_sim = 1 - self.spearman_footrule
        # RBO is already similarity
        return (tau_norm + footrule_sim + self.rank_biased_overlap) / 3

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("RANKING COMPARISON REPORT")]

        # Status
        status = "SIMILAR" if self.are_similar else "DIFFERENT"
        lines.append(f"\nStatus: {status}")

        # Metrics
        lines.append(m._format_section("Similarity Metrics"))
        lines.append(m._format_metric("Kendall Tau", f"{self.kendall_tau:.4f}"))
        lines.append(m._format_metric("Spearman Footrule", f"{self.spearman_footrule:.4f}"))
        lines.append(m._format_metric("Rank-Biased Overlap", f"{self.rank_biased_overlap:.4f}"))
        lines.append(m._format_metric("RBO Parameter (p)", f"{self.rbo_parameter:.2f}"))

        # Size info
        lines.append(m._format_section("Ranking Sizes"))
        lines.append(m._format_metric("Ranking 1 Length", self.len_ranking1))
        lines.append(m._format_metric("Ranking 2 Length", self.len_ranking2))
        lines.append(m._format_metric("Common Items", self.num_common_items))

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        if self.kendall_tau > 0.7:
            lines.append("  Strong ordinal agreement (Kendall tau > 0.7)")
        elif self.kendall_tau > 0.3:
            lines.append("  Moderate ordinal agreement")
        else:
            lines.append("  Weak or no ordinal agreement")

        if self.rank_biased_overlap > 0.8:
            lines.append("  Very similar top items (RBO > 0.8)")
        elif self.rank_biased_overlap > 0.5:
            lines.append("  Moderate top agreement (RBO > 0.5)")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "kendall_tau": self.kendall_tau,
            "spearman_footrule": self.spearman_footrule,
            "rank_biased_overlap": self.rank_biased_overlap,
            "rbo_parameter": self.rbo_parameter,
            "num_common_items": self.num_common_items,
            "len_ranking1": self.len_ranking1,
            "len_ranking2": self.len_ranking2,
            "are_similar": self.are_similar,
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        return f"RankingComparisonResult(tau={self.kendall_tau:.3f}, RBO={self.rank_biased_overlap:.3f})"


# =============================================================================
# PHASE 3 ADDITIONS - STOCHASTIC CHOICE EXTENSIONS
# =============================================================================


@dataclass(frozen=True)
class StochasticTransitivityResult(ResultDisplayMixin, ResultPlotMixin):
    """
    Result of stochastic transitivity tests (WST/MST/SST).

    Tests whether choice probabilities satisfy stochastic transitivity axioms:
    - WST (Weak): P(a,b) > 0.5 and P(b,c) > 0.5 => P(a,c) > 0.5
    - MST (Moderate): P(a,b) > 0.5 and P(b,c) > 0.5 => P(a,c) >= max(P(a,b), P(b,c))
    - SST (Strong): P(a,b) > 0.5 and P(b,c) > 0.5 => P(a,c) >= min(P(a,b), P(b,c))

    Attributes:
        satisfies_wst: True if weak stochastic transitivity holds
        satisfies_mst: True if moderate stochastic transitivity holds
        satisfies_sst: True if strong stochastic transitivity holds
        wst_violations: List of (a, b, c) triples violating WST
        mst_violations: List of (a, b, c) triples violating MST
        sst_violations: List of (a, b, c) triples violating SST
        num_testable_triples: Number of triples tested
        computation_time_ms: Time taken in milliseconds

    References:
        Luce, R. D. (1959). Individual Choice Behavior.
        Tversky, A. (1969). Intransitivity of preferences. Psychological Review.
    """

    satisfies_wst: bool
    satisfies_mst: bool
    satisfies_sst: bool
    wst_violations: list[tuple[int, int, int]]
    mst_violations: list[tuple[int, int, int]]
    sst_violations: list[tuple[int, int, int]]
    num_testable_triples: int
    computation_time_ms: float

    @property
    def strongest_satisfied(self) -> str:
        """Return the strongest transitivity level satisfied."""
        if self.satisfies_sst:
            return "SST"
        elif self.satisfies_mst:
            return "MST"
        elif self.satisfies_wst:
            return "WST"
        else:
            return "None"

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns 1.0 for SST, 0.67 for MST, 0.33 for WST, 0.0 for none.
        """
        if self.satisfies_sst:
            return 1.0
        elif self.satisfies_mst:
            return 0.67
        elif self.satisfies_wst:
            return 0.33
        else:
            return 0.0

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("STOCHASTIC TRANSITIVITY TEST REPORT")]

        # Status
        lines.append(f"\nStrongest Level Satisfied: {self.strongest_satisfied}")

        # Metrics
        lines.append(m._format_section("Transitivity Levels"))
        lines.append(m._format_metric("Weak (WST)", self.satisfies_wst))
        lines.append(m._format_metric("Moderate (MST)", self.satisfies_mst))
        lines.append(m._format_metric("Strong (SST)", self.satisfies_sst))

        # Violations
        lines.append(m._format_section("Violations"))
        lines.append(m._format_metric("WST Violations", len(self.wst_violations)))
        lines.append(m._format_metric("MST Violations", len(self.mst_violations)))
        lines.append(m._format_metric("SST Violations", len(self.sst_violations)))
        lines.append(m._format_metric("Triples Tested", self.num_testable_triples))

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        if self.satisfies_sst:
            lines.append("  Choices satisfy Strong Stochastic Transitivity.")
            lines.append("  This is consistent with a strong utility model.")
        elif self.satisfies_mst:
            lines.append("  Choices satisfy Moderate Stochastic Transitivity.")
            lines.append("  This is consistent with moderate choice consistency.")
        elif self.satisfies_wst:
            lines.append("  Choices satisfy only Weak Stochastic Transitivity.")
            lines.append("  Preferences may have bounded or noisy utility.")
        else:
            lines.append("  Choices violate even Weak Stochastic Transitivity.")
            lines.append("  This suggests substantial preference inconsistency.")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "satisfies_wst": self.satisfies_wst,
            "satisfies_mst": self.satisfies_mst,
            "satisfies_sst": self.satisfies_sst,
            "num_wst_violations": len(self.wst_violations),
            "num_mst_violations": len(self.mst_violations),
            "num_sst_violations": len(self.sst_violations),
            "num_testable_triples": self.num_testable_triples,
            "strongest_satisfied": self.strongest_satisfied,
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        return f"StochasticTransitivityResult(strongest={self.strongest_satisfied})"


@dataclass(frozen=True)
class MinimumCostIndexResult(ResultDisplayMixin, ResultPlotMixin):
    """
    Result of Minimum Cost Index computation.

    The MCI measures the minimum monetary cost to break all GARP cycles.
    It is an alternative severity measure to AEI that has more direct
    economic interpretation.

    Attributes:
        mci_value: Minimum cost to break cycles (0 if consistent)
        mci_normalized: MCI normalized by total expenditure
        adjustments: Dict of observation -> expenditure adjustment
        cycles_broken: Number of cycles eliminated
        total_expenditure: Total expenditure in data
        is_consistent: True if no adjustments needed
        computation_time_ms: Time taken in milliseconds

    References:
        Dean, M., & Martin, D. (2016). Measuring rationality with the
        minimum cost of revealed preference violations.
        Review of Economics and Statistics.
    """

    mci_value: float
    mci_normalized: float
    adjustments: dict[int, float]
    cycles_broken: int
    total_expenditure: float
    is_consistent: bool
    computation_time_ms: float

    @property
    def num_adjustments(self) -> int:
        """Number of observations requiring adjustment."""
        return len(self.adjustments)

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns 1 - mci_normalized.
        """
        return max(0.0, 1.0 - self.mci_normalized)

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("MINIMUM COST INDEX REPORT")]

        # Status
        status = "CONSISTENT" if self.is_consistent else "VIOLATIONS FOUND"
        lines.append(f"\nStatus: {status}")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("MCI (Absolute)", f"{self.mci_value:.4f}"))
        lines.append(m._format_metric("MCI (Normalized)", f"{self.mci_normalized:.4f}"))
        lines.append(m._format_metric("Total Expenditure", f"{self.total_expenditure:.2f}"))
        lines.append(m._format_metric("Cycles Broken", self.cycles_broken))
        lines.append(m._format_metric("Observations Adjusted", self.num_adjustments))

        # Top adjustments
        if self.adjustments:
            lines.append(m._format_section("Largest Adjustments"))
            sorted_adj = sorted(self.adjustments.items(), key=lambda x: -abs(x[1]))
            for obs, adj in sorted_adj[:5]:
                lines.append(f"  Observation {obs}: {adj:.4f}")

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        if self.is_consistent:
            lines.append("  Data is GARP-consistent, no cost adjustments needed.")
        else:
            pct = self.mci_normalized * 100
            lines.append(f"  Removing violations costs ~{pct:.2f}% of expenditure.")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "mci_value": self.mci_value,
            "mci_normalized": self.mci_normalized,
            "num_adjustments": self.num_adjustments,
            "cycles_broken": self.cycles_broken,
            "total_expenditure": self.total_expenditure,
            "is_consistent": self.is_consistent,
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        indicator = "[+]" if self.is_consistent else "[-]"
        return f"MinimumCostIndexResult: {indicator} MCI={self.mci_normalized:.4f}"


@dataclass(frozen=True)
class DecoyEffectResult(ResultDisplayMixin, ResultPlotMixin):
    """
    Result of decoy/attraction effect detection.

    A decoy effect occurs when adding a dominated alternative D
    increases the choice share of the dominating alternative T
    (the "target") relative to a competitor C.

    Attributes:
        has_decoy_effect: True if significant decoy effects detected
        decoy_triples: List of (target, competitor, decoy) triples
        magnitude: Average probability boost to target
        vulnerabilities: Dict mapping item to decoy vulnerability score
        num_menus_tested: Number of menu pairs tested
        computation_time_ms: Time taken in milliseconds

    References:
        Huber, J., Payne, J. W., & Puto, C. (1982). Adding asymmetrically
        dominated alternatives. Journal of Consumer Research.
    """

    has_decoy_effect: bool
    decoy_triples: list[tuple[int, int, int]]
    magnitude: float
    vulnerabilities: dict[int, float]
    num_menus_tested: int
    computation_time_ms: float

    @property
    def num_decoys(self) -> int:
        """Number of decoy relationships detected."""
        return len(self.decoy_triples)

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher = more decoy effect.

        Returns the magnitude of decoy effect.
        """
        return min(1.0, self.magnitude)

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("DECOY EFFECT ANALYSIS REPORT")]

        # Status
        status = "DECOY EFFECT DETECTED" if self.has_decoy_effect else "NO DECOY EFFECT"
        lines.append(f"\nStatus: {status}")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("Has Decoy Effect", self.has_decoy_effect))
        lines.append(m._format_metric("Decoy Triples Found", self.num_decoys))
        lines.append(m._format_metric("Average Magnitude", f"{self.magnitude:.2%}"))
        lines.append(m._format_metric("Menus Tested", self.num_menus_tested))

        # Top decoy triples
        if self.decoy_triples:
            lines.append(m._format_section("Decoy Relationships"))
            for target, comp, decoy in self.decoy_triples[:5]:
                lines.append(f"  Target {target} boosted vs {comp} by decoy {decoy}")

        # Vulnerable items
        if self.vulnerabilities:
            lines.append(m._format_section("Most Vulnerable Items"))
            sorted_vuln = sorted(self.vulnerabilities.items(), key=lambda x: -x[1])
            for item, vuln in sorted_vuln[:3]:
                lines.append(f"  Item {item}: {vuln:.2%} vulnerability")

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        if self.has_decoy_effect:
            lines.append("  Choice behavior is influenced by dominated options.")
            lines.append("  This violates regularity (rational independence).")
        else:
            lines.append("  No significant decoy effects detected.")
            lines.append("  Choices appear to follow regularity.")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "has_decoy_effect": self.has_decoy_effect,
            "num_decoys": self.num_decoys,
            "magnitude": self.magnitude,
            "vulnerabilities": self.vulnerabilities,
            "num_menus_tested": self.num_menus_tested,
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        if self.has_decoy_effect:
            return f"DecoyEffectResult(n={self.num_decoys}, mag={self.magnitude:.2%})"
        return "DecoyEffectResult(none detected)"


@dataclass(frozen=True)
class CompromiseEffectResult(ResultDisplayMixin, ResultPlotMixin):
    """
    Result of compromise effect detection.

    A compromise effect occurs when adding extreme alternatives
    increases the choice share of middle/compromise options.

    Attributes:
        has_compromise_effect: True if significant compromise effects detected
        compromise_items: Items that benefit from being compromise options
        magnitude: Average probability boost to compromise options
        extreme_pairs: List of (extreme1, extreme2, compromise) triples
        num_menus_tested: Number of menu pairs tested
        computation_time_ms: Time taken in milliseconds

    References:
        Simonson, I. (1989). Choice based on reasons: The case of
        attraction and compromise effects. Journal of Consumer Research.
    """

    has_compromise_effect: bool
    compromise_items: list[int]
    magnitude: float
    extreme_pairs: list[tuple[int, int, int]]
    num_menus_tested: int
    computation_time_ms: float

    @property
    def num_compromises(self) -> int:
        """Number of compromise relationships detected."""
        return len(self.extreme_pairs)

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher = more effect."""
        return min(1.0, self.magnitude)

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("COMPROMISE EFFECT ANALYSIS REPORT")]

        # Status
        status = "COMPROMISE EFFECT DETECTED" if self.has_compromise_effect else "NO EFFECT"
        lines.append(f"\nStatus: {status}")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("Has Compromise Effect", self.has_compromise_effect))
        lines.append(m._format_metric("Compromise Items", len(self.compromise_items)))
        lines.append(m._format_metric("Average Magnitude", f"{self.magnitude:.2%}"))
        lines.append(m._format_metric("Menus Tested", self.num_menus_tested))

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        if self.has_compromise_effect:
            lines.append("  Middle options are preferred when extremes are added.")
            lines.append("  This indicates extremeness aversion in choices.")
        else:
            lines.append("  No significant compromise effects detected.")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "has_compromise_effect": self.has_compromise_effect,
            "compromise_items": self.compromise_items,
            "num_compromises": self.num_compromises,
            "magnitude": self.magnitude,
            "num_menus_tested": self.num_menus_tested,
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        if self.has_compromise_effect:
            return f"CompromiseEffectResult(n={self.num_compromises}, mag={self.magnitude:.2%})"
        return "CompromiseEffectResult(none detected)"


@dataclass(frozen=True)
class BootstrapCIResult(ResultDisplayMixin, ResultPlotMixin):
    """
    Result of bootstrap confidence interval computation.

    Provides confidence intervals for RP metrics like AEI, MPI, etc.

    Attributes:
        point_estimate: The original metric value
        ci_lower: Lower bound of confidence interval
        ci_upper: Upper bound of confidence interval
        confidence_level: Confidence level (e.g., 0.95)
        metric_name: Name of the metric
        bootstrap_distribution: Array of bootstrap samples
        std_error: Standard error of bootstrap distribution
        n_bootstrap: Number of bootstrap iterations
        computation_time_ms: Time taken in milliseconds

    References:
        Efron, B., & Tibshirani, R. J. (1994). An Introduction to the Bootstrap.
    """

    point_estimate: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    metric_name: str
    bootstrap_distribution: NDArray[np.float64]
    std_error: float
    n_bootstrap: int
    computation_time_ms: float

    @property
    def ci_width(self) -> float:
        """Width of confidence interval."""
        return self.ci_upper - self.ci_lower

    @property
    def is_precise(self) -> bool:
        """True if CI width is less than 0.1."""
        return self.ci_width < 0.1

    def score(self) -> float:
        """Return the point estimate as score."""
        return self.point_estimate

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header(f"BOOTSTRAP CI REPORT: {self.metric_name.upper()}")]

        # Status
        pct = int(self.confidence_level * 100)
        lines.append(f"\n{pct}% Confidence Interval: [{self.ci_lower:.4f}, {self.ci_upper:.4f}]")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("Point Estimate", f"{self.point_estimate:.4f}"))
        lines.append(m._format_metric("CI Lower", f"{self.ci_lower:.4f}"))
        lines.append(m._format_metric("CI Upper", f"{self.ci_upper:.4f}"))
        lines.append(m._format_metric("CI Width", f"{self.ci_width:.4f}"))
        lines.append(m._format_metric("Standard Error", f"{self.std_error:.4f}"))
        lines.append(m._format_metric("Bootstrap Samples", self.n_bootstrap))

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        if self.is_precise:
            lines.append("  Precise estimate (narrow confidence interval).")
        else:
            lines.append("  Wide confidence interval - interpret with caution.")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "point_estimate": self.point_estimate,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "confidence_level": self.confidence_level,
            "metric_name": self.metric_name,
            "std_error": self.std_error,
            "n_bootstrap": self.n_bootstrap,
            "ci_width": self.ci_width,
            "computation_time_ms": self.computation_time_ms,
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        pct = int(self.confidence_level * 100)
        return f"BootstrapCIResult({self.metric_name}: {self.point_estimate:.3f} [{self.ci_lower:.3f}, {self.ci_upper:.3f}] {pct}%)"


@dataclass(frozen=True)
class PredictiveSuccessResult(ResultDisplayMixin, ResultPlotMixin):
    """
    Result of predictive success computation.

    Measures how well a model predicts choices beyond chance,
    following Selten (1991)'s measure.

    Attributes:
        predictive_success: Hit rate minus false alarm rate
        hit_rate: Proportion of correct predictions
        false_alarm_rate: Proportion of incorrect predictions
        model_name: Name of the model tested
        num_predictions: Number of predictions made
        computation_time_ms: Time taken in milliseconds

    References:
        Selten, R. (1991). Properties of a measure of predictive success.
        Mathematical Social Sciences.
    """

    predictive_success: float
    hit_rate: float
    false_alarm_rate: float
    model_name: str
    num_predictions: int
    computation_time_ms: float

    def score(self) -> float:
        """Return predictive success as score."""
        return max(0.0, self.predictive_success)

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header(f"PREDICTIVE SUCCESS: {self.model_name.upper()}")]

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("Predictive Success", f"{self.predictive_success:.4f}"))
        lines.append(m._format_metric("Hit Rate", f"{self.hit_rate:.4f}"))
        lines.append(m._format_metric("False Alarm Rate", f"{self.false_alarm_rate:.4f}"))
        lines.append(m._format_metric("Predictions Made", self.num_predictions))

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        if self.predictive_success > 0.5:
            lines.append("  Model has strong predictive power.")
        elif self.predictive_success > 0:
            lines.append("  Model predicts better than chance.")
        else:
            lines.append("  Model does not improve on chance.")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "predictive_success": self.predictive_success,
            "hit_rate": self.hit_rate,
            "false_alarm_rate": self.false_alarm_rate,
            "model_name": self.model_name,
            "num_predictions": self.num_predictions,
            "computation_time_ms": self.computation_time_ms,
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        return f"PredictiveSuccessResult({self.model_name}: {self.predictive_success:.3f})"

