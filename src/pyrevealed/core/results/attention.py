from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pyrevealed.core.types import Cycle
from pyrevealed.core.mixins import ResultSummaryMixin
from pyrevealed.core.display import ResultDisplayMixin, ResultPlotMixin

__all__ = [
    "WARPLAResult",
    "RandomAttentionResult",
    "RUMConsistencyResult",
    "RevealedAttentionResult",
    "StochasticAttentionResult",
    "MixtureRationalityResult",
]


@dataclass(frozen=True)
class WARPLAResult(ResultDisplayMixin, ResultPlotMixin):
    """
    Result of WARP with Limited Attention test (Masatlioglu et al. 2012).

    WARP(LA) is a weakening of WARP that allows for limited attention.
    The axiom states: for any nonempty S, there exists x* in S such that
    for any T including x*, if c(T) is in S and c(T) != c(T\\x*), then c(T) = x*.

    This is equivalent to testing if the revealed preference relation P
    (where xPy iff exists T with c(T)=x != c(T\\y)) is acyclic.

    Attributes:
        satisfies_warp_la: True if data satisfies WARP(LA)
        revealed_preference: List of (x, y) pairs where x is revealed preferred to y
        transitive_closure: List of (x, y) pairs in the transitive closure of P
        attention_filter: Dict mapping menus to inferred consideration sets (if consistent)
        recovered_preference: Tuple representing a compatible preference ordering (if consistent)
        violations: List of cycles in the revealed preference relation
        num_observations: Number of choice observations
        computation_time_ms: Time taken in milliseconds
    """

    satisfies_warp_la: bool
    revealed_preference: list[tuple[int, int]]
    transitive_closure: list[tuple[int, int]]
    attention_filter: dict[frozenset[int], set[int]] | None
    recovered_preference: tuple[int, ...] | None
    violations: list[tuple[int, ...]]
    num_observations: int
    computation_time_ms: float

    @property
    def num_violations(self) -> int:
        """Number of violation cycles in revealed preference."""
        return len(self.violations)

    @property
    def is_consistent(self) -> bool:
        """Alias for satisfies_warp_la for consistency with other results."""
        return self.satisfies_warp_la

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns 1.0 if consistent, 0.0 if violations exist.
        """
        return 1.0 if self.satisfies_warp_la else 0.0

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("WARP(LA) TEST REPORT")]

        # Status
        status = m._format_status(
            self.satisfies_warp_la,
            "CONSISTENT WITH LIMITED ATTENTION",
            "VIOLATIONS FOUND"
        )
        lines.append(f"\nStatus: {status}")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("Satisfies WARP(LA)", self.satisfies_warp_la))
        lines.append(m._format_metric("Observations", self.num_observations))
        lines.append(m._format_metric("Revealed Preferences", len(self.revealed_preference)))
        lines.append(m._format_metric("Transitive Closure Size", len(self.transitive_closure)))
        lines.append(m._format_metric("Violations", self.num_violations))

        # Show recovered preference if available
        if self.recovered_preference:
            lines.append(m._format_section("Recovered Preference"))
            pref_str = " > ".join(str(x) for x in self.recovered_preference)
            lines.append(f"  {pref_str}")

        # Show first violation if any
        if self.violations:
            lines.append(m._format_section("First Violation Cycle"))
            cycle_str = " -> ".join(str(x) for x in self.violations[0])
            lines.append(f"  {cycle_str}")
            if len(self.violations) > 1:
                lines.append(f"  ... and {len(self.violations) - 1} more cycles")

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        if self.satisfies_warp_la:
            lines.append("  Behavior is consistent with preference maximization")
            lines.append("  under limited attention (attention filter model).")
        else:
            lines.append("  Behavior cannot be rationalized even with limited attention.")
            lines.append(f"  Found {self.num_violations} preference cycle(s).")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "satisfies_warp_la": self.satisfies_warp_la,
            "num_violations": self.num_violations,
            "violations": [list(c) for c in self.violations],
            "revealed_preference": self.revealed_preference,
            "transitive_closure": self.transitive_closure,
            "recovered_preference": list(self.recovered_preference) if self.recovered_preference else None,
            "num_observations": self.num_observations,
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        status = "consistent" if self.satisfies_warp_la else f"{self.num_violations} violations"
        return f"WARPLAResult({status}, {self.computation_time_ms:.2f}ms)"


@dataclass(frozen=True)
class RandomAttentionResult(ResultDisplayMixin, ResultPlotMixin):
    """
    Result of Random Attention Model (RAM) estimation (Cattaneo et al. 2020).

    RAM extends the limited attention model to stochastic choice data.
    The model assumes a fixed preference ordering and random attention:
    each item has a probability of being considered, and the choice
    is the most preferred item among those considered.

    Attributes:
        is_ram_consistent: True if data is consistent with RAM
        preference_ranking: Estimated preference ordering (best to worst)
        attention_bounds: Dict mapping (menu, item) to (lower, upper) attention bounds
        item_attention_scores: Array of estimated attention probability per item
        test_statistic: Test statistic for RAM consistency
        p_value: P-value for the test (if bootstrap performed)
        compatible_preferences: List of preference orderings compatible with the data
        assumption: RAM variant used ("monotonic", "independent", "general")
        num_observations: Number of observations
        computation_time_ms: Time taken in milliseconds
    """

    is_ram_consistent: bool
    preference_ranking: tuple[int, ...] | None
    attention_bounds: dict[tuple[frozenset[int], int], tuple[float, float]]
    item_attention_scores: NDArray[np.float64]
    test_statistic: float
    p_value: float
    compatible_preferences: list[tuple[int, ...]]
    assumption: str
    num_observations: int
    computation_time_ms: float

    @property
    def num_items(self) -> int:
        """Number of distinct items."""
        return len(self.item_attention_scores)

    @property
    def num_compatible_preferences(self) -> int:
        """Number of preference orderings compatible with data."""
        return len(self.compatible_preferences)

    def get_attention_ranking(self) -> list[int]:
        """Return items sorted by attention score (highest first)."""
        return list(np.argsort(self.item_attention_scores)[::-1])

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns 1 - test_statistic (capped at 0-1 range).
        """
        return max(0.0, min(1.0, 1.0 - self.test_statistic))

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("RANDOM ATTENTION MODEL REPORT")]

        # Status
        status = m._format_status(
            self.is_ram_consistent,
            "RAM CONSISTENT",
            "RAM INCONSISTENT"
        )
        lines.append(f"\nStatus: {status}")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("RAM Consistent", self.is_ram_consistent))
        lines.append(m._format_metric("Assumption", self.assumption))
        lines.append(m._format_metric("Test Statistic", self.test_statistic))
        lines.append(m._format_metric("P-Value", self.p_value))
        lines.append(m._format_metric("Observations", self.num_observations))
        lines.append(m._format_metric("Compatible Preferences", self.num_compatible_preferences))

        # Show estimated preference if available
        if self.preference_ranking:
            lines.append(m._format_section("Estimated Preference"))
            pref_str = " > ".join(str(x) for x in self.preference_ranking)
            lines.append(f"  {pref_str}")

        # Attention scores
        if len(self.item_attention_scores) > 0:
            lines.append(m._format_section("Attention Scores (top 5)"))
            sorted_idx = np.argsort(self.item_attention_scores)[::-1]
            for i in sorted_idx[:5]:
                lines.append(f"  Item {i}: {self.item_attention_scores[i]:.4f}")

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        if self.is_ram_consistent:
            lines.append("  Stochastic choices are consistent with random attention.")
            lines.append("  Consumer maximizes fixed preference among considered items.")
        else:
            lines.append("  Data cannot be explained by random attention model.")
            lines.append(f"  Test statistic: {self.test_statistic:.4f}")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "is_ram_consistent": self.is_ram_consistent,
            "preference_ranking": list(self.preference_ranking) if self.preference_ranking else None,
            "item_attention_scores": self.item_attention_scores.tolist(),
            "test_statistic": self.test_statistic,
            "p_value": self.p_value,
            "num_compatible_preferences": self.num_compatible_preferences,
            "assumption": self.assumption,
            "num_observations": self.num_observations,
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        status = "consistent" if self.is_ram_consistent else "inconsistent"
        return f"RandomAttentionResult({status}, p={self.p_value:.4f})"


# =============================================================================
# P1: RUM CONSISTENCY RESULTS (Block & Marschak 1960, Smeulders et al. 2021)
# =============================================================================


@dataclass(frozen=True)
class RUMConsistencyResult(ResultDisplayMixin, ResultPlotMixin):
    """
    Result of Random Utility Model (RUM) consistency test.

    Tests whether stochastic choice data can be rationalized by ANY mixture
    of preference orderings (i.e., whether a RUM representation exists).

    Uses column generation algorithm from Smeulders et al. (2021) for efficiency.

    Attributes:
        is_rum_consistent: True if data is RUM-rationalizable
        distance_to_rum: L1 distance to nearest RUM (0 = perfectly consistent)
        regularity_satisfied: True if regularity condition holds
        num_orderings_used: Number of orderings in the sparse representation
        rationalizing_distribution: Dict mapping preference orderings to probabilities
        num_iterations: Number of column generation iterations
        constraint_violations: List of violated constraints (if inconsistent)
        computation_time_ms: Time taken in milliseconds
    """

    is_rum_consistent: bool
    distance_to_rum: float
    regularity_satisfied: bool
    num_orderings_used: int
    rationalizing_distribution: dict[tuple[int, ...], float] | None
    num_iterations: int
    constraint_violations: list[str]
    computation_time_ms: float

    @property
    def is_consistent(self) -> bool:
        """Alias for is_rum_consistent."""
        return self.is_rum_consistent

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns 1 - distance_to_rum (capped at 0-1).
        """
        return max(0.0, 1.0 - min(1.0, self.distance_to_rum))

    def get_top_orderings(self, n: int = 5) -> list[tuple[tuple[int, ...], float]]:
        """Return top n orderings by probability."""
        if not self.rationalizing_distribution:
            return []
        sorted_orderings = sorted(
            self.rationalizing_distribution.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_orderings[:n]

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("RUM CONSISTENCY TEST REPORT")]

        # Status
        status = m._format_status(
            self.is_rum_consistent,
            "RUM CONSISTENT",
            "RUM INCONSISTENT"
        )
        lines.append(f"\nStatus: {status}")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("RUM Consistent", self.is_rum_consistent))
        lines.append(m._format_metric("Distance to RUM", self.distance_to_rum))
        lines.append(m._format_metric("Regularity Satisfied", self.regularity_satisfied))
        lines.append(m._format_metric("Orderings Used", self.num_orderings_used))
        lines.append(m._format_metric("Column Gen Iterations", self.num_iterations))

        # Show top orderings if available
        if self.rationalizing_distribution:
            lines.append(m._format_section("Top Orderings"))
            for ordering, prob in self.get_top_orderings(3):
                pref_str = " > ".join(str(x) for x in ordering)
                lines.append(f"  {pref_str}: {prob:.4f}")

        # Show violations if any
        if self.constraint_violations:
            lines.append(m._format_section("Constraint Violations"))
            for v in self.constraint_violations[:3]:
                lines.append(f"  {v}")

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        if self.is_rum_consistent:
            lines.append("  Stochastic choices can be rationalized by a mixture")
            lines.append("  of utility-maximizing preference orderings.")
            lines.append(f"  Sparse representation uses {self.num_orderings_used} orderings.")
        else:
            lines.append("  Data cannot be explained by ANY random utility model.")
            lines.append(f"  Distance to nearest RUM: {self.distance_to_rum:.4f}")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        dist_dict = None
        if self.rationalizing_distribution:
            dist_dict = {
                str(k): v for k, v in self.rationalizing_distribution.items()
            }
        return {
            "is_rum_consistent": self.is_rum_consistent,
            "distance_to_rum": self.distance_to_rum,
            "regularity_satisfied": self.regularity_satisfied,
            "num_orderings_used": self.num_orderings_used,
            "rationalizing_distribution": dist_dict,
            "num_iterations": self.num_iterations,
            "constraint_violations": self.constraint_violations,
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        status = "consistent" if self.is_rum_consistent else f"dist={self.distance_to_rum:.4f}"
        return f"RUMConsistencyResult({status}, {self.num_orderings_used} orderings)"


# Tech-friendly aliases for new results
RevealedAttentionResult = WARPLAResult
"""Tech-friendly alias for WARP(LA) result."""

StochasticAttentionResult = RandomAttentionResult
"""Tech-friendly alias for Random Attention Model result."""

MixtureRationalityResult = RUMConsistencyResult
"""Tech-friendly alias for RUM consistency result."""

