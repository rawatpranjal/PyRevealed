from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pyrevealed.core.types import Cycle
from pyrevealed.core.mixins import ResultSummaryMixin
from pyrevealed.core.display import ResultDisplayMixin, ResultPlotMixin

__all__ = [
    "AbstractWARPResult",
    "AbstractSARPResult",
    "CongruenceResult",
    "HoutmanMaksAbstractResult",
    "OrdinalUtilityResult",
    "MenuWARPResult",
    "MenuSARPResult",
    "MenuConsistencyResult",
    "MenuEfficiencyResult",
    "MenuPreferenceResult",
]


@dataclass(frozen=True)
class AbstractWARPResult(ResultDisplayMixin, ResultPlotMixin):
    """
    Result of WARP (Weak Axiom of Revealed Preference) check for menu-based choices.

    WARP for abstract choice: if x is chosen from a menu containing y,
    then y cannot be chosen from any menu containing x.

    Attributes:
        is_consistent: True if data satisfies abstract WARP
        violations: List of (t1, t2) pairs where choice at t1 reveals
            preference over choice at t2, but t2's choice was preferred to t1's
        revealed_preference_pairs: List of (x, y) pairs where x is revealed
            preferred to y (x chosen when y was available)
        computation_time_ms: Time taken in milliseconds
    """

    is_consistent: bool
    violations: list[tuple[int, int]]
    revealed_preference_pairs: list[tuple[int, int]]
    computation_time_ms: float

    @property
    def num_violations(self) -> int:
        """Number of WARP violations found."""
        return len(self.violations)

    @property
    def num_revealed_preferences(self) -> int:
        """Number of revealed preference relations found."""
        return len(self.revealed_preference_pairs)

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns 1.0 if consistent, 0.0 if violations exist.
        """
        return 1.0 if self.is_consistent else 0.0

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("ABSTRACT WARP TEST REPORT")]

        # Status
        status = m._format_status(self.is_consistent, "CONSISTENT", "VIOLATIONS FOUND")
        lines.append(f"\nStatus: {status}")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("Consistent", self.is_consistent))
        lines.append(m._format_metric("Violations", self.num_violations))
        lines.append(m._format_metric("Revealed Preferences", self.num_revealed_preferences))

        # Show violations if any
        if self.violations:
            lines.append(m._format_section("Violations"))
            lines.append(m._format_list(self.violations, max_items=5, item_name="pair"))

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        if self.is_consistent:
            lines.append("  No direct preference reversals in menu choices.")
            lines.append("  Satisfies Weak Axiom for abstract choice.")
        else:
            lines.append(f"  {self.num_violations} direct preference reversal(s) found.")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "is_consistent": self.is_consistent,
            "num_violations": self.num_violations,
            "num_revealed_preferences": self.num_revealed_preferences,
            "violations": self.violations,
            "revealed_preference_pairs": self.revealed_preference_pairs,
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        status = "consistent" if self.is_consistent else f"{self.num_violations} violations"
        return f"AbstractWARPResult({status})"


@dataclass(frozen=True)
class AbstractSARPResult(ResultDisplayMixin, ResultPlotMixin):
    """
    Result of SARP (Strict Axiom of Revealed Preference) check for menu-based choices.

    SARP for abstract choice: the transitive closure of revealed preference
    must be acyclic. Equivalently, there should be no cycle i1 R i2 R ... R in R i1
    where R is the revealed preference relation.

    Attributes:
        is_consistent: True if data satisfies abstract SARP
        violations: List of cycles found (each cycle is a tuple of item indices)
        revealed_preference_matrix: N x N boolean matrix R where R[x,y] = True
            iff x is directly revealed preferred to y
        transitive_closure: N x N boolean matrix R* (transitive closure of R)
        computation_time_ms: Time taken in milliseconds
    """

    is_consistent: bool
    violations: list[Cycle]
    revealed_preference_matrix: NDArray[np.bool_]
    transitive_closure: NDArray[np.bool_]
    computation_time_ms: float

    @property
    def num_violations(self) -> int:
        """Number of SARP violation cycles found."""
        return len(self.violations)

    @property
    def num_items(self) -> int:
        """Number of items in the analysis."""
        return self.revealed_preference_matrix.shape[0]

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns 1.0 if consistent, 0.0 if violations exist.
        """
        return 1.0 if self.is_consistent else 0.0

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("ABSTRACT SARP TEST REPORT")]

        # Status
        status = m._format_status(self.is_consistent, "CONSISTENT", "VIOLATIONS FOUND")
        lines.append(f"\nStatus: {status}")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("Consistent", self.is_consistent))
        lines.append(m._format_metric("Violations", self.num_violations))
        lines.append(m._format_metric("Items", self.num_items))

        # Show violations if any
        if self.violations:
            lines.append(m._format_section("Violation Cycles"))
            lines.append(m._format_list(self.violations, max_items=5, item_name="cycle"))

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        if self.is_consistent:
            lines.append("  No preference cycles in menu choices.")
            lines.append("  Choices are rationalizable by a preference ordering.")
        else:
            lines.append(f"  {self.num_violations} preference cycle(s) found.")
            lines.append("  Choices cannot be rationalized by any strict ordering.")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "is_consistent": self.is_consistent,
            "num_violations": self.num_violations,
            "num_items": self.num_items,
            "violations": [list(c) for c in self.violations],
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        status = "consistent" if self.is_consistent else f"{self.num_violations} violations"
        return f"AbstractSARPResult({status}, items={self.num_items})"


@dataclass(frozen=True)
class CongruenceResult(ResultDisplayMixin, ResultPlotMixin):
    """
    Result of Congruence (full rationalizability) check for menu-based choices.

    Congruence requires:
    1. SARP: No cycles in revealed preference
    2. Maximality: If x is chosen and y is in the menu, then x R* y
       (the choice must be maximal under the transitive preference relation)

    A dataset is rationalizable by a preference order iff it satisfies Congruence.

    Attributes:
        is_congruent: True if data satisfies Congruence axiom
        is_rationalizable: Alias for is_congruent (data is rationalizable)
        satisfies_sarp: True if SARP is satisfied
        maximality_violations: List of (t, unchosen_item) pairs where
            unchosen_item was not dominated by the choice
        sarp_result: Detailed SARP result
        computation_time_ms: Time taken in milliseconds
    """

    is_congruent: bool
    satisfies_sarp: bool
    maximality_violations: list[tuple[int, int]]
    sarp_result: AbstractSARPResult
    computation_time_ms: float

    @property
    def is_rationalizable(self) -> bool:
        """True if data can be rationalized by a preference ordering."""
        return self.is_congruent

    @property
    def num_maximality_violations(self) -> int:
        """Number of maximality violations found."""
        return len(self.maximality_violations)

    @property
    def num_sarp_violations(self) -> int:
        """Number of SARP violations found."""
        return self.sarp_result.num_violations

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns 1.0 if rationalizable, 0.0 otherwise.
        """
        return 1.0 if self.is_congruent else 0.0

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("CONGRUENCE TEST REPORT")]

        # Status
        status = m._format_status(self.is_congruent, "RATIONALIZABLE", "NOT RATIONALIZABLE")
        lines.append(f"\nStatus: {status}")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("Is Congruent", self.is_congruent))
        lines.append(m._format_metric("Satisfies SARP", self.satisfies_sarp))
        lines.append(m._format_metric("SARP Violations", self.num_sarp_violations))
        lines.append(m._format_metric("Maximality Violations", self.num_maximality_violations))

        # Show violations if any
        if self.maximality_violations:
            lines.append(m._format_section("Maximality Violations"))
            lines.append(m._format_list(self.maximality_violations, max_items=5, item_name="pair"))

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        if self.is_congruent:
            lines.append("  Choices are fully rationalizable by a preference ordering.")
            lines.append("  Both SARP and maximality conditions satisfied.")
        else:
            if not self.satisfies_sarp:
                lines.append("  SARP violated - preference cycles exist.")
            if self.maximality_violations:
                lines.append("  Maximality violated - chosen items not maximal.")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "is_congruent": self.is_congruent,
            "is_rationalizable": self.is_rationalizable,
            "satisfies_sarp": self.satisfies_sarp,
            "num_sarp_violations": self.num_sarp_violations,
            "num_maximality_violations": self.num_maximality_violations,
            "maximality_violations": self.maximality_violations,
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        if self.is_congruent:
            return "CongruenceResult(rationalizable)"
        return f"CongruenceResult(sarp={self.num_sarp_violations}, max={self.num_maximality_violations})"


@dataclass(frozen=True)
class HoutmanMaksAbstractResult(ResultDisplayMixin, ResultPlotMixin):
    """
    Result of Houtman-Maks efficiency index for menu-based choices.

    The Houtman-Maks index measures the minimum fraction of observations
    that must be removed to make the remaining data satisfy SARP/Congruence.

    Attributes:
        efficiency_index: 1 - (removed/total), in [0, 1]. Higher = more efficient.
        removed_observations: List of observation indices to remove
        remaining_observations: List of observation indices that are consistent
        num_total: Total number of observations
        computation_time_ms: Time taken in milliseconds
    """

    efficiency_index: float
    removed_observations: list[int]
    remaining_observations: list[int]
    num_total: int
    computation_time_ms: float

    @property
    def num_removed(self) -> int:
        """Number of observations to remove for consistency."""
        return len(self.removed_observations)

    @property
    def fraction_removed(self) -> float:
        """Fraction of observations removed (1 - efficiency_index)."""
        return 1.0 - self.efficiency_index

    @property
    def is_consistent(self) -> bool:
        """True if no observations need to be removed."""
        return self.num_removed == 0

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns the efficiency index directly.
        """
        return self.efficiency_index

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("HOUTMAN-MAKS ABSTRACT INDEX REPORT")]

        # Status
        if self.is_consistent:
            status = "FULLY CONSISTENT"
        elif self.efficiency_index >= 0.9:
            status = "MOSTLY CONSISTENT"
        elif self.efficiency_index >= 0.7:
            status = "MODERATE INCONSISTENCY"
        else:
            status = "HIGH INCONSISTENCY"
        lines.append(f"\nStatus: {status}")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("Efficiency Index", self.efficiency_index))
        lines.append(m._format_metric("Fraction Removed", self.fraction_removed))
        lines.append(m._format_metric("Total Observations", self.num_total))
        lines.append(m._format_metric("Removed Observations", self.num_removed))
        lines.append(m._format_metric("Remaining Observations", len(self.remaining_observations)))

        # Show removed observations
        if self.removed_observations:
            lines.append(m._format_section("Removed Observation Indices"))
            lines.append(m._format_list(self.removed_observations, max_items=10, item_name="observation"))

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        if self.is_consistent:
            lines.append("  All menu choices are consistent - no removal needed.")
        else:
            pct = self.fraction_removed * 100
            lines.append(f"  Remove {self.num_removed} observations ({pct:.1f}%) for consistency.")
            lines.append("  Remaining observations satisfy SARP/Congruence.")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "efficiency_index": self.efficiency_index,
            "fraction_removed": self.fraction_removed,
            "num_total": self.num_total,
            "num_removed": self.num_removed,
            "is_consistent": self.is_consistent,
            "removed_observations": self.removed_observations,
            "remaining_observations": self.remaining_observations,
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        if self.is_consistent:
            return f"HoutmanMaksAbstractResult(consistent, n={self.num_total})"
        return f"HoutmanMaksAbstractResult(eff={self.efficiency_index:.4f}, removed={self.num_removed})"


@dataclass(frozen=True)
class OrdinalUtilityResult(ResultDisplayMixin, ResultPlotMixin):
    """
    Result of ordinal utility (preference ranking) recovery for menu-based choices.

    Unlike cardinal utility recovery, this only produces an ordinal ranking
    of items based on revealed preferences. No numerical utility values.

    Attributes:
        success: True if preferences could be ranked (SARP satisfied)
        utility_ranking: Dict mapping item index to rank (0 = most preferred)
        utility_values: Optional array of utility values if computed via LP
        preference_order: List of item indices from most to least preferred
        num_items: Number of items ranked
        is_complete: True if all items could be ranked (no incomparable pairs)
        computation_time_ms: Time taken in milliseconds
    """

    success: bool
    utility_ranking: dict[int, int] | None
    utility_values: NDArray[np.float64] | None
    preference_order: list[int] | None
    num_items: int
    is_complete: bool
    computation_time_ms: float

    @property
    def most_preferred(self) -> int | None:
        """Index of the most preferred item, or None if failed."""
        if self.preference_order is not None and len(self.preference_order) > 0:
            return self.preference_order[0]
        return None

    @property
    def least_preferred(self) -> int | None:
        """Index of the least preferred item, or None if failed."""
        if self.preference_order is not None and len(self.preference_order) > 0:
            return self.preference_order[-1]
        return None

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns 1.0 if successful, 0.0 if failed.
        """
        return 1.0 if self.success else 0.0

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("ORDINAL UTILITY RECOVERY REPORT")]

        # Status
        status = m._format_status(self.success, "SUCCESS", "FAILED")
        lines.append(f"\nStatus: {status}")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("Recovery Successful", self.success))
        lines.append(m._format_metric("Number of Items", self.num_items))
        lines.append(m._format_metric("Complete Ranking", self.is_complete))

        if self.success:
            lines.append(m._format_metric("Most Preferred", self.most_preferred))
            lines.append(m._format_metric("Least Preferred", self.least_preferred))

        # Show preference order
        if self.preference_order:
            lines.append(m._format_section("Preference Order (most to least)"))
            lines.append(f"  {' > '.join(str(i) for i in self.preference_order[:10])}")
            if len(self.preference_order) > 10:
                lines.append(f"  ... ({len(self.preference_order) - 10} more items)")

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        if self.success:
            lines.append("  Ordinal preference ranking successfully recovered.")
            if self.is_complete:
                lines.append("  All items fully ranked (no incomparable pairs).")
            else:
                lines.append("  Some items are incomparable (partial ordering).")
        else:
            lines.append("  Failed to recover ordinal preferences.")
            lines.append("  Data may contain preference cycles (SARP violation).")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        result = {
            "success": self.success,
            "num_items": self.num_items,
            "is_complete": self.is_complete,
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }
        if self.preference_order is not None:
            result["preference_order"] = self.preference_order
            result["most_preferred"] = self.most_preferred
            result["least_preferred"] = self.least_preferred
        if self.utility_ranking is not None:
            result["utility_ranking"] = self.utility_ranking
        if self.utility_values is not None:
            result["utility_values"] = self.utility_values.tolist()
        return result

    def __repr__(self) -> str:
        """Compact string representation."""
        if self.success:
            return f"OrdinalUtilityResult(success, items={self.num_items})"
        return "OrdinalUtilityResult(failed)"


# =============================================================================
# TECH-FRIENDLY ALIASES FOR ABSTRACT CHOICE RESULTS
# =============================================================================

# MenuWARPResult: Tech-friendly alias for AbstractWARPResult
MenuWARPResult = AbstractWARPResult
"""
Tech-friendly alias for AbstractWARPResult.

Use this to check if menu-based choices are WARP-consistent
(no direct preference reversals).
"""

# MenuSARPResult: Tech-friendly alias for AbstractSARPResult
MenuSARPResult = AbstractSARPResult
"""
Tech-friendly alias for AbstractSARPResult.

Use this to check if menu-based choices are SARP-consistent
(no transitive preference cycles).
"""

# MenuConsistencyResult: Tech-friendly alias for CongruenceResult
MenuConsistencyResult = CongruenceResult
"""
Tech-friendly alias for CongruenceResult.

Use this to check if menu-based choices are fully rationalizable
by a preference ordering.
"""

# MenuEfficiencyResult: Tech-friendly alias for HoutmanMaksAbstractResult
MenuEfficiencyResult = HoutmanMaksAbstractResult
"""
Tech-friendly alias for HoutmanMaksAbstractResult.

Measures what fraction of observations must be removed for consistency.
"""

# MenuPreferenceResult: Tech-friendly alias for OrdinalUtilityResult
MenuPreferenceResult = OrdinalUtilityResult
"""
Tech-friendly alias for OrdinalUtilityResult.

Contains the recovered ordinal preference ranking over items.
"""
