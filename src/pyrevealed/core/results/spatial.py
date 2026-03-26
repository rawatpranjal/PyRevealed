from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pyrevealed.core.types import Cycle
from pyrevealed.core.mixins import ResultSummaryMixin
from pyrevealed.core.display import ResultDisplayMixin, ResultPlotMixin

__all__ = [
    "IdealPointResult",
    "SeparabilityResult",
    "PreferenceAnchorResult",
    "FeatureIndependenceResult",
]


@dataclass(frozen=True)
class IdealPointResult(ResultDisplayMixin, ResultPlotMixin):
    """
    Result of ideal point estimation in feature space.

    The ideal point model assumes the user prefers items closer to their
    ideal location in the feature space (Euclidean preferences).

    Attributes:
        ideal_point: D-dimensional vector representing user's ideal location
        is_euclidean_rational: True if all choices are consistent with some ideal point
        violations: List of (choice_set_idx, unchosen_item_idx) pairs where the
            unchosen item was actually closer to the estimated ideal point
        num_violations: Number of choices inconsistent with Euclidean preferences
        explained_variance: Fraction of choice variance explained by ideal point model
        mean_distance_to_chosen: Average distance from ideal point to chosen items
        computation_time_ms: Time taken in milliseconds
    """

    ideal_point: NDArray[np.float64]
    is_euclidean_rational: bool
    violations: list[tuple[int, int]]
    num_violations: int
    explained_variance: float
    mean_distance_to_chosen: float
    computation_time_ms: float

    @property
    def num_dimensions(self) -> int:
        """Number of feature dimensions D."""
        return len(self.ideal_point)

    @property
    def violation_rate(self) -> float:
        """Fraction of choices that violate Euclidean preferences."""
        if self.num_violations == 0:
            return 0.0
        total = len(self.violations) + self.num_violations  # Approximation
        return self.num_violations / max(total, 1)

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns explained_variance (fraction of choices explained by model).
        """
        return self.explained_variance

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("IDEAL POINT ESTIMATION REPORT")]

        # Status
        status = m._format_status(self.is_euclidean_rational,
                                  "EUCLIDEAN RATIONAL", "VIOLATIONS FOUND")
        lines.append(f"\nStatus: {status}")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("Euclidean Rational", self.is_euclidean_rational))
        lines.append(m._format_metric("Explained Variance", self.explained_variance))
        lines.append(m._format_metric("Number of Violations", self.num_violations))
        lines.append(m._format_metric("Mean Distance to Chosen", self.mean_distance_to_chosen))
        lines.append(m._format_metric("Dimensions", self.num_dimensions))

        # Ideal point coordinates
        lines.append(m._format_section("Ideal Point Coordinates"))
        for i, coord in enumerate(self.ideal_point):
            lines.append(f"  Dimension {i}: {coord:.4f}")

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        if self.is_euclidean_rational:
            lines.append("  All choices consistent with Euclidean preferences.")
            lines.append("  User consistently prefers items closer to ideal point.")
        else:
            lines.append(f"  {self.num_violations} choices inconsistent with Euclidean model.")
            lines.append(f"  Model explains {self.explained_variance*100:.1f}% of choice variance.")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "ideal_point": self.ideal_point.tolist(),
            "is_euclidean_rational": self.is_euclidean_rational,
            "num_violations": self.num_violations,
            "explained_variance": self.explained_variance,
            "mean_distance_to_chosen": self.mean_distance_to_chosen,
            "num_dimensions": self.num_dimensions,
            "violations": self.violations,
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation with [+]/[-] indicator."""
        indicator = "[+]" if self.is_euclidean_rational else "[-]"
        status = "RATIONAL" if self.is_euclidean_rational else f"{self.num_violations} violations"
        return f"IdealPointResult: {indicator} {status} (dims={self.num_dimensions}, var={self.explained_variance:.4f})"

    def short_summary(self) -> str:
        """Return one-liner with [+]/[-] indicator."""
        indicator = "[+]" if self.is_euclidean_rational else "[-]"
        return f"Ideal Point: {indicator} var={self.explained_variance:.4f}"


@dataclass(frozen=True)
class SeparabilityResult(ResultDisplayMixin, ResultPlotMixin):
    """
    Result of separability test for groups of goods.

    Tests whether utility can be decomposed as U(x_A, x_B) = V(u_A(x_A), u_B(x_B)),
    meaning the goods can be priced independently.

    Attributes:
        is_separable: True if groups can be treated independently
        group_a_indices: Indices of goods in Group A
        group_b_indices: Indices of goods in Group B
        cross_effect_strength: Measure of how much Group A affects Group B demand
            (0 = fully independent, 1 = fully dependent)
        within_group_a_consistency: GARP consistency score within Group A
        within_group_b_consistency: GARP consistency score within Group B
        recommendation: Strategy recommendation string
        computation_time_ms: Time taken in milliseconds
    """

    is_separable: bool
    group_a_indices: list[int]
    group_b_indices: list[int]
    cross_effect_strength: float
    within_group_a_consistency: float
    within_group_b_consistency: float
    recommendation: str
    computation_time_ms: float

    @property
    def can_price_independently(self) -> bool:
        """True if groups can be priced without considering cross-effects."""
        return self.is_separable and self.cross_effect_strength < 0.1

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns 1 - cross_effect_strength (lower cross-effects = better separability).
        """
        return max(0.0, 1.0 - self.cross_effect_strength)

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("SEPARABILITY TEST REPORT")]

        # Status
        status = m._format_status(self.is_separable, "SEPARABLE", "NOT SEPARABLE")
        lines.append(f"\nStatus: {status}")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("Is Separable", self.is_separable))
        lines.append(m._format_metric("Cross-Effect Strength", self.cross_effect_strength))
        lines.append(m._format_metric("Can Price Independently", self.can_price_independently))

        # Group details
        lines.append(m._format_section("Group Details"))
        lines.append(m._format_metric("Group A Indices", str(self.group_a_indices)))
        lines.append(m._format_metric("Group A Consistency", self.within_group_a_consistency))
        lines.append(m._format_metric("Group B Indices", str(self.group_b_indices)))
        lines.append(m._format_metric("Group B Consistency", self.within_group_b_consistency))

        # Recommendation
        lines.append(m._format_section("Recommendation"))
        lines.append(f"  {self.recommendation}")

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        if self.is_separable:
            lines.append("  Goods can be grouped independently for pricing/optimization.")
            lines.append("  Cross-price effects are negligible between groups.")
        else:
            lines.append("  Significant cross-effects exist between groups.")
            lines.append("  Groups should be analyzed together, not separately.")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "is_separable": self.is_separable,
            "group_a_indices": self.group_a_indices,
            "group_b_indices": self.group_b_indices,
            "cross_effect_strength": self.cross_effect_strength,
            "within_group_a_consistency": self.within_group_a_consistency,
            "within_group_b_consistency": self.within_group_b_consistency,
            "can_price_independently": self.can_price_independently,
            "recommendation": self.recommendation,
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation with [+]/[-] indicator."""
        indicator = "[+]" if self.is_separable else "[-]"
        status = "SEPARABLE" if self.is_separable else "COUPLED"
        return f"SeparabilityResult: {indicator} {status} (cross_effect={self.cross_effect_strength:.4f})"

    def short_summary(self) -> str:
        """Return one-liner with [+]/[-] indicator."""
        indicator = "[+]" if self.is_separable else "[-]"
        status = "SEPARABLE" if self.is_separable else "COUPLED"
        return f"Separability: {indicator} {status}"


# PreferenceAnchorResult: Result of preference anchor (ideal point) estimation
PreferenceAnchorResult = IdealPointResult
"""
Tech-friendly alias for IdealPointResult.

The preference anchor is the user's ideal location in feature space.
Useful for recommendation explainability and personalization.
"""

# FeatureIndependenceResult: Result of feature independence test
FeatureIndependenceResult = SeparabilityResult
"""
Tech-friendly alias for SeparabilityResult.

Tests whether feature groups (e.g., product categories) can be
priced/optimized independently without cross-effects.
"""
