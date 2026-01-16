"""Result dataclasses for behavioral signal analysis.

This module provides result containers for various behavioral analysis algorithms.

Tech-Friendly Names (Primary):
    - ConsistencyResult: Behavioral consistency check result
    - IntegrityResult: Behavioral integrity/noise score result
    - ConfusionResult: Confusion/exploitability metric result
    - LatentValueResult: Latent preference value extraction result
    - PreferenceAnchorResult: Preference anchor (ideal point) result
    - FeatureIndependenceResult: Feature independence/separability result

Economics Names (Deprecated Aliases):
    - GARPResult -> ConsistencyResult
    - AEIResult -> IntegrityResult
    - MPIResult -> ConfusionResult
    - UtilityRecoveryResult -> LatentValueResult
    - IdealPointResult -> PreferenceAnchorResult
    - SeparabilityResult -> FeatureIndependenceResult
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pyrevealed.core.types import Cycle
from pyrevealed.core.mixins import ResultSummaryMixin


@dataclass(frozen=True)
class GARPResult:
    """
    Result of GARP (Generalized Axiom of Revealed Preference) consistency test.

    GARP is satisfied when there are no cycles in the revealed preference
    relation that include at least one strict preference. A violation indicates
    the consumer made inconsistent choices.

    Attributes:
        is_consistent: True if data satisfies GARP (no violations found)
        violations: List of violation cycles (tuples of observation indices)
        direct_revealed_preference: T x T boolean matrix R where R[i,j] = True
            iff bundle i is directly revealed preferred to bundle j
            (i.e., p_i @ x_i >= p_i @ x_j)
        transitive_closure: T x T boolean matrix R* (transitive closure of R)
        strict_revealed_preference: T x T boolean matrix P where P[i,j] = True
            iff bundle i is strictly revealed preferred to bundle j
            (i.e., p_i @ x_i > p_i @ x_j)
        computation_time_ms: Time taken to compute result in milliseconds
    """

    is_consistent: bool
    violations: list[Cycle]
    direct_revealed_preference: NDArray[np.bool_]
    transitive_closure: NDArray[np.bool_]
    strict_revealed_preference: NDArray[np.bool_]
    computation_time_ms: float

    @property
    def num_violations(self) -> int:
        """Number of violation cycles found."""
        return len(self.violations)

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns 1.0 if consistent (no violations), 0.0 if violations exist.
        """
        return 1.0 if self.is_consistent else 0.0

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("GARP CONSISTENCY REPORT")]

        # Status
        status = m._format_status(self.is_consistent, "CONSISTENT", "VIOLATIONS FOUND")
        lines.append(f"\nStatus: {status}")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("Consistent", self.is_consistent))
        lines.append(m._format_metric("Violations", self.num_violations))
        n_obs = self.direct_revealed_preference.shape[0]
        lines.append(m._format_metric("Observations", n_obs))

        # Show first violation if any
        if self.violations:
            lines.append(m._format_section("First Violation Cycle"))
            lines.append(f"  {self.violations[0]}")
            if len(self.violations) > 1:
                lines.append(f"  ... and {len(self.violations) - 1} more cycles")

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        if self.is_consistent:
            lines.append("  Behavior is consistent with utility maximization.")
            lines.append("  No revealed preference cycles detected.")
        else:
            lines.append("  Behavior violates GARP - inconsistent with utility maximization.")
            lines.append(f"  Found {self.num_violations} preference cycle(s).")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "is_consistent": self.is_consistent,
            "num_violations": self.num_violations,
            "violations": [list(c) for c in self.violations],
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        status = "consistent" if self.is_consistent else f"{self.num_violations} violations"
        return f"GARPResult({status}, {self.computation_time_ms:.2f}ms)"


@dataclass(frozen=True)
class AEIResult:
    """
    Result of Afriat Efficiency Index computation.

    The AEI measures how close consumer behavior is to perfect rationality.
    It is defined as: sup{e in [0,1] : <R_e, P_e> is acyclic}
    where R_e is the revealed preference relation with budgets deflated by e.

    An AEI of 1.0 means perfectly consistent behavior.
    An AEI of 0.5 means the consumer wastes ~50% of their budget on
    inconsistent choices.

    Attributes:
        efficiency_index: The computed AEI score in [0, 1]
        is_perfectly_consistent: True if AEI = 1.0 (data satisfies GARP)
        garp_result_at_threshold: GARP result at the efficiency threshold
        binary_search_iterations: Number of iterations in binary search
        tolerance: Convergence tolerance used
        computation_time_ms: Time taken in milliseconds
    """

    efficiency_index: float
    is_perfectly_consistent: bool
    garp_result_at_threshold: GARPResult
    binary_search_iterations: int
    tolerance: float
    computation_time_ms: float

    @property
    def waste_fraction(self) -> float:
        """Fraction of budget wasted on inconsistent choices (1 - AEI)."""
        return 1.0 - self.efficiency_index

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns the efficiency index directly (already in [0, 1]).
        """
        return self.efficiency_index

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("AFRIAT EFFICIENCY INDEX REPORT")]

        # Status
        if self.is_perfectly_consistent:
            status = "PERFECT (AEI = 1.0)"
        elif self.efficiency_index >= 0.95:
            status = "EXCELLENT"
        elif self.efficiency_index >= 0.9:
            status = "GOOD"
        elif self.efficiency_index >= 0.7:
            status = "MODERATE"
        else:
            status = "LOW"
        lines.append(f"\nStatus: {status}")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("Efficiency Index (AEI)", self.efficiency_index))
        lines.append(m._format_metric("Waste Fraction", self.waste_fraction))
        lines.append(m._format_metric("Perfectly Consistent", self.is_perfectly_consistent))
        lines.append(m._format_metric("Binary Search Iterations", self.binary_search_iterations))
        lines.append(m._format_metric("Tolerance", self.tolerance))

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        lines.append(f"  {m._format_interpretation(self.efficiency_index, 'efficiency')}")
        if self.waste_fraction > 0:
            pct = self.waste_fraction * 100
            lines.append(f"  Approximately {pct:.1f}% of budget on inconsistent choices.")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "efficiency_index": self.efficiency_index,
            "is_perfectly_consistent": self.is_perfectly_consistent,
            "waste_fraction": self.waste_fraction,
            "binary_search_iterations": self.binary_search_iterations,
            "tolerance": self.tolerance,
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        return f"AEIResult(aei={self.efficiency_index:.4f}, {self.computation_time_ms:.2f}ms)"


@dataclass(frozen=True)
class MPIResult:
    """
    Result of Money Pump Index computation.

    The MPI measures the percentage of total expenditure that could be
    "pumped" from a consumer exhibiting cyclic preferences by an arbitrager.

    For a cycle k1 -> k2 -> ... -> kn -> k1:
    MPI = sum(p_ki @ (x_ki - x_{ki+1})) / sum(p_ki @ x_ki)

    Attributes:
        mpi_value: Maximum MPI across all violation cycles (0 if consistent)
        worst_cycle: The cycle with highest MPI (None if consistent)
        cycle_costs: List of (cycle, mpi) pairs for all violation cycles
        total_expenditure: Sum of all expenditures in the session
        computation_time_ms: Time taken in milliseconds
    """

    mpi_value: float
    worst_cycle: Cycle | None
    cycle_costs: list[tuple[Cycle, float]]
    total_expenditure: float
    computation_time_ms: float

    @property
    def is_consistent(self) -> bool:
        """True if no money pump exists (MPI = 0)."""
        return self.mpi_value == 0.0

    @property
    def num_cycles(self) -> int:
        """Number of violation cycles found."""
        return len(self.cycle_costs)

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns 1 - mpi_value (inverted so higher is better).
        MPI is capped at 1.0 for score calculation.
        """
        return max(0.0, 1.0 - min(1.0, self.mpi_value))

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("MONEY PUMP INDEX REPORT")]

        # Status
        if self.is_consistent:
            status = "NO EXPLOITABILITY"
        elif self.mpi_value < 0.05:
            status = "VERY LOW EXPLOITABILITY"
        elif self.mpi_value < 0.15:
            status = "LOW EXPLOITABILITY"
        else:
            status = "HIGH EXPLOITABILITY"
        lines.append(f"\nStatus: {status}")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("Money Pump Index (MPI)", self.mpi_value))
        lines.append(m._format_metric("Exploitability %", self.mpi_value * 100))
        lines.append(m._format_metric("Number of Cycles", self.num_cycles))
        lines.append(m._format_metric("Total Expenditure", self.total_expenditure))

        # Show worst cycle if any
        if self.worst_cycle:
            lines.append(m._format_section("Worst Violation Cycle"))
            lines.append(f"  Cycle: {self.worst_cycle}")
            # Find MPI for worst cycle
            for cycle, mpi in self.cycle_costs:
                if cycle == self.worst_cycle:
                    lines.append(f"  Cycle MPI: {mpi:.4f}")
                    break

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        lines.append(f"  {m._format_interpretation(self.score(), 'mpi')}")
        if self.mpi_value > 0:
            pct = self.mpi_value * 100
            lines.append(f"  An arbitrager could extract ~{pct:.1f}% of expenditure.")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "mpi_value": self.mpi_value,
            "is_consistent": self.is_consistent,
            "num_cycles": self.num_cycles,
            "worst_cycle": list(self.worst_cycle) if self.worst_cycle else None,
            "total_expenditure": self.total_expenditure,
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        status = "consistent" if self.is_consistent else f"mpi={self.mpi_value:.4f}"
        return f"MPIResult({status}, {self.computation_time_ms:.2f}ms)"


@dataclass(frozen=True)
class UtilityRecoveryResult:
    """
    Result of utility recovery via linear programming (Afriat's inequalities).

    If the data satisfies GARP, we can recover utility values U_k and
    Lagrange multipliers (marginal utility of money) lambda_k such that:
    U_k <= U_l + lambda_l * p_l @ (x_k - x_l) for all k, l

    The recovered utility function is piecewise linear and concave.

    Attributes:
        success: True if LP found a feasible solution
        utility_values: Array of U_k values (utility at each observation)
        lagrange_multipliers: Array of lambda_k values (marginal utility of money)
        lp_status: Status message from the LP solver
        residuals: Matrix of Afriat inequality residuals (for verification)
        computation_time_ms: Time taken in milliseconds
    """

    success: bool
    utility_values: NDArray[np.float64] | None
    lagrange_multipliers: NDArray[np.float64] | None
    lp_status: str
    residuals: NDArray[np.float64] | None
    computation_time_ms: float

    @property
    def mean_marginal_utility(self) -> float | None:
        """Average marginal utility of money across observations."""
        if self.lagrange_multipliers is None:
            return None
        return float(np.mean(self.lagrange_multipliers))

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns 1.0 if utility recovery succeeded, 0.0 if failed.
        """
        return 1.0 if self.success else 0.0

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("UTILITY RECOVERY REPORT")]

        # Status
        status = m._format_status(self.success, "SUCCESS", "FAILED")
        lines.append(f"\nStatus: {status}")
        lines.append(f"LP Status: {self.lp_status}")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("Recovery Successful", self.success))

        if self.success and self.utility_values is not None:
            lines.append(m._format_metric("Utility Values Range",
                                          f"[{np.min(self.utility_values):.4f}, {np.max(self.utility_values):.4f}]"))
            lines.append(m._format_metric("Mean Utility", float(np.mean(self.utility_values))))
            lines.append(m._format_metric("Num Observations", len(self.utility_values)))

        if self.lagrange_multipliers is not None:
            lines.append(m._format_metric("Mean Marginal Utility", self.mean_marginal_utility))
            lines.append(m._format_metric("Lagrange Range",
                                          f"[{np.min(self.lagrange_multipliers):.4f}, {np.max(self.lagrange_multipliers):.4f}]"))

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        if self.success:
            lines.append("  Utility function successfully recovered.")
            lines.append("  The recovered utility is piecewise linear and concave.")
        else:
            lines.append("  Utility recovery failed - data may violate GARP.")
            lines.append("  Run consistency check first to identify violations.")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        result = {
            "success": self.success,
            "lp_status": self.lp_status,
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }
        if self.utility_values is not None:
            result["utility_values"] = self.utility_values.tolist()
        if self.lagrange_multipliers is not None:
            result["lagrange_multipliers"] = self.lagrange_multipliers.tolist()
            result["mean_marginal_utility"] = self.mean_marginal_utility
        return result

    def __repr__(self) -> str:
        """Compact string representation."""
        if self.success:
            n = len(self.utility_values) if self.utility_values is not None else 0
            return f"UtilityRecoveryResult(success, n={n}, {self.computation_time_ms:.2f}ms)"
        return f"UtilityRecoveryResult(failed, {self.computation_time_ms:.2f}ms)"


@dataclass(frozen=True)
class RiskProfileResult:
    """
    Result of risk profile analysis from choices under uncertainty.

    Classifies decision-makers as risk-seeking, risk-neutral, or risk-averse
    based on their revealed preferences between safe and risky options.

    Attributes:
        risk_aversion_coefficient: Arrow-Pratt coefficient ρ
            - ρ > 0: risk averse (prefers certainty)
            - ρ ≈ 0: risk neutral (maximizes expected value)
            - ρ < 0: risk seeking (prefers gambles)
        risk_category: Classification string: "risk_seeking" | "risk_neutral" | "risk_averse"
        certainty_equivalents: Array of CEs for each lottery (amount of certain money
            equivalent to the risky option for this decision-maker)
        utility_curvature: Estimated curvature of utility function
        consistency_score: How well the CRRA model fits the choices (0-1)
        num_consistent_choices: Number of choices consistent with estimated ρ
        num_total_choices: Total number of choice observations
        computation_time_ms: Time taken in milliseconds
    """

    risk_aversion_coefficient: float
    risk_category: str
    certainty_equivalents: NDArray[np.float64]
    utility_curvature: float
    consistency_score: float
    num_consistent_choices: int
    num_total_choices: int
    computation_time_ms: float

    @property
    def is_risk_seeking(self) -> bool:
        """True if decision-maker is classified as risk-seeking."""
        return self.risk_category == "risk_seeking"

    @property
    def is_risk_averse(self) -> bool:
        """True if decision-maker is classified as risk-averse."""
        return self.risk_category == "risk_averse"

    @property
    def consistency_fraction(self) -> float:
        """Fraction of choices consistent with estimated risk profile."""
        return self.num_consistent_choices / self.num_total_choices

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns the consistency score (how well the CRRA model fits).
        """
        return self.consistency_score

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("RISK PROFILE ANALYSIS REPORT")]

        # Status
        lines.append(f"\nRisk Category: {self.risk_category.upper().replace('_', ' ')}")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("Risk Aversion Coefficient", self.risk_aversion_coefficient))
        lines.append(m._format_metric("Utility Curvature", self.utility_curvature))
        lines.append(m._format_metric("Consistency Score", self.consistency_score))
        lines.append(m._format_metric("Consistent Choices", f"{self.num_consistent_choices}/{self.num_total_choices}"))

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        if self.is_risk_averse:
            lines.append("  Decision-maker is risk averse - prefers certainty over gambles.")
            lines.append(f"  Coefficient {self.risk_aversion_coefficient:.4f} > 0 indicates risk aversion.")
        elif self.is_risk_seeking:
            lines.append("  Decision-maker is risk seeking - prefers gambles over certainty.")
            lines.append(f"  Coefficient {self.risk_aversion_coefficient:.4f} < 0 indicates risk seeking.")
        else:
            lines.append("  Decision-maker is approximately risk neutral.")
            lines.append("  Maximizes expected value without risk premium.")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "risk_aversion_coefficient": self.risk_aversion_coefficient,
            "risk_category": self.risk_category,
            "utility_curvature": self.utility_curvature,
            "consistency_score": self.consistency_score,
            "num_consistent_choices": self.num_consistent_choices,
            "num_total_choices": self.num_total_choices,
            "certainty_equivalents": self.certainty_equivalents.tolist(),
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        return f"RiskProfileResult({self.risk_category}, rho={self.risk_aversion_coefficient:.4f})"


@dataclass(frozen=True)
class IdealPointResult:
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
        """Compact string representation."""
        status = "rational" if self.is_euclidean_rational else f"{self.num_violations} violations"
        return f"IdealPointResult({status}, dims={self.num_dimensions}, var={self.explained_variance:.4f})"


@dataclass(frozen=True)
class SeparabilityResult:
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
        """Compact string representation."""
        status = "separable" if self.is_separable else "coupled"
        return f"SeparabilityResult({status}, cross_effect={self.cross_effect_strength:.4f})"


@dataclass(frozen=True)
class WARPResult:
    """
    Result of WARP (Weak Axiom of Revealed Preference) consistency test.

    WARP is a weaker condition than GARP. It only checks for direct (length-2)
    violations: if bundle i is directly revealed preferred to bundle j,
    then bundle j cannot be strictly revealed preferred to bundle i.

    Attributes:
        is_consistent: True if data satisfies WARP (no direct violations)
        violations: List of (i, j) pairs where WARP is violated
        computation_time_ms: Time taken to compute result in milliseconds
    """

    is_consistent: bool
    violations: list[tuple[int, int]]
    computation_time_ms: float

    @property
    def num_violations(self) -> int:
        """Number of WARP violations found."""
        return len(self.violations)

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns 1.0 if consistent (no violations), 0.0 if violations exist.
        """
        return 1.0 if self.is_consistent else 0.0

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("WARP CONSISTENCY REPORT")]

        # Status
        status = m._format_status(self.is_consistent, "CONSISTENT", "VIOLATIONS FOUND")
        lines.append(f"\nStatus: {status}")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("Consistent", self.is_consistent))
        lines.append(m._format_metric("Violations", self.num_violations))

        # Show violations if any
        if self.violations:
            lines.append(m._format_section("Violation Pairs"))
            lines.append(m._format_list(self.violations, max_items=5, item_name="pair"))

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        if self.is_consistent:
            lines.append("  No direct preference reversals detected.")
            lines.append("  Behavior satisfies the Weak Axiom of Revealed Preference.")
        else:
            lines.append(f"  {self.num_violations} direct preference reversal(s) found.")
            lines.append("  WARP is weaker than GARP - consider running full GARP check.")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "is_consistent": self.is_consistent,
            "num_violations": self.num_violations,
            "violations": self.violations,
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        status = "consistent" if self.is_consistent else f"{self.num_violations} violations"
        return f"WARPResult({status}, {self.computation_time_ms:.2f}ms)"


@dataclass(frozen=True)
class SARPResult:
    """
    Result of SARP (Strict Axiom of Revealed Preference) consistency test.

    SARP is violated if there exist observations t, s with mutual revealed
    preference (both x^t R* x^s and x^s R* x^t) where x^t != x^s.

    Attributes:
        is_consistent: True if data satisfies SARP (no mutual preferences)
        violations: List of cycles representing mutual preference violations
        computation_time_ms: Time taken to compute result in milliseconds
    """

    is_consistent: bool
    violations: list[Cycle]
    computation_time_ms: float

    @property
    def num_violations(self) -> int:
        """Number of SARP violations found."""
        return len(self.violations)

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns 1.0 if consistent (no violations), 0.0 if violations exist.
        """
        return 1.0 if self.is_consistent else 0.0

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("SARP CONSISTENCY REPORT")]

        # Status
        status = m._format_status(self.is_consistent, "CONSISTENT", "VIOLATIONS FOUND")
        lines.append(f"\nStatus: {status}")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("Consistent", self.is_consistent))
        lines.append(m._format_metric("Violations", self.num_violations))

        # Show violations if any
        if self.violations:
            lines.append(m._format_section("Violation Cycles"))
            lines.append(m._format_list(self.violations, max_items=5, item_name="cycle"))

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        if self.is_consistent:
            lines.append("  No mutual preference cycles detected.")
            lines.append("  Behavior satisfies the Strict Axiom of Revealed Preference.")
        else:
            lines.append(f"  {self.num_violations} mutual preference cycle(s) found.")
            lines.append("  Choices exhibit indifference cycles (x R* y and y R* x).")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "is_consistent": self.is_consistent,
            "num_violations": self.num_violations,
            "violations": [list(c) for c in self.violations],
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        status = "consistent" if self.is_consistent else f"{self.num_violations} violations"
        return f"SARPResult({status}, {self.computation_time_ms:.2f}ms)"


@dataclass(frozen=True)
class HoutmanMaksResult:
    """
    Result of Houtman-Maks index computation.

    The Houtman-Maks index is the minimum fraction of observations that
    must be removed to make the remaining data satisfy GARP. It measures
    how many "bad" observations are causing inconsistency.

    Attributes:
        fraction: Fraction of observations to remove (0 = consistent, 1 = all bad)
        removed_observations: List of observation indices to remove
        computation_time_ms: Time taken to compute result in milliseconds
    """

    fraction: float
    removed_observations: list[int]
    computation_time_ms: float

    @property
    def num_removed(self) -> int:
        """Number of observations that must be removed."""
        return len(self.removed_observations)

    @property
    def is_consistent(self) -> bool:
        """True if no observations need to be removed (data satisfies GARP)."""
        return self.fraction == 0.0

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns 1 - fraction (fraction of observations that are consistent).
        """
        return 1.0 - self.fraction

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("HOUTMAN-MAKS INDEX REPORT")]

        # Status
        if self.is_consistent:
            status = "FULLY CONSISTENT"
        elif self.fraction < 0.1:
            status = "MOSTLY CONSISTENT"
        elif self.fraction < 0.3:
            status = "MODERATE INCONSISTENCY"
        else:
            status = "HIGH INCONSISTENCY"
        lines.append(f"\nStatus: {status}")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("Fraction Removed", self.fraction))
        lines.append(m._format_metric("Fraction Consistent", 1.0 - self.fraction))
        lines.append(m._format_metric("Observations Removed", self.num_removed))

        # Show removed observations
        if self.removed_observations:
            lines.append(m._format_section("Removed Observation Indices"))
            lines.append(m._format_list(self.removed_observations, max_items=10, item_name="observation"))

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        if self.is_consistent:
            lines.append("  All observations are consistent - no removal needed.")
        else:
            pct = self.fraction * 100
            lines.append(f"  Remove {self.num_removed} observations ({pct:.1f}%) for consistency.")
            lines.append("  Remaining observations satisfy GARP.")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "fraction": self.fraction,
            "num_removed": self.num_removed,
            "is_consistent": self.is_consistent,
            "removed_observations": self.removed_observations,
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        if self.is_consistent:
            return f"HoutmanMaksResult(consistent, {self.computation_time_ms:.2f}ms)"
        return f"HoutmanMaksResult(remove={self.num_removed}, frac={self.fraction:.4f})"


# =============================================================================
# TECH-FRIENDLY ALIASES (Primary names)
# =============================================================================

# ConsistencyResult: Result of behavioral consistency validation
ConsistencyResult = GARPResult
"""
Tech-friendly alias for GARPResult.

Use this to check if user behavior is internally consistent.
Consistent behavior = not a bot, single user account.
"""

# IntegrityResult: Result of integrity/noise score computation
IntegrityResult = AEIResult
"""
Tech-friendly alias for AEIResult.

The integrity score (0-1) indicates data quality:
- 1.0 = Perfect signal, fully consistent user
- 0.5 = Noisy signal, possible bot or confused user
- <0.5 = Very noisy, likely bot or shared account
"""

# ConfusionResult: Result of confusion/exploitability metric
ConfusionResult = MPIResult
"""
Tech-friendly alias for MPIResult.

The confusion score indicates how exploitable the user's decisions are.
High confusion = bad UX causing irrational choices.
"""

# LatentValueResult: Result of latent preference extraction
LatentValueResult = UtilityRecoveryResult
"""
Tech-friendly alias for UtilityRecoveryResult.

Contains extracted latent preference values that can be used as
features for ML models or for counterfactual simulations.
"""

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


# =============================================================================
# NEW ALGORITHMS - RESULT DATACLASSES
# =============================================================================


@dataclass(frozen=True)
class BronarsPowerResult:
    """
    Result of Bronars' Power Index computation.

    Bronars' Power Index measures the statistical power of the GARP test.
    It answers: "If this user passed GARP, is that meaningful?"

    The test simulates random behavior on the observed budget constraints
    and checks what fraction of random behaviors violate GARP. High power
    means passing GARP is statistically significant.

    Attributes:
        power_index: Fraction of random simulations that violate GARP (0-1)
            - 1.0 = All random behaviors violate (high power, test is informative)
            - 0.5 = Half violate (moderate power)
            - 0.0 = No randoms violate (no power, test uninformative)
        is_significant: True if power_index > 0.5 (test has discriminatory power)
        n_simulations: Number of random simulations performed
        n_violations: Number of simulations that violated GARP
        mean_integrity_random: Average integrity score (AEI) across random simulations
        simulation_integrity_values: Array of AEI values for each simulation
        computation_time_ms: Time taken in milliseconds
    """

    power_index: float
    is_significant: bool
    n_simulations: int
    n_violations: int
    mean_integrity_random: float
    simulation_integrity_values: NDArray[np.float64] | None
    computation_time_ms: float

    @property
    def violation_rate(self) -> float:
        """Fraction of random simulations that violated GARP."""
        return self.n_violations / self.n_simulations if self.n_simulations > 0 else 0.0

    @property
    def pass_rate_random(self) -> float:
        """Fraction of random simulations that passed GARP."""
        return 1.0 - self.violation_rate

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns the power index (fraction of random behaviors that violate GARP).
        """
        return self.power_index

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("BRONARS POWER INDEX REPORT")]

        # Status
        if self.is_significant:
            status = "HIGH POWER (INFORMATIVE TEST)"
        elif self.power_index >= 0.3:
            status = "MODERATE POWER"
        else:
            status = "LOW POWER (TEST MAY NOT BE INFORMATIVE)"
        lines.append(f"\nStatus: {status}")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("Power Index", self.power_index))
        lines.append(m._format_metric("Is Significant", self.is_significant))
        lines.append(m._format_metric("Simulations", self.n_simulations))
        lines.append(m._format_metric("Violations", self.n_violations))
        lines.append(m._format_metric("Random Pass Rate", self.pass_rate_random))
        lines.append(m._format_metric("Mean Random Integrity", self.mean_integrity_random))

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        lines.append(f"  {m._format_interpretation(self.power_index, 'power')}")
        pct = self.power_index * 100
        lines.append(f"  {pct:.1f}% of random behaviors violate GARP on these budgets.")
        if self.is_significant:
            lines.append("  Passing GARP is statistically meaningful.")
        else:
            lines.append("  Passing GARP may not indicate true rationality.")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        result = {
            "power_index": self.power_index,
            "is_significant": self.is_significant,
            "n_simulations": self.n_simulations,
            "n_violations": self.n_violations,
            "violation_rate": self.violation_rate,
            "pass_rate_random": self.pass_rate_random,
            "mean_integrity_random": self.mean_integrity_random,
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }
        if self.simulation_integrity_values is not None:
            result["simulation_integrity_values"] = self.simulation_integrity_values.tolist()
        return result

    def __repr__(self) -> str:
        """Compact string representation."""
        status = "significant" if self.is_significant else "low power"
        return f"BronarsPowerResult(power={self.power_index:.4f}, {status})"


@dataclass(frozen=True)
class HARPResult:
    """
    Result of HARP (Homothetic Axiom of Revealed Preference) test.

    HARP tests whether preferences are homothetic - demand scales
    proportionally with income. This is a stronger condition than GARP.

    For homothetic preferences, the product of expenditure ratios around
    any cycle must be <= 1. Violations indicate non-homothetic behavior.

    Attributes:
        is_consistent: True if data satisfies HARP (homothetic preferences)
        violations: List of (cycle, product_ratio) for violating cycles
        max_cycle_product: Maximum product of ratios around any cycle
            (1.0 if consistent, >1.0 if violations exist)
        expenditure_ratio_matrix: T x T matrix R[i,j] = (p_i @ x_i) / (p_i @ x_j)
        log_ratio_matrix: T x T matrix of log expenditure ratios
        garp_result: GARP result for comparison (GARP is weaker than HARP)
        computation_time_ms: Time taken in milliseconds
    """

    is_consistent: bool
    violations: list[tuple[Cycle, float]]
    max_cycle_product: float
    expenditure_ratio_matrix: NDArray[np.float64]
    log_ratio_matrix: NDArray[np.float64]
    garp_result: GARPResult
    computation_time_ms: float

    @property
    def is_homothetic(self) -> bool:
        """True if preferences are homothetic (HARP satisfied)."""
        return self.is_consistent

    @property
    def num_violations(self) -> int:
        """Number of violation cycles found."""
        return len(self.violations)

    @property
    def max_violation_severity(self) -> float:
        """Maximum deviation from homotheticity (max_cycle_product - 1)."""
        return max(0.0, self.max_cycle_product - 1.0)

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns 1.0 if homothetic, or 1/max_cycle_product if violations exist.
        """
        if self.is_consistent:
            return 1.0
        return 1.0 / max(1.0, self.max_cycle_product)

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("HARP (HOMOTHETICITY) TEST REPORT")]

        # Status
        status = m._format_status(self.is_consistent, "HOMOTHETIC", "NON-HOMOTHETIC")
        lines.append(f"\nStatus: {status}")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("Homothetic (HARP)", self.is_consistent))
        lines.append(m._format_metric("GARP Consistent", self.garp_result.is_consistent))
        lines.append(m._format_metric("HARP Violations", self.num_violations))
        lines.append(m._format_metric("Max Cycle Product", self.max_cycle_product))
        lines.append(m._format_metric("Max Violation Severity", self.max_violation_severity))

        # Show worst violation if any
        if self.violations:
            worst_cycle, worst_product = max(self.violations, key=lambda x: x[1])
            lines.append(m._format_section("Worst Violation"))
            lines.append(f"  Cycle: {worst_cycle}")
            lines.append(f"  Product: {worst_product:.4f}")

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        if self.is_consistent:
            lines.append("  Preferences are homothetic - demand scales with income.")
            lines.append("  Budget shares are constant across income levels.")
        else:
            lines.append("  Preferences are non-homothetic.")
            lines.append("  Demand does not scale proportionally with income.")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "is_consistent": self.is_consistent,
            "is_homothetic": self.is_homothetic,
            "num_violations": self.num_violations,
            "max_cycle_product": self.max_cycle_product,
            "max_violation_severity": self.max_violation_severity,
            "garp_consistent": self.garp_result.is_consistent,
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        status = "homothetic" if self.is_consistent else f"{self.num_violations} violations"
        return f"HARPResult({status}, max_prod={self.max_cycle_product:.4f})"


@dataclass(frozen=True)
class QuasilinearityResult:
    """
    Result of quasilinearity (cyclic monotonicity) test.

    Quasilinear preferences have no income effects - money has constant
    marginal utility. This is tested via cyclic monotonicity:
    For any cycle, the sum of price-weighted quantity changes must be >= 0.

    Attributes:
        is_quasilinear: True if data satisfies cyclic monotonicity
        violations: List of cycles that violate cyclic monotonicity
        worst_violation_magnitude: Largest violation (most negative cycle sum)
        cycle_sums: Dict mapping cycle tuples to their sums
        num_cycles_tested: Total number of cycles examined
        computation_time_ms: Time taken in milliseconds
    """

    is_quasilinear: bool
    violations: list[Cycle]
    worst_violation_magnitude: float
    cycle_sums: dict[Cycle, float]
    num_cycles_tested: int
    computation_time_ms: float

    @property
    def num_violations(self) -> int:
        """Number of cycles that violate cyclic monotonicity."""
        return len(self.violations)

    @property
    def has_income_effects(self) -> bool:
        """True if income effects detected (quasilinearity violated)."""
        return not self.is_quasilinear

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns 1.0 if quasilinear, 0.0 if violations exist.
        """
        return 1.0 if self.is_quasilinear else 0.0

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("QUASILINEARITY TEST REPORT")]

        # Status
        status = m._format_status(self.is_quasilinear, "QUASILINEAR", "NON-QUASILINEAR")
        lines.append(f"\nStatus: {status}")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("Quasilinear", self.is_quasilinear))
        lines.append(m._format_metric("Has Income Effects", self.has_income_effects))
        lines.append(m._format_metric("Violations", self.num_violations))
        lines.append(m._format_metric("Cycles Tested", self.num_cycles_tested))
        lines.append(m._format_metric("Worst Violation", self.worst_violation_magnitude))

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        if self.is_quasilinear:
            lines.append("  Preferences satisfy cyclic monotonicity.")
            lines.append("  Money has constant marginal utility - no income effects.")
        else:
            lines.append("  Cyclic monotonicity violated - income effects detected.")
            lines.append("  Demand changes with income level at constant prices.")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "is_quasilinear": self.is_quasilinear,
            "has_income_effects": self.has_income_effects,
            "num_violations": self.num_violations,
            "num_cycles_tested": self.num_cycles_tested,
            "worst_violation_magnitude": self.worst_violation_magnitude,
            "violations": [list(c) for c in self.violations],
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        status = "quasilinear" if self.is_quasilinear else f"{self.num_violations} violations"
        return f"QuasilinearityResult({status})"


@dataclass(frozen=True)
class GrossSubstitutesResult:
    """
    Result of gross substitutes test between two goods.

    Tests whether two goods are substitutes (price of A up → demand for B up)
    or complements (price of A up → demand for B down).

    Attributes:
        are_substitutes: True if goods appear to be gross substitutes
        are_complements: True if goods appear to be complements
        relationship: Classification: "substitutes", "complements", "independent", "inconclusive"
        supporting_pairs: List of (obs_i, obs_j) pairs supporting the relationship
        violating_pairs: List of (obs_i, obs_j) pairs violating the relationship
        confidence_score: Fraction of informative pairs supporting the relationship (0-1)
        good_g_index: Index of first good
        good_h_index: Index of second good
        computation_time_ms: Time taken in milliseconds
    """

    are_substitutes: bool
    are_complements: bool
    relationship: str
    supporting_pairs: list[tuple[int, int]]
    violating_pairs: list[tuple[int, int]]
    confidence_score: float
    good_g_index: int
    good_h_index: int
    computation_time_ms: float

    @property
    def is_conclusive(self) -> bool:
        """True if the test gave a conclusive result."""
        return self.relationship != "inconclusive"

    @property
    def num_informative_pairs(self) -> int:
        """Number of observation pairs that informed the relationship."""
        return len(self.supporting_pairs) + len(self.violating_pairs)

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns the confidence score for the relationship.
        """
        return self.confidence_score

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("GROSS SUBSTITUTES TEST REPORT")]

        # Status
        lines.append(f"\nRelationship: {self.relationship.upper()}")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("Good G Index", self.good_g_index))
        lines.append(m._format_metric("Good H Index", self.good_h_index))
        lines.append(m._format_metric("Are Substitutes", self.are_substitutes))
        lines.append(m._format_metric("Are Complements", self.are_complements))
        lines.append(m._format_metric("Confidence Score", self.confidence_score))
        lines.append(m._format_metric("Supporting Pairs", len(self.supporting_pairs)))
        lines.append(m._format_metric("Violating Pairs", len(self.violating_pairs)))

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        if self.are_substitutes:
            lines.append(f"  Goods {self.good_g_index} and {self.good_h_index} are substitutes.")
            lines.append("  Price increase in one leads to demand increase in the other.")
        elif self.are_complements:
            lines.append(f"  Goods {self.good_g_index} and {self.good_h_index} are complements.")
            lines.append("  Price increase in one leads to demand decrease in the other.")
        elif self.relationship == "independent":
            lines.append(f"  Goods {self.good_g_index} and {self.good_h_index} are independent.")
            lines.append("  No significant cross-price effects detected.")
        else:
            lines.append("  Relationship is inconclusive - insufficient evidence.")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "good_g_index": self.good_g_index,
            "good_h_index": self.good_h_index,
            "relationship": self.relationship,
            "are_substitutes": self.are_substitutes,
            "are_complements": self.are_complements,
            "confidence_score": self.confidence_score,
            "num_supporting_pairs": len(self.supporting_pairs),
            "num_violating_pairs": len(self.violating_pairs),
            "is_conclusive": self.is_conclusive,
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        return f"GrossSubstitutesResult(g={self.good_g_index}, h={self.good_h_index}, {self.relationship})"


@dataclass(frozen=True)
class SubstitutionMatrixResult:
    """
    Result of pairwise substitution analysis for all goods.

    Contains an N x N matrix of relationships between all pairs of goods.

    Attributes:
        relationship_matrix: N x N matrix where entry [g,h] is the relationship
            between goods g and h ("substitutes", "complements", "independent", etc.)
        confidence_matrix: N x N matrix of confidence scores for each relationship
        num_goods: Number of goods N
        computation_time_ms: Time taken in milliseconds
    """

    relationship_matrix: NDArray[np.object_]
    confidence_matrix: NDArray[np.float64]
    num_goods: int
    computation_time_ms: float

    @property
    def substitute_pairs(self) -> list[tuple[int, int]]:
        """List of (g, h) pairs that are substitutes."""
        pairs = []
        for g in range(self.num_goods):
            for h in range(g + 1, self.num_goods):
                if self.relationship_matrix[g, h] == "substitutes":
                    pairs.append((g, h))
        return pairs

    @property
    def complement_pairs(self) -> list[tuple[int, int]]:
        """List of (g, h) pairs that are complements."""
        pairs = []
        for g in range(self.num_goods):
            for h in range(g + 1, self.num_goods):
                if self.relationship_matrix[g, h] == "complements":
                    pairs.append((g, h))
        return pairs

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns mean confidence across all pairs.
        """
        # Compute mean of upper triangle (excluding diagonal)
        total = 0.0
        count = 0
        for g in range(self.num_goods):
            for h in range(g + 1, self.num_goods):
                total += self.confidence_matrix[g, h]
                count += 1
        return total / count if count > 0 else 0.0

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("SUBSTITUTION MATRIX REPORT")]

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("Number of Goods", self.num_goods))
        lines.append(m._format_metric("Substitute Pairs", len(self.substitute_pairs)))
        lines.append(m._format_metric("Complement Pairs", len(self.complement_pairs)))
        lines.append(m._format_metric("Mean Confidence", self.score()))

        # Show pairs
        if self.substitute_pairs:
            lines.append(m._format_section("Substitute Pairs"))
            lines.append(m._format_list(self.substitute_pairs, max_items=10, item_name="pair"))

        if self.complement_pairs:
            lines.append(m._format_section("Complement Pairs"))
            lines.append(m._format_list(self.complement_pairs, max_items=10, item_name="pair"))

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "num_goods": self.num_goods,
            "substitute_pairs": self.substitute_pairs,
            "complement_pairs": self.complement_pairs,
            "confidence_matrix": self.confidence_matrix.tolist(),
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        return f"SubstitutionMatrixResult(n={self.num_goods}, subs={len(self.substitute_pairs)}, comps={len(self.complement_pairs)})"


@dataclass(frozen=True)
class VEIResult:
    """
    Result of Varian's Efficiency Index (per-observation efficiency) computation.

    Unlike AEI which gives one global efficiency score, VEI provides
    individual efficiency scores for each observation. This identifies
    which specific observations are problematic.

    Attributes:
        efficiency_vector: Array of e_i values for each observation (0-1)
        mean_efficiency: Average efficiency across observations
        min_efficiency: Lowest efficiency (worst observation)
        worst_observation: Index of observation with lowest efficiency
        problematic_observations: Indices where e_i < threshold (default 0.9)
        total_inefficiency: Sum of (1 - e_i) across all observations
        optimization_success: True if optimization converged
        optimization_status: Status message from optimizer
        computation_time_ms: Time taken in milliseconds
    """

    efficiency_vector: NDArray[np.float64]
    mean_efficiency: float
    min_efficiency: float
    worst_observation: int
    problematic_observations: list[int]
    total_inefficiency: float
    optimization_success: bool
    optimization_status: str
    computation_time_ms: float

    @property
    def num_observations(self) -> int:
        """Number of observations."""
        return len(self.efficiency_vector)

    @property
    def is_perfectly_consistent(self) -> bool:
        """True if all observations have efficiency = 1."""
        return self.min_efficiency >= 1.0 - 1e-6

    @property
    def num_problematic(self) -> int:
        """Number of problematic observations."""
        return len(self.problematic_observations)

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns the mean efficiency across all observations.
        """
        return self.mean_efficiency

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("VARIAN EFFICIENCY INDEX (VEI) REPORT")]

        # Status
        if self.is_perfectly_consistent:
            status = "PERFECT CONSISTENCY"
        elif self.mean_efficiency >= 0.95:
            status = "EXCELLENT"
        elif self.mean_efficiency >= 0.9:
            status = "GOOD"
        elif self.mean_efficiency >= 0.7:
            status = "MODERATE"
        else:
            status = "LOW"
        lines.append(f"\nStatus: {status}")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("Mean Efficiency", self.mean_efficiency))
        lines.append(m._format_metric("Min Efficiency", self.min_efficiency))
        lines.append(m._format_metric("Total Inefficiency", self.total_inefficiency))
        lines.append(m._format_metric("Observations", self.num_observations))
        lines.append(m._format_metric("Problematic Obs", self.num_problematic))
        lines.append(m._format_metric("Worst Observation", self.worst_observation))
        lines.append(m._format_metric("Optimization Success", self.optimization_success))

        # Show problematic observations
        if self.problematic_observations:
            lines.append(m._format_section("Problematic Observations"))
            lines.append(m._format_list(self.problematic_observations, max_items=10, item_name="observation"))

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        lines.append(f"  {m._format_interpretation(self.mean_efficiency, 'efficiency')}")
        if self.num_problematic > 0:
            pct = (self.num_problematic / self.num_observations) * 100
            lines.append(f"  {self.num_problematic} observations ({pct:.1f}%) below efficiency threshold.")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "mean_efficiency": self.mean_efficiency,
            "min_efficiency": self.min_efficiency,
            "worst_observation": self.worst_observation,
            "total_inefficiency": self.total_inefficiency,
            "num_observations": self.num_observations,
            "num_problematic": self.num_problematic,
            "problematic_observations": self.problematic_observations,
            "is_perfectly_consistent": self.is_perfectly_consistent,
            "optimization_success": self.optimization_success,
            "optimization_status": self.optimization_status,
            "efficiency_vector": self.efficiency_vector.tolist(),
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        if self.is_perfectly_consistent:
            return f"VEIResult(perfect, n={self.num_observations})"
        return f"VEIResult(mean={self.mean_efficiency:.4f}, min={self.min_efficiency:.4f})"


# =============================================================================
# TECH-FRIENDLY ALIASES FOR NEW RESULTS
# =============================================================================

# TestPowerResult: Result of statistical test power computation
TestPowerResult = BronarsPowerResult
"""
Tech-friendly alias for BronarsPowerResult.

Use this to validate that consistency test results are statistically meaningful.
"""

# ProportionalScalingResult: Result of proportional scaling (homotheticity) test
ProportionalScalingResult = HARPResult
"""
Tech-friendly alias for HARPResult.

Tests if user preferences scale proportionally with budget.
"""

# IncomeInvarianceResult: Result of income invariance (quasilinearity) test
IncomeInvarianceResult = QuasilinearityResult
"""
Tech-friendly alias for QuasilinearityResult.

Tests if user behavior is invariant to income changes.
"""

# CrossPriceResult: Result of cross-price effect analysis
CrossPriceResult = GrossSubstitutesResult
"""
Tech-friendly alias for GrossSubstitutesResult.

Analyzes how price changes in one good affect demand for another.
"""

# GranularIntegrityResult: Result of per-observation integrity computation
GranularIntegrityResult = VEIResult
"""
Tech-friendly alias for VEIResult.

Provides per-observation integrity scores instead of one global score.
"""


# =============================================================================
# 2024 SURVEY ALGORITHMS - RESULT DATACLASSES
# =============================================================================


@dataclass(frozen=True)
class DifferentiableResult:
    """
    Result of differentiable rationality (smooth preferences) test.

    Tests whether preferences are smooth/differentiable, which requires:
    1. SARP (Strict Axiom): No indifferent preference cycles
    2. Price-Quantity Uniqueness: Different prices imply different quantities

    Differentiable preferences allow for comparative statics and demand
    function derivatives. Violations indicate piecewise-linear preferences.

    Based on Chiappori & Rochet (1987).

    Attributes:
        is_differentiable: True if both SARP and uniqueness conditions hold
        satisfies_sarp: True if no indifferent cycles exist
        satisfies_uniqueness: True if price differences imply quantity differences
        sarp_violations: List of indifferent preference cycles
        uniqueness_violations: List of (t, s) pairs where p^t != p^s but x^t = x^s
        direct_revealed_preference: T x T boolean matrix R
        transitive_closure: T x T boolean matrix R*
        computation_time_ms: Time taken in milliseconds
    """

    is_differentiable: bool
    satisfies_sarp: bool
    satisfies_uniqueness: bool
    sarp_violations: list[Cycle]
    uniqueness_violations: list[tuple[int, int]]
    direct_revealed_preference: NDArray[np.bool_]
    transitive_closure: NDArray[np.bool_]
    computation_time_ms: float

    @property
    def num_sarp_violations(self) -> int:
        """Number of SARP (indifferent cycle) violations."""
        return len(self.sarp_violations)

    @property
    def num_uniqueness_violations(self) -> int:
        """Number of price-quantity uniqueness violations."""
        return len(self.uniqueness_violations)

    @property
    def is_piecewise_linear(self) -> bool:
        """True if preferences appear piecewise-linear (not smooth)."""
        return not self.is_differentiable

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns 1.0 if differentiable, 0.0 if violations exist.
        """
        return 1.0 if self.is_differentiable else 0.0

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("DIFFERENTIABILITY TEST REPORT")]

        # Status
        status = m._format_status(self.is_differentiable,
                                  "DIFFERENTIABLE (SMOOTH)", "PIECEWISE-LINEAR")
        lines.append(f"\nStatus: {status}")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("Is Differentiable", self.is_differentiable))
        lines.append(m._format_metric("Satisfies SARP", self.satisfies_sarp))
        lines.append(m._format_metric("Satisfies Uniqueness", self.satisfies_uniqueness))
        lines.append(m._format_metric("SARP Violations", self.num_sarp_violations))
        lines.append(m._format_metric("Uniqueness Violations", self.num_uniqueness_violations))

        # Show violations if any
        if self.sarp_violations:
            lines.append(m._format_section("SARP Violations"))
            lines.append(m._format_list(self.sarp_violations, max_items=5, item_name="cycle"))

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        if self.is_differentiable:
            lines.append("  Preferences are smooth/differentiable.")
            lines.append("  Demand function derivatives are well-defined.")
        else:
            lines.append("  Preferences are piecewise-linear (not smooth).")
            if not self.satisfies_sarp:
                lines.append("  SARP violations indicate indifference cycles.")
            if not self.satisfies_uniqueness:
                lines.append("  Different prices led to identical quantities.")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "is_differentiable": self.is_differentiable,
            "satisfies_sarp": self.satisfies_sarp,
            "satisfies_uniqueness": self.satisfies_uniqueness,
            "num_sarp_violations": self.num_sarp_violations,
            "num_uniqueness_violations": self.num_uniqueness_violations,
            "is_piecewise_linear": self.is_piecewise_linear,
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        status = "smooth" if self.is_differentiable else "piecewise-linear"
        return f"DifferentiableResult({status})"


@dataclass(frozen=True)
class AcyclicalPResult:
    """
    Result of Acyclical P test (strict preference acyclicity).

    Tests whether the strict revealed preference relation P is acyclic.
    This is MORE LENIENT than GARP - it passes when GARP might fail,
    because it ignores weak preference violations.

    A consumer passes Acyclical P if there are no strict preference cycles,
    even if weak preference cycles exist. This indicates "approximately
    rational" behavior where apparent violations are due to indifference.

    Based on Dziewulski (2023).

    Attributes:
        is_consistent: True if no strict preference cycles exist
        violations: List of strict preference cycles found
        strict_preference_matrix: T x T boolean matrix P where P[t,s] = True
            iff p^t @ x^s < p^t @ x^t (bundle s was strictly cheaper)
        transitive_closure: T x T boolean matrix P* (transitive closure of P)
        num_strict_preferences: Count of strict preference edges
        garp_consistent: True if also passes GARP (for comparison)
        computation_time_ms: Time taken in milliseconds
    """

    is_consistent: bool
    violations: list[Cycle]
    strict_preference_matrix: NDArray[np.bool_]
    transitive_closure: NDArray[np.bool_]
    num_strict_preferences: int
    garp_consistent: bool
    computation_time_ms: float

    @property
    def num_violations(self) -> int:
        """Number of strict preference cycles found."""
        return len(self.violations)

    @property
    def is_approximately_rational(self) -> bool:
        """True if behavior is approximately rational (passes Acyclical P)."""
        return self.is_consistent

    @property
    def strict_violations_only(self) -> bool:
        """True if GARP fails but Acyclical P passes (weak violations only)."""
        return self.is_consistent and not self.garp_consistent

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns 1.0 if consistent (no strict preference cycles), 0.0 otherwise.
        """
        return 1.0 if self.is_consistent else 0.0

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("ACYCLICAL P TEST REPORT")]

        # Status
        status = m._format_status(self.is_consistent,
                                  "NO STRICT CYCLES (APPROX RATIONAL)",
                                  "STRICT CYCLES FOUND")
        lines.append(f"\nStatus: {status}")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("Acyclical P Consistent", self.is_consistent))
        lines.append(m._format_metric("GARP Consistent", self.garp_consistent))
        lines.append(m._format_metric("Strict Violations", self.num_violations))
        lines.append(m._format_metric("Strict Preferences", self.num_strict_preferences))
        lines.append(m._format_metric("Approximately Rational", self.is_approximately_rational))

        # Show violations if any
        if self.violations:
            lines.append(m._format_section("Strict Preference Cycles"))
            lines.append(m._format_list(self.violations, max_items=5, item_name="cycle"))

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        if self.is_consistent:
            if self.garp_consistent:
                lines.append("  Fully rational - passes both GARP and Acyclical P.")
            else:
                lines.append("  Approximately rational - passes Acyclical P but not GARP.")
                lines.append("  Apparent violations due to indifference, not irrationality.")
        else:
            lines.append("  Strict preference cycles exist - not approximately rational.")
            lines.append("  Behavior cannot be explained by any utility function.")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "is_consistent": self.is_consistent,
            "garp_consistent": self.garp_consistent,
            "num_violations": self.num_violations,
            "num_strict_preferences": self.num_strict_preferences,
            "is_approximately_rational": self.is_approximately_rational,
            "strict_violations_only": self.strict_violations_only,
            "violations": [list(c) for c in self.violations],
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        if self.is_consistent:
            return f"AcyclicalPResult(approx_rational, garp={'yes' if self.garp_consistent else 'no'})"
        return f"AcyclicalPResult({self.num_violations} strict cycles)"


@dataclass(frozen=True)
class GAPPResult:
    """
    Result of GAPP (Generalized Axiom of Price Preference) test.

    GAPP tests consistency of revealed PRICE preferences, which is the
    dual perspective to GARP's quantity preferences. Instead of asking
    "does the consumer prefer bundle A to bundle B?", GAPP asks
    "does the consumer prefer price vector A to price vector B?"

    Price s is revealed preferred to price t if the bundle bought at t
    would have been cheaper under prices s. Consistent price preferences
    indicate the consumer is "shopping around" rationally.

    Based on Deb et al. (2022).

    Attributes:
        is_consistent: True if no price preference cycles exist
        violations: List of (s, t) pairs where GAPP is violated
        price_preference_matrix: T x T matrix R_p where R_p[s,t] = True
            iff p^s @ x^t <= p^t @ x^t (price s is preferred to t)
        strict_price_preference: T x T matrix P_p where P_p[s,t] = True
            iff p^s @ x^t < p^t @ x^t (price s strictly preferred to t)
        transitive_closure: T x T matrix R_p* (transitive closure of R_p)
        num_price_preferences: Count of price preference relations
        garp_consistent: True if also passes GARP (for comparison)
        computation_time_ms: Time taken in milliseconds
    """

    is_consistent: bool
    violations: list[tuple[int, int]]
    price_preference_matrix: NDArray[np.bool_]
    strict_price_preference: NDArray[np.bool_]
    transitive_closure: NDArray[np.bool_]
    num_price_preferences: int
    garp_consistent: bool
    computation_time_ms: float

    @property
    def num_violations(self) -> int:
        """Number of GAPP violations found."""
        return len(self.violations)

    @property
    def prefers_lower_prices(self) -> bool:
        """True if price preferences are consistent (GAPP satisfied)."""
        return self.is_consistent

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns 1.0 if consistent price preferences, 0.0 otherwise.
        """
        return 1.0 if self.is_consistent else 0.0

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("GAPP (PRICE PREFERENCE) TEST REPORT")]

        # Status
        status = m._format_status(self.is_consistent,
                                  "CONSISTENT PRICE PREFERENCES",
                                  "INCONSISTENT PRICE PREFERENCES")
        lines.append(f"\nStatus: {status}")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("GAPP Consistent", self.is_consistent))
        lines.append(m._format_metric("GARP Consistent", self.garp_consistent))
        lines.append(m._format_metric("Price Preference Relations", self.num_price_preferences))
        lines.append(m._format_metric("Violations", self.num_violations))

        # Show violations if any
        if self.violations:
            lines.append(m._format_section("GAPP Violations"))
            lines.append(m._format_list(self.violations, max_items=5, item_name="pair"))

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        if self.is_consistent:
            lines.append("  Consumer has consistent price preferences.")
            lines.append("  They rationally prefer situations where items are cheaper.")
        else:
            lines.append("  Inconsistent price preferences detected.")
            lines.append("  Consumer does not consistently prefer lower prices.")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "is_consistent": self.is_consistent,
            "garp_consistent": self.garp_consistent,
            "num_violations": self.num_violations,
            "num_price_preferences": self.num_price_preferences,
            "prefers_lower_prices": self.prefers_lower_prices,
            "violations": self.violations,
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        status = "consistent" if self.is_consistent else f"{self.num_violations} violations"
        return f"GAPPResult({status})"


# Tech-friendly aliases for 2024 survey algorithms

SmoothPreferencesResult = DifferentiableResult
"""
Tech-friendly alias for DifferentiableResult.

Tests if user preferences are smooth (differentiable), enabling
demand function derivatives for price sensitivity analysis.
"""

StrictConsistencyResult = AcyclicalPResult
"""
Tech-friendly alias for AcyclicalPResult.

Tests strict consistency only - more lenient than full consistency check.
Useful for identifying "approximately rational" behavior.
"""

PricePreferencesResult = GAPPResult
"""
Tech-friendly alias for GAPPResult.

Tests if the user has consistent price preferences - do they
prefer situations where their desired items are cheaper?
"""


@dataclass(frozen=True)
class LancasterResult:
    """
    Valuation report from Lancaster characteristics model.

    The Lancaster model transforms product-space data into characteristics-space,
    computing shadow prices (implicit valuations) for underlying product attributes.
    This result provides business insights from the shadow price analysis.

    Attributes:
        mean_shadow_prices: K-length array of average shadow price per characteristic
        shadow_price_std: K-length array of shadow price standard deviations
        shadow_price_cv: K-length array of coefficient of variation (volatility)
        total_spend_on_characteristics: K-length array of total spend per characteristic
        spend_shares: K-length array of spend share per characteristic (sums to 1)
        mean_nnls_residual: Average NNLS fit residual (lower = better fit)
        max_nnls_residual: Maximum NNLS residual (flags problematic observations)
        problematic_observations: List of observation indices with high residuals
        attribute_matrix_rank: Rank of A matrix (for diagnostics)
        is_well_specified: True if A has full column rank (K <= N and rank = K)
        characteristic_names: Optional names for characteristics (from metadata)
        computation_time_ms: Time taken in milliseconds
    """

    mean_shadow_prices: NDArray[np.float64]
    shadow_price_std: NDArray[np.float64]
    shadow_price_cv: NDArray[np.float64]
    total_spend_on_characteristics: NDArray[np.float64]
    spend_shares: NDArray[np.float64]
    mean_nnls_residual: float
    max_nnls_residual: float
    problematic_observations: list[int]
    attribute_matrix_rank: int
    is_well_specified: bool
    characteristic_names: list[str] | None
    computation_time_ms: float

    @property
    def num_characteristics(self) -> int:
        """Number of characteristics K."""
        return len(self.mean_shadow_prices)

    @property
    def most_valued_characteristic(self) -> int:
        """Index of characteristic with highest mean shadow price."""
        return int(np.argmax(self.mean_shadow_prices))

    @property
    def most_volatile_characteristic(self) -> int:
        """Index of characteristic with highest price volatility (CV)."""
        return int(np.argmax(self.shadow_price_cv))

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns 1 - mean_nnls_residual (capped at 0, higher = better fit).
        """
        return max(0.0, 1.0 - self.mean_nnls_residual)

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("LANCASTER CHARACTERISTICS REPORT")]

        # Status
        if self.is_well_specified:
            status = "WELL-SPECIFIED MODEL"
        else:
            status = "MODEL MAY BE UNDER-IDENTIFIED"
        lines.append(f"\nStatus: {status}")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("Characteristics", self.num_characteristics))
        lines.append(m._format_metric("Matrix Rank", self.attribute_matrix_rank))
        lines.append(m._format_metric("Well-Specified", self.is_well_specified))
        lines.append(m._format_metric("Mean NNLS Residual", self.mean_nnls_residual))
        lines.append(m._format_metric("Max NNLS Residual", self.max_nnls_residual))
        lines.append(m._format_metric("Problematic Obs", len(self.problematic_observations)))

        # Shadow prices
        lines.append(m._format_section("Shadow Prices (Implicit Valuations)"))
        for i in range(self.num_characteristics):
            name = self.characteristic_names[i] if self.characteristic_names else f"Char {i}"
            lines.append(f"  {name}: mean={self.mean_shadow_prices[i]:.4f}, "
                         f"std={self.shadow_price_std[i]:.4f}, "
                         f"share={self.spend_shares[i]*100:.1f}%")

        # Key insights
        lines.append(m._format_section("Key Insights"))
        most_valued = self.most_valued_characteristic
        most_volatile = self.most_volatile_characteristic
        val_name = self.characteristic_names[most_valued] if self.characteristic_names else f"Char {most_valued}"
        vol_name = self.characteristic_names[most_volatile] if self.characteristic_names else f"Char {most_volatile}"
        lines.append(f"  Most valued: {val_name} (shadow price {self.mean_shadow_prices[most_valued]:.4f})")
        lines.append(f"  Most volatile: {vol_name} (CV {self.shadow_price_cv[most_volatile]:.4f})")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "num_characteristics": self.num_characteristics,
            "mean_shadow_prices": self.mean_shadow_prices.tolist(),
            "shadow_price_std": self.shadow_price_std.tolist(),
            "shadow_price_cv": self.shadow_price_cv.tolist(),
            "spend_shares": self.spend_shares.tolist(),
            "total_spend_on_characteristics": self.total_spend_on_characteristics.tolist(),
            "mean_nnls_residual": self.mean_nnls_residual,
            "max_nnls_residual": self.max_nnls_residual,
            "is_well_specified": self.is_well_specified,
            "attribute_matrix_rank": self.attribute_matrix_rank,
            "most_valued_characteristic": self.most_valued_characteristic,
            "most_volatile_characteristic": self.most_volatile_characteristic,
            "characteristic_names": self.characteristic_names,
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        return f"LancasterResult(k={self.num_characteristics}, residual={self.mean_nnls_residual:.4f})"


# CharacteristicsValuationResult: Tech-friendly alias for LancasterResult
CharacteristicsValuationResult = LancasterResult
"""
Tech-friendly alias for LancasterResult.

Contains insights from characteristics-space analysis of user behavior,
including shadow prices (implicit valuations) for product attributes.
"""


# =============================================================================
# ABSTRACT CHOICE THEORY - RESULT DATACLASSES
# =============================================================================


@dataclass(frozen=True)
class AbstractWARPResult:
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
class AbstractSARPResult:
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
class CongruenceResult:
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
class HoutmanMaksAbstractResult:
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
class OrdinalUtilityResult:
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


# =============================================================================
# CHAMBERS & ECHENIQUE (2016) - NEW ALGORITHM RESULT TYPES
# =============================================================================


@dataclass(frozen=True)
class IntegrabilityResult:
    """
    Result of integrability conditions test (Chapter 6.4-6.5).

    Tests whether a demand function is integrable to a utility function.
    Based on the Slutsky matrix conditions: symmetry and negative semi-definiteness.

    Attributes:
        is_symmetric: True if Slutsky matrix is symmetric (S[i,j] = S[j,i])
        is_negative_semidefinite: True if all eigenvalues of S are <= 0
        is_integrable: True if both conditions hold (demand is integrable)
        slutsky_matrix: Estimated N x N Slutsky matrix
        eigenvalues: Eigenvalues of the Slutsky matrix (should all be <= 0)
        symmetry_violations: List of (i, j) pairs where S[i,j] != S[j,i]
        max_eigenvalue: Largest eigenvalue (should be <= 0 for NSD)
        symmetry_deviation: Max deviation from symmetry
        computation_time_ms: Time taken in milliseconds
    """

    is_symmetric: bool
    is_negative_semidefinite: bool
    is_integrable: bool
    slutsky_matrix: NDArray[np.float64]
    eigenvalues: NDArray[np.float64]
    symmetry_violations: list[tuple[int, int]]
    max_eigenvalue: float
    symmetry_deviation: float
    computation_time_ms: float

    @property
    def num_goods(self) -> int:
        """Number of goods N."""
        return self.slutsky_matrix.shape[0]

    @property
    def num_symmetry_violations(self) -> int:
        """Number of pairs violating symmetry."""
        return len(self.symmetry_violations)

    @property
    def passes_slutsky_conditions(self) -> bool:
        """True if both Slutsky conditions hold."""
        return self.is_integrable

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns 1.0 if integrable, 0.5 if one condition fails, 0.0 if both fail.
        """
        if self.is_integrable:
            return 1.0
        elif self.is_symmetric or self.is_negative_semidefinite:
            return 0.5
        return 0.0

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("INTEGRABILITY TEST REPORT")]

        # Status
        status = m._format_status(self.is_integrable, "INTEGRABLE", "NOT INTEGRABLE")
        lines.append(f"\nStatus: {status}")

        # Metrics
        lines.append(m._format_section("Slutsky Conditions"))
        lines.append(m._format_metric("Is Integrable", self.is_integrable))
        lines.append(m._format_metric("Symmetric", self.is_symmetric))
        lines.append(m._format_metric("Negative Semi-Definite", self.is_negative_semidefinite))
        lines.append(m._format_metric("Symmetry Violations", self.num_symmetry_violations))
        lines.append(m._format_metric("Max Eigenvalue", self.max_eigenvalue))
        lines.append(m._format_metric("Symmetry Deviation", self.symmetry_deviation))
        lines.append(m._format_metric("Number of Goods", self.num_goods))

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        if self.is_integrable:
            lines.append("  Demand function is integrable to a utility function.")
            lines.append("  Both Slutsky symmetry and NSD conditions satisfied.")
        else:
            if not self.is_symmetric:
                lines.append("  Slutsky symmetry violated - cross-price effects asymmetric.")
            if not self.is_negative_semidefinite:
                lines.append(f"  Not NSD - max eigenvalue {self.max_eigenvalue:.4f} > 0.")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "is_integrable": self.is_integrable,
            "is_symmetric": self.is_symmetric,
            "is_negative_semidefinite": self.is_negative_semidefinite,
            "num_goods": self.num_goods,
            "max_eigenvalue": self.max_eigenvalue,
            "symmetry_deviation": self.symmetry_deviation,
            "num_symmetry_violations": self.num_symmetry_violations,
            "eigenvalues": self.eigenvalues.tolist(),
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        if self.is_integrable:
            return f"IntegrabilityResult(integrable, n={self.num_goods})"
        return f"IntegrabilityResult(sym={self.is_symmetric}, nsd={self.is_negative_semidefinite})"


@dataclass(frozen=True)
class WelfareResult:
    """
    Result of welfare analysis (Chapter 7.3-7.4).

    Measures consumer welfare changes from policy or price changes using
    compensating variation (CV) and equivalent variation (EV).

    Attributes:
        compensating_variation: Money needed to restore original utility after change
        equivalent_variation: Money equivalent to the utility change
        welfare_direction: "improved", "worsened", or "ambiguous"
        baseline_utility: Estimated utility at baseline prices/quantities
        policy_utility: Estimated utility at policy prices/quantities
        baseline_expenditure: Total expenditure at baseline
        policy_expenditure: Total expenditure under policy
        hicksian_surplus: Consumer surplus measure
        computation_time_ms: Time taken in milliseconds
    """

    compensating_variation: float
    equivalent_variation: float
    welfare_direction: str
    baseline_utility: float
    policy_utility: float
    baseline_expenditure: float
    policy_expenditure: float
    hicksian_surplus: float
    computation_time_ms: float

    @property
    def welfare_improved(self) -> bool:
        """True if welfare improved under policy."""
        return self.welfare_direction == "improved"

    @property
    def welfare_worsened(self) -> bool:
        """True if welfare worsened under policy."""
        return self.welfare_direction == "worsened"

    @property
    def mean_variation(self) -> float:
        """Average of CV and EV (common approximation)."""
        return (self.compensating_variation + self.equivalent_variation) / 2

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns 1.0 if welfare improved, 0.5 if ambiguous, 0.0 if worsened.
        """
        if self.welfare_improved:
            return 1.0
        elif self.welfare_worsened:
            return 0.0
        return 0.5

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("WELFARE ANALYSIS REPORT")]

        # Status
        lines.append(f"\nWelfare Direction: {self.welfare_direction.upper()}")

        # Welfare measures
        lines.append(m._format_section("Welfare Measures"))
        lines.append(m._format_metric("Compensating Variation (CV)", self.compensating_variation))
        lines.append(m._format_metric("Equivalent Variation (EV)", self.equivalent_variation))
        lines.append(m._format_metric("Mean Variation", self.mean_variation))
        lines.append(m._format_metric("Hicksian Surplus", self.hicksian_surplus))

        # Utility comparison
        lines.append(m._format_section("Utility Comparison"))
        lines.append(m._format_metric("Baseline Utility", self.baseline_utility))
        lines.append(m._format_metric("Policy Utility", self.policy_utility))
        lines.append(m._format_metric("Baseline Expenditure", self.baseline_expenditure))
        lines.append(m._format_metric("Policy Expenditure", self.policy_expenditure))

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        if self.welfare_improved:
            lines.append("  Consumer welfare improved under the policy.")
            lines.append(f"  Equivalent to receiving ${abs(self.equivalent_variation):.2f}.")
        elif self.welfare_worsened:
            lines.append("  Consumer welfare worsened under the policy.")
            lines.append(f"  Would need ${abs(self.compensating_variation):.2f} to restore utility.")
        else:
            lines.append("  Welfare change is ambiguous (CV and EV have different signs).")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "welfare_direction": self.welfare_direction,
            "compensating_variation": self.compensating_variation,
            "equivalent_variation": self.equivalent_variation,
            "mean_variation": self.mean_variation,
            "hicksian_surplus": self.hicksian_surplus,
            "baseline_utility": self.baseline_utility,
            "policy_utility": self.policy_utility,
            "baseline_expenditure": self.baseline_expenditure,
            "policy_expenditure": self.policy_expenditure,
            "welfare_improved": self.welfare_improved,
            "welfare_worsened": self.welfare_worsened,
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        return f"WelfareResult({self.welfare_direction}, cv={self.compensating_variation:.2f})"


@dataclass(frozen=True)
class AdditivityResult:
    """
    Result of additive separability test (Chapter 9.3).

    Tests whether utility has the additive form U(x) = Σ u_i(x_i),
    which is stronger than quasilinearity.

    Attributes:
        is_additive: True if data is consistent with additive utility
        additive_groups: List of good indices that separate additively
        cross_effects_matrix: N x N matrix of cross-price effects (should be diagonal)
        max_cross_effect: Largest off-diagonal cross-price effect
        violations: List of (i, j) pairs showing cross-effects
        num_violations: Number of cross-effect violations
        computation_time_ms: Time taken in milliseconds
    """

    is_additive: bool
    additive_groups: list[set[int]]
    cross_effects_matrix: NDArray[np.float64]
    max_cross_effect: float
    violations: list[tuple[int, int]]
    num_violations: int
    computation_time_ms: float

    @property
    def num_goods(self) -> int:
        """Number of goods N."""
        return self.cross_effects_matrix.shape[0]

    @property
    def num_groups(self) -> int:
        """Number of additively separable groups."""
        return len(self.additive_groups)

    @property
    def is_fully_separable(self) -> bool:
        """True if each good is in its own group (fully additive)."""
        return self.num_groups == self.num_goods

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns 1 - max_cross_effect (lower cross-effects = more additive).
        """
        return max(0.0, 1.0 - min(1.0, self.max_cross_effect))

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("ADDITIVE SEPARABILITY REPORT")]

        # Status
        status = m._format_status(self.is_additive, "ADDITIVE", "NOT ADDITIVE")
        lines.append(f"\nStatus: {status}")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("Is Additive", self.is_additive))
        lines.append(m._format_metric("Fully Separable", self.is_fully_separable))
        lines.append(m._format_metric("Number of Goods", self.num_goods))
        lines.append(m._format_metric("Additive Groups", self.num_groups))
        lines.append(m._format_metric("Max Cross-Effect", self.max_cross_effect))
        lines.append(m._format_metric("Violations", self.num_violations))

        # Show groups
        if self.additive_groups:
            lines.append(m._format_section("Additive Groups"))
            for i, group in enumerate(self.additive_groups):
                lines.append(f"  Group {i}: {sorted(group)}")

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        if self.is_additive:
            lines.append("  Utility is additively separable: U(x) = Σ u_i(x_i).")
            lines.append("  No significant cross-price effects between groups.")
        else:
            lines.append("  Utility is not additively separable.")
            lines.append(f"  {self.num_violations} significant cross-effect(s) detected.")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "is_additive": self.is_additive,
            "is_fully_separable": self.is_fully_separable,
            "num_goods": self.num_goods,
            "num_groups": self.num_groups,
            "max_cross_effect": self.max_cross_effect,
            "num_violations": self.num_violations,
            "additive_groups": [list(g) for g in self.additive_groups],
            "violations": self.violations,
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        if self.is_additive:
            return f"AdditivityResult(additive, groups={self.num_groups})"
        return f"AdditivityResult(not additive, violations={self.num_violations})"


@dataclass(frozen=True)
class CompensatedDemandResult:
    """
    Result of compensated (Hicksian) demand analysis (Chapter 10.3).

    Decomposes price effects into substitution and income effects.

    Attributes:
        substitution_effects: N x N matrix of Hicksian substitution effects
        income_effects: N x N matrix of income effects
        satisfies_compensated_law: True if compensated law of demand holds
        own_price_elasticities: Dict mapping good index to own-price elasticity
        cross_price_elasticities: N x N matrix of cross-price elasticities
        violations: List of (i, j) pairs violating compensated law
        computation_time_ms: Time taken in milliseconds
    """

    substitution_effects: NDArray[np.float64]
    income_effects: NDArray[np.float64]
    satisfies_compensated_law: bool
    own_price_elasticities: dict[int, float]
    cross_price_elasticities: NDArray[np.float64]
    violations: list[tuple[int, int]]
    computation_time_ms: float

    @property
    def num_goods(self) -> int:
        """Number of goods N."""
        return self.substitution_effects.shape[0]

    @property
    def num_violations(self) -> int:
        """Number of compensated law violations."""
        return len(self.violations)

    @property
    def mean_own_elasticity(self) -> float:
        """Average own-price elasticity across goods."""
        if not self.own_price_elasticities:
            return 0.0
        return sum(self.own_price_elasticities.values()) / len(self.own_price_elasticities)

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns 1.0 if compensated law holds, 0.0 otherwise.
        """
        return 1.0 if self.satisfies_compensated_law else 0.0

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("COMPENSATED DEMAND ANALYSIS REPORT")]

        # Status
        status = m._format_status(self.satisfies_compensated_law,
                                  "COMPENSATED LAW SATISFIED", "VIOLATIONS FOUND")
        lines.append(f"\nStatus: {status}")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("Compensated Law", self.satisfies_compensated_law))
        lines.append(m._format_metric("Number of Goods", self.num_goods))
        lines.append(m._format_metric("Violations", self.num_violations))
        lines.append(m._format_metric("Mean Own Elasticity", self.mean_own_elasticity))

        # Elasticities
        if self.own_price_elasticities:
            lines.append(m._format_section("Own-Price Elasticities"))
            for good, elast in list(self.own_price_elasticities.items())[:5]:
                lines.append(f"  Good {good}: {elast:.4f}")
            if len(self.own_price_elasticities) > 5:
                lines.append(f"  ... ({len(self.own_price_elasticities) - 5} more goods)")

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        if self.satisfies_compensated_law:
            lines.append("  Compensated (Hicksian) demand is well-behaved.")
            lines.append("  Substitution effects are negative (law of demand holds).")
        else:
            lines.append(f"  {self.num_violations} Giffen-like good pair(s) detected.")
            lines.append("  Some substitution effects have wrong sign.")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "satisfies_compensated_law": self.satisfies_compensated_law,
            "num_goods": self.num_goods,
            "num_violations": self.num_violations,
            "mean_own_elasticity": self.mean_own_elasticity,
            "own_price_elasticities": self.own_price_elasticities,
            "violations": self.violations,
            "substitution_effects": self.substitution_effects.tolist(),
            "income_effects": self.income_effects.tolist(),
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        if self.satisfies_compensated_law:
            return f"CompensatedDemandResult(law_ok, n={self.num_goods})"
        return f"CompensatedDemandResult({self.num_violations} violations)"


@dataclass(frozen=True)
class GeneralMetricResult:
    """
    Result of general metric space ideal point analysis (Chapter 11.3-11.4).

    Extends Euclidean preferences to arbitrary metric spaces.

    Attributes:
        ideal_point: D-dimensional ideal point in feature space
        metric_type: Type of metric used ("L1", "L2", "Linf", "minkowski")
        metric_params: Parameters for the metric (e.g., p for Minkowski)
        is_rationalizable: True if choices are rationalizable under this metric
        violations: List of (choice_set_idx, unchosen_item_idx) violations
        best_metric: The metric type that best fits the data
        metric_comparison: Dict mapping metric type to violation count
        explained_variance: Fraction of choices explained by the model
        computation_time_ms: Time taken in milliseconds
    """

    ideal_point: NDArray[np.float64]
    metric_type: str
    metric_params: dict[str, float]
    is_rationalizable: bool
    violations: list[tuple[int, int]]
    best_metric: str
    metric_comparison: dict[str, int]
    explained_variance: float
    computation_time_ms: float

    @property
    def num_violations(self) -> int:
        """Number of violations under the chosen metric."""
        return len(self.violations)

    @property
    def num_dimensions(self) -> int:
        """Number of feature dimensions D."""
        return len(self.ideal_point)

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns explained_variance.
        """
        return self.explained_variance

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("GENERAL METRIC PREFERENCES REPORT")]

        # Status
        status = m._format_status(self.is_rationalizable, "RATIONALIZABLE", "VIOLATIONS FOUND")
        lines.append(f"\nStatus: {status}")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("Rationalizable", self.is_rationalizable))
        lines.append(m._format_metric("Metric Type", self.metric_type))
        lines.append(m._format_metric("Best Metric", self.best_metric))
        lines.append(m._format_metric("Explained Variance", self.explained_variance))
        lines.append(m._format_metric("Violations", self.num_violations))
        lines.append(m._format_metric("Dimensions", self.num_dimensions))

        # Ideal point
        lines.append(m._format_section("Ideal Point Coordinates"))
        for i, coord in enumerate(self.ideal_point[:5]):
            lines.append(f"  Dimension {i}: {coord:.4f}")
        if self.num_dimensions > 5:
            lines.append(f"  ... ({self.num_dimensions - 5} more dimensions)")

        # Metric comparison
        if self.metric_comparison:
            lines.append(m._format_section("Metric Comparison (violations)"))
            for metric, viol in self.metric_comparison.items():
                lines.append(f"  {metric}: {viol}")

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        lines.append(f"  Best fit achieved with {self.best_metric} metric.")
        lines.append(f"  Model explains {self.explained_variance*100:.1f}% of choice variance.")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "is_rationalizable": self.is_rationalizable,
            "metric_type": self.metric_type,
            "best_metric": self.best_metric,
            "explained_variance": self.explained_variance,
            "num_violations": self.num_violations,
            "num_dimensions": self.num_dimensions,
            "ideal_point": self.ideal_point.tolist(),
            "metric_params": self.metric_params,
            "metric_comparison": self.metric_comparison,
            "violations": self.violations,
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        return f"GeneralMetricResult({self.best_metric}, var={self.explained_variance:.4f})"


@dataclass(frozen=True)
class StochasticChoiceResult:
    """
    Result of stochastic/random utility model fitting (Chapter 13).

    Fits probabilistic choice models like logit, probit, or Luce model.

    Attributes:
        model_type: Type of model ("logit", "probit", "luce", "rum")
        parameters: Model parameters (e.g., temperature, scale)
        satisfies_iia: True if Independence of Irrelevant Alternatives holds
        choice_probabilities: Array of predicted choice probabilities
        log_likelihood: Log-likelihood of the fitted model
        aic: Akaike Information Criterion
        bic: Bayesian Information Criterion
        regularity_violations: List of observations violating regularity
        computation_time_ms: Time taken in milliseconds
    """

    model_type: str
    parameters: dict[str, float]
    satisfies_iia: bool
    choice_probabilities: NDArray[np.float64]
    log_likelihood: float
    aic: float
    bic: float
    regularity_violations: list[int]
    computation_time_ms: float

    @property
    def num_regularity_violations(self) -> int:
        """Number of regularity axiom violations."""
        return len(self.regularity_violations)

    @property
    def is_random_utility(self) -> bool:
        """True if consistent with random utility maximization."""
        return self.satisfies_iia and len(self.regularity_violations) == 0

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns normalized log-likelihood (scaled by number of observations).
        Uses 1.0 if random utility model holds, else scaled by AIC.
        """
        if self.is_random_utility:
            return 1.0
        # Use inverse of normalized AIC as score
        return max(0.0, 1.0 / (1.0 + abs(self.aic) / 1000))

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("STOCHASTIC CHOICE MODEL REPORT")]

        # Status
        status = m._format_status(self.is_random_utility,
                                  "RANDOM UTILITY MODEL FITS", "RUM VIOLATIONS")
        lines.append(f"\nStatus: {status}")
        lines.append(f"Model Type: {self.model_type}")

        # Metrics
        lines.append(m._format_section("Model Fit"))
        lines.append(m._format_metric("Log-Likelihood", self.log_likelihood))
        lines.append(m._format_metric("AIC", self.aic))
        lines.append(m._format_metric("BIC", self.bic))
        lines.append(m._format_metric("Satisfies IIA", self.satisfies_iia))
        lines.append(m._format_metric("Regularity Violations", self.num_regularity_violations))

        # Parameters
        if self.parameters:
            lines.append(m._format_section("Model Parameters"))
            for param, value in self.parameters.items():
                lines.append(f"  {param}: {value:.4f}")

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        if self.is_random_utility:
            lines.append("  Choices consistent with random utility maximization.")
            lines.append(f"  {self.model_type.capitalize()} model provides good fit.")
        else:
            if not self.satisfies_iia:
                lines.append("  IIA violated - choice probabilities context-dependent.")
            if self.regularity_violations:
                lines.append(f"  {self.num_regularity_violations} regularity violation(s) detected.")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "model_type": self.model_type,
            "parameters": self.parameters,
            "satisfies_iia": self.satisfies_iia,
            "is_random_utility": self.is_random_utility,
            "log_likelihood": self.log_likelihood,
            "aic": self.aic,
            "bic": self.bic,
            "num_regularity_violations": self.num_regularity_violations,
            "regularity_violations": self.regularity_violations,
            "choice_probabilities": self.choice_probabilities.tolist(),
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        rum = "RUM" if self.is_random_utility else "non-RUM"
        return f"StochasticChoiceResult({self.model_type}, {rum}, aic={self.aic:.2f})"


@dataclass(frozen=True)
class AttentionResult:
    """
    Result of limited attention model estimation (Chapter 14).

    Models consideration sets - items the consumer actually considers.

    Attributes:
        consideration_sets: List of estimated consideration sets per observation
        attention_parameter: Fraction of items typically considered (0-1)
        is_attention_rational: True if rationalizable with limited attention
        salience_weights: Array of attention weights per item
        default_option: Index of default option (if any)
        inattention_rate: Fraction of observations with limited attention
        rationalizable_observations: List of observations rationalizable with attention
        computation_time_ms: Time taken in milliseconds
    """

    consideration_sets: list[set[int]]
    attention_parameter: float
    is_attention_rational: bool
    salience_weights: NDArray[np.float64]
    default_option: int | None
    inattention_rate: float
    rationalizable_observations: list[int]
    computation_time_ms: float

    @property
    def num_observations(self) -> int:
        """Number of choice observations."""
        return len(self.consideration_sets)

    @property
    def mean_consideration_size(self) -> float:
        """Average size of consideration sets."""
        if not self.consideration_sets:
            return 0.0
        return sum(len(cs) for cs in self.consideration_sets) / len(self.consideration_sets)

    @property
    def rationalizability_rate(self) -> float:
        """Fraction of observations rationalizable with attention."""
        return len(self.rationalizable_observations) / max(self.num_observations, 1)

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns the rationalizability rate.
        """
        return self.rationalizability_rate

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("LIMITED ATTENTION MODEL REPORT")]

        # Status
        status = m._format_status(self.is_attention_rational,
                                  "ATTENTION RATIONAL", "ATTENTION VIOLATIONS")
        lines.append(f"\nStatus: {status}")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("Attention Rational", self.is_attention_rational))
        lines.append(m._format_metric("Attention Parameter", self.attention_parameter))
        lines.append(m._format_metric("Inattention Rate", self.inattention_rate))
        lines.append(m._format_metric("Observations", self.num_observations))
        lines.append(m._format_metric("Mean Consideration Size", self.mean_consideration_size))
        lines.append(m._format_metric("Rationalizable Obs", len(self.rationalizable_observations)))
        lines.append(m._format_metric("Rationalizability Rate", self.rationalizability_rate))
        if self.default_option is not None:
            lines.append(m._format_metric("Default Option", self.default_option))

        # Salience weights
        if len(self.salience_weights) > 0:
            lines.append(m._format_section("Salience Weights (top 5)"))
            sorted_idx = np.argsort(self.salience_weights)[::-1]
            for i in sorted_idx[:5]:
                lines.append(f"  Item {i}: {self.salience_weights[i]:.4f}")

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        if self.is_attention_rational:
            lines.append("  Behavior is rationalizable with limited attention.")
            lines.append(f"  Consumer considers ~{self.mean_consideration_size:.1f} items on average.")
        else:
            lines.append("  Behavior cannot be explained by limited attention alone.")
            lines.append(f"  {self.rationalizability_rate*100:.1f}% of observations rationalizable.")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "is_attention_rational": self.is_attention_rational,
            "attention_parameter": self.attention_parameter,
            "inattention_rate": self.inattention_rate,
            "num_observations": self.num_observations,
            "mean_consideration_size": self.mean_consideration_size,
            "rationalizability_rate": self.rationalizability_rate,
            "default_option": self.default_option,
            "salience_weights": self.salience_weights.tolist(),
            "rationalizable_observations": self.rationalizable_observations,
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        status = "rational" if self.is_attention_rational else "irrational"
        return f"AttentionResult({status}, attn={self.attention_parameter:.4f})"


@dataclass(frozen=True)
class ProductionGARPResult:
    """
    Result of production theory GARP test (Chapter 15).

    Tests profit maximization consistency for firm behavior.

    Attributes:
        is_profit_maximizing: True if data satisfies production GARP
        violations: List of violation cycles
        cost_efficiency_score: Efficiency score for cost minimization (0-1)
        returns_to_scale: "increasing", "constant", "decreasing", or "variable"
        profit_efficiency: Overall profit efficiency score (0-1)
        input_efficiency_vector: Per-input efficiency scores
        output_efficiency_vector: Per-output efficiency scores
        technical_efficiency: Technical efficiency score
        computation_time_ms: Time taken in milliseconds
    """

    is_profit_maximizing: bool
    violations: list[Cycle]
    cost_efficiency_score: float
    returns_to_scale: str
    profit_efficiency: float
    input_efficiency_vector: NDArray[np.float64]
    output_efficiency_vector: NDArray[np.float64]
    technical_efficiency: float
    computation_time_ms: float

    @property
    def is_consistent(self) -> bool:
        """True if behavior is profit-maximizing consistent."""
        return self.is_profit_maximizing

    @property
    def num_violations(self) -> int:
        """Number of violation cycles."""
        return len(self.violations)

    @property
    def is_cost_minimizing(self) -> bool:
        """True if firm is cost-minimizing (dual of profit max)."""
        return self.cost_efficiency_score >= 1.0 - 1e-6

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns the profit efficiency score.
        """
        return self.profit_efficiency

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("PRODUCTION GARP TEST REPORT")]

        # Status
        status = m._format_status(self.is_profit_maximizing,
                                  "PROFIT MAXIMIZING", "VIOLATIONS FOUND")
        lines.append(f"\nStatus: {status}")
        lines.append(f"Returns to Scale: {self.returns_to_scale}")

        # Metrics
        lines.append(m._format_section("Efficiency Metrics"))
        lines.append(m._format_metric("Profit Maximizing", self.is_profit_maximizing))
        lines.append(m._format_metric("Cost Minimizing", self.is_cost_minimizing))
        lines.append(m._format_metric("Profit Efficiency", self.profit_efficiency))
        lines.append(m._format_metric("Cost Efficiency", self.cost_efficiency_score))
        lines.append(m._format_metric("Technical Efficiency", self.technical_efficiency))
        lines.append(m._format_metric("Violations", self.num_violations))

        # Input/Output efficiencies
        if len(self.input_efficiency_vector) > 0:
            lines.append(m._format_section("Input Efficiencies"))
            for i, eff in enumerate(self.input_efficiency_vector[:5]):
                lines.append(f"  Input {i}: {eff:.4f}")
            if len(self.input_efficiency_vector) > 5:
                lines.append(f"  ... ({len(self.input_efficiency_vector) - 5} more inputs)")

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        if self.is_profit_maximizing:
            lines.append("  Firm behavior is consistent with profit maximization.")
            lines.append(f"  Returns to scale: {self.returns_to_scale}.")
        else:
            lines.append(f"  {self.num_violations} violation(s) of profit maximization.")
            lines.append(f"  Profit efficiency is {self.profit_efficiency*100:.1f}%.")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "is_profit_maximizing": self.is_profit_maximizing,
            "is_cost_minimizing": self.is_cost_minimizing,
            "returns_to_scale": self.returns_to_scale,
            "profit_efficiency": self.profit_efficiency,
            "cost_efficiency_score": self.cost_efficiency_score,
            "technical_efficiency": self.technical_efficiency,
            "num_violations": self.num_violations,
            "input_efficiency_vector": self.input_efficiency_vector.tolist(),
            "output_efficiency_vector": self.output_efficiency_vector.tolist(),
            "violations": [list(c) for c in self.violations],
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        if self.is_profit_maximizing:
            return f"ProductionGARPResult(profit_max, {self.returns_to_scale})"
        return f"ProductionGARPResult({self.num_violations} violations, eff={self.profit_efficiency:.4f})"


# =============================================================================
# TECH-FRIENDLY ALIASES FOR NEW ALGORITHM RESULTS
# =============================================================================

# SlutskyConditionsResult: Tech-friendly alias for IntegrabilityResult
SlutskyConditionsResult = IntegrabilityResult
"""
Tech-friendly alias for IntegrabilityResult.

Tests if demand can be derived from utility maximization via Slutsky conditions.
"""

# WelfareChangeResult: Tech-friendly alias for WelfareResult
WelfareChangeResult = WelfareResult
"""
Tech-friendly alias for WelfareResult.

Measures welfare impact of price or policy changes.
"""

# AdditiveUtilityResult: Tech-friendly alias for AdditivityResult
AdditiveUtilityResult = AdditivityResult
"""
Tech-friendly alias for AdditivityResult.

Tests if preferences are additively separable across goods.
"""

# HicksianDemandResult: Tech-friendly alias for CompensatedDemandResult
HicksianDemandResult = CompensatedDemandResult
"""
Tech-friendly alias for CompensatedDemandResult.

Contains Slutsky decomposition of price effects.
"""

# MetricPreferencesResult: Tech-friendly alias for GeneralMetricResult
MetricPreferencesResult = GeneralMetricResult
"""
Tech-friendly alias for GeneralMetricResult.

Ideal point model with general distance metrics.
"""

# RandomUtilityResult: Tech-friendly alias for StochasticChoiceResult
RandomUtilityResult = StochasticChoiceResult
"""
Tech-friendly alias for StochasticChoiceResult.

Random utility model parameters and fit statistics.
"""

# ConsiderationSetResult: Tech-friendly alias for AttentionResult
ConsiderationSetResult = AttentionResult
"""
Tech-friendly alias for AttentionResult.

Limited attention model with consideration sets.
"""

# FirmBehaviorResult: Tech-friendly alias for ProductionGARPResult
FirmBehaviorResult = ProductionGARPResult
"""
Tech-friendly alias for ProductionGARPResult.

Production theory consistency test for firm behavior.
"""
