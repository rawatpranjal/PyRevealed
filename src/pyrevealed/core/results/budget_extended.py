from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pyrevealed.core.types import Cycle
from pyrevealed.core.mixins import ResultSummaryMixin
from pyrevealed.core.display import ResultDisplayMixin, ResultPlotMixin

__all__ = [
    "WARPResult",
    "SARPResult",
    "HoutmanMaksResult",
    "BronarsPowerResult",
    "HARPResult",
    "QuasilinearityResult",
    "GrossSubstitutesResult",
    "SubstitutionMatrixResult",
    "VEIResult",
    "DifferentiableResult",
    "AcyclicalPResult",
    "GAPPResult",
    "LancasterResult",
    "TestPowerResult",
    "ProportionalScalingResult",
    "IncomeInvarianceResult",
    "CrossPriceResult",
    "GranularIntegrityResult",
    "SmoothPreferencesResult",
    "StrictConsistencyResult",
    "PricePreferencesResult",
    "CharacteristicsValuationResult",
]

# Need GARPResult for type references
from pyrevealed.core.results.budget_core import GARPResult


@dataclass(frozen=True)
class WARPResult(ResultDisplayMixin, ResultPlotMixin):
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
        """Compact string representation with [+]/[-] indicator."""
        indicator = "[+]" if self.is_consistent else "[-]"
        status = "CONSISTENT" if self.is_consistent else f"{self.num_violations} violations"
        return f"WARPResult: {indicator} {status} ({self.computation_time_ms:.2f}ms)"

    def short_summary(self) -> str:
        """Return one-liner with [+]/[-] indicator."""
        indicator = "[+]" if self.is_consistent else "[-]"
        status = "PASS" if self.is_consistent else "FAIL"
        return f"WARP: {indicator} {status}"


@dataclass(frozen=True)
class SARPResult(ResultDisplayMixin, ResultPlotMixin):
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
        """Compact string representation with [+]/[-] indicator."""
        indicator = "[+]" if self.is_consistent else "[-]"
        status = "CONSISTENT" if self.is_consistent else f"{self.num_violations} violations"
        return f"SARPResult: {indicator} {status} ({self.computation_time_ms:.2f}ms)"

    def short_summary(self) -> str:
        """Return one-liner with [+]/[-] indicator."""
        indicator = "[+]" if self.is_consistent else "[-]"
        status = "PASS" if self.is_consistent else "FAIL"
        return f"SARP: {indicator} {status}"


@dataclass(frozen=True)
class HoutmanMaksResult(ResultDisplayMixin, ResultPlotMixin):
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
        """Compact string representation with [+]/[-] indicator."""
        indicator = "[+]" if self.is_consistent else "[-]"
        if self.is_consistent:
            return f"HoutmanMaksResult: {indicator} CONSISTENT ({self.computation_time_ms:.2f}ms)"
        return f"HoutmanMaksResult: {indicator} remove={self.num_removed} (frac={self.fraction:.4f})"

    def short_summary(self) -> str:
        """Return one-liner with [+]/[-] indicator."""
        indicator = "[+]" if self.is_consistent else "[-]"
        return f"Houtman-Maks: {indicator} {1.0 - self.fraction:.4f}"


@dataclass(frozen=True)
class BronarsPowerResult(ResultDisplayMixin, ResultPlotMixin):
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
        """Compact string representation with [+]/[-] indicator."""
        indicator = "[+]" if self.is_significant else "[-]"
        status = "SIGNIFICANT" if self.is_significant else "LOW POWER"
        return f"BronarsPowerResult: {indicator} {status} (power={self.power_index:.4f})"

    def short_summary(self) -> str:
        """Return one-liner with [+]/[-] indicator."""
        indicator = "[+]" if self.is_significant else "[-]"
        return f"Power: {indicator} {self.power_index:.4f}"


@dataclass(frozen=True)
class HARPResult(ResultDisplayMixin, ResultPlotMixin):
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
        """Compact string representation with [+]/[-] indicator."""
        indicator = "[+]" if self.is_consistent else "[-]"
        status = "HOMOTHETIC" if self.is_consistent else f"{self.num_violations} violations"
        return f"HARPResult: {indicator} {status} (max_prod={self.max_cycle_product:.4f})"

    def short_summary(self) -> str:
        """Return one-liner with [+]/[-] indicator."""
        indicator = "[+]" if self.is_consistent else "[-]"
        return f"HARP: {indicator} {'PASS' if self.is_consistent else 'FAIL'}"


@dataclass(frozen=True)
class QuasilinearityResult(ResultDisplayMixin, ResultPlotMixin):
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
        """Compact string representation with [+]/[-] indicator."""
        indicator = "[+]" if self.is_quasilinear else "[-]"
        status = "QUASILINEAR" if self.is_quasilinear else f"{self.num_violations} violations"
        return f"QuasilinearityResult: {indicator} {status}"

    def short_summary(self) -> str:
        """Return one-liner with [+]/[-] indicator."""
        indicator = "[+]" if self.is_quasilinear else "[-]"
        return f"Quasilinear: {indicator} {'PASS' if self.is_quasilinear else 'FAIL'}"


@dataclass(frozen=True)
class GrossSubstitutesResult(ResultDisplayMixin, ResultPlotMixin):
    """
    Result of gross substitutes test between two goods.

    Tests whether two goods are substitutes (price of A up \u2192 demand for B up)
    or complements (price of A up \u2192 demand for B down).

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
        """Compact string representation with [+]/[-] indicator."""
        indicator = "[+]" if self.is_conclusive else "[~]"
        return f"GrossSubstitutesResult: {indicator} g{self.good_g_index}<->g{self.good_h_index}: {self.relationship}"

    def short_summary(self) -> str:
        """Return one-liner with [+]/[-] indicator."""
        return f"Gross Substitutes: g{self.good_g_index}<->g{self.good_h_index}: {self.relationship}"


@dataclass(frozen=True)
class SubstitutionMatrixResult(ResultDisplayMixin, ResultPlotMixin):
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
class VEIResult(ResultDisplayMixin, ResultPlotMixin):
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
class DifferentiableResult(ResultDisplayMixin, ResultPlotMixin):
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
class AcyclicalPResult(ResultDisplayMixin, ResultPlotMixin):
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
class GAPPResult(ResultDisplayMixin, ResultPlotMixin):
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
class LancasterResult(ResultDisplayMixin, ResultPlotMixin):
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
