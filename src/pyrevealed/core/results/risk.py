from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pyrevealed.core.types import Cycle
from pyrevealed.core.mixins import ResultSummaryMixin
from pyrevealed.core.display import ResultDisplayMixin, ResultPlotMixin

__all__ = [
    "RiskProfileResult",
    "ExpectedUtilityResult",
    "RankDependentUtilityResult",
]


@dataclass(frozen=True)
class RiskProfileResult(ResultDisplayMixin, ResultPlotMixin):
    """
    Result of risk profile analysis from choices under uncertainty.

    Classifies decision-makers as risk-seeking, risk-neutral, or risk-averse
    based on their revealed preferences between safe and risky options.

    Attributes:
        risk_aversion_coefficient: Arrow-Pratt coefficient \u03c1
            - \u03c1 > 0: risk averse (prefers certainty)
            - \u03c1 \u2248 0: risk neutral (maximizes expected value)
            - \u03c1 < 0: risk seeking (prefers gambles)
        risk_category: Classification string: "risk_seeking" | "risk_neutral" | "risk_averse"
        certainty_equivalents: Array of CEs for each lottery (amount of certain money
            equivalent to the risky option for this decision-maker)
        utility_curvature: Estimated curvature of utility function
        consistency_score: How well the CRRA model fits the choices (0-1)
        num_consistent_choices: Number of choices consistent with estimated \u03c1
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
        """Compact string representation with [+]/[-] indicator."""
        indicator = "[+]" if self.consistency_score >= 0.9 else "[-]"
        return f"RiskProfileResult: {indicator} {self.risk_category} (rho={self.risk_aversion_coefficient:.4f})"

    def short_summary(self) -> str:
        """Return one-liner with [+]/[-] indicator."""
        indicator = "[+]" if self.consistency_score >= 0.9 else "[-]"
        return f"Risk Profile: {indicator} {self.risk_category}"


@dataclass(frozen=True)
class ExpectedUtilityResult(ResultDisplayMixin, ResultPlotMixin):
    """
    Result of Expected Utility axiom test using GRID method.

    Tests whether lottery choices can be rationalized by Expected Utility (EU)
    theory using the GRID (Generalized Restriction of Infinite Domains) method
    from Polisson, Quah & Renou (2020).

    Attributes:
        is_consistent: True if choices satisfy EU axioms
        risk_attitude: Detected risk attitude ("any", "averse", "seeking", "neutral")
        violations: List of choice pairs violating EU
        violation_severity: Severity measure of violations
        num_choices: Total number of lottery choices tested
        num_violations: Number of EU violations found
        computation_time_ms: Time taken in milliseconds
    """

    is_consistent: bool
    risk_attitude: str
    violations: list[tuple[int, int]]
    violation_severity: float
    num_choices: int
    num_violations: int
    computation_time_ms: float

    def score(self) -> float:
        """Return consistency score (1 = consistent, lower = more violations)."""
        if self.num_choices == 0:
            return 1.0
        return 1.0 - (self.num_violations / max(1, self.num_choices))

    def summary(self) -> str:
        """Return formatted summary of the Expected Utility test result."""
        m = _formatting()
        lines = [m._format_header("EXPECTED UTILITY TEST RESULT")]

        # Status
        status = "CONSISTENT" if self.is_consistent else "VIOLATED"
        lines.append(f"\nStatus: {status}")
        lines.append(f"Risk Attitude: {self.risk_attitude}")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("Choices Tested", self.num_choices))
        lines.append(m._format_metric("Violations Found", self.num_violations))
        lines.append(m._format_metric("Violation Severity", self.violation_severity))
        lines.append(m._format_metric("Consistency Score", self.score()))

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation."""
        return {
            "is_consistent": self.is_consistent,
            "risk_attitude": self.risk_attitude,
            "violations": self.violations,
            "violation_severity": self.violation_severity,
            "num_choices": self.num_choices,
            "num_violations": self.num_violations,
            "score": self.score(),
            "computation_time_ms": self.computation_time_ms,
        }

    def __repr__(self) -> str:
        indicator = "[+]" if self.is_consistent else "[-]"
        return f"ExpectedUtilityResult: {indicator} EU-{self.risk_attitude} ({self.num_violations} violations)"


@dataclass(frozen=True)
class RankDependentUtilityResult(ResultDisplayMixin, ResultPlotMixin):
    """
    Result of Rank-Dependent Utility (RDU) axiom test.

    Tests whether lottery choices can be rationalized by Rank-Dependent
    Utility theory, which allows for probability weighting.

    Attributes:
        is_consistent: True if choices satisfy RDU axioms
        probability_weighting: Detected probability weighting type
        violations: List of choice pairs violating RDU
        violation_severity: Severity measure of violations
        num_choices: Total number of lottery choices tested
        num_violations: Number of RDU violations found
        computation_time_ms: Time taken in milliseconds
    """

    is_consistent: bool
    probability_weighting: str
    violations: list[tuple[int, int]]
    violation_severity: float
    num_choices: int
    num_violations: int
    computation_time_ms: float

    def score(self) -> float:
        """Return consistency score (1 = consistent, lower = more violations)."""
        if self.num_choices == 0:
            return 1.0
        return 1.0 - (self.num_violations / max(1, self.num_choices))

    def summary(self) -> str:
        """Return formatted summary of the RDU test result."""
        m = _formatting()
        lines = [m._format_header("RANK-DEPENDENT UTILITY TEST RESULT")]

        # Status
        status = "CONSISTENT" if self.is_consistent else "VIOLATED"
        lines.append(f"\nStatus: {status}")
        lines.append(f"Probability Weighting: {self.probability_weighting}")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("Choices Tested", self.num_choices))
        lines.append(m._format_metric("Violations Found", self.num_violations))
        lines.append(m._format_metric("Violation Severity", self.violation_severity))
        lines.append(m._format_metric("Consistency Score", self.score()))

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation."""
        return {
            "is_consistent": self.is_consistent,
            "probability_weighting": self.probability_weighting,
            "violations": self.violations,
            "violation_severity": self.violation_severity,
            "num_choices": self.num_choices,
            "num_violations": self.num_violations,
            "score": self.score(),
            "computation_time_ms": self.computation_time_ms,
        }

    def __repr__(self) -> str:
        indicator = "[+]" if self.is_consistent else "[-]"
        return f"RankDependentUtilityResult: {indicator} RDU-{self.probability_weighting} ({self.num_violations} violations)"
