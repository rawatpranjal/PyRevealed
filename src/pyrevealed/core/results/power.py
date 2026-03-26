from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pyrevealed.core.types import Cycle
from pyrevealed.core.mixins import ResultSummaryMixin
from pyrevealed.core.display import ResultDisplayMixin, ResultPlotMixin

__all__ = [
    "SeltenMeasureResult",
    "RelativeAreaResult",
    "SmoothedHitRateResult",
    "BayesianCredibilityResult",
    "OptimalEfficiencyResult",
]


@dataclass(frozen=True)
class SeltenMeasureResult(ResultDisplayMixin, ResultPlotMixin):
    """
    Result of Selten's predictive success measure computation.

    Selten's measure m = r - a captures how well utility maximization
    performs beyond chance:
    - r: Pass rate (1 if data passes GARP, 0 otherwise)
    - a: Relative area (probability random behavior passes GARP)

    Interpretation:
    - m → 1: Demanding restrictions + data satisfies (strong evidence)
    - m ≈ 0: Model performs about as well as random (uninformative)
    - m → -1: Easy restrictions + data fails (bad fit)

    Attributes:
        measure: Selten's m = r - a
        pass_rate: Binary pass rate (1 or 0)
        relative_area: Probability random behavior passes GARP
        n_simulations: Number of Monte Carlo simulations
        algorithm: Bundle generation algorithm used (1, 2, or 3)
        is_meaningful: Whether measure exceeds threshold (m > 0.05)
        computation_time_ms: Time taken in milliseconds

    References:
        Beatty, T. K., & Crawford, I. A. (2011). How demanding is the
        revealed preference approach to demand? AER, 101(6), 2782-2795.
    """

    measure: float
    pass_rate: float
    relative_area: float
    n_simulations: int
    algorithm: int
    is_meaningful: bool
    computation_time_ms: float

    @property
    def bronars_power(self) -> float:
        """Bronars power = 1 - relative_area."""
        return 1.0 - self.relative_area

    def score(self) -> float:
        """Return scikit-learn style score. Higher is better."""
        return (self.measure + 1) / 2  # Normalize from [-1, 1] to [0, 1]

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("SELTEN'S PREDICTIVE SUCCESS REPORT")]

        # Status
        if self.measure > 0.1:
            status = "MEANINGFUL (m > 0.1)"
        elif self.measure > 0.05:
            status = "MARGINAL"
        elif self.measure > 0:
            status = "WEAK POSITIVE"
        elif self.measure > -0.05:
            status = "UNINFORMATIVE"
        else:
            status = "NEGATIVE (model worse than random)"
        lines.append(f"\nStatus: {status}")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("Selten's m", f"{self.measure:.4f}"))
        lines.append(m._format_metric("Pass Rate (r)", f"{self.pass_rate:.1f}"))
        lines.append(m._format_metric("Relative Area (a)", f"{self.relative_area:.4f}"))
        lines.append(m._format_metric("Bronars Power", f"{self.bronars_power:.4f}"))
        lines.append(m._format_metric("Simulations", self.n_simulations))
        lines.append(m._format_metric("Algorithm", self.algorithm))

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        if self.pass_rate == 1.0:
            lines.append("  Data satisfies GARP.")
        else:
            lines.append("  Data violates GARP.")

        area_pct = self.relative_area * 100
        lines.append(f"  {area_pct:.1f}% of random behavior would also pass GARP.")

        if self.measure > 0.1:
            lines.append("  Strong evidence: behavior meaningfully better than random.")
        elif self.measure > 0:
            lines.append("  Weak evidence: behavior slightly better than random.")
        else:
            lines.append("  No evidence: passing GARP is uninformative for this data.")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "measure": self.measure,
            "pass_rate": self.pass_rate,
            "relative_area": self.relative_area,
            "bronars_power": self.bronars_power,
            "n_simulations": self.n_simulations,
            "algorithm": self.algorithm,
            "is_meaningful": self.is_meaningful,
            "computation_time_ms": self.computation_time_ms,
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        indicator = "[+]" if self.is_meaningful else "[-]"
        return f"SeltenMeasureResult: {indicator} m={self.measure:.4f} (r={self.pass_rate:.0f}, a={self.relative_area:.3f})"

    def short_summary(self) -> str:
        """Return one-liner with key metrics."""
        indicator = "[+]" if self.is_meaningful else "[-]"
        return f"Selten: {indicator} m={self.measure:.4f}"


@dataclass(frozen=True)
class RelativeAreaResult(ResultDisplayMixin, ResultPlotMixin):
    """
    Result of relative area computation.

    The relative area 'a' measures how demanding the GARP test is for
    a given price configuration. It equals the probability that random
    behavior would satisfy GARP.

    Interpretation:
    - a ≈ 1: GARP very easy to satisfy (low power, test uninformative)
    - a ≈ 0: GARP very hard to satisfy (high power, test demanding)

    Relationship to Bronars power: a ≈ 1 - Bronars_power

    Attributes:
        relative_area: Proportion of random behavior that passes GARP
        std_error: Standard error of area estimate
        ci_lower: Lower bound of 95% confidence interval
        ci_upper: Upper bound of 95% confidence interval
        n_simulations: Number of Monte Carlo simulations
        n_consistent: Number of simulations that passed GARP
        algorithm: Bundle generation algorithm used
        computation_time_ms: Time taken in milliseconds

    References:
        Beatty, T. K., & Crawford, I. A. (2011). How demanding is the
        revealed preference approach to demand? AER, 101(6), 2782-2795.
    """

    relative_area: float
    std_error: float
    ci_lower: float
    ci_upper: float
    n_simulations: int
    n_consistent: int
    algorithm: int
    computation_time_ms: float

    @property
    def bronars_power(self) -> float:
        """Bronars power = 1 - relative_area."""
        return 1.0 - self.relative_area

    @property
    def is_demanding(self) -> bool:
        """Test is demanding if area < 0.5."""
        return self.relative_area < 0.5

    def score(self) -> float:
        """Return scikit-learn style score. Lower area = higher power = better."""
        return 1.0 - self.relative_area

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("RELATIVE AREA REPORT")]

        # Status
        if self.relative_area < 0.3:
            status = "HIGHLY DEMANDING TEST"
        elif self.relative_area < 0.5:
            status = "DEMANDING TEST"
        elif self.relative_area < 0.7:
            status = "MODERATE TEST"
        elif self.relative_area < 0.9:
            status = "WEAK TEST"
        else:
            status = "UNMISSABLE TARGET (a > 0.9)"
        lines.append(f"\nStatus: {status}")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("Relative Area (a)", f"{self.relative_area:.4f}"))
        lines.append(m._format_metric("Standard Error", f"{self.std_error:.4f}"))
        lines.append(m._format_metric("95% CI", f"[{self.ci_lower:.4f}, {self.ci_upper:.4f}]"))
        lines.append(m._format_metric("Bronars Power", f"{self.bronars_power:.4f}"))
        lines.append(m._format_metric("Simulations", self.n_simulations))
        lines.append(m._format_metric("Consistent Count", self.n_consistent))
        lines.append(m._format_metric("Algorithm", self.algorithm))

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        pct = self.relative_area * 100
        lines.append(f"  {pct:.1f}% of random behaviors would satisfy GARP.")

        if self.relative_area > 0.9:
            lines.append("  Warning: Budget sets barely intersect.")
            lines.append("  GARP is an 'unmissable target' - test is uninformative.")
        elif self.relative_area > 0.5:
            lines.append("  GARP test has limited discriminatory power.")
        else:
            lines.append("  GARP test has good discriminatory power.")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "relative_area": self.relative_area,
            "std_error": self.std_error,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "bronars_power": self.bronars_power,
            "n_simulations": self.n_simulations,
            "n_consistent": self.n_consistent,
            "algorithm": self.algorithm,
            "is_demanding": self.is_demanding,
            "computation_time_ms": self.computation_time_ms,
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        indicator = "[+]" if self.is_demanding else "[-]"
        return f"RelativeAreaResult: {indicator} a={self.relative_area:.4f} ± {self.std_error:.4f}"

    def short_summary(self) -> str:
        """Return one-liner."""
        indicator = "[+]" if self.is_demanding else "[-]"
        return f"Area: {indicator} a={self.relative_area:.4f}"


@dataclass(frozen=True)
class SmoothedHitRateResult(ResultDisplayMixin, ResultPlotMixin):
    """
    Result of smoothed hit rate computation for GARP violators.

    For data that violates GARP, the smoothed hit rate rd = 1 - d/d_max
    distinguishes near-misses from wild misses:
    - d: Distance from observed data to GARP-consistent region
    - d_max: Maximum possible distance

    Interpretation:
    - rd ≈ 1: Near miss (almost rational)
    - rd ≈ 0: Wild miss (far from rational)

    Attributes:
        smoothed_rate: rd = 1 - d/d_max ≈ AEI
        distance: Distance from GARP-consistent region
        max_distance: Maximum possible distance
        aei: Afriat Efficiency Index (basis for distance)
        is_consistent: Whether data satisfies GARP
        computation_time_ms: Time taken in milliseconds

    References:
        Beatty, T. K., & Crawford, I. A. (2011). Section IV.
    """

    smoothed_rate: float
    distance: float
    max_distance: float
    aei: float
    is_consistent: bool
    computation_time_ms: float

    @property
    def is_near_miss(self) -> bool:
        """Data is a near-miss if smoothed rate > 0.9."""
        return not self.is_consistent and self.smoothed_rate > 0.9

    def score(self) -> float:
        """Return scikit-learn style score. Higher is better."""
        return self.smoothed_rate

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("SMOOTHED HIT RATE REPORT")]

        # Status
        if self.is_consistent:
            status = "CONSISTENT (rd = 1.0)"
        elif self.smoothed_rate > 0.95:
            status = "NEAR MISS"
        elif self.smoothed_rate > 0.9:
            status = "CLOSE"
        elif self.smoothed_rate > 0.7:
            status = "MODERATE DEVIATION"
        else:
            status = "LARGE DEVIATION"
        lines.append(f"\nStatus: {status}")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("Smoothed Hit Rate (rd)", f"{self.smoothed_rate:.4f}"))
        lines.append(m._format_metric("Distance (d)", f"{self.distance:.4f}"))
        lines.append(m._format_metric("Max Distance", f"{self.max_distance:.4f}"))
        lines.append(m._format_metric("AEI", f"{self.aei:.4f}"))
        lines.append(m._format_metric("Is Consistent", self.is_consistent))

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        if self.is_consistent:
            lines.append("  Data satisfies GARP exactly.")
        else:
            pct = (1 - self.smoothed_rate) * 100
            lines.append(f"  Data deviates {pct:.1f}% from perfect rationality.")
            if self.is_near_miss:
                lines.append("  This is a 'near miss' - behavior is almost rational.")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "smoothed_rate": self.smoothed_rate,
            "distance": self.distance,
            "max_distance": self.max_distance,
            "aei": self.aei,
            "is_consistent": self.is_consistent,
            "is_near_miss": self.is_near_miss,
            "computation_time_ms": self.computation_time_ms,
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        if self.is_consistent:
            return f"SmoothedHitRateResult: [+] CONSISTENT (rd=1.0)"
        indicator = "[~]" if self.is_near_miss else "[-]"
        return f"SmoothedHitRateResult: {indicator} rd={self.smoothed_rate:.4f} (AEI={self.aei:.4f})"

    def short_summary(self) -> str:
        """Return one-liner."""
        if self.is_consistent:
            return "Smoothed: [+] rd=1.0"
        indicator = "[~]" if self.is_near_miss else "[-]"
        return f"Smoothed: {indicator} rd={self.smoothed_rate:.4f}"


@dataclass(frozen=True)
class BayesianCredibilityResult(ResultDisplayMixin, ResultPlotMixin):
    """
    Result of Bayesian credibility computation.

    Computes the posterior probability of utility maximization using
    Bayes' rule: P(R|G) = P(G|R)P(R) / [P(G|R)P(R) + P(G|~R)P(~R)]

    Key insight: If test has low power (P(G|~R) ≈ 1), passing GARP
    doesn't update beliefs much. Need high power for strong inference.

    Attributes:
        posterior: P(Rational | GARP result)
        prior: Prior P(Rational) used
        likelihood_ratio: P(G|R) / P(G|~R)
        bayes_factor: Same as likelihood_ratio
        p_pass_given_rational: P(Pass | Rational) = 1
        p_pass_given_random: P(Pass | Random) = relative_area
        passes_garp: Whether data passed GARP
        evidence_strength: Qualitative interpretation (Jeffreys scale)
        computation_time_ms: Time taken in milliseconds

    References:
        Crawford, I. (2019). Mini Course on Empirical Revealed Preference.
    """

    posterior: float
    prior: float
    likelihood_ratio: float
    bayes_factor: float
    p_pass_given_rational: float
    p_pass_given_random: float
    passes_garp: bool
    evidence_strength: str
    computation_time_ms: float

    @property
    def belief_update(self) -> float:
        """How much posterior differs from prior."""
        return self.posterior - self.prior

    def score(self) -> float:
        """Return scikit-learn style score. Higher is better."""
        return self.posterior

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("BAYESIAN CREDIBILITY REPORT")]

        # Status
        if self.posterior > 0.95:
            status = "STRONG EVIDENCE FOR RATIONALITY"
        elif self.posterior > 0.8:
            status = "MODERATE EVIDENCE"
        elif self.posterior > 0.5:
            status = "WEAK EVIDENCE"
        else:
            status = "EVIDENCE AGAINST RATIONALITY"
        lines.append(f"\nStatus: {status}")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("Posterior P(Rational)", f"{self.posterior:.4f}"))
        lines.append(m._format_metric("Prior P(Rational)", f"{self.prior:.4f}"))
        lines.append(m._format_metric("Belief Update", f"{self.belief_update:+.4f}"))
        lines.append(m._format_metric("Bayes Factor", f"{self.bayes_factor:.2f}"))
        lines.append(m._format_metric("Evidence Strength", self.evidence_strength))
        lines.append(m._format_metric("Passes GARP", self.passes_garp))

        # Likelihoods
        lines.append(m._format_section("Likelihoods"))
        lines.append(m._format_metric("P(Pass | Rational)", f"{self.p_pass_given_rational:.4f}"))
        lines.append(m._format_metric("P(Pass | Random)", f"{self.p_pass_given_random:.4f}"))

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        if self.passes_garp:
            if self.bayes_factor > 10:
                lines.append("  Strong Bayesian evidence: data supports rationality.")
            elif self.bayes_factor > 3:
                lines.append("  Moderate evidence: data somewhat supports rationality.")
            else:
                lines.append("  Weak evidence: passing GARP is not very informative here.")
        else:
            lines.append("  Data violates GARP.")
            lines.append("  Under strict model, posterior = 0.")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "posterior": self.posterior,
            "prior": self.prior,
            "belief_update": self.belief_update,
            "likelihood_ratio": self.likelihood_ratio,
            "bayes_factor": self.bayes_factor,
            "p_pass_given_rational": self.p_pass_given_rational,
            "p_pass_given_random": self.p_pass_given_random,
            "passes_garp": self.passes_garp,
            "evidence_strength": self.evidence_strength,
            "computation_time_ms": self.computation_time_ms,
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        indicator = "[+]" if self.posterior > 0.5 else "[-]"
        return f"BayesianCredibilityResult: {indicator} P(R|data)={self.posterior:.4f} (BF={self.bayes_factor:.2f})"

    def short_summary(self) -> str:
        """Return one-liner."""
        indicator = "[+]" if self.posterior > 0.5 else "[-]"
        return f"Credibility: {indicator} P={self.posterior:.4f}"


@dataclass(frozen=True)
class OptimalEfficiencyResult(ResultDisplayMixin, ResultPlotMixin):
    """
    Result of optimal Afriat efficiency computation.

    Instead of testing GARP at efficiency e=1, finds e* that maximizes
    predictive success: e* = argmax_e [r(e) - a(e)]

    Where:
    - r(e) = 1 if data passes GARP at efficiency e, else 0
    - a(e) = relative area at efficiency e (proportion of random that passes)

    This identifies the efficiency level at which the utility maximization
    hypothesis provides the strongest evidence relative to random behavior.

    Attributes:
        optimal_efficiency: The efficiency level e* maximizing predictive success
        optimal_measure: The predictive success m(e*) at optimal efficiency
        efficiency_levels: Array of efficiency levels tested
        measures: Array of predictive success values at each level
        pass_rates: Array of pass rates (0 or 1) at each level
        relative_areas: Array of relative areas at each level
        aei: Original Afriat Efficiency Index (max e where data passes GARP)
        computation_time_ms: Time taken in milliseconds

    References:
        Beatty, T. K., & Crawford, I. A. (2011). How demanding is the
        revealed preference approach to demand? AER, 101(6), 2782-2795.
    """

    optimal_efficiency: float
    optimal_measure: float
    efficiency_levels: list[float]
    measures: list[float]
    pass_rates: list[float]
    relative_areas: list[float]
    aei: float
    computation_time_ms: float

    @property
    def is_meaningful(self) -> bool:
        """Whether optimal measure exceeds threshold (m > 0.05)."""
        return self.optimal_measure > 0.05

    @property
    def bronars_power_at_optimal(self) -> float:
        """Bronars power at optimal efficiency level."""
        # Find the relative area at optimal efficiency
        for i, e in enumerate(self.efficiency_levels):
            if abs(e - self.optimal_efficiency) < 1e-6:
                return 1.0 - self.relative_areas[i]
        return 0.0

    def score(self) -> float:
        """Return scikit-learn style score. Higher is better."""
        return (self.optimal_measure + 1) / 2  # Normalize from [-1, 1] to [0, 1]

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("OPTIMAL EFFICIENCY REPORT")]

        # Status
        if self.optimal_measure > 0.1:
            status = "MEANINGFUL (m > 0.1)"
        elif self.optimal_measure > 0.05:
            status = "MARGINAL"
        elif self.optimal_measure > 0:
            status = "WEAK POSITIVE"
        elif self.optimal_measure > -0.05:
            status = "UNINFORMATIVE"
        else:
            status = "NEGATIVE"
        lines.append(f"\nStatus: {status}")

        # Metrics
        lines.append(m._format_section("Optimal Values"))
        lines.append(m._format_metric("Optimal Efficiency (e*)", f"{self.optimal_efficiency:.4f}"))
        lines.append(m._format_metric("Optimal Measure (m*)", f"{self.optimal_measure:.4f}"))
        lines.append(m._format_metric("Bronars Power at e*", f"{self.bronars_power_at_optimal:.4f}"))

        lines.append(m._format_section("Reference Values"))
        lines.append(m._format_metric("Afriat Efficiency Index", f"{self.aei:.4f}"))
        lines.append(m._format_metric("Levels Tested", len(self.efficiency_levels)))

        # Find measure at AEI for comparison
        aei_measure = None
        for i, e in enumerate(self.efficiency_levels):
            if abs(e - self.aei) < 0.06:  # Closest level to AEI
                aei_measure = self.measures[i]
                break

        if aei_measure is not None:
            lines.append(m._format_metric("Measure at AEI", f"{aei_measure:.4f}"))

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        lines.append(f"  Optimal efficiency e*={self.optimal_efficiency:.3f} maximizes")
        lines.append(f"  predictive success m={self.optimal_measure:.3f}.")

        if self.optimal_efficiency < self.aei - 0.05:
            lines.append("  Note: e* < AEI suggests relaxing efficiency improves signal.")
        elif self.optimal_efficiency >= 0.99:
            lines.append("  Full efficiency (e=1) provides strongest signal.")

        if self.optimal_measure > 0.1:
            lines.append("  Strong evidence for utility maximization at this e*.")
        elif self.optimal_measure > 0:
            lines.append("  Weak evidence: behavior slightly better than random at e*.")
        else:
            lines.append("  No efficiency level provides meaningful evidence.")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "optimal_efficiency": self.optimal_efficiency,
            "optimal_measure": self.optimal_measure,
            "efficiency_levels": self.efficiency_levels,
            "measures": self.measures,
            "pass_rates": self.pass_rates,
            "relative_areas": self.relative_areas,
            "aei": self.aei,
            "bronars_power_at_optimal": self.bronars_power_at_optimal,
            "is_meaningful": self.is_meaningful,
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        indicator = "[+]" if self.is_meaningful else "[-]"
        return f"OptimalEfficiencyResult: {indicator} e*={self.optimal_efficiency:.3f}, m*={self.optimal_measure:.3f}"

    def short_summary(self) -> str:
        """Return one-liner."""
        indicator = "[+]" if self.is_meaningful else "[-]"
        return f"Optimal: {indicator} e*={self.optimal_efficiency:.3f}, m*={self.optimal_measure:.3f}"
