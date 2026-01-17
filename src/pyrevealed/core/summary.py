"""BehavioralSummary: Unified summary of all behavioral tests.

This module provides a statsmodels-style unified summary for behavioral
analysis results, combining consistency tests and goodness-of-fit metrics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pyrevealed.core.display import ResultDisplayMixin
from pyrevealed.core.mixins import ResultSummaryMixin

if TYPE_CHECKING:
    from pyrevealed.core.session import (
        BehaviorLog,
        MenuChoiceLog,
        RiskChoiceLog,
        StochasticChoiceLog,
        ProductionLog,
    )
    from pyrevealed.core.result import (
        GARPResult,
        AEIResult,
        MPIResult,
        WARPResult,
        SARPResult,
        HoutmanMaksResult,
        RiskProfileResult,
        RUMConsistencyResult,
        RegularityResult,
        StochasticTransitivityResult,
        StochasticChoiceResult,
        ProductionGARPResult,
        OptimalEfficiencyResult,
    )


@dataclass
class BehavioralSummary(ResultDisplayMixin):
    """Unified summary of all behavioral tests (statsmodels-style).

    Provides a comprehensive overview of behavioral consistency analysis,
    combining multiple tests and metrics in a single, professional output.

    Attributes:
        garp_result: GARP consistency test result
        warp_result: WARP consistency test result (optional)
        sarp_result: SARP consistency test result (optional)
        aei_result: Afriat Efficiency Index result
        mpi_result: Money Pump Index result
        houtman_maks_result: Houtman-Maks efficiency result (optional)
        num_observations: Number of observations in the dataset
        num_goods: Number of goods/dimensions
        computation_time_ms: Total computation time in milliseconds

    Example:
        >>> from pyrevealed import BehaviorLog, BehavioralSummary
        >>> log = BehaviorLog(prices, quantities)
        >>> summary = BehavioralSummary.from_log(log)
        >>> print(summary.summary())
    """

    garp_result: GARPResult
    warp_result: WARPResult | None
    sarp_result: SARPResult | None
    aei_result: AEIResult
    mpi_result: MPIResult
    houtman_maks_result: HoutmanMaksResult | None
    optimal_efficiency_result: "OptimalEfficiencyResult | None"
    num_observations: int
    num_goods: int
    computation_time_ms: float

    @property
    def is_consistent(self) -> bool:
        """True if data passes GARP consistency test."""
        return self.garp_result.is_consistent

    @property
    def efficiency_index(self) -> float:
        """Afriat Efficiency Index (AEI) score."""
        return self.aei_result.efficiency_index

    @property
    def mpi_value(self) -> float:
        """Money Pump Index value."""
        return self.mpi_result.mpi_value

    def score(self) -> float:
        """Return aggregate scikit-learn style score in [0, 1].

        Combines AEI and (1 - MPI) with equal weighting.
        """
        aei = self.aei_result.efficiency_index
        mpi = min(1.0, self.mpi_result.mpi_value)
        return (aei + (1.0 - mpi)) / 2.0

    def summary(self) -> str:
        """Return formatted summary table (statsmodels-style).

        Returns a professional text summary including:
        - Data statistics
        - Consistency test results with [+]/[-] indicators
        - Goodness-of-fit metrics
        - Interpretation guidance

        Returns:
            Multi-line formatted string suitable for printing.
        """
        m = ResultSummaryMixin
        width = 60

        lines = []
        lines.append("=" * width)
        lines.append(" " * ((width - 18) // 2) + "BEHAVIORAL SUMMARY")
        lines.append("=" * width)

        # Data section
        lines.append("")
        lines.append("Data:")
        lines.append("-" * 5)
        lines.append(m._format_metric("Observations", self.num_observations, width - 4))
        lines.append(m._format_metric("Goods", self.num_goods, width - 4))

        # Consistency Tests section
        lines.append("")
        lines.append("Consistency Tests:")
        lines.append("-" * 18)

        # GARP
        garp_indicator = "[+]" if self.garp_result.is_consistent else "[-]"
        garp_status = "PASS" if self.garp_result.is_consistent else "FAIL"
        lines.append(f"  GARP {'.' * (width - 18)} {garp_indicator} {garp_status}")

        # WARP (if available)
        if self.warp_result is not None:
            warp_indicator = "[+]" if self.warp_result.is_consistent else "[-]"
            warp_status = "PASS" if self.warp_result.is_consistent else "FAIL"
            lines.append(f"  WARP {'.' * (width - 18)} {warp_indicator} {warp_status}")

        # SARP (if available)
        if self.sarp_result is not None:
            sarp_indicator = "[+]" if self.sarp_result.is_consistent else "[-]"
            sarp_status = "PASS" if self.sarp_result.is_consistent else "FAIL"
            lines.append(f"  SARP {'.' * (width - 18)} {sarp_indicator} {sarp_status}")

        # Goodness-of-Fit section
        lines.append("")
        lines.append("Goodness-of-Fit:")
        lines.append("-" * 16)
        lines.append(m._format_metric("Afriat Efficiency (AEI)", self.aei_result.efficiency_index, width - 4))
        lines.append(m._format_metric("Money Pump Index (MPI)", self.mpi_result.mpi_value, width - 4))

        if self.houtman_maks_result is not None:
            hm_score = 1.0 - self.houtman_maks_result.fraction
            lines.append(m._format_metric("Houtman-Maks Index", hm_score, width - 4))

        # Power Analysis section (optional)
        if self.optimal_efficiency_result is not None:
            lines.append("")
            lines.append("Power Analysis:")
            lines.append("-" * 15)
            power_result = self.optimal_efficiency_result
            # Bronars Power = 1 - relative_area at e=1
            bronars_power = 1.0 - power_result.relative_areas[-1] if power_result.relative_areas else 0.0
            lines.append(m._format_metric("Bronars Power", bronars_power, width - 4))
            lines.append(m._format_metric("Optimal Efficiency (e*)", power_result.optimal_efficiency, width - 4))
            lines.append(m._format_metric("Optimal Measure (m*)", power_result.optimal_measure, width - 4))

        # Interpretation section
        lines.append("")
        lines.append("Interpretation:")
        lines.append("-" * 15)
        lines.append(f"  {m._format_interpretation(self.aei_result.efficiency_index, 'efficiency')}")

        # Footer
        lines.append("")
        if self.computation_time_ms < 1000:
            time_str = f"{self.computation_time_ms:.2f} ms"
        else:
            time_str = f"{self.computation_time_ms / 1000:.2f} s"
        lines.append(f"Computation Time: {time_str}")
        lines.append("=" * width)

        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """Return HTML representation for Jupyter notebook display."""
        from pyrevealed.viz.html_templates import render_behavioral_summary_html

        # Prepare consistency tests
        consistency_tests = [
            ("GARP", self.garp_result.is_consistent),
        ]
        if self.warp_result is not None:
            consistency_tests.append(("WARP", self.warp_result.is_consistent))
        if self.sarp_result is not None:
            consistency_tests.append(("SARP", self.sarp_result.is_consistent))

        # Prepare goodness metrics
        goodness_metrics = [
            ("Afriat Efficiency (AEI)", self.aei_result.efficiency_index),
            ("Money Pump Index (MPI)", self.mpi_result.mpi_value),
        ]
        if self.houtman_maks_result is not None:
            hm_score = 1.0 - self.houtman_maks_result.fraction
            goodness_metrics.append(("Houtman-Maks Index", hm_score))

        return render_behavioral_summary_html(
            num_observations=self.num_observations,
            num_goods=self.num_goods,
            consistency_tests=consistency_tests,
            goodness_metrics=goodness_metrics,
            computation_time_ms=self.computation_time_ms,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        result = {
            "is_consistent": self.is_consistent,
            "efficiency_index": self.efficiency_index,
            "mpi_value": self.mpi_value,
            "num_observations": self.num_observations,
            "num_goods": self.num_goods,
            "score": self.score(),
            "computation_time_ms": self.computation_time_ms,
            "garp": self.garp_result.to_dict(),
            "aei": self.aei_result.to_dict(),
            "mpi": self.mpi_result.to_dict(),
        }
        if self.warp_result is not None:
            result["warp"] = self.warp_result.to_dict()
        if self.sarp_result is not None:
            result["sarp"] = self.sarp_result.to_dict()
        if self.houtman_maks_result is not None:
            result["houtman_maks"] = self.houtman_maks_result.to_dict()
        if self.optimal_efficiency_result is not None:
            power_result = self.optimal_efficiency_result
            bronars_power = 1.0 - power_result.relative_areas[-1] if power_result.relative_areas else 0.0
            result["power_analysis"] = {
                "bronars_power": bronars_power,
                "optimal_efficiency": power_result.optimal_efficiency,
                "optimal_measure": power_result.optimal_measure,
            }
        return result

    def short_summary(self) -> str:
        """Return one-liner summary."""
        indicator = "[+]" if self.is_consistent else "[-]"
        return f"BehavioralSummary: {indicator} AEI={self.efficiency_index:.4f}, MPI={self.mpi_value:.4f}"

    def __repr__(self) -> str:
        """Compact string representation."""
        indicator = "[+]" if self.is_consistent else "[-]"
        return (
            f"BehavioralSummary: {indicator} "
            f"n={self.num_observations}, "
            f"AEI={self.efficiency_index:.4f}, "
            f"MPI={self.mpi_value:.4f}"
        )

    def __str__(self) -> str:
        """Return formatted summary table when printed."""
        return self.summary()

    @classmethod
    def from_log(
        cls,
        log: BehaviorLog,
        include_warp: bool = True,
        include_sarp: bool = True,
        include_power: bool = False,
    ) -> "BehavioralSummary":
        """Create BehavioralSummary by running all tests on a BehaviorLog.

        This factory method runs GARP, AEI, MPI, and optionally WARP, SARP,
        Houtman-Maks, and power analysis tests, combining results into a unified summary.

        Args:
            log: BehaviorLog containing the behavioral data
            include_warp: Whether to include WARP test (default: True)
            include_sarp: Whether to include SARP test (default: True)
            include_power: Whether to include power analysis (default: False)

        Returns:
            BehavioralSummary instance with all test results

        Example:
            >>> summary = BehavioralSummary.from_log(behavior_log)
            >>> print(summary)
            >>> # With power analysis
            >>> summary = BehavioralSummary.from_log(behavior_log, include_power=True)
            >>> print(summary)
        """
        start_time = time.perf_counter()

        # Import algorithms here to avoid circular imports
        from pyrevealed.algorithms.garp import validate_consistency, check_warp
        from pyrevealed.algorithms.differentiable import validate_sarp
        from pyrevealed.algorithms.aei import compute_integrity_score
        from pyrevealed.algorithms.mpi import compute_confusion_metric, compute_houtman_maks_index

        # Run required tests
        garp_result = validate_consistency(log)
        aei_result = compute_integrity_score(log)
        mpi_result = compute_confusion_metric(log)

        # Run optional tests
        warp_result = None
        if include_warp:
            warp_result = check_warp(log)

        sarp_result = None
        if include_sarp:
            sarp_result = validate_sarp(log)

        # Houtman-Maks if there are violations
        houtman_maks_result = None
        if not garp_result.is_consistent:
            houtman_maks_result = compute_houtman_maks_index(log)

        # Power analysis (optional, computationally expensive)
        optimal_efficiency_result = None
        if include_power:
            from pyrevealed.algorithms.power_analysis import compute_optimal_efficiency
            optimal_efficiency_result = compute_optimal_efficiency(
                log, n_simulations=200, n_efficiency_levels=10
            )

        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000

        return cls(
            garp_result=garp_result,
            warp_result=warp_result,
            sarp_result=sarp_result,
            aei_result=aei_result,
            mpi_result=mpi_result,
            houtman_maks_result=houtman_maks_result,
            optimal_efficiency_result=optimal_efficiency_result,
            num_observations=log.num_observations,
            num_goods=log.num_goods,
            computation_time_ms=total_time_ms,
        )


@dataclass
class MenuChoiceSummary(ResultDisplayMixin):
    """Unified summary of menu-based choice analysis.

    Attributes:
        warp_result: WARP consistency result
        sarp_result: SARP consistency result
        congruence_result: Congruence (full rationalizability) result
        efficiency_result: Houtman-Maks efficiency result
        utility_result: Ordinal utility recovery result (optional)
        num_observations: Number of choice observations
        num_alternatives: Number of unique alternatives
        computation_time_ms: Total computation time in milliseconds
    """

    warp_result: Any  # AbstractWARPResult
    sarp_result: Any  # AbstractSARPResult
    congruence_result: Any  # CongruenceResult
    efficiency_result: Any  # HoutmanMaksAbstractResult
    utility_result: Any | None  # OrdinalUtilityResult
    num_observations: int
    num_alternatives: int
    computation_time_ms: float

    @property
    def is_rationalizable(self) -> bool:
        """True if choices are fully rationalizable."""
        return bool(self.congruence_result.is_rationalizable)

    @property
    def efficiency_score(self) -> float:
        """Houtman-Maks efficiency score."""
        return float(self.efficiency_result.efficiency_index)

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]."""
        return self.efficiency_score

    def summary(self) -> str:
        """Return formatted summary table."""
        m = ResultSummaryMixin
        width = 60

        lines = []
        lines.append("=" * width)
        lines.append(" " * ((width - 22) // 2) + "MENU CHOICE SUMMARY")
        lines.append("=" * width)

        # Data section
        lines.append("")
        lines.append("Data:")
        lines.append("-" * 5)
        lines.append(m._format_metric("Observations", self.num_observations, width - 4))
        lines.append(m._format_metric("Alternatives", self.num_alternatives, width - 4))

        # Consistency Tests section
        lines.append("")
        lines.append("Consistency Tests:")
        lines.append("-" * 18)

        warp_indicator = "[+]" if self.warp_result.is_consistent else "[-]"
        warp_status = "PASS" if self.warp_result.is_consistent else "FAIL"
        lines.append(f"  WARP {'.' * (width - 18)} {warp_indicator} {warp_status}")

        sarp_indicator = "[+]" if self.sarp_result.is_consistent else "[-]"
        sarp_status = "PASS" if self.sarp_result.is_consistent else "FAIL"
        lines.append(f"  SARP {'.' * (width - 18)} {sarp_indicator} {sarp_status}")

        cong_indicator = "[+]" if self.congruence_result.is_rationalizable else "[-]"
        cong_status = "PASS" if self.congruence_result.is_rationalizable else "FAIL"
        lines.append(f"  Congruence {'.' * (width - 24)} {cong_indicator} {cong_status}")

        # Goodness-of-Fit section
        lines.append("")
        lines.append("Goodness-of-Fit:")
        lines.append("-" * 16)
        lines.append(m._format_metric("Houtman-Maks Efficiency", self.efficiency_score, width - 4))

        # Utility recovery if available
        if self.utility_result is not None and self.utility_result.success:
            lines.append("")
            lines.append("Preference Order:")
            lines.append("-" * 17)
            if self.utility_result.preference_order:
                order_str = " > ".join(str(i) for i in self.utility_result.preference_order[:8])
                lines.append(f"  {order_str}")
                if len(self.utility_result.preference_order) > 8:
                    lines.append(f"  ... ({len(self.utility_result.preference_order) - 8} more)")

        # Footer
        lines.append("")
        if self.computation_time_ms < 1000:
            time_str = f"{self.computation_time_ms:.2f} ms"
        else:
            time_str = f"{self.computation_time_ms / 1000:.2f} s"
        lines.append(f"Computation Time: {time_str}")
        lines.append("=" * width)

        return "\n".join(lines)

    def __repr__(self) -> str:
        """Compact string representation."""
        indicator = "[+]" if self.is_rationalizable else "[-]"
        return (
            f"MenuChoiceSummary: {indicator} "
            f"n={self.num_observations}, "
            f"efficiency={self.efficiency_score:.4f}"
        )

    def __str__(self) -> str:
        """Return formatted summary table when printed."""
        return self.summary()

    @classmethod
    def from_log(cls, log: MenuChoiceLog) -> "MenuChoiceSummary":
        """Create MenuChoiceSummary by running all tests on a MenuChoiceLog.

        Args:
            log: MenuChoiceLog containing the choice data

        Returns:
            MenuChoiceSummary instance with all test results
        """
        start_time = time.perf_counter()

        from pyrevealed.algorithms.abstract_choice import (
            validate_menu_warp,
            validate_menu_sarp,
            validate_menu_consistency,
            compute_menu_efficiency,
            fit_menu_preferences,
        )

        warp_result = validate_menu_warp(log)
        sarp_result = validate_menu_sarp(log)
        congruence_result = validate_menu_consistency(log)
        efficiency_result = compute_menu_efficiency(log)

        # Try to recover preferences
        utility_result = None
        if sarp_result.is_consistent:
            utility_result = fit_menu_preferences(log)

        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000

        return cls(
            warp_result=warp_result,
            sarp_result=sarp_result,
            congruence_result=congruence_result,
            efficiency_result=efficiency_result,
            utility_result=utility_result,
            num_observations=log.num_observations,
            num_alternatives=log.num_alternatives,
            computation_time_ms=total_time_ms,
        )


@dataclass
class RiskChoiceSummary(ResultDisplayMixin):
    """Unified summary of risk choice analysis.

    Provides a comprehensive overview of risk preferences analysis,
    combining risk profile estimation with Expected Utility axiom tests.

    Attributes:
        risk_profile_result: Result of CRRA risk profile estimation
        eu_axioms_satisfied: Whether Expected Utility axioms hold
        eu_violations: List of EU axiom violations
        num_observations: Number of choice observations
        num_risk_seeking_choices: Choices where risky option with lower EV was chosen
        num_risk_averse_choices: Choices where safe option with lower EV was chosen
        computation_time_ms: Total computation time in milliseconds

    Example:
        >>> from pyrevealed import RiskChoiceLog, RiskChoiceSummary
        >>> log = RiskChoiceLog(safe_values, risky_outcomes, risky_probs, choices)
        >>> summary = RiskChoiceSummary.from_log(log)
        >>> print(summary.summary())
    """

    risk_profile_result: "RiskProfileResult"
    eu_axioms_satisfied: bool
    eu_violations: list[str]
    num_observations: int
    num_risk_seeking_choices: int
    num_risk_averse_choices: int
    computation_time_ms: float

    @property
    def risk_category(self) -> str:
        """Risk category: 'risk_seeking', 'risk_neutral', or 'risk_averse'."""
        return str(self.risk_profile_result.risk_category)

    @property
    def risk_aversion_coefficient(self) -> float:
        """Arrow-Pratt coefficient of relative risk aversion (rho)."""
        return float(self.risk_profile_result.risk_aversion_coefficient)

    @property
    def consistency_score(self) -> float:
        """Fraction of choices consistent with the estimated risk profile."""
        return float(self.risk_profile_result.consistency_score)

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]."""
        return self.consistency_score

    def summary(self) -> str:
        """Return formatted summary table (statsmodels-style)."""
        m = ResultSummaryMixin
        width = 60

        lines = []
        lines.append("=" * width)
        lines.append(" " * ((width - 22) // 2) + "RISK CHOICE SUMMARY")
        lines.append("=" * width)

        # Data section
        lines.append("")
        lines.append("Data:")
        lines.append("-" * 5)
        lines.append(m._format_metric("Observations", self.num_observations, width - 4))
        lines.append(m._format_metric("Risk-Seeking Choices", self.num_risk_seeking_choices, width - 4))
        lines.append(m._format_metric("Risk-Averse Choices", self.num_risk_averse_choices, width - 4))

        # Risk Profile section
        lines.append("")
        lines.append("Risk Profile:")
        lines.append("-" * 13)
        lines.append(m._format_metric("Risk Category", self.risk_category.replace("_", " ").title(), width - 4))
        lines.append(m._format_metric("Risk Aversion (rho)", self.risk_aversion_coefficient, width - 4))
        lines.append(m._format_metric("Consistency Score", self.consistency_score, width - 4))

        # EU Axioms section
        lines.append("")
        lines.append("Expected Utility Axioms:")
        lines.append("-" * 24)
        eu_indicator = "[+]" if self.eu_axioms_satisfied else "[-]"
        eu_status = "SATISFIED" if self.eu_axioms_satisfied else "VIOLATED"
        lines.append(f"  Status {'.' * (width - 22)} {eu_indicator} {eu_status}")

        if not self.eu_axioms_satisfied and self.eu_violations:
            lines.append("")
            lines.append("  Violations:")
            for v in self.eu_violations[:3]:
                lines.append(f"    - {v}")
            if len(self.eu_violations) > 3:
                lines.append(f"    ... and {len(self.eu_violations) - 3} more")

        # Interpretation section
        lines.append("")
        lines.append("Interpretation:")
        lines.append("-" * 15)
        if self.risk_category == "risk_averse":
            lines.append("  Decision-maker prefers certainty over gambles.")
            lines.append(f"  Certainty premium: willing to accept ~{(1-0.5**(1/max(self.risk_aversion_coefficient, 0.1)))*100:.0f}% less for certainty.")
        elif self.risk_category == "risk_seeking":
            lines.append("  Decision-maker prefers gambles over certainty.")
            lines.append("  May accept unfavorable expected value for chance of large gain.")
        else:
            lines.append("  Decision-maker approximately maximizes expected value.")

        # Footer
        lines.append("")
        if self.computation_time_ms < 1000:
            time_str = f"{self.computation_time_ms:.2f} ms"
        else:
            time_str = f"{self.computation_time_ms / 1000:.2f} s"
        lines.append(f"Computation Time: {time_str}")
        lines.append("=" * width)

        return "\n".join(lines)

    def __repr__(self) -> str:
        """Compact string representation."""
        indicator = "[+]" if self.eu_axioms_satisfied else "[-]"
        return (
            f"RiskChoiceSummary: {indicator} "
            f"{self.risk_category}, "
            f"rho={self.risk_aversion_coefficient:.2f}, "
            f"consistency={self.consistency_score:.2f}"
        )

    def __str__(self) -> str:
        """Return formatted summary table when printed."""
        return self.summary()

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "risk_category": self.risk_category,
            "risk_aversion_coefficient": self.risk_aversion_coefficient,
            "consistency_score": self.consistency_score,
            "eu_axioms_satisfied": self.eu_axioms_satisfied,
            "num_eu_violations": len(self.eu_violations),
            "num_observations": self.num_observations,
            "num_risk_seeking_choices": self.num_risk_seeking_choices,
            "num_risk_averse_choices": self.num_risk_averse_choices,
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    @classmethod
    def from_log(cls, log: "RiskChoiceLog") -> "RiskChoiceSummary":
        """Create RiskChoiceSummary by running all tests on a RiskChoiceLog.

        Args:
            log: RiskChoiceLog containing the risk choice data

        Returns:
            RiskChoiceSummary instance with all test results
        """
        start_time = time.perf_counter()

        from pyrevealed.algorithms.risk import (
            compute_risk_profile,
            check_expected_utility_axioms,
        )

        # Run risk profile analysis
        risk_profile_result = compute_risk_profile(log)

        # Check EU axioms
        eu_satisfied, eu_violations = check_expected_utility_axioms(log)

        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000

        return cls(
            risk_profile_result=risk_profile_result,
            eu_axioms_satisfied=eu_satisfied,
            eu_violations=eu_violations,
            num_observations=log.num_observations,
            num_risk_seeking_choices=log.num_risk_seeking_choices,
            num_risk_averse_choices=log.num_risk_averse_choices,
            computation_time_ms=total_time_ms,
        )


@dataclass
class StochasticChoiceSummary(ResultDisplayMixin):
    """Unified summary of stochastic choice analysis.

    Provides a comprehensive overview of probabilistic choice analysis,
    combining RUM consistency tests, regularity tests, and model fitting.

    Attributes:
        rum_result: RUM consistency test result
        regularity_result: Regularity (Luce axiom) test result
        transitivity_result: Stochastic transitivity test result
        iia_satisfied: Whether Independence of Irrelevant Alternatives holds
        model_result: Fitted stochastic choice model (if consistent)
        num_menus: Number of distinct menus
        num_items: Number of unique items
        total_observations: Total number of choice observations
        computation_time_ms: Total computation time in milliseconds

    Example:
        >>> from pyrevealed import StochasticChoiceLog, StochasticChoiceSummary
        >>> log = StochasticChoiceLog(menus, choice_frequencies)
        >>> summary = StochasticChoiceSummary.from_log(log)
        >>> print(summary.summary())
    """

    rum_result: "RUMConsistencyResult"
    regularity_result: "RegularityResult"
    transitivity_result: "StochasticTransitivityResult"
    iia_satisfied: bool
    model_result: "StochasticChoiceResult | None"
    num_menus: int
    num_items: int
    total_observations: int
    computation_time_ms: float

    @property
    def is_rum_consistent(self) -> bool:
        """True if data is consistent with a Random Utility Model."""
        return bool(self.rum_result.is_rum_consistent)

    @property
    def satisfies_regularity(self) -> bool:
        """True if regularity (Luce axiom) is satisfied."""
        return bool(self.regularity_result.satisfies_regularity)

    @property
    def strongest_transitivity(self) -> str:
        """Strongest stochastic transitivity level satisfied (WST/MST/SST/None)."""
        return str(self.transitivity_result.strongest_satisfied)

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]."""
        return float(self.rum_result.score())

    def summary(self) -> str:
        """Return formatted summary table (statsmodels-style)."""
        m = ResultSummaryMixin
        width = 60

        lines = []
        lines.append("=" * width)
        lines.append(" " * ((width - 28) // 2) + "STOCHASTIC CHOICE SUMMARY")
        lines.append("=" * width)

        # Data section
        lines.append("")
        lines.append("Data:")
        lines.append("-" * 5)
        lines.append(m._format_metric("Menus", self.num_menus, width - 4))
        lines.append(m._format_metric("Unique Items", self.num_items, width - 4))
        lines.append(m._format_metric("Total Observations", self.total_observations, width - 4))

        # Consistency Tests section
        lines.append("")
        lines.append("Consistency Tests:")
        lines.append("-" * 18)

        # RUM consistency
        rum_indicator = "[+]" if self.is_rum_consistent else "[-]"
        rum_status = "PASS" if self.is_rum_consistent else "FAIL"
        lines.append(f"  RUM Consistency {'.' * (width - 30)} {rum_indicator} {rum_status}")

        # Regularity
        reg_indicator = "[+]" if self.satisfies_regularity else "[-]"
        reg_status = "PASS" if self.satisfies_regularity else "FAIL"
        lines.append(f"  Regularity (Luce) {'.' * (width - 32)} {reg_indicator} {reg_status}")

        # IIA
        iia_indicator = "[+]" if self.iia_satisfied else "[-]"
        iia_status = "PASS" if self.iia_satisfied else "FAIL"
        lines.append(f"  IIA {'.' * (width - 18)} {iia_indicator} {iia_status}")

        # Stochastic Transitivity section
        lines.append("")
        lines.append("Stochastic Transitivity:")
        lines.append("-" * 24)
        wst_ind = "[+]" if self.transitivity_result.satisfies_wst else "[-]"
        mst_ind = "[+]" if self.transitivity_result.satisfies_mst else "[-]"
        sst_ind = "[+]" if self.transitivity_result.satisfies_sst else "[-]"
        lines.append(f"  Weak (WST) {'.' * (width - 24)} {wst_ind} {'PASS' if self.transitivity_result.satisfies_wst else 'FAIL'}")
        lines.append(f"  Moderate (MST) {'.' * (width - 28)} {mst_ind} {'PASS' if self.transitivity_result.satisfies_mst else 'FAIL'}")
        lines.append(f"  Strong (SST) {'.' * (width - 26)} {sst_ind} {'PASS' if self.transitivity_result.satisfies_sst else 'FAIL'}")

        # Model Fit section (if available)
        if self.model_result is not None:
            lines.append("")
            lines.append("Model Fit:")
            lines.append("-" * 10)
            lines.append(m._format_metric("Model Type", self.model_result.model_type, width - 4))
            lines.append(m._format_metric("Log-Likelihood", self.model_result.log_likelihood, width - 4))
            lines.append(m._format_metric("AIC", self.model_result.aic, width - 4))
            lines.append(m._format_metric("BIC", self.model_result.bic, width - 4))

        # Interpretation section
        lines.append("")
        lines.append("Interpretation:")
        lines.append("-" * 15)
        if self.is_rum_consistent:
            lines.append("  Choices can be rationalized by a random utility model.")
            lines.append(f"  Strongest transitivity: {self.strongest_transitivity}")
        else:
            lines.append("  Choices cannot be explained by any random utility model.")
            lines.append(f"  Distance to nearest RUM: {self.rum_result.distance_to_rum:.4f}")

        # Footer
        lines.append("")
        if self.computation_time_ms < 1000:
            time_str = f"{self.computation_time_ms:.2f} ms"
        else:
            time_str = f"{self.computation_time_ms / 1000:.2f} s"
        lines.append(f"Computation Time: {time_str}")
        lines.append("=" * width)

        return "\n".join(lines)

    def __repr__(self) -> str:
        """Compact string representation."""
        indicator = "[+]" if self.is_rum_consistent else "[-]"
        return (
            f"StochasticChoiceSummary: {indicator} "
            f"menus={self.num_menus}, "
            f"RUM={'consistent' if self.is_rum_consistent else 'inconsistent'}, "
            f"transitivity={self.strongest_transitivity}"
        )

    def __str__(self) -> str:
        """Return formatted summary table when printed."""
        return self.summary()

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        result = {
            "is_rum_consistent": self.is_rum_consistent,
            "satisfies_regularity": self.satisfies_regularity,
            "iia_satisfied": self.iia_satisfied,
            "strongest_transitivity": self.strongest_transitivity,
            "distance_to_rum": self.rum_result.distance_to_rum,
            "num_menus": self.num_menus,
            "num_items": self.num_items,
            "total_observations": self.total_observations,
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }
        if self.model_result is not None:
            result["model_type"] = self.model_result.model_type
            result["log_likelihood"] = self.model_result.log_likelihood
            result["aic"] = self.model_result.aic
            result["bic"] = self.model_result.bic
        return result

    @classmethod
    def from_log(cls, log: "StochasticChoiceLog") -> "StochasticChoiceSummary":
        """Create StochasticChoiceSummary by running all tests on a StochasticChoiceLog.

        Args:
            log: StochasticChoiceLog containing the stochastic choice data

        Returns:
            StochasticChoiceSummary instance with all test results
        """
        start_time = time.perf_counter()

        from pyrevealed.algorithms.stochastic import (
            test_rum_consistency,
            test_regularity,
            test_stochastic_transitivity,
            check_independence_irrelevant_alternatives,
            fit_random_utility_model,
        )

        # Run all tests
        rum_result = test_rum_consistency(log)
        regularity_result = test_regularity(log)
        transitivity_result = test_stochastic_transitivity(log)
        iia_satisfied = check_independence_irrelevant_alternatives(log)

        # Fit model if consistent
        model_result = None
        if rum_result.is_rum_consistent:
            model_result = fit_random_utility_model(log)

        # Calculate total observations
        obs_per_menu = log.total_observations_per_menu or []
        total_observations = sum(obs_per_menu)

        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000

        return cls(
            rum_result=rum_result,
            regularity_result=regularity_result,
            transitivity_result=transitivity_result,
            iia_satisfied=iia_satisfied,
            model_result=model_result,
            num_menus=log.num_menus,
            num_items=log.num_items,
            total_observations=total_observations,
            computation_time_ms=total_time_ms,
        )


@dataclass
class ProductionSummary(ResultDisplayMixin):
    """Unified summary of production/firm behavior analysis.

    Provides a comprehensive overview of production efficiency analysis,
    combining profit maximization tests, cost minimization checks, and
    efficiency metrics.

    Attributes:
        profit_max_result: Profit maximization test result
        cost_min_result: Cost minimization test result
        returns_to_scale: Returns to scale classification
        technical_efficiency: Overall technical efficiency score
        cost_efficiency: Cost efficiency score
        profit_efficiency: Profit efficiency score
        num_observations: Number of production observations
        num_inputs: Number of inputs
        num_outputs: Number of outputs
        computation_time_ms: Total computation time in milliseconds

    Example:
        >>> from pyrevealed import ProductionLog, ProductionSummary
        >>> log = ProductionLog(input_prices, input_quantities, output_prices, output_quantities)
        >>> summary = ProductionSummary.from_log(log)
        >>> print(summary.summary())
    """

    profit_max_result: "ProductionGARPResult"
    cost_min_result: dict[str, Any]
    returns_to_scale: str
    technical_efficiency: float
    cost_efficiency: float
    profit_efficiency: float
    num_observations: int
    num_inputs: int
    num_outputs: int
    computation_time_ms: float

    @property
    def is_profit_maximizing(self) -> bool:
        """True if firm behavior is consistent with profit maximization."""
        return bool(self.profit_max_result.is_profit_maximizing)

    @property
    def is_cost_minimizing(self) -> bool:
        """True if firm behavior is consistent with cost minimization."""
        return bool(self.cost_min_result.get("is_cost_minimizing", False))

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]."""
        return self.profit_efficiency

    def summary(self) -> str:
        """Return formatted summary table (statsmodels-style)."""
        m = ResultSummaryMixin
        width = 60

        lines = []
        lines.append("=" * width)
        lines.append(" " * ((width - 20) // 2) + "PRODUCTION SUMMARY")
        lines.append("=" * width)

        # Data section
        lines.append("")
        lines.append("Data:")
        lines.append("-" * 5)
        lines.append(m._format_metric("Observations", self.num_observations, width - 4))
        lines.append(m._format_metric("Inputs", self.num_inputs, width - 4))
        lines.append(m._format_metric("Outputs", self.num_outputs, width - 4))

        # Consistency Tests section
        lines.append("")
        lines.append("Consistency Tests:")
        lines.append("-" * 18)

        # Profit maximization
        pm_indicator = "[+]" if self.is_profit_maximizing else "[-]"
        pm_status = "PASS" if self.is_profit_maximizing else "FAIL"
        lines.append(f"  Profit Maximization {'.' * (width - 34)} {pm_indicator} {pm_status}")

        # Cost minimization
        cm_indicator = "[+]" if self.is_cost_minimizing else "[-]"
        cm_status = "PASS" if self.is_cost_minimizing else "FAIL"
        lines.append(f"  Cost Minimization {'.' * (width - 32)} {cm_indicator} {cm_status}")

        # Returns to Scale
        lines.append("")
        lines.append(m._format_metric("Returns to Scale", self.returns_to_scale.title(), width - 4))

        # Efficiency Metrics section
        lines.append("")
        lines.append("Efficiency Metrics:")
        lines.append("-" * 19)
        lines.append(m._format_metric("Technical Efficiency", self.technical_efficiency, width - 4))
        lines.append(m._format_metric("Cost Efficiency", self.cost_efficiency, width - 4))
        lines.append(m._format_metric("Profit Efficiency", self.profit_efficiency, width - 4))

        # Per-input efficiency (if available)
        if hasattr(self.profit_max_result, 'input_efficiency_vector'):
            input_eff = self.profit_max_result.input_efficiency_vector
            if len(input_eff) > 0:
                lines.append("")
                lines.append("Per-Input Efficiency:")
                lines.append("-" * 21)
                for i, eff in enumerate(input_eff[:5]):
                    lines.append(m._format_metric(f"Input {i}", eff, width - 4))
                if len(input_eff) > 5:
                    lines.append(f"  ... ({len(input_eff) - 5} more inputs)")

        # Interpretation section
        lines.append("")
        lines.append("Interpretation:")
        lines.append("-" * 15)
        if self.is_profit_maximizing:
            lines.append("  Firm behavior is consistent with profit maximization.")
        else:
            num_violations = self.profit_max_result.num_violations
            lines.append(f"  Found {num_violations} profit maximization violation(s).")

        lines.append(f"  Returns to scale: {self.returns_to_scale}.")
        eff_pct = self.profit_efficiency * 100
        lines.append(f"  Operating at {eff_pct:.1f}% of optimal profit efficiency.")

        # Footer
        lines.append("")
        if self.computation_time_ms < 1000:
            time_str = f"{self.computation_time_ms:.2f} ms"
        else:
            time_str = f"{self.computation_time_ms / 1000:.2f} s"
        lines.append(f"Computation Time: {time_str}")
        lines.append("=" * width)

        return "\n".join(lines)

    def __repr__(self) -> str:
        """Compact string representation."""
        indicator = "[+]" if self.is_profit_maximizing else "[-]"
        return (
            f"ProductionSummary: {indicator} "
            f"n={self.num_observations}, "
            f"RTS={self.returns_to_scale}, "
            f"profit_eff={self.profit_efficiency:.2f}"
        )

    def __str__(self) -> str:
        """Return formatted summary table when printed."""
        return self.summary()

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "is_profit_maximizing": self.is_profit_maximizing,
            "is_cost_minimizing": self.is_cost_minimizing,
            "returns_to_scale": self.returns_to_scale,
            "technical_efficiency": self.technical_efficiency,
            "cost_efficiency": self.cost_efficiency,
            "profit_efficiency": self.profit_efficiency,
            "num_violations": self.profit_max_result.num_violations,
            "num_observations": self.num_observations,
            "num_inputs": self.num_inputs,
            "num_outputs": self.num_outputs,
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    @classmethod
    def from_log(cls, log: "ProductionLog") -> "ProductionSummary":
        """Create ProductionSummary by running all tests on a ProductionLog.

        Args:
            log: ProductionLog containing the production data

        Returns:
            ProductionSummary instance with all test results
        """
        start_time = time.perf_counter()

        from pyrevealed.algorithms.production import (
            test_profit_maximization,
            check_cost_minimization,
        )

        # Run all tests
        profit_max_result = test_profit_maximization(log)
        cost_min_result = check_cost_minimization(log)

        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000

        return cls(
            profit_max_result=profit_max_result,
            cost_min_result=cost_min_result,
            returns_to_scale=profit_max_result.returns_to_scale,
            technical_efficiency=profit_max_result.technical_efficiency,
            cost_efficiency=profit_max_result.cost_efficiency_score,
            profit_efficiency=profit_max_result.profit_efficiency,
            num_observations=log.num_observations,
            num_inputs=log.num_inputs,
            num_outputs=log.num_outputs,
            computation_time_ms=total_time_ms,
        )
