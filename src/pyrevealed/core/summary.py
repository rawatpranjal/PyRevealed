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
    from pyrevealed.core.session import BehaviorLog, MenuChoiceLog
    from pyrevealed.core.result import (
        GARPResult,
        AEIResult,
        MPIResult,
        WARPResult,
        SARPResult,
        HoutmanMaksResult,
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

    @classmethod
    def from_log(cls, log: BehaviorLog, include_warp: bool = True, include_sarp: bool = True) -> "BehavioralSummary":
        """Create BehavioralSummary by running all tests on a BehaviorLog.

        This factory method runs GARP, AEI, MPI, and optionally WARP, SARP,
        and Houtman-Maks tests, combining results into a unified summary.

        Args:
            log: BehaviorLog containing the behavioral data
            include_warp: Whether to include WARP test (default: True)
            include_sarp: Whether to include SARP test (default: True)

        Returns:
            BehavioralSummary instance with all test results

        Example:
            >>> summary = BehavioralSummary.from_log(behavior_log)
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

        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000

        return cls(
            garp_result=garp_result,
            warp_result=warp_result,
            sarp_result=sarp_result,
            aei_result=aei_result,
            mpi_result=mpi_result,
            houtman_maks_result=houtman_maks_result,
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
        return self.congruence_result.is_rationalizable

    @property
    def efficiency_score(self) -> float:
        """Houtman-Maks efficiency score."""
        return self.efficiency_result.efficiency_index

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
