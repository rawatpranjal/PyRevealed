"""BehavioralSummary: Unified summary of all behavioral tests.

This module provides a statsmodels-style unified summary for behavioral
analysis results, combining consistency tests and goodness-of-fit metrics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from prefgraph.core.display import ResultDisplayMixin
from prefgraph.core.mixins import ResultSummaryMixin

if TYPE_CHECKING:
    from prefgraph.core.session import (
        BehaviorLog,
        MenuChoiceLog,
        RiskChoiceLog,
        StochasticChoiceLog,
        ProductionLog,
    )
    from prefgraph.core.result import (
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
        >>> from prefgraph import BehaviorLog, BehavioralSummary
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
    # Rich stats fields (added for statsmodels-style output)
    price_stats: dict[str, float] | None = field(default=None, repr=False)
    quantity_stats: dict[str, float] | None = field(default=None, repr=False)
    expenditure_stats: dict[str, float] | None = field(default=None, repr=False)
    r_density: float | None = field(default=None, repr=False)
    p_density: float | None = field(default=None, repr=False)
    r_star_density: float | None = field(default=None, repr=False)
    violation_pair_count: int | None = field(default=None, repr=False)
    user_id: str | None = field(default=None, repr=False)

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
        - Two-column header with key results
        - Input data statistics (prices, quantities, expenditure)
        - Revealed preference graph density
        - Consistency test results with [+]/[-] indicators
        - Goodness-of-fit metrics with sub-details
        - Interpretation guidance

        Returns:
            Multi-line formatted string suitable for printing.
        """
        m = ResultSummaryMixin
        W = 70
        sep = "-" * W

        def _indicator(passed: bool) -> str:
            return f"[+] {'PASS' if passed else 'FAIL'}" if passed else f"[-] FAIL"

        def _time_str(ms: float) -> str:
            return f"{ms:.2f} ms" if ms < 1000 else f"{ms / 1000:.2f} s"

        lines: list[str] = []

        # === Two-column header ===
        lines.append("=" * W)
        lines.append(" " * ((W - 18) // 2) + "BEHAVIORAL SUMMARY")
        lines.append("=" * W)

        uid = self.user_id or "N/A"
        garp_str = "[+] PASS" if self.garp_result.is_consistent else "[-] FAIL"
        warp_str = "[+] PASS" if (self.warp_result and self.warp_result.is_consistent) else (
            "[-] FAIL" if self.warp_result else "N/A"
        )
        sarp_str = "[+] PASS" if (self.sarp_result and self.sarp_result.is_consistent) else (
            "[-] FAIL" if self.sarp_result else "N/A"
        )

        lines.append(m._format_two_column_row("User ID", uid, "GARP", garp_str, W))
        lines.append(m._format_two_column_row("No. Observations", self.num_observations, "WARP", warp_str, W))
        lines.append(m._format_two_column_row("No. Goods", self.num_goods, "SARP", sarp_str, W))
        lines.append(m._format_two_column_row(
            "Method", "Floyd-Warshall",
            "AEI", f"{self.aei_result.efficiency_index:.4f}", W,
        ))
        lines.append(m._format_two_column_row(
            "Computation Time", _time_str(self.computation_time_ms),
            "MPI", f"{self.mpi_result.mpi_value:.4f}", W,
        ))
        lines.append("=" * W)

        # === Input Data Statistics ===
        if self.price_stats and self.quantity_stats and self.expenditure_stats:
            lines.append("")
            lines.append("Input Data:")
            lines.append(sep)
            lines.append(m._format_descriptive_table({
                "Prices": self.price_stats,
                "Quantities": self.quantity_stats,
                "Expenditure": self.expenditure_stats,
            }, W))

        # === Revealed Preference Graph ===
        if self.r_density is not None:
            T = self.num_observations
            T2 = T * T
            lines.append("")
            lines.append("Revealed Preference Graph:")
            lines.append(sep)
            r_edges = int(round(self.r_density * T2))
            p_edges = int(round(self.p_density * T2)) if self.p_density is not None else 0
            rs_edges = int(round(self.r_star_density * T2)) if self.r_star_density is not None else 0
            lines.append(m._format_matrix_density("R  (direct, p'x >= p'y)", r_edges, T2, W))
            lines.append(m._format_matrix_density("P  (strict, p'x >  p'y)", p_edges, T2, W))
            lines.append(m._format_matrix_density("R* (transitive closure)", rs_edges, T2, W))
            vp = self.violation_pair_count if self.violation_pair_count is not None else 0
            lines.append(m._format_metric("Violation pairs (R* & P')", vp, W - 4))

        # === Consistency Tests ===
        lines.append("")
        lines.append("Consistency Tests:")
        lines.append(sep)
        n_garp = self.garp_result.num_violations
        garp_detail = f" ({n_garp} cycle{'s' if n_garp != 1 else ''})" if n_garp > 0 else ""
        lines.append(m._format_metric("GARP", f"{_indicator(self.garp_result.is_consistent)}{garp_detail}", W - 4))

        if self.warp_result is not None:
            n_warp = self.warp_result.num_violations
            warp_detail = f" ({n_warp} violation{'s' if n_warp != 1 else ''})" if n_warp > 0 else ""
            lines.append(m._format_metric("WARP", f"{_indicator(self.warp_result.is_consistent)}{warp_detail}", W - 4))

        if self.sarp_result is not None:
            n_sarp = self.sarp_result.num_violations if hasattr(self.sarp_result, 'num_violations') else 0
            sarp_detail = f" ({n_sarp} cycle{'s' if n_sarp != 1 else ''})" if n_sarp > 0 else ""
            lines.append(m._format_metric("SARP", f"{_indicator(self.sarp_result.is_consistent)}{sarp_detail}", W - 4))

        # === Goodness-of-Fit ===
        lines.append("")
        lines.append("Goodness-of-Fit:")
        lines.append(sep)

        # AEI with sub-metrics
        aei = self.aei_result
        lines.append(m._format_metric("Afriat Efficiency (AEI)", aei.efficiency_index, W - 4))
        lines.append(m._format_metric("  Binary search iterations", aei.binary_search_iterations, W - 4))
        waste = (1.0 - aei.efficiency_index) * 100
        lines.append(m._format_metric("  Budget waste", f"{waste:.2f}%", W - 4))

        # MPI with sub-metrics
        mpi = self.mpi_result
        lines.append(m._format_metric("Money Pump Index (MPI)", mpi.mpi_value, W - 4))
        lines.append(m._format_metric("  Violation cycles", mpi.num_cycles, W - 4))
        if mpi.worst_cycle is not None:
            worst_cost = max(c for _, c in mpi.cycle_costs) if mpi.cycle_costs else mpi.mpi_value
            lines.append(m._format_metric("  Worst cycle cost", f"{worst_cost:.4f}", W - 4))
        lines.append(m._format_metric("  Total expenditure", f"${mpi.total_expenditure:,.2f}", W - 4))

        # Houtman-Maks
        if self.houtman_maks_result is not None:
            hm = self.houtman_maks_result
            hm_score = 1.0 - hm.fraction
            lines.append(m._format_metric("Houtman-Maks Index", hm_score, W - 4))
            lines.append(m._format_metric(
                "  Observations removed",
                f"{hm.num_removed} / {self.num_observations}", W - 4,
            ))

        # === Power Analysis ===
        if self.optimal_efficiency_result is not None:
            lines.append("")
            lines.append("Power Analysis:")
            lines.append(sep)
            pr = self.optimal_efficiency_result
            bronars = 1.0 - pr.relative_areas[-1] if pr.relative_areas else 0.0
            lines.append(m._format_metric("Bronars Power", bronars, W - 4))
            lines.append(m._format_metric("Optimal Efficiency (e*)", pr.optimal_efficiency, W - 4))
            lines.append(m._format_metric("Optimal Measure (m*)", pr.optimal_measure, W - 4))

        # === Interpretation ===
        lines.append("")
        lines.append("Interpretation:")
        lines.append(sep)
        lines.append(f"  {m._format_interpretation(aei.efficiency_index, 'efficiency')}")
        if not self.garp_result.is_consistent:
            lines.append(f"  ~{waste:.1f}% budget waste; an arbitrager could extract ~{mpi.mpi_value * 100:.1f}%.")
            if self.houtman_maks_result is not None:
                hm = self.houtman_maks_result
                pct = 100.0 * hm.num_removed / self.num_observations if self.num_observations > 0 else 0
                lines.append(f"  {hm.num_removed} observations ({pct:.1f}%) must be removed for full consistency.")

        # === Footer ===
        lines.append("=" * W)

        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """Return HTML representation for Jupyter notebook display."""
        from prefgraph.viz.html_templates import render_behavioral_summary_html

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
        from prefgraph.algorithms.garp import validate_consistency, check_warp
        from prefgraph.algorithms.differentiable import validate_sarp
        from prefgraph.algorithms.aei import compute_integrity_score
        from prefgraph.algorithms.mpi import compute_confusion_metric, compute_houtman_maks_index

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

        # Houtman-Maks: always compute, even for consistent data.
        # Heufer & Hjertstrand (2015), Section 2 (references/papers/md/
        # HeuferHjertstrand2015_ConsistentSubsets.md):
        #   "The HM-index is the maximal fraction of non-zero elements
        #    in the binary vector v such that GARP(v) holds."
        # For consistent data, compute_houtman_maks_index fast-exits with
        # fraction=0.0, removed=[] — trivially correct. Never leave as None
        # because callers (summary display, user code) expect this field
        # to always be populated.
        houtman_maks_result = compute_houtman_maks_index(log)

        # Power analysis (optional, computationally expensive)
        optimal_efficiency_result = None
        if include_power:
            from prefgraph.algorithms.power_analysis import compute_optimal_efficiency
            optimal_efficiency_result = compute_optimal_efficiency(
                log, n_simulations=200, n_efficiency_levels=10
            )

        # Compute rich stats from log and garp_result
        def _array_stats(arr: "np.ndarray") -> dict[str, float]:
            return {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }

        price_stats = _array_stats(log.cost_vectors)
        quantity_stats = _array_stats(log.action_vectors)
        expenditure_stats = _array_stats(log.total_spend)

        T = log.num_observations
        T2 = T * T if T > 0 else 1
        R = garp_result.direct_revealed_preference
        P = garp_result.strict_revealed_preference
        Rstar = garp_result.transitive_closure
        r_density = float(np.sum(R)) / T2
        p_density = float(np.sum(P)) / T2
        r_star_density = float(np.sum(Rstar)) / T2
        violation_pair_count = int(np.sum(Rstar & P.T))

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
            price_stats=price_stats,
            quantity_stats=quantity_stats,
            expenditure_stats=expenditure_stats,
            r_density=r_density,
            p_density=p_density,
            r_star_density=r_star_density,
            violation_pair_count=violation_pair_count,
            user_id=log.user_id,
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
    menu_size_stats: dict[str, float] | None = field(default=None, repr=False)
    choice_diversity: float | None = field(default=None, repr=False)

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
        """Return formatted summary table (statsmodels-style)."""
        m = ResultSummaryMixin
        W = 70
        sep = "-" * W

        def _ind(passed: bool) -> str:
            return "[+] PASS" if passed else "[-] FAIL"

        def _time_str(ms: float) -> str:
            return f"{ms:.2f} ms" if ms < 1000 else f"{ms / 1000:.2f} s"

        lines: list[str] = []

        # Two-column header
        lines.append("=" * W)
        lines.append(" " * ((W - 19) // 2) + "MENU CHOICE SUMMARY")
        lines.append("=" * W)

        lines.append(m._format_two_column_row(
            "No. Observations", self.num_observations,
            "WARP", _ind(self.warp_result.is_consistent), W,
        ))
        lines.append(m._format_two_column_row(
            "No. Alternatives", self.num_alternatives,
            "SARP", _ind(self.sarp_result.is_consistent), W,
        ))
        lines.append(m._format_two_column_row(
            "Computation Time", _time_str(self.computation_time_ms),
            "Congruence", _ind(self.congruence_result.is_rationalizable), W,
        ))
        lines.append("=" * W)

        # Input Data
        if self.menu_size_stats is not None:
            lines.append("")
            lines.append("Input Data:")
            lines.append(sep)
            lines.append(m._format_descriptive_table({"Menu Size": self.menu_size_stats}, W))
            lines.append("")
            if self.choice_diversity is not None:
                items_chosen = int(round(self.choice_diversity * self.num_alternatives))
                lines.append(m._format_metric(
                    "Unique Items Chosen", f"{items_chosen} / {self.num_alternatives}", W - 4,
                ))
                lines.append(m._format_metric("Choice Diversity", self.choice_diversity, W - 4))

        # Consistency Tests with violation counts
        lines.append("")
        lines.append("Consistency Tests:")
        lines.append(sep)
        n_warp = self.warp_result.num_violations if hasattr(self.warp_result, 'num_violations') else 0
        warp_detail = f" ({n_warp} violation{'s' if n_warp != 1 else ''})" if n_warp > 0 else ""
        lines.append(m._format_metric("WARP", f"{_ind(self.warp_result.is_consistent)}{warp_detail}", W - 4))

        n_sarp = self.sarp_result.num_violations if hasattr(self.sarp_result, 'num_violations') else 0
        sarp_detail = f" ({n_sarp} cycle{'s' if n_sarp != 1 else ''})" if n_sarp > 0 else ""
        lines.append(m._format_metric("SARP", f"{_ind(self.sarp_result.is_consistent)}{sarp_detail}", W - 4))
        lines.append(m._format_metric("Congruence", _ind(self.congruence_result.is_rationalizable), W - 4))

        # Goodness-of-Fit
        lines.append("")
        lines.append("Goodness-of-Fit:")
        lines.append(sep)
        lines.append(m._format_metric("Houtman-Maks Efficiency", self.efficiency_score, W - 4))
        if hasattr(self.efficiency_result, 'removed_observations'):
            n_removed = len(self.efficiency_result.removed_observations)
            lines.append(m._format_metric(
                "  Observations removed", f"{n_removed} / {self.num_observations}", W - 4,
            ))

        # Preference Order
        if self.utility_result is not None and self.utility_result.success:
            lines.append("")
            lines.append("Recovered Preference Order:")
            lines.append(sep)
            if self.utility_result.preference_order:
                order_str = " > ".join(str(i) for i in self.utility_result.preference_order[:10])
                lines.append(f"  {order_str}")
                if len(self.utility_result.preference_order) > 10:
                    lines.append(f"  ... ({len(self.utility_result.preference_order) - 10} more)")

        # Interpretation
        lines.append("")
        lines.append("Interpretation:")
        lines.append(sep)
        if self.congruence_result.is_rationalizable:
            lines.append("  Choices are fully rationalizable by a complete preference ordering.")
        elif self.sarp_result.is_consistent:
            lines.append("  Choices satisfy SARP but not Congruence (violates maximality).")
        elif self.warp_result.is_consistent:
            lines.append("  Choices satisfy WARP but not SARP (long preference cycles exist).")
        else:
            lines.append("  Choices violate WARP - direct preference reversals found.")
        lines.append(f"  Efficiency: {self.efficiency_score * 100:.1f}% of observations are consistent.")

        lines.append("=" * W)

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

        from prefgraph.algorithms.abstract_choice import (
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

        # Input data stats
        menu_sizes = np.array([len(m) for m in log.menus], dtype=np.float64)
        menu_size_stats = {
            "mean": float(np.mean(menu_sizes)), "std": float(np.std(menu_sizes)),
            "min": float(np.min(menu_sizes)), "max": float(np.max(menu_sizes)),
        }
        items_chosen = len(set(log.choices))
        n_alt = log.num_alternatives if log.num_alternatives > 0 else 1
        choice_diversity = items_chosen / n_alt

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
            menu_size_stats=menu_size_stats,
            choice_diversity=choice_diversity,
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
        >>> from prefgraph import RiskChoiceLog, RiskChoiceSummary
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
    safe_value_stats: dict[str, float] | None = field(default=None, repr=False)
    risky_ev_stats: dict[str, float] | None = field(default=None, repr=False)

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
        W = 70
        sep = "-" * W

        def _ind(passed: bool) -> str:
            return "[+] PASS" if passed else "[-] FAIL"

        def _time_str(ms: float) -> str:
            return f"{ms:.2f} ms" if ms < 1000 else f"{ms / 1000:.2f} s"

        lines: list[str] = []

        # Two-column header
        lines.append("=" * W)
        lines.append(" " * ((W - 19) // 2) + "RISK CHOICE SUMMARY")
        lines.append("=" * W)

        cat = self.risk_category.replace("_", " ").title()
        lines.append(m._format_two_column_row(
            "No. Observations", self.num_observations,
            "Risk Category", cat, W,
        ))
        lines.append(m._format_two_column_row(
            "Risk-Seeking Choices", self.num_risk_seeking_choices,
            "Risk Aversion (rho)", f"{self.risk_aversion_coefficient:.4f}", W,
        ))
        lines.append(m._format_two_column_row(
            "Risk-Averse Choices", self.num_risk_averse_choices,
            "Consistency", f"{self.consistency_score:.4f}", W,
        ))
        lines.append(m._format_two_column_row(
            "Computation Time", _time_str(self.computation_time_ms),
            "EU Axioms", _ind(self.eu_axioms_satisfied), W,
        ))
        lines.append("=" * W)

        # Input Data
        if self.safe_value_stats is not None and self.risky_ev_stats is not None:
            lines.append("")
            lines.append("Input Data:")
            lines.append(sep)
            lines.append(m._format_descriptive_table({
                "Safe Values": self.safe_value_stats,
                "Risky EV": self.risky_ev_stats,
            }, W))

        # Choice Distribution
        lines.append("")
        lines.append("Choice Distribution:")
        lines.append(sep)
        total = self.num_observations
        if total > 0:
            seek_pct = 100.0 * self.num_risk_seeking_choices / total
            averse_pct = 100.0 * self.num_risk_averse_choices / total
            neutral = total - self.num_risk_seeking_choices - self.num_risk_averse_choices
            neutral_pct = 100.0 * neutral / total
            lines.append(m._format_metric("Risk-Seeking", f"{self.num_risk_seeking_choices} ({seek_pct:.1f}%)", W - 4))
            lines.append(m._format_metric("Risk-Averse", f"{self.num_risk_averse_choices} ({averse_pct:.1f}%)", W - 4))
            lines.append(m._format_metric("Risk-Neutral", f"{neutral} ({neutral_pct:.1f}%)", W - 4))

        # Risk Profile
        lines.append("")
        lines.append("Risk Profile (CRRA):")
        lines.append(sep)
        lines.append(m._format_metric("Risk Category", cat, W - 4))
        lines.append(m._format_metric("Risk Aversion (rho)", self.risk_aversion_coefficient, W - 4))
        lines.append(m._format_metric("Consistency Score", self.consistency_score, W - 4))

        # EU Axioms
        lines.append("")
        lines.append("Expected Utility Axioms:")
        lines.append(sep)
        eu_str = "[+] SATISFIED" if self.eu_axioms_satisfied else "[-] VIOLATED"
        lines.append(m._format_metric("Status", eu_str, W - 4))
        if not self.eu_axioms_satisfied and self.eu_violations:
            lines.append(m._format_metric("  Num. violations", len(self.eu_violations), W - 4))
            for v in self.eu_violations[:3]:
                lines.append(f"    - {v}")
            if len(self.eu_violations) > 3:
                lines.append(f"    ... and {len(self.eu_violations) - 3} more")

        # Interpretation
        lines.append("")
        lines.append("Interpretation:")
        lines.append(sep)
        if self.risk_category == "risk_averse":
            lines.append("  Decision-maker prefers certainty over gambles.")
            rho = max(self.risk_aversion_coefficient, 0.1)
            lines.append(f"  Certainty premium: ~{(1 - 0.5 ** (1 / rho)) * 100:.0f}% less for certainty.")
        elif self.risk_category == "risk_seeking":
            lines.append("  Decision-maker prefers gambles over certainty.")
        else:
            lines.append("  Decision-maker approximately maximizes expected value.")
        lines.append(f"  Model fit: {self.consistency_score * 100:.1f}% of choices consistent with CRRA profile.")

        lines.append("=" * W)
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

        from prefgraph.algorithms.risk import (
            compute_risk_profile,
            check_expected_utility_axioms,
        )

        # Run risk profile analysis
        risk_profile_result = compute_risk_profile(log)

        # Check EU axioms
        eu_satisfied, eu_violations = check_expected_utility_axioms(log)

        # Input data stats
        def _arr_stats(arr: np.ndarray) -> dict[str, float]:
            return {"mean": float(np.mean(arr)), "std": float(np.std(arr)),
                    "min": float(np.min(arr)), "max": float(np.max(arr))}

        safe_value_stats = _arr_stats(log.safe_values)
        risky_evs = np.sum(log.risky_outcomes * log.risky_probabilities, axis=1)
        risky_ev_stats = _arr_stats(risky_evs)

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
            safe_value_stats=safe_value_stats,
            risky_ev_stats=risky_ev_stats,
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
        >>> from prefgraph import StochasticChoiceLog, StochasticChoiceSummary
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
    menu_size_stats: dict[str, float] | None = field(default=None, repr=False)
    obs_per_menu_stats: dict[str, float] | None = field(default=None, repr=False)
    mean_choice_entropy: float | None = field(default=None, repr=False)

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
        W = 70
        sep = "-" * W

        def _ind(passed: bool) -> str:
            return "[+] PASS" if passed else "[-] FAIL"

        def _time_str(ms: float) -> str:
            return f"{ms:.2f} ms" if ms < 1000 else f"{ms / 1000:.2f} s"

        lines: list[str] = []

        # Two-column header
        lines.append("=" * W)
        lines.append(" " * ((W - 25) // 2) + "STOCHASTIC CHOICE SUMMARY")
        lines.append("=" * W)

        lines.append(m._format_two_column_row(
            "No. Menus", self.num_menus,
            "RUM Consistency", _ind(self.is_rum_consistent), W,
        ))
        lines.append(m._format_two_column_row(
            "Unique Items", self.num_items,
            "Regularity", _ind(self.satisfies_regularity), W,
        ))
        lines.append(m._format_two_column_row(
            "Total Observations", self.total_observations,
            "IIA", _ind(self.iia_satisfied), W,
        ))
        lines.append(m._format_two_column_row(
            "Computation Time", _time_str(self.computation_time_ms),
            "Transitivity", self.strongest_transitivity, W,
        ))
        lines.append("=" * W)

        # Input Data
        if self.menu_size_stats is not None:
            lines.append("")
            lines.append("Input Data:")
            lines.append(sep)
            stats_rows: dict[str, dict[str, float]] = {"Menu Size": self.menu_size_stats}
            if self.obs_per_menu_stats is not None:
                stats_rows["Obs per Menu"] = self.obs_per_menu_stats
            lines.append(m._format_descriptive_table(stats_rows, W))
            if self.mean_choice_entropy is not None:
                lines.append("")
                lines.append(m._format_metric("Mean Choice Entropy", f"{self.mean_choice_entropy:.4f}", W - 4))

        # Consistency Tests
        lines.append("")
        lines.append("Consistency Tests:")
        lines.append(sep)
        lines.append(m._format_metric("RUM Consistency", _ind(self.is_rum_consistent), W - 4))
        if hasattr(self.rum_result, 'distance_to_rum'):
            lines.append(m._format_metric("  Distance to nearest RUM", self.rum_result.distance_to_rum, W - 4))
        lines.append(m._format_metric("Regularity (Luce)", _ind(self.satisfies_regularity), W - 4))
        if hasattr(self.regularity_result, 'num_violations'):
            n_reg = self.regularity_result.num_violations
            if n_reg > 0:
                lines.append(m._format_metric("  Regularity violations", n_reg, W - 4))
        lines.append(m._format_metric("IIA", _ind(self.iia_satisfied), W - 4))

        # Stochastic Transitivity
        lines.append("")
        lines.append("Stochastic Transitivity:")
        lines.append(sep)
        tr = self.transitivity_result
        lines.append(m._format_metric("Weak (WST)", _ind(tr.satisfies_wst), W - 4))
        lines.append(m._format_metric("Moderate (MST)", _ind(tr.satisfies_mst), W - 4))
        lines.append(m._format_metric("Strong (SST)", _ind(tr.satisfies_sst), W - 4))
        if hasattr(tr, 'num_triples_tested') and tr.num_triples_tested:
            lines.append(m._format_metric("  Triples tested", tr.num_triples_tested, W - 4))

        # Model Fit
        if self.model_result is not None:
            lines.append("")
            lines.append("Model Fit:")
            lines.append(sep)
            lines.append(m._format_metric("Model Type", self.model_result.model_type, W - 4))
            lines.append(m._format_metric("Log-Likelihood", self.model_result.log_likelihood, W - 4))
            lines.append(m._format_metric("AIC", self.model_result.aic, W - 4))
            lines.append(m._format_metric("BIC", self.model_result.bic, W - 4))

        # Interpretation
        lines.append("")
        lines.append("Interpretation:")
        lines.append(sep)
        if self.is_rum_consistent:
            lines.append("  Choices can be rationalized by a random utility model.")
            lines.append(f"  Strongest transitivity satisfied: {self.strongest_transitivity}")
        else:
            lines.append("  Choices cannot be explained by any random utility model.")
            if hasattr(self.rum_result, 'distance_to_rum'):
                lines.append(f"  Distance to nearest RUM: {self.rum_result.distance_to_rum:.4f}")

        lines.append("=" * W)
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

        from prefgraph.algorithms.stochastic import (
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

        # Input data stats
        menu_sizes = np.array([len(m) for m in log.menus], dtype=np.float64)
        menu_size_stats = {
            "mean": float(np.mean(menu_sizes)), "std": float(np.std(menu_sizes)),
            "min": float(np.min(menu_sizes)), "max": float(np.max(menu_sizes)),
        }
        obs_arr = np.array(obs_per_menu, dtype=np.float64) if obs_per_menu else np.array([0.0])
        obs_per_menu_stats = {
            "mean": float(np.mean(obs_arr)), "std": float(np.std(obs_arr)),
            "min": float(np.min(obs_arr)), "max": float(np.max(obs_arr)),
        }
        entropies = []
        for freq in log.choice_frequencies:
            total = sum(freq.values())
            if total > 0:
                probs = [c / total for c in freq.values() if c > 0]
                entropies.append(-sum(p * np.log2(p) for p in probs))
        mean_choice_entropy = float(np.mean(entropies)) if entropies else 0.0

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
            menu_size_stats=menu_size_stats,
            obs_per_menu_stats=obs_per_menu_stats,
            mean_choice_entropy=mean_choice_entropy,
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
        >>> from prefgraph import ProductionLog, ProductionSummary
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
    input_price_stats: dict[str, float] | None = field(default=None, repr=False)
    output_price_stats: dict[str, float] | None = field(default=None, repr=False)
    profit_stats: dict[str, float] | None = field(default=None, repr=False)

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
        W = 70
        sep = "-" * W

        def _ind(passed: bool) -> str:
            return "[+] PASS" if passed else "[-] FAIL"

        def _time_str(ms: float) -> str:
            return f"{ms:.2f} ms" if ms < 1000 else f"{ms / 1000:.2f} s"

        lines: list[str] = []

        # Two-column header
        lines.append("=" * W)
        lines.append(" " * ((W - 18) // 2) + "PRODUCTION SUMMARY")
        lines.append("=" * W)

        lines.append(m._format_two_column_row(
            "No. Observations", self.num_observations,
            "Profit Max", _ind(self.is_profit_maximizing), W,
        ))
        lines.append(m._format_two_column_row(
            "No. Inputs", self.num_inputs,
            "Cost Min", _ind(self.is_cost_minimizing), W,
        ))
        lines.append(m._format_two_column_row(
            "No. Outputs", self.num_outputs,
            "Returns to Scale", self.returns_to_scale.title(), W,
        ))
        lines.append(m._format_two_column_row(
            "Computation Time", _time_str(self.computation_time_ms),
            "Profit Efficiency", f"{self.profit_efficiency:.4f}", W,
        ))
        lines.append("=" * W)

        # Input Data
        if self.input_price_stats is not None:
            lines.append("")
            lines.append("Input Data:")
            lines.append(sep)
            stats_rows: dict[str, dict[str, float]] = {
                "Input Prices": self.input_price_stats,
            }
            if self.output_price_stats is not None:
                stats_rows["Output Prices"] = self.output_price_stats
            if self.profit_stats is not None:
                stats_rows["Profit"] = self.profit_stats
            lines.append(m._format_descriptive_table(stats_rows, W))

        # Consistency Tests
        lines.append("")
        lines.append("Consistency Tests:")
        lines.append(sep)
        pm_detail = ""
        if not self.is_profit_maximizing:
            pm_detail = f" ({self.profit_max_result.num_violations} violations)"
        lines.append(m._format_metric("Profit Maximization", f"{_ind(self.is_profit_maximizing)}{pm_detail}", W - 4))
        lines.append(m._format_metric("Cost Minimization", _ind(self.is_cost_minimizing), W - 4))
        lines.append(m._format_metric("Returns to Scale", self.returns_to_scale.title(), W - 4))

        # Efficiency Metrics
        lines.append("")
        lines.append("Efficiency Metrics:")
        lines.append(sep)
        lines.append(m._format_metric("Technical Efficiency", self.technical_efficiency, W - 4))
        lines.append(m._format_metric("Cost Efficiency", self.cost_efficiency, W - 4))
        lines.append(m._format_metric("Profit Efficiency", self.profit_efficiency, W - 4))

        # Per-input efficiency
        if hasattr(self.profit_max_result, 'input_efficiency_vector'):
            input_eff = self.profit_max_result.input_efficiency_vector
            if len(input_eff) > 0:
                lines.append("")
                lines.append("Per-Input Efficiency:")
                lines.append(sep)
                for i, eff in enumerate(input_eff[:5]):
                    lines.append(m._format_metric(f"Input {i}", eff, W - 4))
                if len(input_eff) > 5:
                    lines.append(f"  ... ({len(input_eff) - 5} more inputs)")

        # Interpretation
        lines.append("")
        lines.append("Interpretation:")
        lines.append(sep)
        if self.is_profit_maximizing:
            lines.append("  Firm behavior is consistent with profit maximization.")
        else:
            lines.append(f"  Found {self.profit_max_result.num_violations} profit maximization violation(s).")
        lines.append(f"  Returns to scale: {self.returns_to_scale}.")
        lines.append(f"  Operating at {self.profit_efficiency * 100:.1f}% of optimal profit efficiency.")

        lines.append("=" * W)
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

        from prefgraph.algorithms.production import (
            test_profit_maximization,
            check_cost_minimization,
        )

        # Run all tests
        profit_max_result = test_profit_maximization(log)
        cost_min_result = check_cost_minimization(log)

        # Input data stats
        def _arr_stats(arr: np.ndarray) -> dict[str, float]:
            return {"mean": float(np.mean(arr)), "std": float(np.std(arr)),
                    "min": float(np.min(arr)), "max": float(np.max(arr))}

        input_price_stats = _arr_stats(log.input_prices)
        output_price_stats = _arr_stats(log.output_prices)
        profit_stats = _arr_stats(log.profit)

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
            input_price_stats=input_price_stats,
            output_price_stats=output_price_stats,
            profit_stats=profit_stats,
        )


@dataclass
class PanelSummary(ResultDisplayMixin):
    """Aggregate summary for multi-user panel analysis.

    Combines per-user BehavioralSummary results into aggregate statistics:
    consistency rates, efficiency distributions, and identification of
    the most inconsistent users.

    Example:
        >>> from prefgraph import BehaviorPanel
        >>> panel = BehaviorPanel.from_logs(logs)
        >>> ps = panel.summary()
        >>> print(ps)
    """

    user_summaries: dict[str, "BehavioralSummary"] = field(repr=False)
    num_users: int
    total_observations: int
    num_goods: int
    obs_per_user_stats: dict[str, float]
    garp_pass_rate: float
    warp_pass_rate: float | None
    sarp_pass_rate: float | None
    aei_distribution: dict[str, float]
    mpi_distribution: dict[str, float]
    hm_distribution: dict[str, float] | None
    top_inconsistent: list[tuple[str, float, float, int]]  # (uid, aei, mpi, T)
    computation_time_ms: float
    # Period-level breakdown (None if no period structure)
    period_stats: list[dict[str, Any]] | None = field(default=None, repr=False)
    num_periods: int = field(default=0, repr=False)

    @classmethod
    def from_summaries(
        cls,
        user_summaries: dict[str, "BehavioralSummary"],
        period_map: dict[str, tuple[str, str]] | None = None,
    ) -> "PanelSummary":
        """Build PanelSummary from per-user BehavioralSummary results."""
        if not user_summaries:
            raise ValueError("Cannot create PanelSummary from empty dict")

        n = len(user_summaries)
        summaries = list(user_summaries.values())

        # Obs per entry
        obs_counts = np.array([s.num_observations for s in summaries], dtype=np.float64)
        aei_vals = np.array([s.efficiency_index for s in summaries], dtype=np.float64)
        mpi_vals = np.array([s.mpi_value for s in summaries], dtype=np.float64)

        def _dist(arr: np.ndarray) -> dict[str, float]:
            return {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "25%": float(np.percentile(arr, 25)),
                "50%": float(np.percentile(arr, 50)),
                "75%": float(np.percentile(arr, 75)),
                "max": float(np.max(arr)),
            }

        # Count unique users (period_map may have multiple entries per user)
        if period_map is not None:
            unique_users = set(u for u, _ in period_map.values())
            num_users = len(unique_users)
        else:
            num_users = n

        # GARP pass rate
        garp_pass = sum(1 for s in summaries if s.is_consistent)

        # WARP/SARP pass rates (if computed)
        warp_pass_rate = None
        if summaries[0].warp_result is not None:
            warp_pass = sum(1 for s in summaries if s.warp_result and s.warp_result.is_consistent)
            warp_pass_rate = warp_pass / n

        sarp_pass_rate = None
        if summaries[0].sarp_result is not None:
            sarp_pass = sum(1 for s in summaries if s.sarp_result and s.sarp_result.is_consistent)
            sarp_pass_rate = sarp_pass / n

        # Houtman-Maks distribution
        hm_vals = []
        for s in summaries:
            if s.houtman_maks_result is not None:
                hm_vals.append(1.0 - s.houtman_maks_result.fraction)
            else:
                hm_vals.append(1.0)
        hm_arr = np.array(hm_vals, dtype=np.float64)
        hm_distribution = _dist(hm_arr)

        # Top inconsistent entries (sorted by AEI ascending)
        user_list = list(user_summaries.items())
        user_list.sort(key=lambda x: x[1].efficiency_index)
        top_inconsistent = [
            (uid, s.efficiency_index, s.mpi_value, s.num_observations)
            for uid, s in user_list[:5]
        ]

        # Total computation time
        total_time = sum(s.computation_time_ms for s in summaries)

        # Num goods (from first entry)
        num_goods = summaries[0].num_goods

        # Period-level breakdown
        period_stats = None
        num_periods = 0
        if period_map is not None:
            periods_set = sorted(set(p for _, p in period_map.values()))
            num_periods = len(periods_set)
            period_stats = []
            for period in periods_set:
                period_keys = [k for k, (_, p) in period_map.items() if p == period]
                period_summaries = [user_summaries[k] for k in period_keys if k in user_summaries]
                if not period_summaries:
                    continue
                n_p = len(period_summaries)
                p_garp = sum(1 for s in period_summaries if s.is_consistent)
                p_aei = np.mean([s.efficiency_index for s in period_summaries])
                p_mpi = np.mean([s.mpi_value for s in period_summaries])
                period_stats.append({
                    "period": period,
                    "users": n_p,
                    "garp_pass_rate": p_garp / n_p,
                    "mean_aei": float(p_aei),
                    "mean_mpi": float(p_mpi),
                })

        return cls(
            user_summaries=user_summaries,
            num_users=num_users,
            total_observations=int(np.sum(obs_counts)),
            num_goods=num_goods,
            obs_per_user_stats=_dist(obs_counts),
            garp_pass_rate=garp_pass / n,
            warp_pass_rate=warp_pass_rate,
            sarp_pass_rate=sarp_pass_rate,
            aei_distribution=_dist(aei_vals),
            mpi_distribution=_dist(mpi_vals),
            hm_distribution=hm_distribution,
            top_inconsistent=top_inconsistent,
            computation_time_ms=total_time,
            period_stats=period_stats,
            num_periods=num_periods,
        )

    def summary(self) -> str:
        """Return formatted panel summary (statsmodels-style)."""
        m = ResultSummaryMixin
        W = 70
        sep = "-" * W

        def _time_str(ms: float) -> str:
            return f"{ms:.2f} ms" if ms < 1000 else f"{ms / 1000:.2f} s"

        lines: list[str] = []

        # Two-column header
        lines.append("=" * W)
        lines.append(" " * ((W - 13) // 2) + "PANEL SUMMARY")
        lines.append("=" * W)

        n_garp = int(round(self.garp_pass_rate * self.num_users))
        lines.append(m._format_two_column_row(
            "No. Users", f"{self.num_users:,}",
            "GARP Pass Rate", f"{self.garp_pass_rate * 100:.1f}%", W,
        ))
        lines.append(m._format_two_column_row(
            "Total Observations", f"{self.total_observations:,}",
            "Mean AEI", f"{self.aei_distribution['mean']:.4f}", W,
        ))
        lines.append(m._format_two_column_row(
            "No. Goods", self.num_goods,
            "Mean MPI", f"{self.mpi_distribution['mean']:.4f}", W,
        ))
        lines.append(m._format_two_column_row(
            "Obs/User (mean)", f"{self.obs_per_user_stats['mean']:.1f}",
            "Computation Time", _time_str(self.computation_time_ms), W,
        ))
        if self.num_periods > 0:
            lines.append(m._format_two_column_row(
                "No. Periods", self.num_periods,
                "Entries (Users x Periods)", len(self.user_summaries), W,
            ))
        lines.append("=" * W)

        # Consistency Rates
        lines.append("")
        lines.append("Consistency Rates:")
        lines.append(sep)
        n_garp = int(round(self.garp_pass_rate * self.num_users))
        lines.append(m._format_metric(
            "GARP", f"{self.garp_pass_rate * 100:.1f}% ({n_garp:,} / {self.num_users:,})", W - 4,
        ))
        if self.warp_pass_rate is not None:
            n_warp = int(round(self.warp_pass_rate * self.num_users))
            lines.append(m._format_metric(
                "WARP", f"{self.warp_pass_rate * 100:.1f}% ({n_warp:,} / {self.num_users:,})", W - 4,
            ))
        if self.sarp_pass_rate is not None:
            n_sarp = int(round(self.sarp_pass_rate * self.num_users))
            lines.append(m._format_metric(
                "SARP", f"{self.sarp_pass_rate * 100:.1f}% ({n_sarp:,} / {self.num_users:,})", W - 4,
            ))

        # Efficiency Distribution
        lines.append("")
        lines.append("Efficiency Distribution:")
        lines.append(sep)
        dist_rows = {
            "AEI": self.aei_distribution,
            "MPI": self.mpi_distribution,
        }
        if self.hm_distribution is not None:
            dist_rows["HM Index"] = self.hm_distribution
        lines.append(m._format_distribution_table(dist_rows, W))

        # Most Inconsistent Users
        if self.top_inconsistent:
            lines.append("")
            lines.append("Most Inconsistent (Bottom 5):")
            lines.append(sep)
            for i, (uid, aei, mpi, t) in enumerate(self.top_inconsistent):
                lines.append(m._format_metric(
                    f"  {i+1}. {uid}", f"AEI={aei:.3f}, MPI={mpi:.3f}, T={t}", W - 4,
                ))

        # Temporal Breakdown (if period data available)
        if self.period_stats:
            lines.append("")
            lines.append("Temporal Breakdown:")
            lines.append(sep)
            hdr = f"  {'Period':<12s} {'Users':>6s} {'GARP Pass':>10s} {'Mean AEI':>10s} {'Mean MPI':>10s}"
            lines.append(hdr)
            for ps in self.period_stats:
                garp_pct = f"{ps['garp_pass_rate'] * 100:.1f}%"
                lines.append(
                    f"  {ps['period']:<12s} {ps['users']:>6d} {garp_pct:>10s}"
                    f" {ps['mean_aei']:>10.4f} {ps['mean_mpi']:>10.4f}"
                )

        lines.append("=" * W)
        return "\n".join(lines)

    def score(self) -> float:
        """Aggregate score: mean AEI across all users."""
        return self.aei_distribution["mean"]

    def __repr__(self) -> str:
        return (
            f"PanelSummary(users={self.num_users}, "
            f"garp_pass={self.garp_pass_rate * 100:.1f}%, "
            f"mean_aei={self.aei_distribution['mean']:.4f})"
        )

    def __str__(self) -> str:
        return self.summary()

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "num_users": self.num_users,
            "total_observations": self.total_observations,
            "num_goods": self.num_goods,
            "garp_pass_rate": self.garp_pass_rate,
            "warp_pass_rate": self.warp_pass_rate,
            "sarp_pass_rate": self.sarp_pass_rate,
            "aei_distribution": self.aei_distribution,
            "mpi_distribution": self.mpi_distribution,
            "hm_distribution": self.hm_distribution,
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }
