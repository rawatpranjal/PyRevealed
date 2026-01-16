"""BehavioralAuditor: High-level API for behavioral consistency validation.

This module provides a user-friendly, tech-native interface for validating
user behavior consistency. Think of it as a "linter" for behavioral data.

Use this to:
- Measure behavioral consistency (integrity score)
- Measure exploitability of inconsistencies (confusion metric)
- Compare behavior across user segments or time periods
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pyrevealed.core.mixins import ResultSummaryMixin
from pyrevealed.algorithms.garp import validate_consistency
from pyrevealed.algorithms.aei import compute_integrity_score
from pyrevealed.algorithms.mpi import compute_confusion_metric
from pyrevealed.algorithms.bronars import compute_test_power
from pyrevealed.algorithms.harp import validate_proportional_scaling
from pyrevealed.algorithms.vei import compute_granular_integrity
from pyrevealed.algorithms.quasilinear import test_income_invariance
from pyrevealed.algorithms.gross_substitutes import test_cross_price_effect
from pyrevealed.algorithms.differentiable import validate_smooth_preferences
from pyrevealed.algorithms.acyclical_p import validate_strict_consistency
from pyrevealed.algorithms.gapp import validate_price_preferences
from pyrevealed.algorithms.abstract_choice import (
    validate_menu_warp,
    validate_menu_sarp,
    validate_menu_consistency,
    compute_menu_efficiency,
    fit_menu_preferences,
)

if TYPE_CHECKING:
    from pyrevealed.core.session import BehaviorLog, MenuChoiceLog
    from pyrevealed.core.result import (
        ConsistencyResult,
        IntegrityResult,
        ConfusionResult,
        TestPowerResult,
        ProportionalScalingResult,
        GranularIntegrityResult,
        IncomeInvarianceResult,
        CrossPriceResult,
        SmoothPreferencesResult,
        StrictConsistencyResult,
        PricePreferencesResult,
        CongruenceResult,
        OrdinalUtilityResult,
    )


@dataclass
class AuditReport:
    """
    Comprehensive audit report for user behavior.

    Attributes:
        is_consistent: True if behavior passes GARP consistency check
        integrity_score: Afriat Efficiency Index (0-1, higher = more consistent)
        confusion_score: Money Pump Index (0-1, higher = more exploitable)
    """

    is_consistent: bool
    integrity_score: float
    confusion_score: float

    def summary(self) -> str:
        """Return human-readable audit summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("BEHAVIORAL AUDIT REPORT")]

        # Overall status
        status = m._format_status(self.is_consistent, "PASS", "FAIL")
        lines.append(f"\nOverall Status: {status}")

        # Core metrics
        lines.append(m._format_section("Core Metrics"))
        lines.append(m._format_metric("Consistent (GARP)", self.is_consistent))
        lines.append(m._format_metric("Integrity Score (AEI)", self.integrity_score))
        lines.append(m._format_metric("Confusion Score (MPI)", self.confusion_score))

        # Interpretation
        lines.append(m._format_section("Interpretation"))

        # Integrity interpretation
        integrity_interp = m._format_interpretation(self.integrity_score, "efficiency")
        lines.append(f"  Integrity: {integrity_interp}")

        # Confusion interpretation (note: score() inverts MPI)
        confusion_interp = m._format_interpretation(1.0 - self.confusion_score, "mpi")
        lines.append(f"  Confusion: {confusion_interp}")

        # Overall recommendation
        lines.append(m._format_section("Recommendation"))
        if self.is_consistent and self.integrity_score >= 0.95:
            lines.append("  Behavior is highly consistent with utility maximization.")
            lines.append("  User signal is clean and reliable.")
        elif self.integrity_score >= 0.85:
            lines.append("  Behavior shows minor inconsistencies.")
            lines.append("  User signal is generally reliable with some noise.")
        elif self.integrity_score >= 0.70:
            lines.append("  Behavior shows moderate inconsistencies.")
            lines.append("  Consider investigating specific violations.")
        else:
            lines.append("  Behavior shows significant inconsistencies.")
            lines.append("  User may be confused, a bot, or shared account.")

        return "\n".join(lines)

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns the average of integrity score and (1 - confusion score).
        """
        return (self.integrity_score + (1.0 - min(1.0, self.confusion_score))) / 2.0

    def __repr__(self) -> str:
        status = "PASS" if self.is_consistent else "FAIL"
        return (
            f"AuditReport({status}, "
            f"integrity={self.integrity_score:.2f}, "
            f"confusion={self.confusion_score:.2f})"
        )


@dataclass
class MenuAuditReport:
    """
    Comprehensive audit report for menu-based choice behavior.

    Attributes:
        is_warp_consistent: True if behavior passes WARP check
        is_sarp_consistent: True if behavior passes SARP check
        is_rationalizable: True if behavior can be rationalized by a preference order
        efficiency_score: Houtman-Maks efficiency (0-1, higher = fewer observations to remove)
        preference_order: Recovered preference ranking (most to least preferred), or None
    """

    is_warp_consistent: bool
    is_sarp_consistent: bool
    is_rationalizable: bool
    efficiency_score: float
    preference_order: list[int] | None

    def summary(self) -> str:
        """Return human-readable menu audit summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("MENU CHOICE AUDIT REPORT")]

        # Overall status
        status = m._format_status(self.is_rationalizable, "RATIONALIZABLE", "INCONSISTENT")
        lines.append(f"\nOverall Status: {status}")

        # Core metrics
        lines.append(m._format_section("Consistency Tests"))
        lines.append(m._format_metric("WARP Consistent", self.is_warp_consistent))
        lines.append(m._format_metric("SARP Consistent", self.is_sarp_consistent))
        lines.append(m._format_metric("Fully Rationalizable", self.is_rationalizable))
        lines.append(m._format_metric("Efficiency Score (HM)", self.efficiency_score))

        # Preference order
        if self.preference_order is not None and len(self.preference_order) > 0:
            lines.append(m._format_section("Recovered Preference Order"))
            order_str = " > ".join(str(i) for i in self.preference_order[:10])
            lines.append(f"  {order_str}")
            if len(self.preference_order) > 10:
                lines.append(f"  ... ({len(self.preference_order) - 10} more items)")

        # Interpretation
        lines.append(m._format_section("Interpretation"))

        # Efficiency interpretation
        if self.efficiency_score >= 1.0 - 1e-6:
            lines.append("  Efficiency: Perfect - all observations are consistent.")
        elif self.efficiency_score >= 0.9:
            lines.append("  Efficiency: Excellent - minor inconsistencies only.")
        elif self.efficiency_score >= 0.8:
            lines.append("  Efficiency: Good - some observations may need review.")
        else:
            lines.append(f"  Efficiency: Moderate - {(1-self.efficiency_score)*100:.1f}% observations inconsistent.")

        # Overall recommendation
        lines.append(m._format_section("Recommendation"))
        if self.is_rationalizable:
            lines.append("  Choices are fully rationalizable by a preference order.")
            lines.append("  User has stable, consistent preferences.")
        elif self.is_sarp_consistent:
            lines.append("  Choices satisfy SARP but not full congruence.")
            lines.append("  Consider checking maximality condition.")
        elif self.is_warp_consistent:
            lines.append("  Choices satisfy WARP but not SARP.")
            lines.append("  Some transitive preference cycles exist.")
        else:
            lines.append("  Choices violate WARP - direct preference reversals.")
            lines.append("  User preferences appear inconsistent.")

        return "\n".join(lines)

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns the efficiency score directly.
        """
        return self.efficiency_score

    def __repr__(self) -> str:
        status = "RATIONALIZABLE" if self.is_rationalizable else "INCONSISTENT"
        return (
            f"MenuAuditReport({status}, "
            f"warp={self.is_warp_consistent}, "
            f"sarp={self.is_sarp_consistent}, "
            f"efficiency={self.efficiency_score:.2f})"
        )


class BehavioralAuditor:
    """
    Validates behavioral consistency in user action logs.

    BehavioralAuditor is the "linter" for user behavior. It checks if
    a user's historical actions are internally consistent with utility
    maximization.

    Example:
        >>> from pyrevealed import BehavioralAuditor, BehaviorLog
        >>> import numpy as np

        >>> # Create behavior log
        >>> log = BehaviorLog(
        ...     cost_vectors=np.array([[1.0, 2.0], [2.0, 1.0]]),
        ...     action_vectors=np.array([[3.0, 1.0], [1.0, 3.0]]),
        ...     user_id="user_123"
        ... )

        >>> # Run audit
        >>> auditor = BehavioralAuditor()
        >>> if auditor.validate_history(log):
        ...     print("User behavior is consistent")
        ... else:
        ...     print("Inconsistent behavior detected")

        >>> # Get detailed scores
        >>> score = auditor.get_integrity_score(log)
        >>> print(f"Behavioral integrity: {score:.2f}")

    Attributes:
        precision: Numerical precision for consistency checks (default: 1e-6)
    """

    def __init__(self, precision: float = 1e-6) -> None:
        """
        Initialize the auditor.

        Args:
            precision: Numerical precision for floating-point comparisons.
                       Smaller values are more strict.
        """
        self.precision = precision

    def validate_history(self, log: BehaviorLog) -> bool:
        """
        Check if user behavior history is internally consistent.

        A consistent history means the user's choices don't contradict
        each other transitively. Inconsistent behavior suggests:
        - Bot (random choices)
        - Shared account (multiple users)
        - Confused user (bad UX)

        Args:
            log: BehaviorLog containing user's historical actions

        Returns:
            True if behavior is consistent, False otherwise

        Example:
            >>> if auditor.validate_history(user_log):
            ...     trust_level = "high"
            ... else:
            ...     trust_level = "low"
        """
        result = validate_consistency(log, tolerance=self.precision)
        return result.is_consistent

    def get_integrity_score(self, log: BehaviorLog) -> float:
        """
        Get behavioral integrity score (0-1).

        The integrity score measures how "clean" the behavioral signal is:
        - 1.0 = Perfect integrity, fully consistent behavior
        - 0.9+ = High integrity, minor noise
        - 0.7-0.9 = Moderate integrity, some confusion
        - <0.7 = Low integrity, likely bot or multiple users

        Args:
            log: BehaviorLog containing user's historical actions

        Returns:
            Float between 0 (chaotic) and 1 (perfectly consistent)

        Example:
            >>> score = auditor.get_integrity_score(user_log)
            >>> if score < 0.85:
            ...     flag_for_manual_review(user_id)
        """
        result = compute_integrity_score(log, tolerance=self.precision)
        return result.efficiency_index

    def get_confusion_score(self, log: BehaviorLog) -> float:
        """
        Get confusion/exploitability score (0-1).

        The confusion score measures how exploitable the user's
        inconsistencies are. High confusion indicates:
        - User not understanding the options
        - Bad UX causing irrational choices
        - Possible UI dark patterns

        Args:
            log: BehaviorLog containing user's historical actions

        Returns:
            Float between 0 (no confusion) and 1 (highly confused)

        Example:
            >>> confusion = auditor.get_confusion_score(user_log)
            >>> if confusion > 0.15:
            ...     alert_ux_team("User showing high confusion")
        """
        result = compute_confusion_metric(log, tolerance=self.precision)
        return result.mpi_value

    def get_consistency_details(self, log: BehaviorLog) -> ConsistencyResult:
        """
        Get detailed consistency check results.

        Returns the full ConsistencyResult with information about
        specific inconsistencies found.

        Args:
            log: BehaviorLog containing user's historical actions

        Returns:
            ConsistencyResult with is_consistent, violations, etc.
        """
        return validate_consistency(log, tolerance=self.precision)

    def get_integrity_details(self, log: BehaviorLog) -> IntegrityResult:
        """
        Get detailed integrity score results.

        Returns the full IntegrityResult with the underlying
        consistency check at the computed threshold.

        Args:
            log: BehaviorLog containing user's historical actions

        Returns:
            IntegrityResult with integrity_score, waste_fraction, etc.
        """
        return compute_integrity_score(log, tolerance=self.precision)

    def get_confusion_details(self, log: BehaviorLog) -> ConfusionResult:
        """
        Get detailed confusion metric results.

        Returns the full ConfusionResult with information about
        the worst inconsistency cycles.

        Args:
            log: BehaviorLog containing user's historical actions

        Returns:
            ConfusionResult with confusion_score, worst_cycle, etc.
        """
        return compute_confusion_metric(log, tolerance=self.precision)

    def full_audit(self, log: BehaviorLog) -> AuditReport:
        """
        Run comprehensive behavioral audit.

        Computes all core metrics and returns a single report.

        Args:
            log: BehaviorLog containing user's historical actions

        Returns:
            AuditReport with consistency, integrity, and confusion scores

        Example:
            >>> report = auditor.full_audit(user_log)
            >>> print(f"Consistent: {report.is_consistent}")
            >>> print(f"Integrity: {report.integrity_score:.2f}")
            >>> print(f"Confusion: {report.confusion_score:.2f}")
        """
        return AuditReport(
            is_consistent=self.validate_history(log),
            integrity_score=self.get_integrity_score(log),
            confusion_score=self.get_confusion_score(log),
        )

    # =========================================================================
    # NEW METHODS - Extended Analysis
    # =========================================================================

    def compute_test_power(
        self, log: BehaviorLog, n_simulations: int = 1000
    ) -> TestPowerResult:
        """
        Compute statistical power of the consistency test.

        Bronars' Power Index measures whether passing the consistency test
        is statistically meaningful. Low power means even random behavior
        would pass, making the consistency result uninformative.

        Args:
            log: BehaviorLog containing user's historical actions
            n_simulations: Number of random simulations (default: 1000)

        Returns:
            TestPowerResult with power_index and is_significant

        Example:
            >>> result = auditor.compute_test_power(user_log)
            >>> if result.power_index < 0.5:
            ...     print("Warning: GARP test has low power for this data")
        """
        return compute_test_power(
            log, n_simulations=n_simulations, tolerance=self.precision
        )

    def validate_proportional_scaling(
        self, log: BehaviorLog
    ) -> ProportionalScalingResult:
        """
        Test if user preferences scale proportionally with budget.

        Homothetic preferences mean relative preferences don't change
        with budget - useful for demand prediction and user segmentation.

        Args:
            log: BehaviorLog containing user's historical actions

        Returns:
            ProportionalScalingResult with is_homothetic and violations

        Example:
            >>> result = auditor.validate_proportional_scaling(user_log)
            >>> if result.is_homothetic:
            ...     print("User preferences scale with budget")
        """
        return validate_proportional_scaling(log, tolerance=self.precision)

    def compute_granular_integrity(
        self, log: BehaviorLog, efficiency_threshold: float = 0.9
    ) -> GranularIntegrityResult:
        """
        Get per-observation integrity scores.

        Unlike get_integrity_score which gives one global score, this
        identifies which specific observations are problematic.

        Args:
            log: BehaviorLog containing user's historical actions
            efficiency_threshold: Threshold below which observations are flagged (default: 0.9)

        Returns:
            GranularIntegrityResult with efficiency_vector and problematic_observations

        Example:
            >>> result = auditor.compute_granular_integrity(user_log)
            >>> for idx in result.problematic_observations:
            ...     print(f"Investigate observation {idx}")
        """
        return compute_granular_integrity(
            log, tolerance=self.precision, efficiency_threshold=efficiency_threshold
        )

    def test_income_invariance(self, log: BehaviorLog) -> IncomeInvarianceResult:
        """
        Test if user behavior is invariant to income changes.

        Quasilinear preferences have constant marginal utility of money,
        meaning no income effects on goods choices.

        Args:
            log: BehaviorLog containing user's historical actions

        Returns:
            IncomeInvarianceResult with is_quasilinear and violations

        Example:
            >>> result = auditor.test_income_invariance(user_log)
            >>> if result.has_income_effects:
            ...     print("User behavior varies with income")
        """
        return test_income_invariance(log, tolerance=self.precision)

    def test_cross_price_effect(
        self, log: BehaviorLog, good_g: int, good_h: int
    ) -> CrossPriceResult:
        """
        Test cross-price relationship between two goods.

        Determines if goods are substitutes (price of A up → demand for B up)
        or complements (price of A up → demand for B down).

        Args:
            log: BehaviorLog containing user's historical actions
            good_g: Index of first good
            good_h: Index of second good

        Returns:
            CrossPriceResult with relationship classification

        Example:
            >>> result = auditor.test_cross_price_effect(user_log, 0, 1)
            >>> print(f"Goods 0 and 1 are {result.relationship}")
        """
        return test_cross_price_effect(
            log, good_g=good_g, good_h=good_h, tolerance=self.precision
        )

    # =========================================================================
    # 2024 SURVEY ALGORITHMS
    # =========================================================================

    def validate_smooth_preferences(self, log: BehaviorLog) -> SmoothPreferencesResult:
        """
        Test if user preferences are smooth (differentiable).

        Smooth preferences enable demand function derivatives for
        price sensitivity analysis. Requires both SARP (no indifferent
        cycles) and price-quantity uniqueness.

        Args:
            log: BehaviorLog containing user's historical actions

        Returns:
            SmoothPreferencesResult with differentiability status

        Example:
            >>> result = auditor.validate_smooth_preferences(user_log)
            >>> if result.is_differentiable:
            ...     print("Preferences are smooth - can compute price elasticities")
        """
        return validate_smooth_preferences(log, tolerance=self.precision)

    def validate_strict_consistency(self, log: BehaviorLog) -> StrictConsistencyResult:
        """
        Test strict behavioral consistency (more lenient than full check).

        Tests only strict preference cycles. Passes if violations are
        only due to weak preferences (indifference). Useful for
        identifying "approximately rational" behavior.

        Args:
            log: BehaviorLog containing user's historical actions

        Returns:
            StrictConsistencyResult with consistency status

        Example:
            >>> result = auditor.validate_strict_consistency(user_log)
            >>> if result.strict_violations_only:
            ...     print("GARP fails but only due to indifference")
        """
        return validate_strict_consistency(log, tolerance=self.precision)

    def validate_price_preferences(self, log: BehaviorLog) -> PricePreferencesResult:
        """
        Test if user has consistent price preferences.

        The dual of consistency validation - tests if the user prefers
        situations where their desired items are cheaper. Useful for
        understanding price sensitivity.

        Args:
            log: BehaviorLog containing user's historical actions

        Returns:
            PricePreferencesResult with price preference consistency

        Example:
            >>> result = auditor.validate_price_preferences(user_log)
            >>> if result.prefers_lower_prices:
            ...     print("User consistently seeks lower prices")
        """
        return validate_price_preferences(log, tolerance=self.precision)

    # =========================================================================
    # MENU-BASED CHOICE ANALYSIS (Abstract Choice Theory)
    # =========================================================================

    def validate_menu_history(self, log: MenuChoiceLog) -> bool:
        """
        Check if menu-based choice history is consistent (SARP).

        A consistent history means choices don't form preference cycles.
        Inconsistent behavior suggests irrational decision-making.

        Args:
            log: MenuChoiceLog containing menu choices

        Returns:
            True if behavior is SARP-consistent, False otherwise

        Example:
            >>> if auditor.validate_menu_history(menu_log):
            ...     print("Menu choices are consistent")
        """
        result = validate_menu_sarp(log)
        return result.is_consistent

    def get_menu_consistency_details(self, log: MenuChoiceLog) -> CongruenceResult:
        """
        Get detailed menu consistency (Congruence/rationalizability) results.

        Returns the full CongruenceResult with information about
        SARP violations and maximality violations.

        Args:
            log: MenuChoiceLog containing menu choices

        Returns:
            CongruenceResult with is_rationalizable, violations, etc.
        """
        return validate_menu_consistency(log)

    def get_menu_efficiency_score(self, log: MenuChoiceLog) -> float:
        """
        Get menu choice efficiency score (Houtman-Maks index).

        The efficiency score measures what fraction of observations
        are consistent. 1.0 means fully consistent, lower values
        indicate more inconsistencies.

        Args:
            log: MenuChoiceLog containing menu choices

        Returns:
            Float between 0 (all inconsistent) and 1 (perfectly consistent)

        Example:
            >>> score = auditor.get_menu_efficiency_score(menu_log)
            >>> if score < 0.9:
            ...     print("Some inconsistent choices detected")
        """
        result = compute_menu_efficiency(log)
        return result.efficiency_index

    def recover_menu_preferences(self, log: MenuChoiceLog) -> OrdinalUtilityResult:
        """
        Recover ordinal preference ranking from menu choices.

        If the data is SARP-consistent, computes a preference ranking
        using topological sort of the revealed preference graph.

        Args:
            log: MenuChoiceLog containing menu choices

        Returns:
            OrdinalUtilityResult with preference_order and utility_ranking

        Example:
            >>> result = auditor.recover_menu_preferences(menu_log)
            >>> if result.success:
            ...     print(f"Preference order: {result.preference_order}")
        """
        return fit_menu_preferences(log)

    def full_menu_audit(self, log: MenuChoiceLog) -> MenuAuditReport:
        """
        Run comprehensive menu choice audit.

        Computes all menu-based consistency metrics and returns a single report.

        Args:
            log: MenuChoiceLog containing menu choices

        Returns:
            MenuAuditReport with WARP, SARP, efficiency, and preference results

        Example:
            >>> report = auditor.full_menu_audit(menu_log)
            >>> print(f"Rationalizable: {report.is_rationalizable}")
            >>> print(f"Efficiency: {report.efficiency_score:.2f}")
        """
        warp_result = validate_menu_warp(log)
        sarp_result = validate_menu_sarp(log)
        cong_result = validate_menu_consistency(log)
        eff_result = compute_menu_efficiency(log)
        pref_result = fit_menu_preferences(log)

        return MenuAuditReport(
            is_warp_consistent=warp_result.is_consistent,
            is_sarp_consistent=sarp_result.is_consistent,
            is_rationalizable=cong_result.is_rationalizable,
            efficiency_score=eff_result.efficiency_index,
            preference_order=pref_result.preference_order,
        )
