"""BehavioralAuditor: High-level API for behavioral consistency validation.

This module provides a user-friendly, tech-native interface for validating
user behavior consistency. Think of it as a "linter" for behavioral data.

Use this to:
- Detect bots (inconsistent/random behavior)
- Detect account sharing (multiple preference profiles)
- Detect UI confusion (exploitable inconsistencies)
- A/B test UX changes (compare confusion scores)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pyrevealed.algorithms.garp import validate_consistency
from pyrevealed.algorithms.aei import compute_integrity_score
from pyrevealed.algorithms.mpi import compute_confusion_metric
from pyrevealed.algorithms.bronars import compute_test_power
from pyrevealed.algorithms.harp import validate_proportional_scaling
from pyrevealed.algorithms.vei import compute_granular_integrity
from pyrevealed.algorithms.quasilinear import test_income_invariance
from pyrevealed.algorithms.gross_substitutes import test_cross_price_effect

if TYPE_CHECKING:
    from pyrevealed.core.session import BehaviorLog
    from pyrevealed.core.result import (
        ConsistencyResult,
        IntegrityResult,
        ConfusionResult,
        TestPowerResult,
        ProportionalScalingResult,
        GranularIntegrityResult,
        IncomeInvarianceResult,
        CrossPriceResult,
    )


@dataclass
class AuditReport:
    """
    Comprehensive audit report for user behavior.

    Attributes:
        is_consistent: True if behavior passes consistency check
        integrity_score: Behavioral integrity score (0-1, higher = cleaner signal)
        confusion_score: Confusion/exploitability score (0-1, higher = more confused)
        bot_risk: Estimated probability this is bot behavior
        shared_account_risk: Estimated probability of account sharing
        ux_confusion_risk: Estimated probability of UI-caused confusion
    """

    is_consistent: bool
    integrity_score: float
    confusion_score: float
    bot_risk: float
    shared_account_risk: float
    ux_confusion_risk: float

    def __repr__(self) -> str:
        status = "PASS" if self.is_consistent else "FAIL"
        return (
            f"AuditReport({status}, "
            f"integrity={self.integrity_score:.2f}, "
            f"confusion={self.confusion_score:.2f})"
        )


class BehavioralAuditor:
    """
    Validates behavioral consistency in user action logs.

    BehavioralAuditor is the "linter" for user behavior. It checks if
    a user's historical actions are internally consistent, which helps
    identify:

    - **Bots**: Random/automated behavior fails consistency checks
    - **Shared accounts**: Multiple users = multiple inconsistent preferences
    - **UI confusion**: Bad UX leads to exploitable decision patterns

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

        Computes all metrics and returns a single report with
        risk assessments for bot, shared account, and UI confusion.

        Args:
            log: BehaviorLog containing user's historical actions

        Returns:
            AuditReport with all scores and risk assessments

        Example:
            >>> report = auditor.full_audit(user_log)
            >>> if report.bot_risk > 0.7:
            ...     block_user(user_id)
            >>> elif report.shared_account_risk > 0.5:
            ...     prompt_profile_split(user_id)
            >>> elif report.ux_confusion_risk > 0.5:
            ...     enable_user_guidance(user_id)
        """
        is_consistent = self.validate_history(log)
        integrity = self.get_integrity_score(log)
        confusion = self.get_confusion_score(log)

        # Heuristic risk calculations
        # Bot risk: High if low integrity AND inconsistent
        if not is_consistent and integrity < 0.7:
            bot_risk = 0.8
        elif not is_consistent and integrity < 0.85:
            bot_risk = 0.5
        elif not is_consistent:
            bot_risk = 0.3
        else:
            bot_risk = 0.1

        # Shared account risk: Moderate integrity but inconsistent
        if not is_consistent and 0.6 <= integrity <= 0.85:
            shared_account_risk = 0.6
        elif not is_consistent:
            shared_account_risk = 0.3
        else:
            shared_account_risk = 0.1

        # UX confusion risk: High confusion score
        if confusion > 0.2:
            ux_confusion_risk = 0.7
        elif confusion > 0.1:
            ux_confusion_risk = 0.4
        else:
            ux_confusion_risk = 0.15

        return AuditReport(
            is_consistent=is_consistent,
            integrity_score=integrity,
            confusion_score=confusion,
            bot_risk=bot_risk,
            shared_account_risk=shared_account_risk,
            ux_confusion_risk=ux_confusion_risk,
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

    def validate_proportional_scaling(self, log: BehaviorLog) -> ProportionalScalingResult:
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
