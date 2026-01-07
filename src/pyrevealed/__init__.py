"""
PyRevealed: Behavioral Signal Analysis for User Understanding.

Detect bots, shared accounts, and UI confusion using structural consistency
checks on user behavior logs.

## Tech-Friendly API (Primary)

High-Level Classes:
    - BehavioralAuditor: Validate behavior consistency (linter for user data)
    - PreferenceEncoder: Extract latent preferences (encoder for ML pipelines)

Data Containers:
    - BehaviorLog: User behavior history (cost/action pairs)
    - RiskChoiceLog: Choices under uncertainty
    - EmbeddingChoiceLog: Choices in feature/embedding space

Functions:
    - validate_consistency(): Check behavioral consistency
    - compute_integrity_score(): Compute integrity/noise score
    - compute_confusion_metric(): Compute confusion/exploitability score
    - fit_latent_values(): Extract latent preference values
    - find_preference_anchor(): Find preference anchor in embedding space
    - test_feature_independence(): Test feature group independence

## Legacy API (Deprecated but supported)

Economics-based names still work for backward compatibility:
    - ConsumerSession -> BehaviorLog
    - check_garp -> validate_consistency
    - compute_aei -> compute_integrity_score
    - compute_mpi -> compute_confusion_metric
"""

# =============================================================================
# HIGH-LEVEL CLASSES (Primary API)
# =============================================================================

from pyrevealed.auditor import BehavioralAuditor, AuditReport
from pyrevealed.encoder import PreferenceEncoder
from pyrevealed.lancaster import (
    LancasterLog,
    CharacteristicsLog,
    transform_to_characteristics,
)

# =============================================================================
# DATA CONTAINERS - Tech-friendly names (Primary)
# =============================================================================

from pyrevealed.core.session import (
    # Primary tech-friendly names
    BehaviorLog,
    RiskChoiceLog,
    EmbeddingChoiceLog,
    # Legacy names (aliases for backward compatibility)
    ConsumerSession,
    RiskSession,
    SpatialSession,
)

# =============================================================================
# RESULT TYPES - Tech-friendly names (Primary)
# =============================================================================

from pyrevealed.core.result import (
    # Primary tech-friendly names
    ConsistencyResult,
    IntegrityResult,
    ConfusionResult,
    LatentValueResult,
    PreferenceAnchorResult,
    FeatureIndependenceResult,
    # Legacy names (aliases for backward compatibility)
    GARPResult,
    AEIResult,
    MPIResult,
    UtilityRecoveryResult,
    IdealPointResult,
    SeparabilityResult,
    # Risk result (already tech-friendly)
    RiskProfileResult,
    # New result types - tech-friendly
    TestPowerResult,
    ProportionalScalingResult,
    IncomeInvarianceResult,
    CrossPriceResult,
    GranularIntegrityResult,
    # New result types - legacy
    BronarsPowerResult,
    HARPResult,
    QuasilinearityResult,
    GrossSubstitutesResult,
    SubstitutionMatrixResult,
    VEIResult,
    # Lancaster characteristics model
    LancasterResult,
    CharacteristicsValuationResult,
)

# =============================================================================
# FUNCTIONS - Tech-friendly names (Primary)
# =============================================================================

# Consistency validation
from pyrevealed.algorithms.garp import (
    validate_consistency,
    validate_consistency_weak,
    check_garp,  # Legacy
    check_warp,  # Legacy
)

# Integrity/noise score
from pyrevealed.algorithms.aei import (
    compute_integrity_score,
    compute_aei,  # Legacy
    compute_varian_index,
)

# Confusion metric
from pyrevealed.algorithms.mpi import (
    compute_confusion_metric,
    compute_minimal_outlier_fraction,
    compute_mpi,  # Legacy
    compute_houtman_maks_index,  # Legacy
)

# Latent value extraction
from pyrevealed.algorithms.utility import (
    fit_latent_values,
    build_value_function,
    predict_choice,
    recover_utility,  # Legacy
    construct_afriat_utility,  # Legacy
    predict_demand,  # Legacy
)

# Risk profiling (already tech-friendly)
from pyrevealed.algorithms.risk import (
    compute_risk_profile,
    check_expected_utility_axioms,
    classify_risk_type,
)

# Preference anchor / embedding analysis
from pyrevealed.algorithms.spatial import (
    find_preference_anchor,
    validate_embedding_consistency,
    compute_signal_strength,
    find_multiple_anchors,
    find_ideal_point,  # Legacy
    check_euclidean_rationality,  # Legacy
    compute_preference_strength,  # Legacy
    find_multiple_ideal_points,  # Legacy
)

# Feature independence / separability
from pyrevealed.algorithms.separability import (
    test_feature_independence,
    discover_independent_groups,
    compute_cross_impact,
    check_separability,  # Legacy
    find_separable_partition,  # Legacy
    compute_cannibalization,  # Legacy
)

# =============================================================================
# NEW ALGORITHMS
# =============================================================================

# Test power (Bronars)
from pyrevealed.algorithms.bronars import (
    compute_test_power,
    compute_test_power_fast,
    compute_bronars_power,  # Legacy
    compute_bronars_power_fast,  # Legacy
)

# Proportional scaling (HARP)
from pyrevealed.algorithms.harp import (
    validate_proportional_scaling,
    check_harp,  # Legacy
)

# Granular integrity (VEI)
from pyrevealed.algorithms.vei import (
    compute_granular_integrity,
    compute_granular_integrity_l2,
    compute_vei,  # Legacy
    compute_vei_l2,  # Legacy
)

# Income invariance (Quasilinearity)
from pyrevealed.algorithms.quasilinear import (
    test_income_invariance,
    test_income_invariance_exhaustive,
    check_quasilinearity,  # Legacy
    check_quasilinearity_exhaustive,  # Legacy
)

# Cross-price effects (Gross substitutes)
from pyrevealed.algorithms.gross_substitutes import (
    test_cross_price_effect,
    compute_cross_price_matrix,
    check_gross_substitutes,  # Legacy
    compute_substitution_matrix,  # Legacy
    check_law_of_demand,
)

__version__ = "0.3.0"

__all__ = [
    # ==========================================================================
    # HIGH-LEVEL CLASSES (Primary API)
    # ==========================================================================
    "BehavioralAuditor",
    "AuditReport",
    "PreferenceEncoder",
    # Lancaster Characteristics Model
    "LancasterLog",
    "CharacteristicsLog",
    "transform_to_characteristics",

    # ==========================================================================
    # DATA CONTAINERS - Tech-friendly (Primary)
    # ==========================================================================
    "BehaviorLog",
    "RiskChoiceLog",
    "EmbeddingChoiceLog",

    # ==========================================================================
    # RESULT TYPES - Tech-friendly (Primary)
    # ==========================================================================
    "ConsistencyResult",
    "IntegrityResult",
    "ConfusionResult",
    "LatentValueResult",
    "PreferenceAnchorResult",
    "FeatureIndependenceResult",
    "RiskProfileResult",
    # New result types
    "TestPowerResult",
    "ProportionalScalingResult",
    "IncomeInvarianceResult",
    "CrossPriceResult",
    "GranularIntegrityResult",
    # Lancaster characteristics model results
    "LancasterResult",
    "CharacteristicsValuationResult",

    # ==========================================================================
    # FUNCTIONS - Tech-friendly (Primary)
    # ==========================================================================
    # Consistency
    "validate_consistency",
    "validate_consistency_weak",
    # Integrity
    "compute_integrity_score",
    # Confusion
    "compute_confusion_metric",
    "compute_minimal_outlier_fraction",
    # Latent values
    "fit_latent_values",
    "build_value_function",
    "predict_choice",
    # Risk
    "compute_risk_profile",
    "check_expected_utility_axioms",
    "classify_risk_type",
    # Preference anchor
    "find_preference_anchor",
    "validate_embedding_consistency",
    "compute_signal_strength",
    "find_multiple_anchors",
    # Feature independence
    "test_feature_independence",
    "discover_independent_groups",
    "compute_cross_impact",
    # Test power (NEW)
    "compute_test_power",
    "compute_test_power_fast",
    # Proportional scaling (NEW)
    "validate_proportional_scaling",
    # Granular integrity (NEW)
    "compute_granular_integrity",
    "compute_granular_integrity_l2",
    # Income invariance (NEW)
    "test_income_invariance",
    "test_income_invariance_exhaustive",
    # Cross-price effects (NEW)
    "test_cross_price_effect",
    "compute_cross_price_matrix",
    "check_law_of_demand",

    # ==========================================================================
    # LEGACY NAMES (Deprecated - use tech-friendly names above)
    # ==========================================================================
    # Data containers
    "ConsumerSession",
    "RiskSession",
    "SpatialSession",
    # Result types
    "GARPResult",
    "AEIResult",
    "MPIResult",
    "UtilityRecoveryResult",
    "IdealPointResult",
    "SeparabilityResult",
    # New result types - legacy names
    "BronarsPowerResult",
    "HARPResult",
    "QuasilinearityResult",
    "GrossSubstitutesResult",
    "SubstitutionMatrixResult",
    "VEIResult",
    # Functions
    "check_garp",
    "check_warp",
    "compute_aei",
    "compute_varian_index",
    "compute_mpi",
    "compute_houtman_maks_index",
    "recover_utility",
    "construct_afriat_utility",
    "predict_demand",
    "find_ideal_point",
    "check_euclidean_rationality",
    "compute_preference_strength",
    "find_multiple_ideal_points",
    "check_separability",
    "find_separable_partition",
    "compute_cannibalization",
    # New functions - legacy names
    "compute_bronars_power",
    "compute_bronars_power_fast",
    "check_harp",
    "compute_vei",
    "compute_vei_l2",
    "check_quasilinearity",
    "check_quasilinearity_exhaustive",
    "check_gross_substitutes",
    "compute_substitution_matrix",
    # Convenience
    "get_integrity_score",
]


def get_integrity_score(log: BehaviorLog, precision: float = 1e-6) -> float:
    """
    Convenience function to get the behavioral integrity score directly.

    The integrity score measures data quality:
    - 1.0 = Perfect signal, fully consistent user
    - 0.5 = Noisy signal, possible bot or confused user
    - <0.5 = Very noisy, likely bot or shared account

    Args:
        log: BehaviorLog (or ConsumerSession) with user behavior data
        precision: Convergence tolerance for computation

    Returns:
        Float between 0 (chaotic) and 1 (perfectly consistent)

    Example:
        >>> from pyrevealed import BehaviorLog, get_integrity_score
        >>> score = get_integrity_score(user_log)
        >>> if score < 0.85:
        ...     flag_for_review(user_id)
    """
    result = compute_integrity_score(log, tolerance=precision)
    return result.efficiency_index
