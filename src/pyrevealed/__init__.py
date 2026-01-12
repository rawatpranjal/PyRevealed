"""
PyRevealed: Behavioral Signal Analysis for User Understanding.

Analyze behavioral consistency using revealed preference theory.

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
# EXCEPTIONS AND WARNINGS
# =============================================================================

from pyrevealed.core.exceptions import (
    # Base exception
    PyRevealedError,
    # Data validation exceptions
    DataValidationError,
    DimensionError,
    ValueRangeError,
    NaNInfError,
    # Computation exceptions
    OptimizationError,
    NotFittedError,
    InsufficientDataError,
    # Warnings
    DataQualityWarning,
    NumericalInstabilityWarning,
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
    # 2024 Survey algorithms - tech-friendly
    SmoothPreferencesResult,
    StrictConsistencyResult,
    PricePreferencesResult,
    # 2024 Survey algorithms - legacy
    DifferentiableResult,
    AcyclicalPResult,
    GAPPResult,
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

# =============================================================================
# 2024 SURVEY ALGORITHMS
# =============================================================================

# Smooth preferences (Differentiable rationality)
from pyrevealed.algorithms.differentiable import (
    validate_smooth_preferences,
    validate_sarp,
    check_differentiable,  # Legacy
    check_sarp,  # Legacy
)

# Strict consistency (Acyclical P)
from pyrevealed.algorithms.acyclical_p import (
    validate_strict_consistency,
    check_acyclical_p,  # Legacy
)

# Price preferences (GAPP)
from pyrevealed.algorithms.gapp import (
    validate_price_preferences,
    check_gapp,  # Legacy
)

__version__ = "0.3.1"

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
    # EXCEPTIONS AND WARNINGS
    # ==========================================================================
    # Base exception
    "PyRevealedError",
    # Data validation exceptions
    "DataValidationError",
    "DimensionError",
    "ValueRangeError",
    "NaNInfError",
    # Computation exceptions
    "OptimizationError",
    "NotFittedError",
    "InsufficientDataError",
    # Warnings
    "DataQualityWarning",
    "NumericalInstabilityWarning",
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
    # Smooth preferences (2024 Survey)
    "validate_smooth_preferences",
    "validate_sarp",
    # Strict consistency (2024 Survey)
    "validate_strict_consistency",
    # Price preferences (2024 Survey)
    "validate_price_preferences",
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
    # 2024 Survey result types
    "SmoothPreferencesResult",
    "StrictConsistencyResult",
    "PricePreferencesResult",
    "DifferentiableResult",
    "AcyclicalPResult",
    "GAPPResult",
    # 2024 Survey functions - legacy names
    "check_differentiable",
    "check_sarp",
    "check_acyclical_p",
    "check_gapp",
    # Convenience
    "get_integrity_score",
]


def get_integrity_score(log: BehaviorLog, precision: float = 1e-6) -> float:
    """
    Convenience function to get the behavioral integrity score directly.

    The integrity score (Afriat Efficiency Index) measures consistency:
    - 1.0 = Perfectly consistent with utility maximization
    - 0.9+ = Minor deviations from rationality
    - <0.9 = Notable inconsistencies in behavior

    Args:
        log: BehaviorLog (or ConsumerSession) with user behavior data
        precision: Convergence tolerance for computation

    Returns:
        Float between 0 (highly inconsistent) and 1 (perfectly consistent)

    Example:
        >>> from pyrevealed import BehaviorLog, get_integrity_score
        >>> score = get_integrity_score(user_log)
        >>> print(f"Integrity: {score:.2f}")
    """
    result = compute_integrity_score(log, tolerance=precision)
    return result.efficiency_index
