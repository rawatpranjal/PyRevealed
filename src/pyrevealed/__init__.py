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

from pyrevealed.auditor import BehavioralAuditor, AuditReport, MenuAuditReport
from pyrevealed.encoder import PreferenceEncoder, MenuPreferenceEncoder
from pyrevealed.lancaster import (
    LancasterLog,
    CharacteristicsLog,
    transform_to_characteristics,
)
from pyrevealed.core.summary import BehavioralSummary, MenuChoiceSummary

# =============================================================================
# DATA CONTAINERS - Tech-friendly names (Primary)
# =============================================================================

from pyrevealed.core.session import (
    # Primary tech-friendly names
    BehaviorLog,
    RiskChoiceLog,
    EmbeddingChoiceLog,
    MenuChoiceLog,
    # New data structures (Chapters 13, 15)
    StochasticChoiceLog,
    ProbabilisticChoiceLog,
    ProductionLog,
    FirmLog,
    # Legacy names (aliases for backward compatibility)
    ConsumerSession,
    RiskSession,
    SpatialSession,
    ChoiceSession,
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
    # v0.4.0: New result types for API consistency
    WARPResult,
    SARPResult,
    HoutmanMaksResult,
    # Abstract Choice Theory (menu-based)
    AbstractWARPResult,
    MenuWARPResult,
    AbstractSARPResult,
    MenuSARPResult,
    CongruenceResult,
    MenuConsistencyResult,
    HoutmanMaksAbstractResult,
    MenuEfficiencyResult,
    OrdinalUtilityResult,
    MenuPreferenceResult,
    # Chambers & Echenique (2016) - New algorithm results
    IntegrabilityResult,
    SlutskyConditionsResult,
    WelfareResult,
    WelfareChangeResult,
    AdditivityResult,
    AdditiveUtilityResult,
    CompensatedDemandResult,
    HicksianDemandResult,
    GeneralMetricResult,
    MetricPreferencesResult,
    StochasticChoiceResult,
    RandomUtilityResult,
    AttentionResult,
    ConsiderationSetResult,
    ProductionGARPResult,
    FirmBehaviorResult,
    # New revealed attention result types (P0)
    WARPLAResult,
    RandomAttentionResult,
    # RUM consistency result type (P1)
    RUMConsistencyResult,
    # Phase 2 extension result types
    RegularityResult,
    RegularityViolation,
    AttentionOverloadResult,
    SwapsIndexResult,
    ObservationContributionResult,
    StatusQuoBiasResult,
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
    # Phase 2 extensions
    compute_swaps_index,
    compute_observation_contributions,
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

# =============================================================================
# ABSTRACT CHOICE THEORY (Menu-based preferences)
# =============================================================================

from pyrevealed.algorithms.abstract_choice import (
    # Primary tech-friendly names
    validate_menu_warp,
    validate_menu_sarp,
    validate_menu_consistency,
    compute_menu_efficiency,
    fit_menu_preferences,
    # Legacy names (economics terminology)
    check_abstract_warp,
    check_abstract_sarp,
    check_congruence,
    compute_abstract_efficiency,
    recover_ordinal_utility,
)

# =============================================================================
# CHAMBERS & ECHENIQUE (2016) - NEW ALGORITHMS
# =============================================================================

# Integrability conditions (Ch 6.4-6.5)
from pyrevealed.algorithms.integrability import (
    test_integrability,
    compute_slutsky_matrix,
    check_slutsky_symmetry,
    check_slutsky_nsd,
    check_integrability,  # Legacy
)

# Welfare analysis (Ch 7.3-7.4)
from pyrevealed.algorithms.welfare import (
    analyze_welfare_change,
    compute_compensating_variation,
    compute_equivalent_variation,
    recover_cost_function,
    compute_consumer_surplus,
    compute_deadweight_loss,
    compute_cv,  # Legacy
    compute_ev,  # Legacy
)

# Additive separability (Ch 9.3)
from pyrevealed.algorithms.additive import (
    test_additive_separability,
    identify_additive_groups,
    check_no_cross_effects,
    check_additivity,  # Legacy
)

# Compensated demand (Ch 10.3) - extended gross_substitutes
from pyrevealed.algorithms.gross_substitutes import (
    decompose_price_effects,
    compute_hicksian_demand,
    check_compensated_law_of_demand,
    compute_slutsky_decomposition,
    estimate_compensated_demand,
)

# General metric spaces (Ch 11.3-11.4) - extended spatial
from pyrevealed.algorithms.spatial import (
    find_ideal_point_general,
    determine_best_metric,
    test_metric_rationality,
    find_anchor_general,
    select_best_metric,
)

# Stochastic choice (Ch 13)
from pyrevealed.algorithms.stochastic import (
    fit_random_utility_model,
    test_mcfadden_axioms,
    estimate_choice_probabilities,
    check_independence_irrelevant_alternatives,
    fit_luce_model,
    fit_rum,  # Legacy
    check_iia,  # Legacy
    # New RUM consistency functions (P1 - Smeulders et al. 2021)
    test_rum_consistency,
    compute_distance_to_rum,
    fit_rum_distribution,
    check_rum_consistency,  # Legacy alias
    # Phase 2 extensions - standalone regularity test
    test_regularity,
)

# Limited attention (Ch 14)
from pyrevealed.algorithms.attention import (
    test_attention_rationality,
    estimate_consideration_sets,
    compute_salience_weights,
    test_attention_filter,
    identify_attention,  # Legacy
    # New WARP-LA functions (P0 - Masatlioglu et al. 2012)
    test_warp_la,
    recover_preference_with_attention,
    validate_attention_filter_consistency,
    check_warp_la,  # Legacy alias
    # New RAM functions (P0 - Cattaneo et al. 2020)
    fit_random_attention_model,
    test_ram_consistency,
    estimate_attention_probabilities,
    compute_attention_bounds,
    # Phase 2 extensions - attention overload and status quo bias
    test_attention_overload,
    test_status_quo_bias,
)

# Production theory (Ch 15)
from pyrevealed.algorithms.production import (
    test_profit_maximization,
    check_cost_minimization,
    estimate_returns_to_scale,
    compute_technical_efficiency,
    check_production_garp,  # Legacy
)

__version__ = "0.4.2"

__all__ = [
    # ==========================================================================
    # HIGH-LEVEL CLASSES (Primary API)
    # ==========================================================================
    "BehavioralAuditor",
    "AuditReport",
    "MenuAuditReport",
    "PreferenceEncoder",
    "MenuPreferenceEncoder",
    # Summary classes
    "BehavioralSummary",
    "MenuChoiceSummary",
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
    "MenuChoiceLog",
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
    # v0.4.0: New result types for API consistency
    "WARPResult",
    "SARPResult",
    "HoutmanMaksResult",
    # 2024 Survey functions - legacy names
    "check_differentiable",
    "check_sarp",
    "check_acyclical_p",
    "check_gapp",
    # ==========================================================================
    # ABSTRACT CHOICE THEORY (Menu-based)
    # ==========================================================================
    # Data container
    "ChoiceSession",
    # Result types - tech-friendly
    "MenuWARPResult",
    "MenuSARPResult",
    "MenuConsistencyResult",
    "MenuEfficiencyResult",
    "MenuPreferenceResult",
    # Result types - legacy
    "AbstractWARPResult",
    "AbstractSARPResult",
    "CongruenceResult",
    "HoutmanMaksAbstractResult",
    "OrdinalUtilityResult",
    # Functions - tech-friendly
    "validate_menu_warp",
    "validate_menu_sarp",
    "validate_menu_consistency",
    "compute_menu_efficiency",
    "fit_menu_preferences",
    # Functions - legacy
    "check_abstract_warp",
    "check_abstract_sarp",
    "check_congruence",
    "compute_abstract_efficiency",
    "recover_ordinal_utility",
    # ==========================================================================
    # CHAMBERS & ECHENIQUE (2016) - NEW ALGORITHMS
    # ==========================================================================
    # New data structures
    "StochasticChoiceLog",
    "ProbabilisticChoiceLog",
    "ProductionLog",
    "FirmLog",
    # New result types - Chambers & Echenique
    "IntegrabilityResult",
    "SlutskyConditionsResult",
    "WelfareResult",
    "WelfareChangeResult",
    "AdditivityResult",
    "AdditiveUtilityResult",
    "CompensatedDemandResult",
    "HicksianDemandResult",
    "GeneralMetricResult",
    "MetricPreferencesResult",
    "StochasticChoiceResult",
    "RandomUtilityResult",
    "AttentionResult",
    "ConsiderationSetResult",
    "ProductionGARPResult",
    "FirmBehaviorResult",
    # P0: Revealed Attention result types
    "WARPLAResult",
    "RandomAttentionResult",
    # P1: RUM Consistency result type
    "RUMConsistencyResult",
    # Integrability (Ch 6.4-6.5)
    "test_integrability",
    "compute_slutsky_matrix",
    "check_slutsky_symmetry",
    "check_slutsky_nsd",
    "check_integrability",
    # Welfare (Ch 7.3-7.4)
    "analyze_welfare_change",
    "compute_compensating_variation",
    "compute_equivalent_variation",
    "recover_cost_function",
    "compute_consumer_surplus",
    "compute_deadweight_loss",
    "compute_cv",
    "compute_ev",
    # Additive separability (Ch 9.3)
    "test_additive_separability",
    "identify_additive_groups",
    "check_no_cross_effects",
    "check_additivity",
    # Compensated demand (Ch 10.3)
    "decompose_price_effects",
    "compute_hicksian_demand",
    "check_compensated_law_of_demand",
    "compute_slutsky_decomposition",
    "estimate_compensated_demand",
    # General metrics (Ch 11.3-11.4)
    "find_ideal_point_general",
    "determine_best_metric",
    "test_metric_rationality",
    "find_anchor_general",
    "select_best_metric",
    # Stochastic choice (Ch 13)
    "fit_random_utility_model",
    "test_mcfadden_axioms",
    "estimate_choice_probabilities",
    "check_independence_irrelevant_alternatives",
    "fit_luce_model",
    "fit_rum",
    "check_iia",
    # P1: RUM Consistency (Smeulders et al. 2021)
    "test_rum_consistency",
    "compute_distance_to_rum",
    "fit_rum_distribution",
    "check_rum_consistency",
    # Limited attention (Ch 14)
    "test_attention_rationality",
    "estimate_consideration_sets",
    "compute_salience_weights",
    "test_attention_filter",
    "identify_attention",
    # P0: WARP-LA (Masatlioglu et al. 2012)
    "test_warp_la",
    "recover_preference_with_attention",
    "validate_attention_filter_consistency",
    "check_warp_la",
    # P0: Random Attention Model (Cattaneo et al. 2020)
    "fit_random_attention_model",
    "test_ram_consistency",
    "estimate_attention_probabilities",
    "compute_attention_bounds",
    # Production theory (Ch 15)
    "test_profit_maximization",
    "check_cost_minimization",
    "estimate_returns_to_scale",
    "compute_technical_efficiency",
    "check_production_garp",
    # ==========================================================================
    # PHASE 2 EXTENSIONS - Quick-Win Diagnostics
    # ==========================================================================
    # Result types
    "RegularityResult",
    "RegularityViolation",
    "AttentionOverloadResult",
    "SwapsIndexResult",
    "ObservationContributionResult",
    "StatusQuoBiasResult",
    # Functions - Regularity test
    "test_regularity",
    # Functions - Attention overload & status quo bias
    "test_attention_overload",
    "test_status_quo_bias",
    # Functions - Swaps index & observation contributions
    "compute_swaps_index",
    "compute_observation_contributions",
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
