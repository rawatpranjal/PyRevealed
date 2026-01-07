"""Result dataclasses for behavioral signal analysis.

This module provides result containers for various behavioral analysis algorithms.

Tech-Friendly Names (Primary):
    - ConsistencyResult: Behavioral consistency check result
    - IntegrityResult: Behavioral integrity/noise score result
    - ConfusionResult: Confusion/exploitability metric result
    - LatentValueResult: Latent preference value extraction result
    - PreferenceAnchorResult: Preference anchor (ideal point) result
    - FeatureIndependenceResult: Feature independence/separability result

Economics Names (Deprecated Aliases):
    - GARPResult -> ConsistencyResult
    - AEIResult -> IntegrityResult
    - MPIResult -> ConfusionResult
    - UtilityRecoveryResult -> LatentValueResult
    - IdealPointResult -> PreferenceAnchorResult
    - SeparabilityResult -> FeatureIndependenceResult
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from pyrevealed.core.types import Cycle


@dataclass(frozen=True)
class GARPResult:
    """
    Result of GARP (Generalized Axiom of Revealed Preference) consistency test.

    GARP is satisfied when there are no cycles in the revealed preference
    relation that include at least one strict preference. A violation indicates
    the consumer made inconsistent choices.

    Attributes:
        is_consistent: True if data satisfies GARP (no violations found)
        violations: List of violation cycles (tuples of observation indices)
        direct_revealed_preference: T x T boolean matrix R where R[i,j] = True
            iff bundle i is directly revealed preferred to bundle j
            (i.e., p_i @ x_i >= p_i @ x_j)
        transitive_closure: T x T boolean matrix R* (transitive closure of R)
        strict_revealed_preference: T x T boolean matrix P where P[i,j] = True
            iff bundle i is strictly revealed preferred to bundle j
            (i.e., p_i @ x_i > p_i @ x_j)
        computation_time_ms: Time taken to compute result in milliseconds
    """

    is_consistent: bool
    violations: list[Cycle]
    direct_revealed_preference: NDArray[np.bool_]
    transitive_closure: NDArray[np.bool_]
    strict_revealed_preference: NDArray[np.bool_]
    computation_time_ms: float

    @property
    def num_violations(self) -> int:
        """Number of violation cycles found."""
        return len(self.violations)


@dataclass(frozen=True)
class AEIResult:
    """
    Result of Afriat Efficiency Index computation.

    The AEI measures how close consumer behavior is to perfect rationality.
    It is defined as: sup{e in [0,1] : <R_e, P_e> is acyclic}
    where R_e is the revealed preference relation with budgets deflated by e.

    An AEI of 1.0 means perfectly consistent behavior.
    An AEI of 0.5 means the consumer wastes ~50% of their budget on
    inconsistent choices.

    Attributes:
        efficiency_index: The computed AEI score in [0, 1]
        is_perfectly_consistent: True if AEI = 1.0 (data satisfies GARP)
        garp_result_at_threshold: GARP result at the efficiency threshold
        binary_search_iterations: Number of iterations in binary search
        tolerance: Convergence tolerance used
        computation_time_ms: Time taken in milliseconds
    """

    efficiency_index: float
    is_perfectly_consistent: bool
    garp_result_at_threshold: GARPResult
    binary_search_iterations: int
    tolerance: float
    computation_time_ms: float

    @property
    def waste_fraction(self) -> float:
        """Fraction of budget wasted on inconsistent choices (1 - AEI)."""
        return 1.0 - self.efficiency_index


@dataclass(frozen=True)
class MPIResult:
    """
    Result of Money Pump Index computation.

    The MPI measures the percentage of total expenditure that could be
    "pumped" from a consumer exhibiting cyclic preferences by an arbitrager.

    For a cycle k1 -> k2 -> ... -> kn -> k1:
    MPI = sum(p_ki @ (x_ki - x_{ki+1})) / sum(p_ki @ x_ki)

    Attributes:
        mpi_value: Maximum MPI across all violation cycles (0 if consistent)
        worst_cycle: The cycle with highest MPI (None if consistent)
        cycle_costs: List of (cycle, mpi) pairs for all violation cycles
        total_expenditure: Sum of all expenditures in the session
        computation_time_ms: Time taken in milliseconds
    """

    mpi_value: float
    worst_cycle: Cycle | None
    cycle_costs: list[tuple[Cycle, float]]
    total_expenditure: float
    computation_time_ms: float

    @property
    def is_consistent(self) -> bool:
        """True if no money pump exists (MPI = 0)."""
        return self.mpi_value == 0.0

    @property
    def num_cycles(self) -> int:
        """Number of violation cycles found."""
        return len(self.cycle_costs)


@dataclass(frozen=True)
class UtilityRecoveryResult:
    """
    Result of utility recovery via linear programming (Afriat's inequalities).

    If the data satisfies GARP, we can recover utility values U_k and
    Lagrange multipliers (marginal utility of money) lambda_k such that:
    U_k <= U_l + lambda_l * p_l @ (x_k - x_l) for all k, l

    The recovered utility function is piecewise linear and concave.

    Attributes:
        success: True if LP found a feasible solution
        utility_values: Array of U_k values (utility at each observation)
        lagrange_multipliers: Array of lambda_k values (marginal utility of money)
        lp_status: Status message from the LP solver
        residuals: Matrix of Afriat inequality residuals (for verification)
        computation_time_ms: Time taken in milliseconds
    """

    success: bool
    utility_values: NDArray[np.float64] | None
    lagrange_multipliers: NDArray[np.float64] | None
    lp_status: str
    residuals: NDArray[np.float64] | None
    computation_time_ms: float

    @property
    def mean_marginal_utility(self) -> float | None:
        """Average marginal utility of money across observations."""
        if self.lagrange_multipliers is None:
            return None
        return float(np.mean(self.lagrange_multipliers))


@dataclass(frozen=True)
class RiskProfileResult:
    """
    Result of risk profile analysis from choices under uncertainty.

    Classifies decision-makers as risk-seeking, risk-neutral, or risk-averse
    based on their revealed preferences between safe and risky options.

    Attributes:
        risk_aversion_coefficient: Arrow-Pratt coefficient ρ
            - ρ > 0: risk averse (prefers certainty)
            - ρ ≈ 0: risk neutral (maximizes expected value)
            - ρ < 0: risk seeking (prefers gambles)
        risk_category: Classification string: "risk_seeking" | "risk_neutral" | "risk_averse"
        certainty_equivalents: Array of CEs for each lottery (amount of certain money
            equivalent to the risky option for this decision-maker)
        utility_curvature: Estimated curvature of utility function
        consistency_score: How well the CRRA model fits the choices (0-1)
        num_consistent_choices: Number of choices consistent with estimated ρ
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


@dataclass(frozen=True)
class IdealPointResult:
    """
    Result of ideal point estimation in feature space.

    The ideal point model assumes the user prefers items closer to their
    ideal location in the feature space (Euclidean preferences).

    Attributes:
        ideal_point: D-dimensional vector representing user's ideal location
        is_euclidean_rational: True if all choices are consistent with some ideal point
        violations: List of (choice_set_idx, unchosen_item_idx) pairs where the
            unchosen item was actually closer to the estimated ideal point
        num_violations: Number of choices inconsistent with Euclidean preferences
        explained_variance: Fraction of choice variance explained by ideal point model
        mean_distance_to_chosen: Average distance from ideal point to chosen items
        computation_time_ms: Time taken in milliseconds
    """

    ideal_point: NDArray[np.float64]
    is_euclidean_rational: bool
    violations: list[tuple[int, int]]
    num_violations: int
    explained_variance: float
    mean_distance_to_chosen: float
    computation_time_ms: float

    @property
    def num_dimensions(self) -> int:
        """Number of feature dimensions D."""
        return len(self.ideal_point)

    @property
    def violation_rate(self) -> float:
        """Fraction of choices that violate Euclidean preferences."""
        if self.num_violations == 0:
            return 0.0
        total = len(self.violations) + self.num_violations  # Approximation
        return self.num_violations / max(total, 1)


@dataclass(frozen=True)
class SeparabilityResult:
    """
    Result of separability test for groups of goods.

    Tests whether utility can be decomposed as U(x_A, x_B) = V(u_A(x_A), u_B(x_B)),
    meaning the goods can be priced independently.

    Attributes:
        is_separable: True if groups can be treated independently
        group_a_indices: Indices of goods in Group A
        group_b_indices: Indices of goods in Group B
        cross_effect_strength: Measure of how much Group A affects Group B demand
            (0 = fully independent, 1 = fully dependent)
        within_group_a_consistency: GARP consistency score within Group A
        within_group_b_consistency: GARP consistency score within Group B
        recommendation: Strategy recommendation string
        computation_time_ms: Time taken in milliseconds
    """

    is_separable: bool
    group_a_indices: list[int]
    group_b_indices: list[int]
    cross_effect_strength: float
    within_group_a_consistency: float
    within_group_b_consistency: float
    recommendation: str
    computation_time_ms: float

    @property
    def can_price_independently(self) -> bool:
        """True if groups can be priced without considering cross-effects."""
        return self.is_separable and self.cross_effect_strength < 0.1


# =============================================================================
# TECH-FRIENDLY ALIASES (Primary names)
# =============================================================================

# ConsistencyResult: Result of behavioral consistency validation
ConsistencyResult = GARPResult
"""
Tech-friendly alias for GARPResult.

Use this to check if user behavior is internally consistent.
Consistent behavior = not a bot, single user account.
"""

# IntegrityResult: Result of integrity/noise score computation
IntegrityResult = AEIResult
"""
Tech-friendly alias for AEIResult.

The integrity score (0-1) indicates data quality:
- 1.0 = Perfect signal, fully consistent user
- 0.5 = Noisy signal, possible bot or confused user
- <0.5 = Very noisy, likely bot or shared account
"""

# ConfusionResult: Result of confusion/exploitability metric
ConfusionResult = MPIResult
"""
Tech-friendly alias for MPIResult.

The confusion score indicates how exploitable the user's decisions are.
High confusion = bad UX causing irrational choices.
"""

# LatentValueResult: Result of latent preference extraction
LatentValueResult = UtilityRecoveryResult
"""
Tech-friendly alias for UtilityRecoveryResult.

Contains extracted latent preference values that can be used as
features for ML models or for counterfactual simulations.
"""

# PreferenceAnchorResult: Result of preference anchor (ideal point) estimation
PreferenceAnchorResult = IdealPointResult
"""
Tech-friendly alias for IdealPointResult.

The preference anchor is the user's ideal location in feature space.
Useful for recommendation explainability and personalization.
"""

# FeatureIndependenceResult: Result of feature independence test
FeatureIndependenceResult = SeparabilityResult
"""
Tech-friendly alias for SeparabilityResult.

Tests whether feature groups (e.g., product categories) can be
priced/optimized independently without cross-effects.
"""


# =============================================================================
# NEW ALGORITHMS - RESULT DATACLASSES
# =============================================================================


@dataclass(frozen=True)
class BronarsPowerResult:
    """
    Result of Bronars' Power Index computation.

    Bronars' Power Index measures the statistical power of the GARP test.
    It answers: "If this user passed GARP, is that meaningful?"

    The test simulates random behavior on the observed budget constraints
    and checks what fraction of random behaviors violate GARP. High power
    means passing GARP is statistically significant.

    Attributes:
        power_index: Fraction of random simulations that violate GARP (0-1)
            - 1.0 = All random behaviors violate (high power, test is informative)
            - 0.5 = Half violate (moderate power)
            - 0.0 = No randoms violate (no power, test uninformative)
        is_significant: True if power_index > 0.5 (test has discriminatory power)
        n_simulations: Number of random simulations performed
        n_violations: Number of simulations that violated GARP
        mean_integrity_random: Average integrity score (AEI) across random simulations
        simulation_integrity_values: Array of AEI values for each simulation
        computation_time_ms: Time taken in milliseconds
    """

    power_index: float
    is_significant: bool
    n_simulations: int
    n_violations: int
    mean_integrity_random: float
    simulation_integrity_values: NDArray[np.float64] | None
    computation_time_ms: float

    @property
    def violation_rate(self) -> float:
        """Fraction of random simulations that violated GARP."""
        return self.n_violations / self.n_simulations if self.n_simulations > 0 else 0.0

    @property
    def pass_rate_random(self) -> float:
        """Fraction of random simulations that passed GARP."""
        return 1.0 - self.violation_rate


@dataclass(frozen=True)
class HARPResult:
    """
    Result of HARP (Homothetic Axiom of Revealed Preference) test.

    HARP tests whether preferences are homothetic - demand scales
    proportionally with income. This is a stronger condition than GARP.

    For homothetic preferences, the product of expenditure ratios around
    any cycle must be <= 1. Violations indicate non-homothetic behavior.

    Attributes:
        is_consistent: True if data satisfies HARP (homothetic preferences)
        violations: List of (cycle, product_ratio) for violating cycles
        max_cycle_product: Maximum product of ratios around any cycle
            (1.0 if consistent, >1.0 if violations exist)
        expenditure_ratio_matrix: T x T matrix R[i,j] = (p_i @ x_i) / (p_i @ x_j)
        log_ratio_matrix: T x T matrix of log expenditure ratios
        garp_result: GARP result for comparison (GARP is weaker than HARP)
        computation_time_ms: Time taken in milliseconds
    """

    is_consistent: bool
    violations: list[tuple[Cycle, float]]
    max_cycle_product: float
    expenditure_ratio_matrix: NDArray[np.float64]
    log_ratio_matrix: NDArray[np.float64]
    garp_result: GARPResult
    computation_time_ms: float

    @property
    def is_homothetic(self) -> bool:
        """True if preferences are homothetic (HARP satisfied)."""
        return self.is_consistent

    @property
    def num_violations(self) -> int:
        """Number of violation cycles found."""
        return len(self.violations)

    @property
    def max_violation_severity(self) -> float:
        """Maximum deviation from homotheticity (max_cycle_product - 1)."""
        return max(0.0, self.max_cycle_product - 1.0)


@dataclass(frozen=True)
class QuasilinearityResult:
    """
    Result of quasilinearity (cyclic monotonicity) test.

    Quasilinear preferences have no income effects - money has constant
    marginal utility. This is tested via cyclic monotonicity:
    For any cycle, the sum of price-weighted quantity changes must be >= 0.

    Attributes:
        is_quasilinear: True if data satisfies cyclic monotonicity
        violations: List of cycles that violate cyclic monotonicity
        worst_violation_magnitude: Largest violation (most negative cycle sum)
        cycle_sums: Dict mapping cycle tuples to their sums
        num_cycles_tested: Total number of cycles examined
        computation_time_ms: Time taken in milliseconds
    """

    is_quasilinear: bool
    violations: list[Cycle]
    worst_violation_magnitude: float
    cycle_sums: dict[Cycle, float]
    num_cycles_tested: int
    computation_time_ms: float

    @property
    def num_violations(self) -> int:
        """Number of cycles that violate cyclic monotonicity."""
        return len(self.violations)

    @property
    def has_income_effects(self) -> bool:
        """True if income effects detected (quasilinearity violated)."""
        return not self.is_quasilinear


@dataclass(frozen=True)
class GrossSubstitutesResult:
    """
    Result of gross substitutes test between two goods.

    Tests whether two goods are substitutes (price of A up → demand for B up)
    or complements (price of A up → demand for B down).

    Attributes:
        are_substitutes: True if goods appear to be gross substitutes
        are_complements: True if goods appear to be complements
        relationship: Classification: "substitutes", "complements", "independent", "inconclusive"
        supporting_pairs: List of (obs_i, obs_j) pairs supporting the relationship
        violating_pairs: List of (obs_i, obs_j) pairs violating the relationship
        confidence_score: Fraction of informative pairs supporting the relationship (0-1)
        good_g_index: Index of first good
        good_h_index: Index of second good
        computation_time_ms: Time taken in milliseconds
    """

    are_substitutes: bool
    are_complements: bool
    relationship: str
    supporting_pairs: list[tuple[int, int]]
    violating_pairs: list[tuple[int, int]]
    confidence_score: float
    good_g_index: int
    good_h_index: int
    computation_time_ms: float

    @property
    def is_conclusive(self) -> bool:
        """True if the test gave a conclusive result."""
        return self.relationship != "inconclusive"

    @property
    def num_informative_pairs(self) -> int:
        """Number of observation pairs that informed the relationship."""
        return len(self.supporting_pairs) + len(self.violating_pairs)


@dataclass(frozen=True)
class SubstitutionMatrixResult:
    """
    Result of pairwise substitution analysis for all goods.

    Contains an N x N matrix of relationships between all pairs of goods.

    Attributes:
        relationship_matrix: N x N matrix where entry [g,h] is the relationship
            between goods g and h ("substitutes", "complements", "independent", etc.)
        confidence_matrix: N x N matrix of confidence scores for each relationship
        num_goods: Number of goods N
        computation_time_ms: Time taken in milliseconds
    """

    relationship_matrix: NDArray[np.object_]
    confidence_matrix: NDArray[np.float64]
    num_goods: int
    computation_time_ms: float

    @property
    def substitute_pairs(self) -> list[tuple[int, int]]:
        """List of (g, h) pairs that are substitutes."""
        pairs = []
        for g in range(self.num_goods):
            for h in range(g + 1, self.num_goods):
                if self.relationship_matrix[g, h] == "substitutes":
                    pairs.append((g, h))
        return pairs

    @property
    def complement_pairs(self) -> list[tuple[int, int]]:
        """List of (g, h) pairs that are complements."""
        pairs = []
        for g in range(self.num_goods):
            for h in range(g + 1, self.num_goods):
                if self.relationship_matrix[g, h] == "complements":
                    pairs.append((g, h))
        return pairs


@dataclass(frozen=True)
class VEIResult:
    """
    Result of Varian's Efficiency Index (per-observation efficiency) computation.

    Unlike AEI which gives one global efficiency score, VEI provides
    individual efficiency scores for each observation. This identifies
    which specific observations are problematic.

    Attributes:
        efficiency_vector: Array of e_i values for each observation (0-1)
        mean_efficiency: Average efficiency across observations
        min_efficiency: Lowest efficiency (worst observation)
        worst_observation: Index of observation with lowest efficiency
        problematic_observations: Indices where e_i < threshold (default 0.9)
        total_inefficiency: Sum of (1 - e_i) across all observations
        optimization_success: True if optimization converged
        optimization_status: Status message from optimizer
        computation_time_ms: Time taken in milliseconds
    """

    efficiency_vector: NDArray[np.float64]
    mean_efficiency: float
    min_efficiency: float
    worst_observation: int
    problematic_observations: list[int]
    total_inefficiency: float
    optimization_success: bool
    optimization_status: str
    computation_time_ms: float

    @property
    def num_observations(self) -> int:
        """Number of observations."""
        return len(self.efficiency_vector)

    @property
    def is_perfectly_consistent(self) -> bool:
        """True if all observations have efficiency = 1."""
        return self.min_efficiency >= 1.0 - 1e-6

    @property
    def num_problematic(self) -> int:
        """Number of problematic observations."""
        return len(self.problematic_observations)


# =============================================================================
# TECH-FRIENDLY ALIASES FOR NEW RESULTS
# =============================================================================

# TestPowerResult: Result of statistical test power computation
TestPowerResult = BronarsPowerResult
"""
Tech-friendly alias for BronarsPowerResult.

Use this to validate that consistency test results are statistically meaningful.
"""

# ProportionalScalingResult: Result of proportional scaling (homotheticity) test
ProportionalScalingResult = HARPResult
"""
Tech-friendly alias for HARPResult.

Tests if user preferences scale proportionally with budget.
"""

# IncomeInvarianceResult: Result of income invariance (quasilinearity) test
IncomeInvarianceResult = QuasilinearityResult
"""
Tech-friendly alias for QuasilinearityResult.

Tests if user behavior is invariant to income changes.
"""

# CrossPriceResult: Result of cross-price effect analysis
CrossPriceResult = GrossSubstitutesResult
"""
Tech-friendly alias for GrossSubstitutesResult.

Analyzes how price changes in one good affect demand for another.
"""

# GranularIntegrityResult: Result of per-observation integrity computation
GranularIntegrityResult = VEIResult
"""
Tech-friendly alias for VEIResult.

Provides per-observation integrity scores instead of one global score.
"""


# =============================================================================
# 2024 SURVEY ALGORITHMS - RESULT DATACLASSES
# =============================================================================


@dataclass(frozen=True)
class DifferentiableResult:
    """
    Result of differentiable rationality (smooth preferences) test.

    Tests whether preferences are smooth/differentiable, which requires:
    1. SARP (Strict Axiom): No indifferent preference cycles
    2. Price-Quantity Uniqueness: Different prices imply different quantities

    Differentiable preferences allow for comparative statics and demand
    function derivatives. Violations indicate piecewise-linear preferences.

    Based on Chiappori & Rochet (1987).

    Attributes:
        is_differentiable: True if both SARP and uniqueness conditions hold
        satisfies_sarp: True if no indifferent cycles exist
        satisfies_uniqueness: True if price differences imply quantity differences
        sarp_violations: List of indifferent preference cycles
        uniqueness_violations: List of (t, s) pairs where p^t != p^s but x^t = x^s
        direct_revealed_preference: T x T boolean matrix R
        transitive_closure: T x T boolean matrix R*
        computation_time_ms: Time taken in milliseconds
    """

    is_differentiable: bool
    satisfies_sarp: bool
    satisfies_uniqueness: bool
    sarp_violations: list[Cycle]
    uniqueness_violations: list[tuple[int, int]]
    direct_revealed_preference: NDArray[np.bool_]
    transitive_closure: NDArray[np.bool_]
    computation_time_ms: float

    @property
    def num_sarp_violations(self) -> int:
        """Number of SARP (indifferent cycle) violations."""
        return len(self.sarp_violations)

    @property
    def num_uniqueness_violations(self) -> int:
        """Number of price-quantity uniqueness violations."""
        return len(self.uniqueness_violations)

    @property
    def is_piecewise_linear(self) -> bool:
        """True if preferences appear piecewise-linear (not smooth)."""
        return not self.is_differentiable


@dataclass(frozen=True)
class AcyclicalPResult:
    """
    Result of Acyclical P test (strict preference acyclicity).

    Tests whether the strict revealed preference relation P is acyclic.
    This is MORE LENIENT than GARP - it passes when GARP might fail,
    because it ignores weak preference violations.

    A consumer passes Acyclical P if there are no strict preference cycles,
    even if weak preference cycles exist. This indicates "approximately
    rational" behavior where apparent violations are due to indifference.

    Based on Dziewulski (2023).

    Attributes:
        is_consistent: True if no strict preference cycles exist
        violations: List of strict preference cycles found
        strict_preference_matrix: T x T boolean matrix P where P[t,s] = True
            iff p^t @ x^s < p^t @ x^t (bundle s was strictly cheaper)
        transitive_closure: T x T boolean matrix P* (transitive closure of P)
        num_strict_preferences: Count of strict preference edges
        garp_consistent: True if also passes GARP (for comparison)
        computation_time_ms: Time taken in milliseconds
    """

    is_consistent: bool
    violations: list[Cycle]
    strict_preference_matrix: NDArray[np.bool_]
    transitive_closure: NDArray[np.bool_]
    num_strict_preferences: int
    garp_consistent: bool
    computation_time_ms: float

    @property
    def num_violations(self) -> int:
        """Number of strict preference cycles found."""
        return len(self.violations)

    @property
    def is_approximately_rational(self) -> bool:
        """True if behavior is approximately rational (passes Acyclical P)."""
        return self.is_consistent

    @property
    def strict_violations_only(self) -> bool:
        """True if GARP fails but Acyclical P passes (weak violations only)."""
        return self.is_consistent and not self.garp_consistent


@dataclass(frozen=True)
class GAPPResult:
    """
    Result of GAPP (Generalized Axiom of Price Preference) test.

    GAPP tests consistency of revealed PRICE preferences, which is the
    dual perspective to GARP's quantity preferences. Instead of asking
    "does the consumer prefer bundle A to bundle B?", GAPP asks
    "does the consumer prefer price vector A to price vector B?"

    Price s is revealed preferred to price t if the bundle bought at t
    would have been cheaper under prices s. Consistent price preferences
    indicate the consumer is "shopping around" rationally.

    Based on Deb et al. (2022).

    Attributes:
        is_consistent: True if no price preference cycles exist
        violations: List of (s, t) pairs where GAPP is violated
        price_preference_matrix: T x T matrix R_p where R_p[s,t] = True
            iff p^s @ x^t <= p^t @ x^t (price s is preferred to t)
        strict_price_preference: T x T matrix P_p where P_p[s,t] = True
            iff p^s @ x^t < p^t @ x^t (price s strictly preferred to t)
        transitive_closure: T x T matrix R_p* (transitive closure of R_p)
        num_price_preferences: Count of price preference relations
        garp_consistent: True if also passes GARP (for comparison)
        computation_time_ms: Time taken in milliseconds
    """

    is_consistent: bool
    violations: list[tuple[int, int]]
    price_preference_matrix: NDArray[np.bool_]
    strict_price_preference: NDArray[np.bool_]
    transitive_closure: NDArray[np.bool_]
    num_price_preferences: int
    garp_consistent: bool
    computation_time_ms: float

    @property
    def num_violations(self) -> int:
        """Number of GAPP violations found."""
        return len(self.violations)

    @property
    def prefers_lower_prices(self) -> bool:
        """True if price preferences are consistent (GAPP satisfied)."""
        return self.is_consistent


# Tech-friendly aliases for 2024 survey algorithms

SmoothPreferencesResult = DifferentiableResult
"""
Tech-friendly alias for DifferentiableResult.

Tests if user preferences are smooth (differentiable), enabling
demand function derivatives for price sensitivity analysis.
"""

StrictConsistencyResult = AcyclicalPResult
"""
Tech-friendly alias for AcyclicalPResult.

Tests strict consistency only - more lenient than full consistency check.
Useful for identifying "approximately rational" behavior.
"""

PricePreferencesResult = GAPPResult
"""
Tech-friendly alias for GAPPResult.

Tests if the user has consistent price preferences - do they
prefer situations where their desired items are cheaper?
"""


@dataclass(frozen=True)
class LancasterResult:
    """
    Valuation report from Lancaster characteristics model.

    The Lancaster model transforms product-space data into characteristics-space,
    computing shadow prices (implicit valuations) for underlying product attributes.
    This result provides business insights from the shadow price analysis.

    Attributes:
        mean_shadow_prices: K-length array of average shadow price per characteristic
        shadow_price_std: K-length array of shadow price standard deviations
        shadow_price_cv: K-length array of coefficient of variation (volatility)
        total_spend_on_characteristics: K-length array of total spend per characteristic
        spend_shares: K-length array of spend share per characteristic (sums to 1)
        mean_nnls_residual: Average NNLS fit residual (lower = better fit)
        max_nnls_residual: Maximum NNLS residual (flags problematic observations)
        problematic_observations: List of observation indices with high residuals
        attribute_matrix_rank: Rank of A matrix (for diagnostics)
        is_well_specified: True if A has full column rank (K <= N and rank = K)
        characteristic_names: Optional names for characteristics (from metadata)
        computation_time_ms: Time taken in milliseconds
    """

    mean_shadow_prices: NDArray[np.float64]
    shadow_price_std: NDArray[np.float64]
    shadow_price_cv: NDArray[np.float64]
    total_spend_on_characteristics: NDArray[np.float64]
    spend_shares: NDArray[np.float64]
    mean_nnls_residual: float
    max_nnls_residual: float
    problematic_observations: list[int]
    attribute_matrix_rank: int
    is_well_specified: bool
    characteristic_names: list[str] | None
    computation_time_ms: float

    @property
    def num_characteristics(self) -> int:
        """Number of characteristics K."""
        return len(self.mean_shadow_prices)

    @property
    def most_valued_characteristic(self) -> int:
        """Index of characteristic with highest mean shadow price."""
        return int(np.argmax(self.mean_shadow_prices))

    @property
    def most_volatile_characteristic(self) -> int:
        """Index of characteristic with highest price volatility (CV)."""
        return int(np.argmax(self.shadow_price_cv))


# CharacteristicsValuationResult: Tech-friendly alias for LancasterResult
CharacteristicsValuationResult = LancasterResult
"""
Tech-friendly alias for LancasterResult.

Contains insights from characteristics-space analysis of user behavior,
including shadow prices (implicit valuations) for product attributes.
"""
