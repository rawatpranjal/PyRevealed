"""Result dataclasses for revealed preference analysis."""

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
