from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pyrevealed.core.types import Cycle
from pyrevealed.core.mixins import ResultSummaryMixin
from pyrevealed.core.display import ResultDisplayMixin, ResultPlotMixin

__all__ = [
    "IntegrabilityResult",
    "WelfareResult",
    "AdditivityResult",
    "CompensatedDemandResult",
    "GeneralMetricResult",
    "StochasticChoiceResult",
    "AttentionResult",
    "ProductionGARPResult",
    "SlutskyConditionsResult",
    "WelfareChangeResult",
    "AdditiveUtilityResult",
    "HicksianDemandResult",
    "MetricPreferencesResult",
    "RandomUtilityResult",
    "ConsiderationSetResult",
    "FirmBehaviorResult",
]


@dataclass(frozen=True)
class IntegrabilityResult(ResultDisplayMixin, ResultPlotMixin):
    """
    Result of integrability conditions test (Chapter 6.4-6.5).

    Tests whether a demand function is integrable to a utility function.
    Based on the Slutsky matrix conditions: symmetry and negative semi-definiteness.

    Attributes:
        is_symmetric: True if Slutsky matrix is symmetric (S[i,j] = S[j,i])
        is_negative_semidefinite: True if all eigenvalues of S are <= 0
        is_integrable: True if both conditions hold (demand is integrable)
        slutsky_matrix: Estimated N x N Slutsky matrix
        eigenvalues: Eigenvalues of the Slutsky matrix (should all be <= 0)
        symmetry_violations: List of (i, j) pairs where S[i,j] != S[j,i]
        max_eigenvalue: Largest eigenvalue (should be <= 0 for NSD)
        symmetry_deviation: Max deviation from symmetry
        computation_time_ms: Time taken in milliseconds
    """

    is_symmetric: bool
    is_negative_semidefinite: bool
    is_integrable: bool
    slutsky_matrix: NDArray[np.float64]
    eigenvalues: NDArray[np.float64]
    symmetry_violations: list[tuple[int, int]]
    max_eigenvalue: float
    symmetry_deviation: float
    computation_time_ms: float

    @property
    def num_goods(self) -> int:
        """Number of goods N."""
        return self.slutsky_matrix.shape[0]

    @property
    def num_symmetry_violations(self) -> int:
        """Number of pairs violating symmetry."""
        return len(self.symmetry_violations)

    @property
    def passes_slutsky_conditions(self) -> bool:
        """True if both Slutsky conditions hold."""
        return self.is_integrable

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns 1.0 if integrable, 0.5 if one condition fails, 0.0 if both fail.
        """
        if self.is_integrable:
            return 1.0
        elif self.is_symmetric or self.is_negative_semidefinite:
            return 0.5
        return 0.0

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("INTEGRABILITY TEST REPORT")]

        # Status
        status = m._format_status(self.is_integrable, "INTEGRABLE", "NOT INTEGRABLE")
        lines.append(f"\nStatus: {status}")

        # Metrics
        lines.append(m._format_section("Slutsky Conditions"))
        lines.append(m._format_metric("Is Integrable", self.is_integrable))
        lines.append(m._format_metric("Symmetric", self.is_symmetric))
        lines.append(m._format_metric("Negative Semi-Definite", self.is_negative_semidefinite))
        lines.append(m._format_metric("Symmetry Violations", self.num_symmetry_violations))
        lines.append(m._format_metric("Max Eigenvalue", self.max_eigenvalue))
        lines.append(m._format_metric("Symmetry Deviation", self.symmetry_deviation))
        lines.append(m._format_metric("Number of Goods", self.num_goods))

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        if self.is_integrable:
            lines.append("  Demand function is integrable to a utility function.")
            lines.append("  Both Slutsky symmetry and NSD conditions satisfied.")
        else:
            if not self.is_symmetric:
                lines.append("  Slutsky symmetry violated - cross-price effects asymmetric.")
            if not self.is_negative_semidefinite:
                lines.append(f"  Not NSD - max eigenvalue {self.max_eigenvalue:.4f} > 0.")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "is_integrable": self.is_integrable,
            "is_symmetric": self.is_symmetric,
            "is_negative_semidefinite": self.is_negative_semidefinite,
            "num_goods": self.num_goods,
            "max_eigenvalue": self.max_eigenvalue,
            "symmetry_deviation": self.symmetry_deviation,
            "num_symmetry_violations": self.num_symmetry_violations,
            "eigenvalues": self.eigenvalues.tolist(),
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        if self.is_integrable:
            return f"IntegrabilityResult(integrable, n={self.num_goods})"
        return f"IntegrabilityResult(sym={self.is_symmetric}, nsd={self.is_negative_semidefinite})"


@dataclass(frozen=True)
class WelfareResult(ResultDisplayMixin, ResultPlotMixin):
    """
    Result of welfare analysis (Chapter 7.3-7.4).

    Measures consumer welfare changes from policy or price changes using
    compensating variation (CV) and equivalent variation (EV).

    Attributes:
        compensating_variation: Money needed to restore original utility after change
        equivalent_variation: Money equivalent to the utility change
        welfare_direction: "improved", "worsened", or "ambiguous"
        baseline_utility: Estimated utility at baseline prices/quantities
        policy_utility: Estimated utility at policy prices/quantities
        baseline_expenditure: Total expenditure at baseline
        policy_expenditure: Total expenditure under policy
        hicksian_surplus: Consumer surplus measure
        computation_time_ms: Time taken in milliseconds
    """

    compensating_variation: float
    equivalent_variation: float
    welfare_direction: str
    baseline_utility: float
    policy_utility: float
    baseline_expenditure: float
    policy_expenditure: float
    hicksian_surplus: float
    computation_time_ms: float

    @property
    def welfare_improved(self) -> bool:
        """True if welfare improved under policy."""
        return self.welfare_direction == "improved"

    @property
    def welfare_worsened(self) -> bool:
        """True if welfare worsened under policy."""
        return self.welfare_direction == "worsened"

    @property
    def mean_variation(self) -> float:
        """Average of CV and EV (common approximation)."""
        return (self.compensating_variation + self.equivalent_variation) / 2

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns 1.0 if welfare improved, 0.5 if ambiguous, 0.0 if worsened.
        """
        if self.welfare_improved:
            return 1.0
        elif self.welfare_worsened:
            return 0.0
        return 0.5

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("WELFARE ANALYSIS REPORT")]

        # Status
        lines.append(f"\nWelfare Direction: {self.welfare_direction.upper()}")

        # Welfare measures
        lines.append(m._format_section("Welfare Measures"))
        lines.append(m._format_metric("Compensating Variation (CV)", self.compensating_variation))
        lines.append(m._format_metric("Equivalent Variation (EV)", self.equivalent_variation))
        lines.append(m._format_metric("Mean Variation", self.mean_variation))
        lines.append(m._format_metric("Hicksian Surplus", self.hicksian_surplus))

        # Utility comparison
        lines.append(m._format_section("Utility Comparison"))
        lines.append(m._format_metric("Baseline Utility", self.baseline_utility))
        lines.append(m._format_metric("Policy Utility", self.policy_utility))
        lines.append(m._format_metric("Baseline Expenditure", self.baseline_expenditure))
        lines.append(m._format_metric("Policy Expenditure", self.policy_expenditure))

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        if self.welfare_improved:
            lines.append("  Consumer welfare improved under the policy.")
            lines.append(f"  Equivalent to receiving ${abs(self.equivalent_variation):.2f}.")
        elif self.welfare_worsened:
            lines.append("  Consumer welfare worsened under the policy.")
            lines.append(f"  Would need ${abs(self.compensating_variation):.2f} to restore utility.")
        else:
            lines.append("  Welfare change is ambiguous (CV and EV have different signs).")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "welfare_direction": self.welfare_direction,
            "compensating_variation": self.compensating_variation,
            "equivalent_variation": self.equivalent_variation,
            "mean_variation": self.mean_variation,
            "hicksian_surplus": self.hicksian_surplus,
            "baseline_utility": self.baseline_utility,
            "policy_utility": self.policy_utility,
            "baseline_expenditure": self.baseline_expenditure,
            "policy_expenditure": self.policy_expenditure,
            "welfare_improved": self.welfare_improved,
            "welfare_worsened": self.welfare_worsened,
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        return f"WelfareResult({self.welfare_direction}, cv={self.compensating_variation:.2f})"


@dataclass(frozen=True)
class AdditivityResult(ResultDisplayMixin, ResultPlotMixin):
    """
    Result of additive separability test (Chapter 9.3).

    Tests whether utility has the additive form U(x) = Σ u_i(x_i),
    which is stronger than quasilinearity.

    Attributes:
        is_additive: True if data is consistent with additive utility
        additive_groups: List of good indices that separate additively
        cross_effects_matrix: N x N matrix of cross-price effects (should be diagonal)
        max_cross_effect: Largest off-diagonal cross-price effect
        violations: List of (i, j) pairs showing cross-effects
        num_violations: Number of cross-effect violations
        computation_time_ms: Time taken in milliseconds
    """

    is_additive: bool
    additive_groups: list[set[int]]
    cross_effects_matrix: NDArray[np.float64]
    max_cross_effect: float
    violations: list[tuple[int, int]]
    num_violations: int
    computation_time_ms: float

    @property
    def num_goods(self) -> int:
        """Number of goods N."""
        return self.cross_effects_matrix.shape[0]

    @property
    def num_groups(self) -> int:
        """Number of additively separable groups."""
        return len(self.additive_groups)

    @property
    def is_fully_separable(self) -> bool:
        """True if each good is in its own group (fully additive)."""
        return self.num_groups == self.num_goods

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns 1 - max_cross_effect (lower cross-effects = more additive).
        """
        return max(0.0, 1.0 - min(1.0, self.max_cross_effect))

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("ADDITIVE SEPARABILITY REPORT")]

        # Status
        status = m._format_status(self.is_additive, "ADDITIVE", "NOT ADDITIVE")
        lines.append(f"\nStatus: {status}")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("Is Additive", self.is_additive))
        lines.append(m._format_metric("Fully Separable", self.is_fully_separable))
        lines.append(m._format_metric("Number of Goods", self.num_goods))
        lines.append(m._format_metric("Additive Groups", self.num_groups))
        lines.append(m._format_metric("Max Cross-Effect", self.max_cross_effect))
        lines.append(m._format_metric("Violations", self.num_violations))

        # Show groups
        if self.additive_groups:
            lines.append(m._format_section("Additive Groups"))
            for i, group in enumerate(self.additive_groups):
                lines.append(f"  Group {i}: {sorted(group)}")

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        if self.is_additive:
            lines.append("  Utility is additively separable: U(x) = Σ u_i(x_i).")
            lines.append("  No significant cross-price effects between groups.")
        else:
            lines.append("  Utility is not additively separable.")
            lines.append(f"  {self.num_violations} significant cross-effect(s) detected.")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "is_additive": self.is_additive,
            "is_fully_separable": self.is_fully_separable,
            "num_goods": self.num_goods,
            "num_groups": self.num_groups,
            "max_cross_effect": self.max_cross_effect,
            "num_violations": self.num_violations,
            "additive_groups": [list(g) for g in self.additive_groups],
            "violations": self.violations,
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        if self.is_additive:
            return f"AdditivityResult(additive, groups={self.num_groups})"
        return f"AdditivityResult(not additive, violations={self.num_violations})"


@dataclass(frozen=True)
class CompensatedDemandResult(ResultDisplayMixin, ResultPlotMixin):
    """
    Result of compensated (Hicksian) demand analysis (Chapter 10.3).

    Decomposes price effects into substitution and income effects.

    Attributes:
        substitution_effects: N x N matrix of Hicksian substitution effects
        income_effects: N x N matrix of income effects
        satisfies_compensated_law: True if compensated law of demand holds
        own_price_elasticities: Dict mapping good index to own-price elasticity
        cross_price_elasticities: N x N matrix of cross-price elasticities
        violations: List of (i, j) pairs violating compensated law
        computation_time_ms: Time taken in milliseconds
    """

    substitution_effects: NDArray[np.float64]
    income_effects: NDArray[np.float64]
    satisfies_compensated_law: bool
    own_price_elasticities: dict[int, float]
    cross_price_elasticities: NDArray[np.float64]
    violations: list[tuple[int, int]]
    computation_time_ms: float

    @property
    def num_goods(self) -> int:
        """Number of goods N."""
        return self.substitution_effects.shape[0]

    @property
    def num_violations(self) -> int:
        """Number of compensated law violations."""
        return len(self.violations)

    @property
    def mean_own_elasticity(self) -> float:
        """Average own-price elasticity across goods."""
        if not self.own_price_elasticities:
            return 0.0
        return sum(self.own_price_elasticities.values()) / len(self.own_price_elasticities)

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns 1.0 if compensated law holds, 0.0 otherwise.
        """
        return 1.0 if self.satisfies_compensated_law else 0.0

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("COMPENSATED DEMAND ANALYSIS REPORT")]

        # Status
        status = m._format_status(self.satisfies_compensated_law,
                                  "COMPENSATED LAW SATISFIED", "VIOLATIONS FOUND")
        lines.append(f"\nStatus: {status}")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("Compensated Law", self.satisfies_compensated_law))
        lines.append(m._format_metric("Number of Goods", self.num_goods))
        lines.append(m._format_metric("Violations", self.num_violations))
        lines.append(m._format_metric("Mean Own Elasticity", self.mean_own_elasticity))

        # Elasticities
        if self.own_price_elasticities:
            lines.append(m._format_section("Own-Price Elasticities"))
            for good, elast in list(self.own_price_elasticities.items())[:5]:
                lines.append(f"  Good {good}: {elast:.4f}")
            if len(self.own_price_elasticities) > 5:
                lines.append(f"  ... ({len(self.own_price_elasticities) - 5} more goods)")

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        if self.satisfies_compensated_law:
            lines.append("  Compensated (Hicksian) demand is well-behaved.")
            lines.append("  Substitution effects are negative (law of demand holds).")
        else:
            lines.append(f"  {self.num_violations} Giffen-like good pair(s) detected.")
            lines.append("  Some substitution effects have wrong sign.")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "satisfies_compensated_law": self.satisfies_compensated_law,
            "num_goods": self.num_goods,
            "num_violations": self.num_violations,
            "mean_own_elasticity": self.mean_own_elasticity,
            "own_price_elasticities": self.own_price_elasticities,
            "violations": self.violations,
            "substitution_effects": self.substitution_effects.tolist(),
            "income_effects": self.income_effects.tolist(),
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        if self.satisfies_compensated_law:
            return f"CompensatedDemandResult(law_ok, n={self.num_goods})"
        return f"CompensatedDemandResult({self.num_violations} violations)"


@dataclass(frozen=True)
class GeneralMetricResult(ResultDisplayMixin, ResultPlotMixin):
    """
    Result of general metric space ideal point analysis (Chapter 11.3-11.4).

    Extends Euclidean preferences to arbitrary metric spaces.

    Attributes:
        ideal_point: D-dimensional ideal point in feature space
        metric_type: Type of metric used ("L1", "L2", "Linf", "minkowski")
        metric_params: Parameters for the metric (e.g., p for Minkowski)
        is_rationalizable: True if choices are rationalizable under this metric
        violations: List of (choice_set_idx, unchosen_item_idx) violations
        best_metric: The metric type that best fits the data
        metric_comparison: Dict mapping metric type to violation count
        explained_variance: Fraction of choices explained by the model
        computation_time_ms: Time taken in milliseconds
    """

    ideal_point: NDArray[np.float64]
    metric_type: str
    metric_params: dict[str, float]
    is_rationalizable: bool
    violations: list[tuple[int, int]]
    best_metric: str
    metric_comparison: dict[str, int]
    explained_variance: float
    computation_time_ms: float

    @property
    def num_violations(self) -> int:
        """Number of violations under the chosen metric."""
        return len(self.violations)

    @property
    def num_dimensions(self) -> int:
        """Number of feature dimensions D."""
        return len(self.ideal_point)

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns explained_variance.
        """
        return self.explained_variance

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("GENERAL METRIC PREFERENCES REPORT")]

        # Status
        status = m._format_status(self.is_rationalizable, "RATIONALIZABLE", "VIOLATIONS FOUND")
        lines.append(f"\nStatus: {status}")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("Rationalizable", self.is_rationalizable))
        lines.append(m._format_metric("Metric Type", self.metric_type))
        lines.append(m._format_metric("Best Metric", self.best_metric))
        lines.append(m._format_metric("Explained Variance", self.explained_variance))
        lines.append(m._format_metric("Violations", self.num_violations))
        lines.append(m._format_metric("Dimensions", self.num_dimensions))

        # Ideal point
        lines.append(m._format_section("Ideal Point Coordinates"))
        for i, coord in enumerate(self.ideal_point[:5]):
            lines.append(f"  Dimension {i}: {coord:.4f}")
        if self.num_dimensions > 5:
            lines.append(f"  ... ({self.num_dimensions - 5} more dimensions)")

        # Metric comparison
        if self.metric_comparison:
            lines.append(m._format_section("Metric Comparison (violations)"))
            for metric, viol in self.metric_comparison.items():
                lines.append(f"  {metric}: {viol}")

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        lines.append(f"  Best fit achieved with {self.best_metric} metric.")
        lines.append(f"  Model explains {self.explained_variance*100:.1f}% of choice variance.")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "is_rationalizable": self.is_rationalizable,
            "metric_type": self.metric_type,
            "best_metric": self.best_metric,
            "explained_variance": self.explained_variance,
            "num_violations": self.num_violations,
            "num_dimensions": self.num_dimensions,
            "ideal_point": self.ideal_point.tolist(),
            "metric_params": self.metric_params,
            "metric_comparison": self.metric_comparison,
            "violations": self.violations,
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        return f"GeneralMetricResult({self.best_metric}, var={self.explained_variance:.4f})"


@dataclass(frozen=True)
class StochasticChoiceResult(ResultDisplayMixin, ResultPlotMixin):
    """
    Result of stochastic/random utility model fitting (Chapter 13).

    Fits probabilistic choice models like logit, probit, or Luce model.

    Attributes:
        model_type: Type of model ("logit", "probit", "luce", "rum")
        parameters: Model parameters (e.g., temperature, scale)
        satisfies_iia: True if Independence of Irrelevant Alternatives holds
        choice_probabilities: Array of predicted choice probabilities
        log_likelihood: Log-likelihood of the fitted model
        aic: Akaike Information Criterion
        bic: Bayesian Information Criterion
        regularity_violations: List of observations violating regularity
        computation_time_ms: Time taken in milliseconds
    """

    model_type: str
    parameters: dict[str, float]
    satisfies_iia: bool
    choice_probabilities: NDArray[np.float64]
    log_likelihood: float
    aic: float
    bic: float
    regularity_violations: list[int]
    computation_time_ms: float

    @property
    def num_regularity_violations(self) -> int:
        """Number of regularity axiom violations."""
        return len(self.regularity_violations)

    @property
    def is_random_utility(self) -> bool:
        """True if consistent with random utility maximization."""
        return self.satisfies_iia and len(self.regularity_violations) == 0

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns normalized log-likelihood (scaled by number of observations).
        Uses 1.0 if random utility model holds, else scaled by AIC.
        """
        if self.is_random_utility:
            return 1.0
        # Use inverse of normalized AIC as score
        return max(0.0, 1.0 / (1.0 + abs(self.aic) / 1000))

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("STOCHASTIC CHOICE MODEL REPORT")]

        # Status
        status = m._format_status(self.is_random_utility,
                                  "RANDOM UTILITY MODEL FITS", "RUM VIOLATIONS")
        lines.append(f"\nStatus: {status}")
        lines.append(f"Model Type: {self.model_type}")

        # Metrics
        lines.append(m._format_section("Model Fit"))
        lines.append(m._format_metric("Log-Likelihood", self.log_likelihood))
        lines.append(m._format_metric("AIC", self.aic))
        lines.append(m._format_metric("BIC", self.bic))
        lines.append(m._format_metric("Satisfies IIA", self.satisfies_iia))
        lines.append(m._format_metric("Regularity Violations", self.num_regularity_violations))

        # Parameters
        if self.parameters:
            lines.append(m._format_section("Model Parameters"))
            for param, value in self.parameters.items():
                lines.append(f"  {param}: {value:.4f}")

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        if self.is_random_utility:
            lines.append("  Choices consistent with random utility maximization.")
            lines.append(f"  {self.model_type.capitalize()} model provides good fit.")
        else:
            if not self.satisfies_iia:
                lines.append("  IIA violated - choice probabilities context-dependent.")
            if self.regularity_violations:
                lines.append(f"  {self.num_regularity_violations} regularity violation(s) detected.")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "model_type": self.model_type,
            "parameters": self.parameters,
            "satisfies_iia": self.satisfies_iia,
            "is_random_utility": self.is_random_utility,
            "log_likelihood": self.log_likelihood,
            "aic": self.aic,
            "bic": self.bic,
            "num_regularity_violations": self.num_regularity_violations,
            "regularity_violations": self.regularity_violations,
            "choice_probabilities": self.choice_probabilities.tolist(),
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        rum = "RUM" if self.is_random_utility else "non-RUM"
        return f"StochasticChoiceResult({self.model_type}, {rum}, aic={self.aic:.2f})"


@dataclass(frozen=True)
class AttentionResult(ResultDisplayMixin, ResultPlotMixin):
    """
    Result of limited attention model estimation (Chapter 14).

    Models consideration sets - items the consumer actually considers.

    Attributes:
        consideration_sets: List of estimated consideration sets per observation
        attention_parameter: Fraction of items typically considered (0-1)
        is_attention_rational: True if rationalizable with limited attention
        salience_weights: Array of attention weights per item
        default_option: Index of default option (if any)
        inattention_rate: Fraction of observations with limited attention
        rationalizable_observations: List of observations rationalizable with attention
        computation_time_ms: Time taken in milliseconds
    """

    consideration_sets: list[set[int]]
    attention_parameter: float
    is_attention_rational: bool
    salience_weights: NDArray[np.float64]
    default_option: int | None
    inattention_rate: float
    rationalizable_observations: list[int]
    computation_time_ms: float

    @property
    def num_observations(self) -> int:
        """Number of choice observations."""
        return len(self.consideration_sets)

    @property
    def mean_consideration_size(self) -> float:
        """Average size of consideration sets."""
        if not self.consideration_sets:
            return 0.0
        return sum(len(cs) for cs in self.consideration_sets) / len(self.consideration_sets)

    @property
    def rationalizability_rate(self) -> float:
        """Fraction of observations rationalizable with attention."""
        return len(self.rationalizable_observations) / max(self.num_observations, 1)

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns the rationalizability rate.
        """
        return self.rationalizability_rate

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("LIMITED ATTENTION MODEL REPORT")]

        # Status
        status = m._format_status(self.is_attention_rational,
                                  "ATTENTION RATIONAL", "ATTENTION VIOLATIONS")
        lines.append(f"\nStatus: {status}")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("Attention Rational", self.is_attention_rational))
        lines.append(m._format_metric("Attention Parameter", self.attention_parameter))
        lines.append(m._format_metric("Inattention Rate", self.inattention_rate))
        lines.append(m._format_metric("Observations", self.num_observations))
        lines.append(m._format_metric("Mean Consideration Size", self.mean_consideration_size))
        lines.append(m._format_metric("Rationalizable Obs", len(self.rationalizable_observations)))
        lines.append(m._format_metric("Rationalizability Rate", self.rationalizability_rate))
        if self.default_option is not None:
            lines.append(m._format_metric("Default Option", self.default_option))

        # Salience weights
        if len(self.salience_weights) > 0:
            lines.append(m._format_section("Salience Weights (top 5)"))
            sorted_idx = np.argsort(self.salience_weights)[::-1]
            for i in sorted_idx[:5]:
                lines.append(f"  Item {i}: {self.salience_weights[i]:.4f}")

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        if self.is_attention_rational:
            lines.append("  Behavior is rationalizable with limited attention.")
            lines.append(f"  Consumer considers ~{self.mean_consideration_size:.1f} items on average.")
        else:
            lines.append("  Behavior cannot be explained by limited attention alone.")
            lines.append(f"  {self.rationalizability_rate*100:.1f}% of observations rationalizable.")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "is_attention_rational": self.is_attention_rational,
            "attention_parameter": self.attention_parameter,
            "inattention_rate": self.inattention_rate,
            "num_observations": self.num_observations,
            "mean_consideration_size": self.mean_consideration_size,
            "rationalizability_rate": self.rationalizability_rate,
            "default_option": self.default_option,
            "salience_weights": self.salience_weights.tolist(),
            "rationalizable_observations": self.rationalizable_observations,
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        status = "rational" if self.is_attention_rational else "irrational"
        return f"AttentionResult({status}, attn={self.attention_parameter:.4f})"


@dataclass(frozen=True)
class ProductionGARPResult(ResultDisplayMixin, ResultPlotMixin):
    """
    Result of production theory GARP test (Chapter 15).

    Tests profit maximization consistency for firm behavior.

    Attributes:
        is_profit_maximizing: True if data satisfies production GARP
        violations: List of violation cycles
        cost_efficiency_score: Efficiency score for cost minimization (0-1)
        returns_to_scale: "increasing", "constant", "decreasing", or "variable"
        profit_efficiency: Overall profit efficiency score (0-1)
        input_efficiency_vector: Per-input efficiency scores
        output_efficiency_vector: Per-output efficiency scores
        technical_efficiency: Technical efficiency score
        computation_time_ms: Time taken in milliseconds
    """

    is_profit_maximizing: bool
    violations: list[Cycle]
    cost_efficiency_score: float
    returns_to_scale: str
    profit_efficiency: float
    input_efficiency_vector: NDArray[np.float64]
    output_efficiency_vector: NDArray[np.float64]
    technical_efficiency: float
    computation_time_ms: float

    @property
    def is_consistent(self) -> bool:
        """True if behavior is profit-maximizing consistent."""
        return self.is_profit_maximizing

    @property
    def num_violations(self) -> int:
        """Number of violation cycles."""
        return len(self.violations)

    @property
    def is_cost_minimizing(self) -> bool:
        """True if firm is cost-minimizing (dual of profit max)."""
        return self.cost_efficiency_score >= 1.0 - 1e-6

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns the profit efficiency score.
        """
        return self.profit_efficiency

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("PRODUCTION GARP TEST REPORT")]

        # Status
        status = m._format_status(self.is_profit_maximizing,
                                  "PROFIT MAXIMIZING", "VIOLATIONS FOUND")
        lines.append(f"\nStatus: {status}")
        lines.append(f"Returns to Scale: {self.returns_to_scale}")

        # Metrics
        lines.append(m._format_section("Efficiency Metrics"))
        lines.append(m._format_metric("Profit Maximizing", self.is_profit_maximizing))
        lines.append(m._format_metric("Cost Minimizing", self.is_cost_minimizing))
        lines.append(m._format_metric("Profit Efficiency", self.profit_efficiency))
        lines.append(m._format_metric("Cost Efficiency", self.cost_efficiency_score))
        lines.append(m._format_metric("Technical Efficiency", self.technical_efficiency))
        lines.append(m._format_metric("Violations", self.num_violations))

        # Input/Output efficiencies
        if len(self.input_efficiency_vector) > 0:
            lines.append(m._format_section("Input Efficiencies"))
            for i, eff in enumerate(self.input_efficiency_vector[:5]):
                lines.append(f"  Input {i}: {eff:.4f}")
            if len(self.input_efficiency_vector) > 5:
                lines.append(f"  ... ({len(self.input_efficiency_vector) - 5} more inputs)")

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        if self.is_profit_maximizing:
            lines.append("  Firm behavior is consistent with profit maximization.")
            lines.append(f"  Returns to scale: {self.returns_to_scale}.")
        else:
            lines.append(f"  {self.num_violations} violation(s) of profit maximization.")
            lines.append(f"  Profit efficiency is {self.profit_efficiency*100:.1f}%.")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "is_profit_maximizing": self.is_profit_maximizing,
            "is_cost_minimizing": self.is_cost_minimizing,
            "returns_to_scale": self.returns_to_scale,
            "profit_efficiency": self.profit_efficiency,
            "cost_efficiency_score": self.cost_efficiency_score,
            "technical_efficiency": self.technical_efficiency,
            "num_violations": self.num_violations,
            "input_efficiency_vector": self.input_efficiency_vector.tolist(),
            "output_efficiency_vector": self.output_efficiency_vector.tolist(),
            "violations": [list(c) for c in self.violations],
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation."""
        if self.is_profit_maximizing:
            return f"ProductionGARPResult(profit_max, {self.returns_to_scale})"
        return f"ProductionGARPResult({self.num_violations} violations, eff={self.profit_efficiency:.4f})"


# =============================================================================
# TECH-FRIENDLY ALIASES FOR NEW ALGORITHM RESULTS
# =============================================================================

# SlutskyConditionsResult: Tech-friendly alias for IntegrabilityResult
SlutskyConditionsResult = IntegrabilityResult
"""
Tech-friendly alias for IntegrabilityResult.

Tests if demand can be derived from utility maximization via Slutsky conditions.
"""

# WelfareChangeResult: Tech-friendly alias for WelfareResult
WelfareChangeResult = WelfareResult
"""
Tech-friendly alias for WelfareResult.

Measures welfare impact of price or policy changes.
"""

# AdditiveUtilityResult: Tech-friendly alias for AdditivityResult
AdditiveUtilityResult = AdditivityResult
"""
Tech-friendly alias for AdditivityResult.

Tests if preferences are additively separable across goods.
"""

# HicksianDemandResult: Tech-friendly alias for CompensatedDemandResult
HicksianDemandResult = CompensatedDemandResult
"""
Tech-friendly alias for CompensatedDemandResult.

Contains Slutsky decomposition of price effects.
"""

# MetricPreferencesResult: Tech-friendly alias for GeneralMetricResult
MetricPreferencesResult = GeneralMetricResult
"""
Tech-friendly alias for GeneralMetricResult.

Ideal point model with general distance metrics.
"""

# RandomUtilityResult: Tech-friendly alias for StochasticChoiceResult
RandomUtilityResult = StochasticChoiceResult
"""
Tech-friendly alias for StochasticChoiceResult.

Random utility model parameters and fit statistics.
"""

# ConsiderationSetResult: Tech-friendly alias for AttentionResult
ConsiderationSetResult = AttentionResult
"""
Tech-friendly alias for AttentionResult.

Limited attention model with consideration sets.
"""

# FirmBehaviorResult: Tech-friendly alias for ProductionGARPResult
FirmBehaviorResult = ProductionGARPResult
"""
Tech-friendly alias for ProductionGARPResult.

Production theory consistency test for firm behavior.
"""

