"""PreferenceEncoder: High-level API for latent preference extraction.

This module provides a user-friendly, scikit-learn style interface for
extracting latent preference values from user behavior logs.

Use this to:
- Extract features for ML models
- Generate user embeddings
- Run counterfactual simulations
- Predict user choices under new conditions
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
from numpy.typing import NDArray

from pyrevealed.algorithms.utility import (
    fit_latent_values,
    build_value_function,
    predict_choice,
)
from pyrevealed.core.exceptions import NotFittedError

if TYPE_CHECKING:
    from pyrevealed.core.session import BehaviorLog
    from pyrevealed.core.result import LatentValueResult


class PreferenceEncoder:
    """
    Encodes user preferences into latent value representations.

    PreferenceEncoder follows the scikit-learn pattern: fit() to learn
    from data, then extract features or make predictions.

    The encoder solves an optimization problem to find latent values
    that explain the user's observed choices. These values can be used as:
    - Features for downstream ML models
    - User embeddings for similarity calculations
    - Inputs to counterfactual simulations

    Example:
        >>> from pyrevealed import PreferenceEncoder, BehaviorLog
        >>> import numpy as np

        >>> # Create behavior log
        >>> log = BehaviorLog(
        ...     cost_vectors=np.array([[1.0, 2.0], [2.0, 1.0], [1.5, 1.5]]),
        ...     action_vectors=np.array([[3.0, 1.0], [1.0, 3.0], [2.0, 2.0]]),
        ... )

        >>> # Fit encoder
        >>> encoder = PreferenceEncoder()
        >>> encoder.fit(log)

        >>> # Extract latent values as features
        >>> features = encoder.extract_latent_values()
        >>> print(f"Latent values: {features}")

        >>> # Build value function for counterfactuals
        >>> value_fn = encoder.get_value_function()
        >>> print(f"Value of [2, 2]: {value_fn(np.array([2.0, 2.0]))}")

    Attributes:
        precision: Numerical precision for optimization (default: 1e-8)
    """

    def __init__(self, precision: float = 1e-8) -> None:
        """
        Initialize the encoder.

        Args:
            precision: Numerical precision for the LP solver.
        """
        self.precision = precision
        self._result: LatentValueResult | None = None
        self._log: BehaviorLog | None = None
        self._is_fitted: bool = False

    def fit(self, log: BehaviorLog) -> PreferenceEncoder:
        """
        Fit the encoder to a behavior log.

        Solves an optimization problem to find latent preference values
        that explain the user's observed choices.

        Args:
            log: BehaviorLog containing user's historical actions

        Returns:
            self (for method chaining)

        Raises:
            ValueError: If the behavior is too inconsistent to fit

        Example:
            >>> encoder = PreferenceEncoder().fit(user_log)
        """
        result = fit_latent_values(log, tolerance=self.precision)
        self._result = result
        self._log = log
        self._is_fitted = result.success
        return self

    @property
    def is_fitted(self) -> bool:
        """Check if the encoder has been successfully fitted."""
        return self._is_fitted

    def _check_fitted(self) -> None:
        """Raise error if not fitted."""
        if not self._is_fitted:
            raise NotFittedError(
                "Encoder not fitted. Call fit() first, or check if behavior "
                "is too inconsistent (use BehavioralAuditor to check). "
                "Hint: Use compute_integrity_score() to check data consistency before fitting."
            )

    def extract_latent_values(self) -> NDArray[np.float64]:
        """
        Extract latent preference values.

        Returns an array of latent values, one per observation in the
        fitted log. These can be used as features for ML models.

        Returns:
            Array of latent values (T observations)

        Raises:
            ValueError: If not fitted

        Example:
            >>> encoder.fit(user_log)
            >>> features = encoder.extract_latent_values()
            >>> # Use as features in an ML model
            >>> X_train = np.column_stack([other_features, features])
        """
        self._check_fitted()
        return self._result.utility_values.copy()

    def extract_marginal_weights(self) -> NDArray[np.float64]:
        """
        Extract marginal weights (sensitivity to costs).

        Returns an array of marginal weights representing how sensitive
        the user's preferences are to cost changes at each observation.

        Returns:
            Array of marginal weights (T observations)

        Raises:
            ValueError: If not fitted
        """
        self._check_fitted()
        return self._result.lagrange_multipliers.copy()

    def get_value_function(self) -> Callable[[NDArray], float]:
        """
        Get a callable value function.

        Returns a function that estimates the latent value of any
        action vector. Useful for counterfactual analysis.

        Returns:
            Callable that takes an action vector and returns its value

        Raises:
            ValueError: If not fitted

        Example:
            >>> encoder.fit(user_log)
            >>> value_fn = encoder.get_value_function()
            >>> # Estimate value of a hypothetical action
            >>> value = value_fn(np.array([5.0, 3.0]))
        """
        self._check_fitted()
        return build_value_function(self._log, self._result)

    def predict_choice(
        self,
        cost_vector: NDArray[np.float64],
        resource_limit: float,
    ) -> NDArray[np.float64] | None:
        """
        Predict what action the user would take under new conditions.

        Given a new cost vector and resource limit (budget), predicts
        what action vector the user would choose to maximize their
        latent preference value.

        Args:
            cost_vector: Array of costs for each action dimension
            resource_limit: Total budget/resource constraint

        Returns:
            Predicted action vector, or None if prediction failed

        Raises:
            ValueError: If not fitted

        Example:
            >>> encoder.fit(user_log)
            >>> # What would user do with new prices and $100 budget?
            >>> new_costs = np.array([1.5, 2.5])
            >>> predicted_action = encoder.predict_choice(new_costs, 100.0)
        """
        self._check_fitted()
        return predict_choice(
            self._log,
            self._result,
            new_prices=cost_vector,
            budget=resource_limit,
        )

    def get_fit_details(self) -> LatentValueResult:
        """
        Get detailed results from the fitting process.

        Returns the full LatentValueResult with solver status,
        residuals, and other diagnostic information.

        Returns:
            LatentValueResult with full details

        Raises:
            ValueError: If not fitted
        """
        self._check_fitted()
        return self._result

    @property
    def solver_status(self) -> str:
        """Get the solver status message."""
        if self._result is None:
            return "not_fitted"
        return self._result.lp_status

    @property
    def mean_marginal_weight(self) -> float | None:
        """Get the mean marginal weight across observations."""
        if not self._is_fitted:
            return None
        return self._result.mean_marginal_utility
