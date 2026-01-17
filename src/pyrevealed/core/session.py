"""Core data structures for behavioral signal analysis.

This module provides data containers for user behavior logs used in
consistency validation, latent value extraction, and anomaly detection.

Tech-Friendly Names (Primary):
    - BehaviorLog: User behavior history (cost/action pairs)
    - RiskChoiceLog: Choices under uncertainty
    - EmbeddingChoiceLog: Choices in feature/embedding space

Economics Names (Deprecated Aliases):
    - ConsumerSession -> BehaviorLog
    - RiskSession -> RiskChoiceLog
    - SpatialSession -> EmbeddingChoiceLog
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from pyrevealed.core.exceptions import (
    DataQualityWarning,
    DimensionError,
    InsufficientDataError,
    NaNInfError,
    ValueRangeError,
)


@dataclass
class BehaviorLog:
    """
    Represents user behavior history as cost/action pairs over T observations.

    The fundamental unit of analysis for behavioral consistency validation.
    Each row represents one observation (transaction, time period, etc.) where
    the user faced specific costs and took specific actions.

    This is the tech-friendly name for what economists call "ConsumerSession".

    Attributes:
        cost_vectors: T x N matrix of costs (e.g., prices) per observation.
            Can also be passed as `prices` for backward compatibility.
        action_vectors: T x N matrix of actions (e.g., quantities) per observation.
            Can also be passed as `quantities` for backward compatibility.
        user_id: Optional identifier for the user/session.
            Can also be passed as `session_id` for backward compatibility.
        metadata: Optional dictionary for additional attributes.

    Properties:
        spend_matrix: Pre-computed T x T matrix where S[i,j] = cost_i @ action_j
        total_spend: Diagonal of spend matrix (actual spend at each observation)
        num_records: Number of observations T
        num_features: Number of goods/actions N

    Example:
        >>> import numpy as np
        >>> # User faced different prices and bought different quantities
        >>> log = BehaviorLog(
        ...     cost_vectors=np.array([[1.0, 2.0], [2.0, 1.0]]),
        ...     action_vectors=np.array([[3.0, 1.0], [1.0, 3.0]]),
        ...     user_id="user_123"
        ... )
        >>> log.num_records
        2

        >>> # Backward compatible with old parameter names
        >>> log = BehaviorLog(prices=prices_array, quantities=quantities_array)
    """

    # Primary parameter names (tech-friendly)
    cost_vectors: NDArray[np.float64] | None = None
    action_vectors: NDArray[np.float64] | None = None
    user_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # NaN/Inf handling policy
    nan_policy: Literal["raise", "warn", "drop"] = field(default="raise", repr=False)

    # Legacy parameter names (deprecated but supported)
    prices: NDArray[np.float64] | None = field(default=None, repr=False)
    quantities: NDArray[np.float64] | None = field(default=None, repr=False)
    session_id: str | None = field(default=None, repr=False)

    # Cached computed properties
    _expenditure_matrix: NDArray[np.float64] | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Resolve parameter aliases and validate inputs."""
        # Resolve cost_vectors vs prices
        if self.cost_vectors is None and self.prices is not None:
            self.cost_vectors = self.prices
        elif self.cost_vectors is None and self.prices is None:
            raise ValueError("Must provide cost_vectors (or prices)")

        # Resolve action_vectors vs quantities
        if self.action_vectors is None and self.quantities is not None:
            self.action_vectors = self.quantities
        elif self.action_vectors is None and self.quantities is None:
            raise ValueError("Must provide action_vectors (or quantities)")

        # Resolve user_id vs session_id
        if self.user_id is None and self.session_id is not None:
            self.user_id = self.session_id

        # Ensure arrays are float64
        self.cost_vectors = np.asarray(self.cost_vectors, dtype=np.float64)
        self.action_vectors = np.asarray(self.action_vectors, dtype=np.float64)

        # Handle NaN/Inf values according to policy
        self.cost_vectors, self.action_vectors = self._handle_nan_inf(
            self.cost_vectors, self.action_vectors
        )

        # Keep legacy attributes in sync for backward compatibility
        self.prices = self.cost_vectors
        self.quantities = self.action_vectors
        self.session_id = self.user_id

        self._validate()
        self._compute_expenditure_matrix()

    def _handle_nan_inf(
        self, costs: NDArray[np.float64], actions: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Handle NaN and Inf values according to nan_policy.

        Args:
            costs: Cost vectors array
            actions: Action vectors array

        Returns:
            Tuple of (cleaned_costs, cleaned_actions)

        Raises:
            NaNInfError: If nan_policy='raise' and NaN/Inf found
        """
        cost_invalid = ~np.isfinite(costs)
        action_invalid = ~np.isfinite(actions)

        has_invalid_costs = np.any(cost_invalid)
        has_invalid_actions = np.any(action_invalid)

        if not (has_invalid_costs or has_invalid_actions):
            return costs, actions

        # Find affected rows
        affected_rows = np.where(
            np.any(cost_invalid, axis=1) | np.any(action_invalid, axis=1)
        )[0]
        nan_count = int(np.sum(cost_invalid) + np.sum(action_invalid))

        # Build informative message
        row_preview = affected_rows[:5].tolist()
        row_msg = str(row_preview) + ("..." if len(affected_rows) > 5 else "")

        if self.nan_policy == "raise":
            raise NaNInfError(
                f"Found {nan_count} NaN/Inf values in {len(affected_rows)} observations. "
                f"Affected rows: {row_msg}. "
                f"Use nan_policy='drop' to remove affected rows, or "
                f"nan_policy='warn' to drop with a warning."
            )
        elif self.nan_policy == "warn":
            warnings.warn(
                f"Dropping {len(affected_rows)} observations with NaN/Inf values "
                f"(rows: {row_msg}).",
                DataQualityWarning,
                stacklevel=4,
            )
            return self._drop_rows(costs, actions, affected_rows)
        elif self.nan_policy == "drop":
            return self._drop_rows(costs, actions, affected_rows)

        return costs, actions

    def _drop_rows(
        self,
        costs: NDArray[np.float64],
        actions: NDArray[np.float64],
        rows_to_drop: NDArray[np.intp],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Drop specified rows from both arrays."""
        mask = np.ones(costs.shape[0], dtype=bool)
        mask[rows_to_drop] = False
        return costs[mask], actions[mask]

    def _validate(self) -> None:
        """Validate cost and action matrix dimensions and values."""
        if self.cost_vectors.shape != self.action_vectors.shape:
            raise DimensionError(
                f"cost_vectors shape {self.cost_vectors.shape} does not match "
                f"action_vectors shape {self.action_vectors.shape}. "
                f"Both arrays must have shape (T, N) where T=observations and N=features. "
                f"Hint: Check that your price and quantity data have the same dimensions."
            )
        if self.cost_vectors.ndim != 2:
            raise DimensionError(
                f"cost_vectors must be a 2D array (T x N), got {self.cost_vectors.ndim}D "
                f"with shape {self.cost_vectors.shape}. "
                f"Hint: Use .reshape(-1, N) to convert 1D arrays."
            )
        if self.cost_vectors.shape[0] < 1:
            raise InsufficientDataError(
                "Must have at least one observation. "
                "Hint: Check that your data is not empty after preprocessing."
            )
        if self.cost_vectors.shape[1] < 1:
            raise InsufficientDataError(
                "Must have at least one feature/good. "
                "Hint: Check that your data has at least one column."
            )
        if np.any(self.cost_vectors <= 0):
            invalid_positions = np.argwhere(self.cost_vectors <= 0)
            pos_preview = invalid_positions[:5].tolist()
            pos_msg = str(pos_preview) + ("..." if len(invalid_positions) > 5 else "")
            raise ValueRangeError(
                f"Found {len(invalid_positions)} non-positive costs at positions: {pos_msg}. "
                f"All costs must be strictly positive (> 0) for revealed preference analysis. "
                f"Hint: Check for missing data encoded as 0, or filter out zero-cost observations."
            )
        if np.any(self.action_vectors < 0):
            invalid_positions = np.argwhere(self.action_vectors < 0)
            pos_preview = invalid_positions[:5].tolist()
            pos_msg = str(pos_preview) + ("..." if len(invalid_positions) > 5 else "")
            raise ValueRangeError(
                f"Found {len(invalid_positions)} negative actions at positions: {pos_msg}. "
                f"All actions must be non-negative (>= 0). "
                f"Hint: Check for data entry errors or consider using absolute values."
            )

    def _compute_expenditure_matrix(self) -> None:
        """Pre-compute spend matrix S[i,j] = cost_i @ action_j."""
        self._expenditure_matrix = self.cost_vectors @ self.action_vectors.T

    @property
    def spend_matrix(self) -> NDArray[np.float64]:
        """
        T x T matrix where S[i,j] = cost to take action j at costs i.

        This matrix is fundamental to behavioral consistency analysis:
        - If S[i,i] >= S[i,j], then action i is revealed preferred to action j
          at costs i (action j was affordable but not chosen).
        """
        if self._expenditure_matrix is None:
            self._compute_expenditure_matrix()
        return self._expenditure_matrix  # type: ignore

    @property
    def total_spend(self) -> NDArray[np.float64]:
        """
        Actual spend at each observation (diagonal of spend matrix).

        total_spend[i] = cost_i @ action_i = total cost at observation i.
        """
        return np.diag(self.spend_matrix)

    @property
    def num_records(self) -> int:
        """Number of observations/records T."""
        return self.cost_vectors.shape[0]

    @property
    def num_features(self) -> int:
        """Number of features/goods/actions N."""
        return self.cost_vectors.shape[1]

    # Legacy property aliases for backward compatibility
    @property
    def expenditure_matrix(self) -> NDArray[np.float64]:
        """Alias for spend_matrix (deprecated, use spend_matrix)."""
        return self.spend_matrix

    @property
    def own_expenditures(self) -> NDArray[np.float64]:
        """Alias for total_spend (deprecated, use total_spend)."""
        return self.total_spend

    @property
    def num_observations(self) -> int:
        """Alias for num_records (deprecated, use num_records)."""
        return self.num_records

    @property
    def num_goods(self) -> int:
        """Alias for num_features (deprecated, use num_features)."""
        return self.num_features

    @classmethod
    def from_dataframe(
        cls,
        df: Any,  # pandas.DataFrame
        cost_cols: list[str] | None = None,
        action_cols: list[str] | None = None,
        user_id: str | None = None,
        # Legacy parameter names
        price_cols: list[str] | None = None,
        quantity_cols: list[str] | None = None,
        session_id: str | None = None,
    ) -> BehaviorLog:
        """
        Create BehaviorLog from pandas DataFrame.

        Args:
            df: DataFrame with cost and action columns
            cost_cols: Column names for costs (or use price_cols)
            action_cols: Column names for actions (or use quantity_cols)
            user_id: Optional user identifier (or use session_id)

        Returns:
            BehaviorLog instance

        Example:
            >>> import pandas as pd
            >>> df = pd.DataFrame({
            ...     'cost_A': [1.0, 2.0], 'cost_B': [2.0, 1.0],
            ...     'action_A': [3.0, 1.0], 'action_B': [1.0, 3.0]
            ... })
            >>> log = BehaviorLog.from_dataframe(
            ...     df, cost_cols=['cost_A', 'cost_B'],
            ...     action_cols=['action_A', 'action_B']
            ... )
        """
        # Resolve aliases
        cost_cols = cost_cols or price_cols
        action_cols = action_cols or quantity_cols
        user_id = user_id or session_id

        if cost_cols is None:
            raise ValueError("Must provide cost_cols (or price_cols)")
        if action_cols is None:
            raise ValueError("Must provide action_cols (or quantity_cols)")

        costs = df[cost_cols].values
        actions = df[action_cols].values
        return cls(cost_vectors=costs, action_vectors=actions, user_id=user_id)

    @classmethod
    def from_long_format(
        cls,
        df: Any,  # pandas.DataFrame
        time_col: str = "time",
        item_col: str = "item_id",
        cost_col: str | None = None,
        action_col: str | None = None,
        user_id: str | None = None,
        # Legacy parameter names
        price_col: str | None = None,
        qty_col: str | None = None,
        session_id: str | None = None,
    ) -> BehaviorLog:
        """
        Create BehaviorLog from long-format transaction logs.

        Pivots SQL-style transaction data (one row per item per time) into
        wide-format matrices (one row per observation).

        Args:
            df: Long-format DataFrame with one row per item per time
            time_col: Column name for time/observation identifier
            item_col: Column name for item/product identifier
            cost_col: Column name for cost (or use price_col)
            action_col: Column name for action (or use qty_col)
            user_id: Optional user identifier (or use session_id)

        Returns:
            BehaviorLog instance
        """

        # Resolve aliases
        cost_col = cost_col or price_col or "price"
        action_col = action_col or qty_col or "quantity"
        user_id = user_id or session_id

        # Pivot to wide format
        action_pivot = df.pivot(index=time_col, columns=item_col, values=action_col)
        cost_pivot = df.pivot(index=time_col, columns=item_col, values=cost_col)

        # Fill missing actions with 0 (item not taken)
        actions = action_pivot.fillna(0).values

        # Costs should not be missing
        if cost_pivot.isna().any().any():
            missing = cost_pivot.isna().sum().sum()
            raise ValueError(
                f"Found {missing} missing costs. All costs must be provided."
            )
        costs = cost_pivot.values

        return cls(cost_vectors=costs, action_vectors=actions, user_id=user_id)

    def split_by_window(self, window_size: int) -> list[BehaviorLog]:
        """
        Split log into non-overlapping windows.

        Useful for detecting structural breaks or analyzing consistency
        over different time periods.

        Args:
            window_size: Number of observations per window

        Returns:
            List of BehaviorLog instances, one per window
        """
        logs = []
        for i in range(0, self.num_records, window_size):
            end = min(i + window_size, self.num_records)
            if end - i >= 2:  # Need at least 2 observations
                logs.append(
                    BehaviorLog(
                        cost_vectors=self.cost_vectors[i:end],
                        action_vectors=self.action_vectors[i:end],
                        user_id=f"{self.user_id}_window_{i // window_size}"
                        if self.user_id
                        else None,
                    )
                )
        return logs

    def summary(self, include_power: bool = False) -> "BehavioralSummary":
        """Run comprehensive analysis and return unified summary.

        This provides a Stata/statsmodels-style summary of all behavioral
        consistency tests and goodness-of-fit metrics.

        Args:
            include_power: Whether to include power analysis (default: False).
                Power analysis computes Bronars power and optimal efficiency
                via Monte Carlo simulation, which is computationally expensive.

        Returns:
            BehavioralSummary with GARP, WARP, SARP, AEI, MPI results

        Example:
            >>> log = BehaviorLog(prices, quantities)
            >>> print(log.summary())  # Standard report
            >>> print(log.summary(include_power=True))  # With power analysis
        """
        from pyrevealed.core.summary import BehavioralSummary
        return BehavioralSummary.from_log(self, include_power=include_power)


# Alias for RiskSession with tech-friendly name
@dataclass
class RiskChoiceLog:
    """
    Represents choice data between safe and risky options under uncertainty.

    Used for revealed preference analysis of risk attitudes. Each observation
    presents the decision-maker with a safe option (certain payoff) and a
    risky option (lottery with multiple possible outcomes).

    Attributes:
        safe_values: T-length array of certain payoff values for the safe option
        risky_outcomes: T x K matrix of possible outcomes for the risky option
            (K = max number of outcomes per lottery)
        risky_probabilities: T x K matrix of objective probabilities for outcomes
            (must sum to 1 for each row)
        choices: T-length boolean array; True = chose risky, False = chose safe
        session_id: Optional identifier for the session/decision-maker
        metadata: Optional dictionary for additional attributes

    Example:
        >>> import numpy as np
        >>> # Two choices: risky lottery vs certain amount
        >>> safe = np.array([50.0, 100.0])
        >>> outcomes = np.array([[100.0, 0.0], [200.0, 0.0]])
        >>> probs = np.array([[0.5, 0.5], [0.5, 0.5]])
        >>> choices = np.array([True, False])  # Chose risky then safe
        >>> session = RiskSession(safe, outcomes, probs, choices)
    """

    safe_values: NDArray[np.float64]
    risky_outcomes: NDArray[np.float64]
    risky_probabilities: NDArray[np.float64]
    choices: NDArray[np.bool_]
    session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate inputs."""
        self.safe_values = np.asarray(self.safe_values, dtype=np.float64)
        self.risky_outcomes = np.asarray(self.risky_outcomes, dtype=np.float64)
        self.risky_probabilities = np.asarray(
            self.risky_probabilities, dtype=np.float64
        )
        self.choices = np.asarray(self.choices, dtype=np.bool_)

        self._validate()

    def _validate(self) -> None:
        """Validate dimensions and values."""
        # Check for NaN/Inf in numeric arrays
        for name, arr in [
            ("safe_values", self.safe_values),
            ("risky_outcomes", self.risky_outcomes),
            ("risky_probabilities", self.risky_probabilities),
        ]:
            if not np.all(np.isfinite(arr)):
                invalid_count = int(np.sum(~np.isfinite(arr)))
                raise NaNInfError(
                    f"Found {invalid_count} NaN/Inf values in {name}. "
                    f"All values must be finite numbers."
                )

        T = len(self.safe_values)

        if self.safe_values.ndim != 1:
            raise DimensionError(
                f"safe_values must be 1D array, got {self.safe_values.ndim}D. "
                f"Hint: Use .flatten() or .ravel() to convert."
            )
        if self.choices.ndim != 1 or len(self.choices) != T:
            raise DimensionError(
                f"choices must have length {T} (matching safe_values), "
                f"got length {len(self.choices)}."
            )
        if self.risky_outcomes.ndim != 2 or self.risky_outcomes.shape[0] != T:
            raise DimensionError(
                f"risky_outcomes must have shape (T={T}, K), "
                f"got shape {self.risky_outcomes.shape}."
            )
        if self.risky_probabilities.shape != self.risky_outcomes.shape:
            raise DimensionError(
                f"risky_probabilities shape {self.risky_probabilities.shape} must match "
                f"risky_outcomes shape {self.risky_outcomes.shape}."
            )

        # Check probabilities sum to 1
        prob_sums = self.risky_probabilities.sum(axis=1)
        if not np.allclose(prob_sums, 1.0):
            bad_rows = np.where(~np.isclose(prob_sums, 1.0))[0]
            raise ValueRangeError(
                f"risky_probabilities must sum to 1 for each observation. "
                f"Rows with invalid sums: {bad_rows[:5].tolist()}"
                f"{'...' if len(bad_rows) > 5 else ''}. "
                f"Hint: Normalize each row with probs / probs.sum(axis=1, keepdims=True)."
            )

        # Check non-negative probabilities
        if np.any(self.risky_probabilities < 0):
            invalid_positions = np.argwhere(self.risky_probabilities < 0)
            raise ValueRangeError(
                f"Found {len(invalid_positions)} negative probabilities. "
                f"All probabilities must be non-negative."
            )

    @property
    def num_observations(self) -> int:
        """Number of choice observations T."""
        return len(self.safe_values)

    @property
    def num_outcomes(self) -> int:
        """Maximum number of outcomes per lottery K."""
        return self.risky_outcomes.shape[1]

    @property
    def expected_values(self) -> NDArray[np.float64]:
        """Expected value of each risky lottery."""
        return np.sum(self.risky_outcomes * self.risky_probabilities, axis=1)

    @property
    def risk_neutral_choices(self) -> NDArray[np.bool_]:
        """What a risk-neutral agent would choose (True if EV > safe)."""
        return self.expected_values > self.safe_values

    @property
    def num_risk_seeking_choices(self) -> int:
        """Count of choices where risky was chosen despite lower EV."""
        chose_risky = self.choices
        risky_has_lower_ev = self.expected_values < self.safe_values
        return int(np.sum(chose_risky & risky_has_lower_ev))

    @property
    def num_risk_averse_choices(self) -> int:
        """Count of choices where safe was chosen despite lower EV."""
        chose_safe = ~self.choices
        safe_has_lower_ev = self.expected_values > self.safe_values
        return int(np.sum(chose_safe & safe_has_lower_ev))

    def summary(self) -> "RiskChoiceSummary":
        """Run comprehensive analysis and return unified summary.

        This provides a Stata/statsmodels-style summary of risk profile
        analysis and Expected Utility axiom tests.

        Returns:
            RiskChoiceSummary with risk profile and EU axiom results

        Example:
            >>> log = RiskChoiceLog(safe_values, risky_outcomes, risky_probs, choices)
            >>> print(log.summary())  # Full report
        """
        from pyrevealed.core.summary import RiskChoiceSummary
        return RiskChoiceSummary.from_log(self)


@dataclass
class SpatialSession:
    """
    Represents choice data in a feature/embedding space for ideal point analysis.

    Used to find a user's "ideal point" in a D-dimensional feature space.
    The model assumes the user prefers items closer to their ideal point
    (Euclidean preference model).

    Attributes:
        item_features: M x D matrix of item embeddings (M items, D dimensions)
        choice_sets: List of T choice sets, each is a list of item indices
        choices: T-length list of chosen item indices (one per choice set)
        session_id: Optional identifier
        metadata: Optional dictionary for additional attributes

    Example:
        >>> import numpy as np
        >>> # 5 items in 2D space
        >>> features = np.array([
        ...     [0.0, 0.0],  # Item 0
        ...     [1.0, 0.0],  # Item 1
        ...     [0.0, 1.0],  # Item 2
        ...     [1.0, 1.0],  # Item 3
        ...     [0.5, 0.5],  # Item 4
        ... ])
        >>> choice_sets = [[0, 1, 2], [1, 3, 4], [0, 4]]
        >>> choices = [0, 4, 4]  # Always chose item closest to origin
        >>> session = SpatialSession(features, choice_sets, choices)
    """

    item_features: NDArray[np.float64]
    choice_sets: list[list[int]]
    choices: list[int]
    session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate inputs."""
        self.item_features = np.asarray(self.item_features, dtype=np.float64)
        self._validate()

    def _validate(self) -> None:
        """Validate dimensions and indices."""
        # Check for NaN/Inf in item features
        if not np.all(np.isfinite(self.item_features)):
            invalid_count = int(np.sum(~np.isfinite(self.item_features)))
            raise NaNInfError(
                f"Found {invalid_count} NaN/Inf values in item_features. "
                f"All feature values must be finite numbers."
            )

        if self.item_features.ndim != 2:
            raise DimensionError(
                f"item_features must be 2D (M items x D dimensions), "
                f"got {self.item_features.ndim}D with shape {self.item_features.shape}."
            )

        M = self.item_features.shape[0]
        T = len(self.choice_sets)

        if len(self.choices) != T:
            raise DimensionError(
                f"choices length {len(self.choices)} must equal number of "
                f"choice sets ({T})."
            )

        for t, (choice_set, chosen) in enumerate(zip(self.choice_sets, self.choices)):
            if len(choice_set) < 2:
                raise InsufficientDataError(
                    f"Choice set {t} has only {len(choice_set)} item(s). "
                    f"Each choice set must have at least 2 items for preference analysis."
                )
            if chosen not in choice_set:
                raise ValueRangeError(
                    f"Chosen item {chosen} is not in choice set {t}: {choice_set}. "
                    f"The chosen item must be one of the available options."
                )
            for idx in choice_set:
                if idx < 0 or idx >= M:
                    raise ValueRangeError(
                        f"Item index {idx} in choice set {t} is out of range [0, {M}). "
                        f"Hint: Ensure all indices refer to valid items in item_features."
                    )

    @property
    def num_items(self) -> int:
        """Number of items M."""
        return self.item_features.shape[0]

    @property
    def num_dimensions(self) -> int:
        """Number of feature dimensions D."""
        return self.item_features.shape[1]

    @property
    def num_observations(self) -> int:
        """Number of choice observations T."""
        return len(self.choice_sets)


# =============================================================================
# TECH-FRIENDLY ALIASES (Primary names)
# =============================================================================

# EmbeddingChoiceLog is an alias for SpatialSession
EmbeddingChoiceLog = SpatialSession
"""Alias for SpatialSession (tech-friendly name for embedding space choices)."""


# =============================================================================
# MENU-BASED CHOICE DATA (Abstract Choice Theory)
# =============================================================================


@dataclass
class MenuChoiceLog:
    """
    Represents menu-based choice data for abstract choice theory analysis.

    This data structure supports analyzing discrete choices from menus
    without price/budget constraints. Used for voting, surveys, recommendations,
    and any domain where items are chosen from finite sets.

    Based on Chapters 1-2 of Chambers & Echenique (2016) "Revealed Preference Theory".

    Attributes:
        menus: List of T menus, each a frozenset of item indices.
            Each menu represents the available options at that observation.
        choices: List of T chosen item indices (one per menu).
            choices[t] must be an element of menus[t].
        item_labels: Optional list of N item names for display.
        user_id: Optional identifier for the user/session.
        metadata: Optional dictionary for additional attributes.

    Properties:
        num_observations: Number of choice observations T
        num_items: Number of unique items N
        all_items: Set of all item indices that appear in any menu

    Example:
        >>> # User chooses from restaurant menus
        >>> log = MenuChoiceLog(
        ...     menus=[
        ...         frozenset({0, 1, 2}),  # Menu 1: Pizza, Burger, Salad
        ...         frozenset({1, 2, 3}),  # Menu 2: Burger, Salad, Pasta
        ...         frozenset({0, 3}),      # Menu 3: Pizza, Pasta
        ...     ],
        ...     choices=[0, 1, 0],  # Chose Pizza, Burger, Pizza
        ...     item_labels=["Pizza", "Burger", "Salad", "Pasta"],
        ... )
        >>> log.num_observations
        3

        >>> # Check if choice pattern is SARP-consistent
        >>> from pyrevealed import validate_menu_sarp
        >>> result = validate_menu_sarp(log)
        >>> print(f"Consistent: {result.is_consistent}")
    """

    menus: list[frozenset[int]]
    choices: list[int]
    item_labels: list[str] | None = None
    user_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate inputs."""
        self._validate()

    def _validate(self) -> None:
        """Validate menu and choice data."""
        # Check lengths match
        if len(self.menus) != len(self.choices):
            raise DimensionError(
                f"Number of menus ({len(self.menus)}) must match "
                f"number of choices ({len(self.choices)}). "
                f"Hint: Each observation needs exactly one menu and one choice."
            )

        if len(self.menus) < 1:
            raise InsufficientDataError(
                "Must have at least one observation. "
                "Hint: Check that your data is not empty."
            )

        # Check each menu and choice
        for t, (menu, choice) in enumerate(zip(self.menus, self.choices)):
            # Check menu is not empty
            if len(menu) < 1:
                raise InsufficientDataError(
                    f"Menu at observation {t} is empty. "
                    f"Each menu must have at least one item."
                )

            # Check choice is in menu
            if choice not in menu:
                raise ValueRangeError(
                    f"Choice {choice} at observation {t} is not in menu {menu}. "
                    f"The chosen item must be one of the available options."
                )

            # Check for negative indices
            if any(idx < 0 for idx in menu):
                raise ValueRangeError(
                    f"Menu at observation {t} contains negative indices. "
                    f"All item indices must be non-negative."
                )

        # Check item labels if provided
        if self.item_labels is not None:
            max_idx = max(max(menu) for menu in self.menus)
            if len(self.item_labels) <= max_idx:
                raise DimensionError(
                    f"item_labels has {len(self.item_labels)} items but "
                    f"menus contain index {max_idx}. "
                    f"Hint: Ensure item_labels covers all item indices."
                )

    @property
    def num_observations(self) -> int:
        """Number of choice observations T."""
        return len(self.menus)

    @property
    def all_items(self) -> frozenset[int]:
        """Set of all item indices that appear in any menu."""
        all_items: set[int] = set()
        for menu in self.menus:
            all_items.update(menu)
        return frozenset(all_items)

    @property
    def num_items(self) -> int:
        """Number of unique items N."""
        return len(self.all_items)

    def get_item_label(self, idx: int) -> str:
        """Get label for an item index."""
        if self.item_labels is not None and idx < len(self.item_labels):
            return self.item_labels[idx]
        return f"item_{idx}"

    def get_choice_label(self, t: int) -> str:
        """Get label for the choice at observation t."""
        return self.get_item_label(self.choices[t])

    def get_menu_labels(self, t: int) -> list[str]:
        """Get labels for all items in menu at observation t."""
        return [self.get_item_label(idx) for idx in sorted(self.menus[t])]

    @classmethod
    def from_dataframe(
        cls,
        df: Any,  # pandas.DataFrame
        menu_col: str = "menu",
        choice_col: str = "choice",
        user_id: str | None = None,
    ) -> MenuChoiceLog:
        """
        Create MenuChoiceLog from pandas DataFrame.

        Expects a DataFrame where each row is an observation with:
        - A menu column containing sets/lists of item indices
        - A choice column containing the chosen item index

        Args:
            df: DataFrame with menu and choice columns
            menu_col: Column name for menus (default: "menu")
            choice_col: Column name for choices (default: "choice")
            user_id: Optional user identifier

        Returns:
            MenuChoiceLog instance

        Example:
            >>> import pandas as pd
            >>> df = pd.DataFrame({
            ...     'menu': [{0, 1, 2}, {1, 2, 3}, {0, 3}],
            ...     'choice': [0, 1, 0]
            ... })
            >>> log = MenuChoiceLog.from_dataframe(df)
        """
        menus = [frozenset(m) for m in df[menu_col]]
        choices = list(df[choice_col])
        return cls(menus=menus, choices=choices, user_id=user_id)

    @classmethod
    def from_recommendations(
        cls,
        shown_items: list[list[int]],
        clicked_items: list[int],
        item_labels: list[str] | None = None,
        user_id: str | None = None,
    ) -> MenuChoiceLog:
        """
        Create MenuChoiceLog from recommendation system data.

        Converts a sequence of (shown items, clicked item) pairs
        into menu choice format. Common in recommender systems.

        Args:
            shown_items: List of T lists, each containing item indices shown to user
            clicked_items: List of T chosen item indices
            item_labels: Optional item names
            user_id: Optional user identifier

        Returns:
            MenuChoiceLog instance

        Example:
            >>> # User saw 3 recommendation slates and clicked one item each time
            >>> shown = [[0, 1, 2, 3], [1, 2, 4, 5], [0, 3, 4]]
            >>> clicked = [1, 4, 0]
            >>> log = MenuChoiceLog.from_recommendations(shown, clicked)
        """
        menus = [frozenset(items) for items in shown_items]
        return cls(
            menus=menus,
            choices=clicked_items,
            item_labels=item_labels,
            user_id=user_id,
        )

    def summary(self) -> "MenuChoiceSummary":
        """Run comprehensive analysis and return unified summary.

        This provides a Stata/statsmodels-style summary of menu-based
        choice consistency tests and efficiency metrics.

        Returns:
            MenuChoiceSummary with WARP, SARP, Congruence, and efficiency results

        Example:
            >>> log = MenuChoiceLog(menus, choices)
            >>> print(log.summary())  # Full report
        """
        from pyrevealed.core.summary import MenuChoiceSummary
        return MenuChoiceSummary.from_log(self)

    @property
    def num_alternatives(self) -> int:
        """Alias for num_items."""
        return self.num_items


# =============================================================================
# STOCHASTIC CHOICE DATA (Chapter 13)
# =============================================================================


@dataclass
class StochasticChoiceLog:
    """
    Represents stochastic/probabilistic choice data for random utility models.

    This data structure supports analyzing choices where the same decision-maker
    may make different choices in the same situation (probabilistic choice).
    Based on Chapter 13 of Chambers & Echenique (2016).

    Attributes:
        menus: List of T menus, each a frozenset of item indices.
        choice_frequencies: List of T dictionaries mapping item -> choice count.
            Each dict shows how many times each item was chosen from that menu.
        item_labels: Optional list of N item names for display.
        total_observations_per_menu: Optional list of total observations per menu.
        user_id: Optional identifier for the user/session.
        metadata: Optional dictionary for additional attributes.

    Example:
        >>> # Same menu presented 100 times, user chose item 0 60 times, item 1 40 times
        >>> log = StochasticChoiceLog(
        ...     menus=[frozenset({0, 1, 2})],
        ...     choice_frequencies=[{0: 60, 1: 40, 2: 0}],
        ... )
        >>> log.get_choice_probability(0, 0)  # P(choose 0 from menu 0)
        0.6
    """

    menus: list[frozenset[int]]
    choice_frequencies: list[dict[int, int]]
    item_labels: list[str] | None = None
    total_observations_per_menu: list[int] | None = None
    user_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate inputs."""
        self._validate()
        # Compute total observations if not provided
        if self.total_observations_per_menu is None:
            self.total_observations_per_menu = [
                sum(freq.values()) for freq in self.choice_frequencies
            ]

    def _validate(self) -> None:
        """Validate stochastic choice data."""
        if len(self.menus) != len(self.choice_frequencies):
            raise DimensionError(
                f"Number of menus ({len(self.menus)}) must match "
                f"number of choice frequency dicts ({len(self.choice_frequencies)})."
            )

        if len(self.menus) < 1:
            raise InsufficientDataError(
                "Must have at least one menu observation."
            )

        for t, (menu, freqs) in enumerate(zip(self.menus, self.choice_frequencies)):
            # Check all chosen items are in the menu
            for item in freqs:
                if item not in menu:
                    raise ValueRangeError(
                        f"Item {item} in frequencies at observation {t} "
                        f"is not in menu {menu}."
                    )
            # Check for negative frequencies
            for item, count in freqs.items():
                if count < 0:
                    raise ValueRangeError(
                        f"Negative frequency {count} for item {item} at observation {t}."
                    )

    @property
    def num_menus(self) -> int:
        """Number of distinct menus T."""
        return len(self.menus)

    @property
    def all_items(self) -> frozenset[int]:
        """Set of all item indices that appear in any menu."""
        all_items: set[int] = set()
        for menu in self.menus:
            all_items.update(menu)
        return frozenset(all_items)

    @property
    def num_items(self) -> int:
        """Number of unique items N."""
        return len(self.all_items)

    def get_choice_probability(self, menu_idx: int, item: int) -> float:
        """Get probability of choosing item from menu at menu_idx."""
        if menu_idx >= len(self.menus):
            raise IndexError(f"Menu index {menu_idx} out of range.")
        total = self.total_observations_per_menu[menu_idx]
        if total == 0:
            return 0.0
        return self.choice_frequencies[menu_idx].get(item, 0) / total

    def get_choice_probabilities(self, menu_idx: int) -> dict[int, float]:
        """Get all choice probabilities for a menu."""
        return {
            item: self.get_choice_probability(menu_idx, item)
            for item in self.menus[menu_idx]
        }

    @classmethod
    def from_repeated_choices(
        cls,
        menus: list[frozenset[int]],
        choices: list[int],
    ) -> "StochasticChoiceLog":
        """
        Create StochasticChoiceLog from repeated deterministic choice data.

        Groups observations by menu and computes choice frequencies.

        Args:
            menus: List of menus (may have duplicates)
            choices: List of chosen items (one per observation)

        Returns:
            StochasticChoiceLog with aggregated frequencies
        """
        from collections import defaultdict

        # Group by menu
        menu_to_choices: dict[frozenset[int], list[int]] = defaultdict(list)
        for menu, choice in zip(menus, choices):
            menu_to_choices[menu].append(choice)

        # Convert to frequencies
        unique_menus = list(menu_to_choices.keys())
        choice_frequencies = []
        for menu in unique_menus:
            freq: dict[int, int] = defaultdict(int)
            for choice in menu_to_choices[menu]:
                freq[choice] += 1
            choice_frequencies.append(dict(freq))

        return cls(menus=unique_menus, choice_frequencies=choice_frequencies)

    def summary(self) -> "StochasticChoiceSummary":
        """Run comprehensive analysis and return unified summary.

        This provides a Stata/statsmodels-style summary of stochastic
        choice analysis including RUM consistency, regularity, and
        transitivity tests.

        Returns:
            StochasticChoiceSummary with RUM, regularity, and transitivity results

        Example:
            >>> log = StochasticChoiceLog(menus, choice_frequencies)
            >>> print(log.summary())  # Full report
        """
        from pyrevealed.core.summary import StochasticChoiceSummary
        return StochasticChoiceSummary.from_log(self)


# =============================================================================
# PRODUCTION DATA (Chapter 15)
# =============================================================================


@dataclass
class ProductionLog:
    """
    Represents firm production data for production theory analysis.

    This data structure supports analyzing firm behavior using revealed
    preference methods for profit maximization and cost minimization tests.
    Based on Chapter 15 of Chambers & Echenique (2016).

    Attributes:
        input_prices: T x N_inputs matrix of input prices per observation.
        input_quantities: T x N_inputs matrix of input quantities used.
        output_prices: T x N_outputs matrix of output prices per observation.
        output_quantities: T x N_outputs matrix of output quantities produced.
        firm_id: Optional identifier for the firm.
        metadata: Optional dictionary for additional attributes.

    Properties:
        profit: Profit at each observation (revenue - cost)
        total_cost: Total input cost at each observation
        total_revenue: Total output revenue at each observation

    Example:
        >>> import numpy as np
        >>> # Firm uses 2 inputs to produce 1 output over 3 periods
        >>> log = ProductionLog(
        ...     input_prices=np.array([[1.0, 2.0], [1.5, 2.0], [1.0, 2.5]]),
        ...     input_quantities=np.array([[10.0, 5.0], [8.0, 6.0], [12.0, 4.0]]),
        ...     output_prices=np.array([[10.0], [12.0], [11.0]]),
        ...     output_quantities=np.array([[5.0], [4.0], [6.0]]),
        ... )
        >>> log.profit  # Revenue - Cost
        array([30., 26., 46.])
    """

    input_prices: NDArray[np.float64]
    input_quantities: NDArray[np.float64]
    output_prices: NDArray[np.float64]
    output_quantities: NDArray[np.float64]
    firm_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Cached computed properties
    _cost_matrix: NDArray[np.float64] | None = field(
        default=None, repr=False, compare=False
    )
    _revenue_matrix: NDArray[np.float64] | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Validate inputs and compute matrices."""
        # Ensure arrays are float64
        self.input_prices = np.asarray(self.input_prices, dtype=np.float64)
        self.input_quantities = np.asarray(self.input_quantities, dtype=np.float64)
        self.output_prices = np.asarray(self.output_prices, dtype=np.float64)
        self.output_quantities = np.asarray(self.output_quantities, dtype=np.float64)

        self._validate()
        self._compute_matrices()

    def _validate(self) -> None:
        """Validate production data dimensions and values."""
        # Check input dimensions
        if self.input_prices.shape != self.input_quantities.shape:
            raise DimensionError(
                f"input_prices shape {self.input_prices.shape} does not match "
                f"input_quantities shape {self.input_quantities.shape}."
            )

        # Check output dimensions
        if self.output_prices.shape != self.output_quantities.shape:
            raise DimensionError(
                f"output_prices shape {self.output_prices.shape} does not match "
                f"output_quantities shape {self.output_quantities.shape}."
            )

        # Check same number of observations
        T_input = self.input_prices.shape[0]
        T_output = self.output_prices.shape[0]
        if T_input != T_output:
            raise DimensionError(
                f"Number of input observations ({T_input}) must match "
                f"number of output observations ({T_output})."
            )

        # Check for NaN/Inf
        for name, arr in [
            ("input_prices", self.input_prices),
            ("input_quantities", self.input_quantities),
            ("output_prices", self.output_prices),
            ("output_quantities", self.output_quantities),
        ]:
            if not np.all(np.isfinite(arr)):
                invalid_count = int(np.sum(~np.isfinite(arr)))
                raise NaNInfError(
                    f"Found {invalid_count} NaN/Inf values in {name}."
                )

        # Check for positive prices
        if np.any(self.input_prices <= 0):
            raise ValueRangeError("All input prices must be strictly positive.")
        if np.any(self.output_prices <= 0):
            raise ValueRangeError("All output prices must be strictly positive.")

        # Check for non-negative quantities
        if np.any(self.input_quantities < 0):
            raise ValueRangeError("Input quantities cannot be negative.")
        if np.any(self.output_quantities < 0):
            raise ValueRangeError("Output quantities cannot be negative.")

    def _compute_matrices(self) -> None:
        """Pre-compute cost and revenue matrices."""
        # Cost matrix: C[i,j] = w_i @ x_j (cost of using inputs j at prices i)
        self._cost_matrix = self.input_prices @ self.input_quantities.T
        # Revenue matrix: R[i,j] = p_i @ y_j (revenue from outputs j at prices i)
        self._revenue_matrix = self.output_prices @ self.output_quantities.T

    @property
    def num_observations(self) -> int:
        """Number of observations T."""
        return self.input_prices.shape[0]

    @property
    def num_inputs(self) -> int:
        """Number of inputs N_inputs."""
        return self.input_prices.shape[1]

    @property
    def num_outputs(self) -> int:
        """Number of outputs N_outputs."""
        return self.output_prices.shape[1]

    @property
    def total_cost(self) -> NDArray[np.float64]:
        """Total input cost at each observation."""
        return np.diag(self._cost_matrix)

    @property
    def total_revenue(self) -> NDArray[np.float64]:
        """Total output revenue at each observation."""
        return np.diag(self._revenue_matrix)

    @property
    def profit(self) -> NDArray[np.float64]:
        """Profit at each observation (revenue - cost)."""
        return self.total_revenue - self.total_cost

    @property
    def cost_matrix(self) -> NDArray[np.float64]:
        """T x T matrix where C[i,j] = cost of using inputs j at prices i."""
        return self._cost_matrix

    @property
    def revenue_matrix(self) -> NDArray[np.float64]:
        """T x T matrix where R[i,j] = revenue from outputs j at prices i."""
        return self._revenue_matrix

    def summary(self) -> "ProductionSummary":
        """Run comprehensive analysis and return unified summary.

        This provides a Stata/statsmodels-style summary of production
        analysis including profit maximization, cost minimization,
        and efficiency metrics.

        Returns:
            ProductionSummary with profit max, cost min, and efficiency results

        Example:
            >>> log = ProductionLog(input_prices, input_quantities, output_prices, output_quantities)
            >>> print(log.summary())  # Full report
        """
        from pyrevealed.core.summary import ProductionSummary
        return ProductionSummary.from_log(self)


# =============================================================================
# TECH-FRIENDLY ALIASES (Primary names)
# =============================================================================

# ChoiceSession is an alias for MenuChoiceLog (economics name)
ChoiceSession = MenuChoiceLog
"""Alias for MenuChoiceLog (economics terminology from abstract choice theory)."""

# ProbabilisticChoiceLog is an alias for StochasticChoiceLog
ProbabilisticChoiceLog = StochasticChoiceLog
"""Tech-friendly alias for StochasticChoiceLog."""

# FirmLog is an alias for ProductionLog
FirmLog = ProductionLog
"""Tech-friendly alias for ProductionLog for firm behavior analysis."""


# =============================================================================
# LEGACY ALIASES (Deprecated - use tech-friendly names instead)
# =============================================================================

# ConsumerSession is now an alias for BehaviorLog
ConsumerSession = BehaviorLog
"""
Deprecated: Use BehaviorLog instead.

ConsumerSession is the legacy name from economics literature.
BehaviorLog is the preferred tech-friendly name.
"""

# RiskSession is now an alias for RiskChoiceLog
RiskSession = RiskChoiceLog
"""
Deprecated: Use RiskChoiceLog instead.

RiskSession is the legacy name from economics literature.
RiskChoiceLog is the preferred tech-friendly name.
"""
