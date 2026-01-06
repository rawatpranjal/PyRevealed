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
from typing import Any

import numpy as np
from numpy.typing import NDArray


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

        # Keep legacy attributes in sync for backward compatibility
        self.prices = self.cost_vectors
        self.quantities = self.action_vectors
        self.session_id = self.user_id

        self._validate()
        self._compute_expenditure_matrix()

    def _validate(self) -> None:
        """Validate cost and action matrix dimensions and values."""
        if self.cost_vectors.shape != self.action_vectors.shape:
            raise ValueError(
                f"cost_vectors shape {self.cost_vectors.shape} must match "
                f"action_vectors shape {self.action_vectors.shape}"
            )
        if self.cost_vectors.ndim != 2:
            raise ValueError(
                f"cost_vectors must be a 2D array (T x N), got {self.cost_vectors.ndim}D"
            )
        if self.cost_vectors.shape[0] < 1:
            raise ValueError("Must have at least one observation")
        if self.cost_vectors.shape[1] < 1:
            raise ValueError("Must have at least one feature")
        if np.any(self.cost_vectors <= 0):
            raise ValueError("All costs must be strictly positive")
        if np.any(self.action_vectors < 0):
            raise ValueError("All actions must be non-negative")

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
        import pandas as pd

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
                        user_id=f"{self.user_id}_window_{i//window_size}"
                        if self.user_id
                        else None,
                    )
                )
        return logs


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
        self.risky_probabilities = np.asarray(self.risky_probabilities, dtype=np.float64)
        self.choices = np.asarray(self.choices, dtype=np.bool_)

        self._validate()

    def _validate(self) -> None:
        """Validate dimensions and values."""
        T = len(self.safe_values)

        if self.safe_values.ndim != 1:
            raise ValueError("safe_values must be 1D array")
        if self.choices.ndim != 1 or len(self.choices) != T:
            raise ValueError(f"choices must have length {T}")
        if self.risky_outcomes.ndim != 2 or self.risky_outcomes.shape[0] != T:
            raise ValueError(f"risky_outcomes must have shape (T={T}, K)")
        if self.risky_probabilities.shape != self.risky_outcomes.shape:
            raise ValueError("risky_probabilities must match risky_outcomes shape")

        # Check probabilities sum to 1
        prob_sums = self.risky_probabilities.sum(axis=1)
        if not np.allclose(prob_sums, 1.0):
            raise ValueError("risky_probabilities must sum to 1 for each observation")

        # Check non-negative probabilities
        if np.any(self.risky_probabilities < 0):
            raise ValueError("Probabilities must be non-negative")

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
        if self.item_features.ndim != 2:
            raise ValueError("item_features must be 2D (M items x D dimensions)")

        M = self.item_features.shape[0]
        T = len(self.choice_sets)

        if len(self.choices) != T:
            raise ValueError(f"choices length {len(self.choices)} != {T} choice sets")

        for t, (choice_set, chosen) in enumerate(zip(self.choice_sets, self.choices)):
            if len(choice_set) < 2:
                raise ValueError(f"Choice set {t} must have at least 2 items")
            if chosen not in choice_set:
                raise ValueError(f"Chosen item {chosen} not in choice set {t}")
            for idx in choice_set:
                if idx < 0 or idx >= M:
                    raise ValueError(f"Item index {idx} out of range [0, {M})")

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
