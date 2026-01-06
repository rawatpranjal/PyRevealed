"""Core data structures for revealed preference analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class RiskSession:
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


@dataclass
class ConsumerSession:
    """
    Represents observed consumer choice data across T observations and N goods.

    The fundamental unit of analysis in revealed preference theory. Each row
    represents one observation (shopping trip, time period, etc.) where the
    consumer faced a specific price vector and chose a specific bundle.

    Attributes:
        prices: T x N matrix of prices (each row is a price vector for one observation)
        quantities: T x N matrix of quantities (each row is the chosen bundle)
        session_id: Optional identifier for the session/consumer
        metadata: Optional dictionary for additional attributes

    Properties:
        expenditure_matrix: Pre-computed T x T matrix E where E[i,j] = p_i @ q_j
        own_expenditures: Diagonal of expenditure matrix (actual spend at each obs)
        num_observations: Number of observations T
        num_goods: Number of goods N

    Example:
        >>> import numpy as np
        >>> prices = np.array([[1.0, 2.0], [2.0, 1.0]])
        >>> quantities = np.array([[3.0, 1.0], [1.0, 3.0]])
        >>> session = ConsumerSession(prices=prices, quantities=quantities)
        >>> session.num_observations
        2
    """

    prices: NDArray[np.float64]
    quantities: NDArray[np.float64]
    session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Cached computed properties (not part of equality/hash)
    _expenditure_matrix: NDArray[np.float64] | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Validate inputs and compute cached properties."""
        # Ensure arrays are float64
        self.prices = np.asarray(self.prices, dtype=np.float64)
        self.quantities = np.asarray(self.quantities, dtype=np.float64)

        self._validate()
        self._compute_expenditure_matrix()

    def _validate(self) -> None:
        """Validate price and quantity matrix dimensions and values."""
        if self.prices.shape != self.quantities.shape:
            raise ValueError(
                f"Prices shape {self.prices.shape} must match "
                f"quantities shape {self.quantities.shape}"
            )
        if self.prices.ndim != 2:
            raise ValueError(
                f"Prices must be a 2D array (T x N), got {self.prices.ndim}D"
            )
        if self.prices.shape[0] < 1:
            raise ValueError("Must have at least one observation")
        if self.prices.shape[1] < 1:
            raise ValueError("Must have at least one good")
        if np.any(self.prices <= 0):
            raise ValueError("All prices must be strictly positive")
        if np.any(self.quantities < 0):
            raise ValueError("All quantities must be non-negative")

    def _compute_expenditure_matrix(self) -> None:
        """
        Pre-compute expenditure matrix E[i,j] = p_i @ q_j.

        E[i,j] represents the cost of buying bundle j at prices i.
        The diagonal E[i,i] is the actual expenditure at observation i.
        """
        self._expenditure_matrix = self.prices @ self.quantities.T

    @property
    def expenditure_matrix(self) -> NDArray[np.float64]:
        """
        T x T matrix where E[i,j] = expenditure to buy bundle j at prices i.

        This matrix is fundamental to revealed preference analysis:
        - If E[i,i] >= E[i,j], then bundle i is revealed preferred to bundle j
          at prices i (bundle j was affordable but not chosen).
        """
        if self._expenditure_matrix is None:
            self._compute_expenditure_matrix()
        return self._expenditure_matrix  # type: ignore

    @property
    def own_expenditures(self) -> NDArray[np.float64]:
        """
        Actual expenditure at each observation (diagonal of expenditure matrix).

        own_expenditures[i] = p_i @ q_i = total money spent at observation i.
        """
        return np.diag(self.expenditure_matrix)

    @property
    def num_observations(self) -> int:
        """Number of observations T."""
        return self.prices.shape[0]

    @property
    def num_goods(self) -> int:
        """Number of goods N."""
        return self.prices.shape[1]

    @classmethod
    def from_dataframe(
        cls,
        df: Any,  # pandas.DataFrame
        price_cols: list[str],
        quantity_cols: list[str],
        session_id: str | None = None,
    ) -> ConsumerSession:
        """
        Create ConsumerSession from pandas DataFrame.

        Args:
            df: DataFrame with price and quantity columns
            price_cols: List of column names for prices (in order of goods)
            quantity_cols: List of column names for quantities (in order of goods)
            session_id: Optional session identifier

        Returns:
            ConsumerSession instance

        Example:
            >>> import pandas as pd
            >>> df = pd.DataFrame({
            ...     'p_A': [1.0, 2.0], 'p_B': [2.0, 1.0],
            ...     'q_A': [3.0, 1.0], 'q_B': [1.0, 3.0]
            ... })
            >>> session = ConsumerSession.from_dataframe(
            ...     df, price_cols=['p_A', 'p_B'], quantity_cols=['q_A', 'q_B']
            ... )
        """
        prices = df[price_cols].values
        quantities = df[quantity_cols].values
        return cls(prices=prices, quantities=quantities, session_id=session_id)

    @classmethod
    def from_long_format(
        cls,
        df: Any,  # pandas.DataFrame
        time_col: str = "time",
        item_col: str = "item_id",
        price_col: str = "price",
        qty_col: str = "quantity",
        session_id: str | None = None,
    ) -> ConsumerSession:
        """
        Create ConsumerSession from long-format transaction logs.

        Pivots SQL-style transaction data (one row per item per time) into
        wide-format matrices (one row per observation).

        Args:
            df: Long-format DataFrame with one row per item per time
            time_col: Column name for time/observation identifier
            item_col: Column name for item/product identifier
            price_col: Column name for price
            qty_col: Column name for quantity
            session_id: Optional session identifier

        Returns:
            ConsumerSession instance

        Note:
            Prices for items not purchased at a given time must be present
            in the data. Missing values will be filled with 0 for quantities
            but will raise an error if prices are missing.
        """
        import pandas as pd

        # Pivot to wide format
        q_pivot = df.pivot(index=time_col, columns=item_col, values=qty_col)
        p_pivot = df.pivot(index=time_col, columns=item_col, values=price_col)

        # Fill missing quantities with 0 (item not purchased)
        quantities = q_pivot.fillna(0).values

        # Prices should not be missing - need to know opportunity cost
        if p_pivot.isna().any().any():
            missing = p_pivot.isna().sum().sum()
            raise ValueError(
                f"Found {missing} missing prices. All prices must be provided "
                "to calculate opportunity costs for unbought items."
            )
        prices = p_pivot.values

        return cls(prices=prices, quantities=quantities, session_id=session_id)

    def split_by_window(self, window_size: int) -> list[ConsumerSession]:
        """
        Split session into non-overlapping windows.

        Useful for detecting structural breaks or analyzing consistency
        over different time periods.

        Args:
            window_size: Number of observations per window

        Returns:
            List of ConsumerSession instances, one per window
        """
        sessions = []
        for i in range(0, self.num_observations, window_size):
            end = min(i + window_size, self.num_observations)
            if end - i >= 2:  # Need at least 2 observations for RP analysis
                sessions.append(
                    ConsumerSession(
                        prices=self.prices[i:end],
                        quantities=self.quantities[i:end],
                        session_id=f"{self.session_id}_window_{i//window_size}"
                        if self.session_id
                        else None,
                    )
                )
        return sessions
