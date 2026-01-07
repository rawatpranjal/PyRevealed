"""Lancaster Characteristics Model for attribute-level preference analysis.

This module provides the LancasterLog class which transforms product-space
behavioral data into characteristics-space, enabling revealed preference
analysis at the attribute level rather than product level.

The Lancaster model (Lancaster, 1966) represents goods as bundles of
underlying characteristics. Instead of analyzing choices over products,
we analyze choices over the characteristics those products deliver.

Core transformations:
    - Z = X @ A (characteristics quantities)
    - Pi = NNLS(A, P[t]) for each t (shadow prices via hedonic regression)

Example:
    >>> import numpy as np
    >>> from pyrevealed import LancasterLog, validate_consistency
    >>> # Products: Apple (95 cal, 4.4g fiber), Banana (105 cal, 3.1g fiber)
    >>> A = np.array([[95.0, 4.4], [105.0, 3.1]])
    >>> prices = np.array([[1.0, 0.5], [0.8, 0.6]])
    >>> quantities = np.array([[2, 3], [4, 1]])
    >>> log = LancasterLog(cost_vectors=prices, action_vectors=quantities, attribute_matrix=A)
    >>> result = validate_consistency(log.behavior_log)

References:
    Lancaster, K. J. (1966). A new approach to consumer theory.
    Journal of Political Economy, 74(2), 132-157.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import nnls

from pyrevealed.core.session import BehaviorLog
from pyrevealed.core.result import LancasterResult
from pyrevealed.core.exceptions import (
    DataQualityWarning,
    DimensionError,
    InsufficientDataError,
    NaNInfError,
    ValueRangeError,
)


@dataclass
class LancasterLog:
    """
    Transform product-space behavior to characteristics-space for analysis.

    The Lancaster Characteristics Model represents goods as bundles of
    underlying characteristics. Instead of analyzing choices over products,
    we analyze choices over the characteristics those products deliver.

    This is useful when:
    - Products share common attributes (e.g., calories, protein, storage, speed)
    - You want to understand preferences over attributes, not specific products
    - You need to predict demand for new product configurations
    - Users appear inconsistent in product-space but may be rational in
      characteristics-space (comparing specs, not brands)

    Attributes:
        cost_vectors: T x N matrix of product prices
        action_vectors: T x N matrix of product quantities
        attribute_matrix: N x K matrix where A[n,k] = amount of characteristic k
            in one unit of product n
        user_id: Optional identifier for the user/session
        metadata: Optional additional attributes (e.g., characteristic_names)

    Properties:
        characteristics_quantities: T x K matrix Z = X @ A
        shadow_prices: T x K matrix of implied characteristic prices
        nnls_residuals: T-length array of NNLS fit residuals
        behavior_log: BehaviorLog in characteristics space for use with algorithms

    Example:
        >>> import numpy as np
        >>> from pyrevealed import LancasterLog
        >>> # 3 products (servers), 2 characteristics (vCPU, RAM_GB)
        >>> A = np.array([[2, 2], [2, 8], [4, 8]])  # t3.small, m5.large, c5.xlarge
        >>> prices = np.array([[0.02, 0.09, 0.17], [0.02, 0.08, 0.16]])
        >>> quantities = np.array([[10, 0, 0], [0, 5, 0]])
        >>> log = LancasterLog(
        ...     cost_vectors=prices,
        ...     action_vectors=quantities,
        ...     attribute_matrix=A,
        ...     metadata={"characteristic_names": ["vCPU", "RAM_GB"]}
        ... )
        >>> char_log = log.behavior_log
        >>> # Now use char_log with any pyrevealed algorithm
    """

    # Input data (with defaults for legacy alias support)
    cost_vectors: NDArray[np.float64] | None = None
    action_vectors: NDArray[np.float64] | None = None
    attribute_matrix: NDArray[np.float64] | None = None
    user_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Legacy aliases (for consistency with BehaviorLog)
    prices: NDArray[np.float64] | None = field(default=None, repr=False)
    quantities: NDArray[np.float64] | None = field(default=None, repr=False)

    # Cached results (internal)
    _characteristics_quantities: NDArray[np.float64] | None = field(
        default=None, repr=False, compare=False, init=False
    )
    _shadow_prices: NDArray[np.float64] | None = field(
        default=None, repr=False, compare=False, init=False
    )
    _nnls_residuals: NDArray[np.float64] | None = field(
        default=None, repr=False, compare=False, init=False
    )
    _behavior_log: BehaviorLog | None = field(
        default=None, repr=False, compare=False, init=False
    )

    def __post_init__(self) -> None:
        """Validate and transform data."""
        # Resolve aliases (same pattern as BehaviorLog)
        if self.cost_vectors is None and self.prices is not None:
            object.__setattr__(self, "cost_vectors", self.prices)
        if self.action_vectors is None and self.quantities is not None:
            object.__setattr__(self, "action_vectors", self.quantities)

        # Validate required fields are provided
        if self.cost_vectors is None:
            raise ValueError("cost_vectors (or prices) is required")
        if self.action_vectors is None:
            raise ValueError("action_vectors (or quantities) is required")
        if self.attribute_matrix is None:
            raise ValueError("attribute_matrix is required")

        # Ensure arrays and float64
        object.__setattr__(
            self, "cost_vectors", np.asarray(self.cost_vectors, dtype=np.float64)
        )
        object.__setattr__(
            self, "action_vectors", np.asarray(self.action_vectors, dtype=np.float64)
        )
        object.__setattr__(
            self,
            "attribute_matrix",
            np.asarray(self.attribute_matrix, dtype=np.float64),
        )

        # Keep legacy aliases in sync
        object.__setattr__(self, "prices", self.cost_vectors)
        object.__setattr__(self, "quantities", self.action_vectors)

        self._validate()
        self._compute_transformation()

    def _validate(self) -> None:
        """Validate input dimensions and values."""
        # Check for NaN/Inf in all arrays
        for name, arr in [
            ("cost_vectors", self.cost_vectors),
            ("action_vectors", self.action_vectors),
            ("attribute_matrix", self.attribute_matrix),
        ]:
            if not np.all(np.isfinite(arr)):
                invalid_count = int(np.sum(~np.isfinite(arr)))
                raise NaNInfError(
                    f"Found {invalid_count} NaN/Inf values in {name}. "
                    f"All values must be finite numbers. "
                    f"Hint: Use np.nan_to_num() or filter invalid values before creating LancasterLog."
                )

        # Check cost_vectors shape
        if self.cost_vectors.ndim != 2:
            raise DimensionError(
                f"cost_vectors must be 2D (T x N), got {self.cost_vectors.ndim}D "
                f"with shape {self.cost_vectors.shape}."
            )

        T, N = self.cost_vectors.shape

        # Check action_vectors shape
        if self.action_vectors.shape != self.cost_vectors.shape:
            raise DimensionError(
                f"action_vectors shape {self.action_vectors.shape} must match "
                f"cost_vectors shape {self.cost_vectors.shape}. "
                f"Hint: Both matrices must have the same T observations and N products."
            )

        # Check attribute_matrix shape
        if self.attribute_matrix.ndim != 2:
            raise DimensionError(
                f"attribute_matrix must be 2D (N x K), got {self.attribute_matrix.ndim}D "
                f"with shape {self.attribute_matrix.shape}."
            )

        if self.attribute_matrix.shape[0] != N:
            raise DimensionError(
                f"attribute_matrix rows ({self.attribute_matrix.shape[0]}) must match "
                f"number of products N ({N}). "
                f"The attribute matrix should have one row per product."
            )

        K = self.attribute_matrix.shape[1]

        # Minimum size requirements
        if T < 1:
            raise InsufficientDataError(
                "Must have at least one observation. "
                "Hint: Check that your data is not empty after preprocessing."
            )
        if N < 1:
            raise InsufficientDataError(
                "Must have at least one product. "
                "Hint: Check that your data has at least one product column."
            )
        if K < 1:
            raise InsufficientDataError(
                "Must have at least one characteristic. "
                "Hint: The attribute matrix must have at least one column."
            )

        # Value checks
        if np.any(self.cost_vectors <= 0):
            invalid_positions = np.argwhere(self.cost_vectors <= 0)
            pos_preview = invalid_positions[:5].tolist()
            raise ValueRangeError(
                f"Found {len(invalid_positions)} non-positive costs at positions: "
                f"{pos_preview}{'...' if len(invalid_positions) > 5 else ''}. "
                f"All costs/prices must be strictly positive."
            )
        if np.any(self.action_vectors < 0):
            invalid_positions = np.argwhere(self.action_vectors < 0)
            pos_preview = invalid_positions[:5].tolist()
            raise ValueRangeError(
                f"Found {len(invalid_positions)} negative actions at positions: "
                f"{pos_preview}{'...' if len(invalid_positions) > 5 else ''}. "
                f"All actions/quantities must be non-negative."
            )
        if np.any(self.attribute_matrix < 0):
            invalid_positions = np.argwhere(self.attribute_matrix < 0)
            pos_preview = invalid_positions[:5].tolist()
            raise ValueRangeError(
                f"Found {len(invalid_positions)} negative attribute values at positions: "
                f"{pos_preview}{'...' if len(invalid_positions) > 5 else ''}. "
                f"All attribute values must be non-negative."
            )

        # Check for zero rows in A (products with no characteristics)
        zero_rows = np.where(self.attribute_matrix.sum(axis=1) == 0)[0]
        if len(zero_rows) > 0:
            raise ValueRangeError(
                f"Products {zero_rows.tolist()} have no characteristics (all zeros in A). "
                "Every product must deliver at least one characteristic. "
                "Hint: Remove products with no attributes or add missing attribute values."
            )

        # Check for zero columns in A (unused characteristics) - warn only
        zero_cols = np.where(self.attribute_matrix.sum(axis=0) == 0)[0]
        if len(zero_cols) > 0:
            warnings.warn(
                f"Characteristics {zero_cols.tolist()} are not present in any product. "
                "Consider removing these columns from the attribute matrix.",
                DataQualityWarning,
                stacklevel=3,
            )

        # Rank check (warning, not error)
        rank = np.linalg.matrix_rank(self.attribute_matrix)
        if rank < K:
            warnings.warn(
                f"Attribute matrix is rank-deficient (rank={rank}, K={K}). "
                "Shadow prices may not be unique. Consider reducing characteristics "
                "or adding more products.",
                DataQualityWarning,
                stacklevel=3,
            )

    def _compute_transformation(self) -> None:
        """Compute characteristics quantities and shadow prices."""
        T = self.num_observations

        # Step 1: Compute characteristics quantities Z = X @ A
        # Z[t,k] = sum_n X[t,n] * A[n,k]
        object.__setattr__(
            self,
            "_characteristics_quantities",
            self.action_vectors @ self.attribute_matrix,
        )

        # Step 2: Compute shadow prices via NNLS for each observation
        # For each t: find Pi[t] >= 0 such that A @ Pi[t] ≈ P[t]
        # This is: minimize ||A @ pi - p_t||^2 subject to pi >= 0
        K = self.num_characteristics
        shadow_prices = np.zeros((T, K))
        nnls_residuals = np.zeros(T)

        for t in range(T):
            # scipy.optimize.nnls(A, b) solves min||Ax - b||^2 s.t. x >= 0
            # We have: product price p_n = sum_k A[n,k] * pi_k
            # In matrix form: p = A @ pi (p is N-vec, A is N x K, pi is K-vec)
            pi, residual = nnls(self.attribute_matrix, self.cost_vectors[t])
            shadow_prices[t] = pi
            nnls_residuals[t] = residual

        object.__setattr__(self, "_shadow_prices", shadow_prices)
        object.__setattr__(self, "_nnls_residuals", nnls_residuals)

    @property
    def characteristics_quantities(self) -> NDArray[np.float64]:
        """T x K matrix of characteristics consumed: Z = X @ A."""
        return self._characteristics_quantities

    @property
    def shadow_prices(self) -> NDArray[np.float64]:
        """T x K matrix of implied characteristic prices via NNLS."""
        return self._shadow_prices

    @property
    def nnls_residuals(self) -> NDArray[np.float64]:
        """T-length array of NNLS fit residuals per observation."""
        return self._nnls_residuals

    @property
    def behavior_log(self) -> BehaviorLog:
        """BehaviorLog in characteristics space for use with algorithms.

        Note: BehaviorLog requires strictly positive costs. If NNLS produces
        zero shadow prices for some characteristics (meaning those characteristics
        don't contribute to pricing), we add a tiny epsilon to avoid validation
        errors. This has negligible effect on consistency analysis.
        """
        if self._behavior_log is None:
            # Ensure all shadow prices are strictly positive for BehaviorLog
            # NNLS can produce zeros, which would fail BehaviorLog validation
            shadow_prices = self._shadow_prices.copy()
            min_positive = (
                shadow_prices[shadow_prices > 0].min()
                if np.any(shadow_prices > 0)
                else 1e-10
            )
            epsilon = min_positive * 1e-6  # Tiny relative to smallest positive price
            shadow_prices = np.maximum(shadow_prices, epsilon)

            object.__setattr__(
                self,
                "_behavior_log",
                BehaviorLog(
                    cost_vectors=shadow_prices,
                    action_vectors=self._characteristics_quantities,
                    user_id=f"{self.user_id}_characteristics" if self.user_id else None,
                    metadata={
                        **self.metadata,
                        "lancaster_source": True,
                        "num_products": self.num_products,
                        "num_characteristics": self.num_characteristics,
                    },
                ),
            )
        return self._behavior_log

    @property
    def num_observations(self) -> int:
        """Number of observations T."""
        return self.cost_vectors.shape[0]

    @property
    def num_products(self) -> int:
        """Number of products N."""
        return self.cost_vectors.shape[1]

    @property
    def num_characteristics(self) -> int:
        """Number of characteristics K."""
        return self.attribute_matrix.shape[1]

    # Tech-friendly aliases
    @property
    def num_records(self) -> int:
        """Alias for num_observations."""
        return self.num_observations

    @property
    def num_features(self) -> int:
        """Alias for num_characteristics."""
        return self.num_characteristics

    def valuation_report(
        self,
        residual_threshold: float = 0.1,
    ) -> LancasterResult:
        """
        Generate business insights from shadow price analysis.

        This method provides actionable insights including:
        - Mean shadow prices: How much users implicitly value each attribute
        - Spend shares: What fraction of budget goes to each characteristic
        - Model diagnostics: How well the hedonic model fits the price data

        Args:
            residual_threshold: Relative residual threshold for flagging
                observations as problematic. Default is 0.1 (10%).

        Returns:
            LancasterResult with valuation insights and diagnostics.

        Example:
            >>> report = lancaster_log.valuation_report()
            >>> print(f"Users value {report.characteristic_names[0]} at "
            ...       f"${report.mean_shadow_prices[0]:.2f} per unit")
        """
        start_time = time.perf_counter()

        K = self.num_characteristics

        # Shadow price statistics
        mean_prices = np.mean(self._shadow_prices, axis=0)
        std_prices = np.std(self._shadow_prices, axis=0)

        # Coefficient of variation (handle zero means)
        with np.errstate(divide="ignore", invalid="ignore"):
            cv_prices = np.where(mean_prices > 1e-10, std_prices / mean_prices, 0.0)

        # Spend on each characteristic: sum_t (pi[t,k] * z[t,k])
        characteristic_spend = np.sum(
            self._shadow_prices * self._characteristics_quantities, axis=0
        )
        total_spend = characteristic_spend.sum()
        spend_shares = (
            characteristic_spend / total_spend if total_spend > 0 else np.zeros(K)
        )

        # Residual analysis
        mean_residual = float(np.mean(self._nnls_residuals))
        max_residual = float(np.max(self._nnls_residuals))

        # Flag observations with high relative residual
        total_prices = self.cost_vectors.sum(axis=1)
        relative_residuals = self._nnls_residuals / np.maximum(total_prices, 1e-10)
        problematic = np.where(relative_residuals > residual_threshold)[0].tolist()

        # Matrix diagnostics
        rank = int(np.linalg.matrix_rank(self.attribute_matrix))
        is_well_specified = rank == K and K <= self.num_products

        # Get characteristic names from metadata if available
        char_names = self.metadata.get("characteristic_names")

        computation_time = (time.perf_counter() - start_time) * 1000

        return LancasterResult(
            mean_shadow_prices=mean_prices,
            shadow_price_std=std_prices,
            shadow_price_cv=cv_prices,
            total_spend_on_characteristics=characteristic_spend,
            spend_shares=spend_shares,
            mean_nnls_residual=mean_residual,
            max_nnls_residual=max_residual,
            problematic_observations=problematic,
            attribute_matrix_rank=rank,
            is_well_specified=is_well_specified,
            characteristic_names=char_names,
            computation_time_ms=computation_time,
        )


def transform_to_characteristics(
    log: BehaviorLog,
    attribute_matrix: NDArray[np.float64],
    characteristic_names: list[str] | None = None,
) -> LancasterLog:
    """
    Transform a BehaviorLog to characteristics space.

    Convenience function for creating a LancasterLog from an existing
    BehaviorLog and attribute matrix. This is useful when you already
    have product-space data and want to analyze it at the attribute level.

    Args:
        log: Existing BehaviorLog in product space
        attribute_matrix: N x K matrix where A[n,k] = amount of characteristic k
            in one unit of product n
        characteristic_names: Optional names for characteristics (e.g.,
            ["calories", "protein", "fiber"])

    Returns:
        LancasterLog with characteristics-space transformation ready for
        use with all standard pyrevealed algorithms.

    Example:
        >>> from pyrevealed import BehaviorLog, transform_to_characteristics
        >>> # Existing user behavior log
        >>> user_log = BehaviorLog(cost_vectors=prices, action_vectors=quantities)
        >>> # Attribute matrix: products x characteristics
        >>> A = np.array([[95, 4.4], [105, 3.1]])  # calories, fiber per product
        >>> # Transform and analyze
        >>> char_log = transform_to_characteristics(user_log, A, ["calories", "fiber"])
        >>> result = validate_consistency(char_log.behavior_log)
    """
    metadata = {**log.metadata}
    if characteristic_names:
        metadata["characteristic_names"] = characteristic_names

    return LancasterLog(
        cost_vectors=log.cost_vectors,
        action_vectors=log.action_vectors,
        attribute_matrix=attribute_matrix,
        user_id=log.user_id,
        metadata=metadata,
    )


# =============================================================================
# LEGACY ALIASES (for backward compatibility with economics terminology)
# =============================================================================

# CharacteristicsLog: Alias for LancasterLog
CharacteristicsLog = LancasterLog
"""
Alias for LancasterLog.

Use LancasterLog for the canonical name referencing the economics literature.
"""
