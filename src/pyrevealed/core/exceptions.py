"""Custom exceptions and warnings for PyRevealed.

This module provides a hierarchy of exceptions for specific error types,
all inheriting from ValueError for backward compatibility with existing
error handling code.

Exception Hierarchy:
    PyRevealedError (ValueError)
    ├── DataValidationError
    │   ├── DimensionError
    │   ├── ValueRangeError
    │   └── NaNInfError
    ├── OptimizationError
    ├── SolverError
    ├── RegressionError
    ├── StatisticalError
    ├── ComputationalLimitError
    ├── NotFittedError
    └── InsufficientDataError

Warning Classes:
    DataQualityWarning (UserWarning)
    NumericalInstabilityWarning (UserWarning)
"""

from __future__ import annotations


# =============================================================================
# BASE EXCEPTION
# =============================================================================


class PyRevealedError(ValueError):
    """Base exception for all PyRevealed errors.

    Inherits from ValueError for backward compatibility - existing code
    that catches ValueError will continue to work.

    Example:
        >>> try:
        ...     log = BehaviorLog(prices, quantities)
        ... except PyRevealedError as e:
        ...     print(f"PyRevealed error: {e}")
        ... except ValueError as e:  # Also catches PyRevealedError
        ...     print(f"Value error: {e}")
    """

    pass


# =============================================================================
# DATA VALIDATION EXCEPTIONS
# =============================================================================


class DataValidationError(PyRevealedError):
    """Raised when input data fails validation checks.

    This is the base class for all data-related validation errors.
    Use more specific subclasses when possible.

    Common causes:
        - Mismatched array dimensions
        - Invalid value ranges
        - Missing or corrupted data
    """

    pass


class DimensionError(DataValidationError):
    """Raised when array dimensions are incompatible.

    Common causes:
        - cost_vectors and action_vectors have different shapes
        - Arrays are not 2D (T x N)
        - Empty arrays (T=0 or N=0)

    Example:
        >>> prices = np.array([[1, 2, 3]])      # shape (1, 3)
        >>> quantities = np.array([[1, 2]])     # shape (1, 2)
        >>> BehaviorLog(prices, quantities)
        DimensionError: cost_vectors shape (1, 3) does not match
        action_vectors shape (1, 2)...
    """

    pass


class ValueRangeError(DataValidationError):
    """Raised when values are outside expected ranges.

    Common causes:
        - Negative or zero prices/costs
        - Negative quantities/actions
        - Probabilities outside [0, 1]

    Example:
        >>> prices = np.array([[1, -2]])  # negative price!
        >>> BehaviorLog(prices, quantities)
        ValueRangeError: Found 1 non-positive costs at positions [(0, 1)]...
    """

    pass


class NaNInfError(DataValidationError):
    """Raised when NaN or Inf values are detected in input data.

    By default, PyRevealed raises this error when NaN/Inf values are found.
    Use nan_policy='drop' or nan_policy='warn' to handle them automatically.

    Common causes:
        - Missing data encoded as NaN
        - Division by zero in preprocessing
        - Numeric overflow producing Inf

    Example:
        >>> prices = np.array([[1, np.nan]])
        >>> BehaviorLog(prices, quantities)
        NaNInfError: Found 1 NaN/Inf values in 1 observations...

        >>> # Solution: use nan_policy to handle automatically
        >>> BehaviorLog(prices, quantities, nan_policy='drop')
    """

    pass


# =============================================================================
# COMPUTATION EXCEPTIONS
# =============================================================================


class OptimizationError(PyRevealedError):
    """Raised when an optimization solver fails to find a solution.

    This typically occurs when:
        - Data is too inconsistent for utility recovery
        - Linear programming constraints are infeasible
        - Numerical issues prevent convergence

    Common causes:
        - Behavior has low integrity score (< 0.7)
        - Extreme values causing numerical instability
        - Too few observations for the operation

    Suggested fixes:
        1. Check data quality first with compute_integrity_score()
        2. Filter highly inconsistent observations
        3. Scale data to avoid extreme values
    """

    pass


class SolverError(PyRevealedError):
    """Raised when a linear programming or optimization solver fails.

    This occurs when scipy.optimize.linprog or similar solvers cannot find
    a feasible solution. Unlike OptimizationError (which covers general
    numerical optimization), SolverError is specific to LP/MILP solvers.

    Common causes:
        - Infeasible constraints (no solution exists)
        - Unbounded problem (solution goes to infinity)
        - Numerical issues in the constraint matrix
        - Data violates required consistency conditions

    Suggested fixes:
        1. Check data consistency with validate_consistency()
        2. Verify input data doesn't have extreme values
        3. Ensure problem is well-posed for the algorithm
    """

    pass


class RegressionError(PyRevealedError):
    """Raised when a statistical regression fails.

    This occurs when OLS, 2SLS, or other regression methods cannot
    produce valid estimates.

    Common causes:
        - Singular or near-singular design matrix
        - Perfect multicollinearity among regressors
        - Insufficient degrees of freedom
        - All-zero dependent variable

    Suggested fixes:
        1. Check for multicollinearity in predictors
        2. Ensure sufficient observations relative to parameters
        3. Verify data has variation in both X and Y
    """

    pass


class StatisticalError(PyRevealedError):
    """Raised when a statistical computation fails.

    This covers failures in statistical tests, p-value calculations,
    bootstrap procedures, and similar statistical methods.

    Common causes:
        - Division by zero in test statistics
        - Degenerate distributions (zero variance)
        - Bootstrap samples all identical
        - Invalid degrees of freedom

    Suggested fixes:
        1. Ensure data has sufficient variation
        2. Check for degenerate cases (all same value)
        3. Increase sample size if possible
    """

    pass


class ComputationalLimitError(PyRevealedError):
    """Raised when a problem exceeds computational feasibility.

    Some algorithms have exponential or factorial complexity and cannot
    be run on large inputs without specialized solvers.

    Common causes:
        - Kemeny aggregation for n > 10 alternatives (NP-hard)
        - RAM search for n > 6 items (factorial enumeration)
        - Exact Houtman-Maks without ILP solver available

    The error message indicates what solver or approach is needed,
    or suggests using an approximate method if available as a
    separate explicit function.
    """

    pass


class NotFittedError(PyRevealedError):
    """Raised when an operation requires a fitted model.

    This occurs when calling transform or prediction methods
    on a PreferenceEncoder before fitting it.

    Example:
        >>> encoder = PreferenceEncoder()
        >>> encoder.transform(log)  # forgot to fit first!
        NotFittedError: Encoder not fitted. Call fit() first...
    """

    pass


class InsufficientDataError(PyRevealedError):
    """Raised when there is not enough data for the requested operation.

    Some operations require minimum amounts of data:
        - At least 2 observations for preference comparisons
        - At least 3 observations for cycle detection
        - Multiple observations per group for separability testing

    Example:
        >>> log = BehaviorLog(prices[:1], quantities[:1])  # only 1 obs
        >>> compute_granular_integrity(log)
        InsufficientDataError: Need at least 3 observations for VEI...
    """

    pass


# =============================================================================
# WARNINGS
# =============================================================================


class DataQualityWarning(UserWarning):
    """Warning for data quality issues that don't prevent computation.

    Emitted when:
        - Rows with NaN/Inf are dropped (nan_policy='warn')
        - Attribute matrix is rank-deficient
        - Characteristics have zero values across all products

    These issues may affect results but don't prevent computation.
    Address them for best results.

    Example:
        >>> import warnings
        >>> # Suppress data quality warnings
        >>> warnings.filterwarnings('ignore', category=DataQualityWarning)
        >>>
        >>> # Or promote to errors
        >>> warnings.filterwarnings('error', category=DataQualityWarning)
    """

    pass


class NumericalInstabilityWarning(UserWarning):
    """Warning for potential numerical issues in computations.

    Emitted when:
        - Division involves near-zero denominators
        - Optimization converges slowly or hits iteration limit
        - Matrix operations may be ill-conditioned

    Results may be less reliable when this warning appears.
    Consider adjusting tolerance parameters or scaling data.
    """

    pass
