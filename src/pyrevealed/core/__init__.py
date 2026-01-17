"""Core data structures for PyRevealed."""

from pyrevealed.core.session import ConsumerSession
from pyrevealed.core.result import (
    GARPResult,
    AEIResult,
    MPIResult,
    UtilityRecoveryResult,
)
from pyrevealed.core.exceptions import (
    PyRevealedError,
    DataValidationError,
    DimensionError,
    ValueRangeError,
    NaNInfError,
    OptimizationError,
    SolverError,
    RegressionError,
    StatisticalError,
    ComputationalLimitError,
    NotFittedError,
    InsufficientDataError,
    DataQualityWarning,
    NumericalInstabilityWarning,
)

__all__ = [
    "ConsumerSession",
    "GARPResult",
    "AEIResult",
    "MPIResult",
    "UtilityRecoveryResult",
    # Exceptions
    "PyRevealedError",
    "DataValidationError",
    "DimensionError",
    "ValueRangeError",
    "NaNInfError",
    "OptimizationError",
    "SolverError",
    "RegressionError",
    "StatisticalError",
    "ComputationalLimitError",
    "NotFittedError",
    "InsufficientDataError",
    # Warnings
    "DataQualityWarning",
    "NumericalInstabilityWarning",
]
