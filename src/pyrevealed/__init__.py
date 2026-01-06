"""
PyRevealed: Revealed Preference Analysis for Consumer Behavior.

Non-parametric behavioral consistency metrics for fraud detection and user segmentation.
"""

from pyrevealed.core.session import ConsumerSession, RiskSession, SpatialSession
from pyrevealed.core.result import (
    GARPResult,
    AEIResult,
    MPIResult,
    UtilityRecoveryResult,
    RiskProfileResult,
    IdealPointResult,
    SeparabilityResult,
)
from pyrevealed.algorithms.garp import check_garp
from pyrevealed.algorithms.aei import compute_aei
from pyrevealed.algorithms.mpi import compute_mpi
from pyrevealed.algorithms.utility import recover_utility, construct_afriat_utility
from pyrevealed.algorithms.risk import (
    compute_risk_profile,
    check_expected_utility_axioms,
    classify_risk_type,
)
from pyrevealed.algorithms.spatial import (
    find_ideal_point,
    check_euclidean_rationality,
    compute_preference_strength,
    find_multiple_ideal_points,
)
from pyrevealed.algorithms.separability import (
    check_separability,
    find_separable_partition,
    compute_cannibalization,
)

__version__ = "0.2.0"

__all__ = [
    # Data structures
    "ConsumerSession",
    "RiskSession",
    "SpatialSession",
    # Result types
    "GARPResult",
    "AEIResult",
    "MPIResult",
    "UtilityRecoveryResult",
    "RiskProfileResult",
    "IdealPointResult",
    "SeparabilityResult",
    # Core algorithms (GARP, AEI, MPI)
    "check_garp",
    "compute_aei",
    "compute_mpi",
    "recover_utility",
    "construct_afriat_utility",
    # Risk profile analysis
    "compute_risk_profile",
    "check_expected_utility_axioms",
    "classify_risk_type",
    # Spatial/ideal point analysis
    "find_ideal_point",
    "check_euclidean_rationality",
    "compute_preference_strength",
    "find_multiple_ideal_points",
    # Separability analysis
    "check_separability",
    "find_separable_partition",
    "compute_cannibalization",
    # Convenience
    "get_integrity_score",
]


def get_integrity_score(session: ConsumerSession, tolerance: float = 1e-6) -> float:
    """
    Convenience function to get the Afriat Efficiency Index score directly.

    Args:
        session: ConsumerSession with prices and quantities
        tolerance: Convergence tolerance for binary search

    Returns:
        Float between 0 (irrational) and 1 (perfectly rational)
    """
    result = compute_aei(session, tolerance=tolerance)
    return result.efficiency_index
