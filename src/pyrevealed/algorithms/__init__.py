"""Core algorithms for revealed preference analysis."""

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

__all__ = [
    # Core consistency
    "check_garp",
    "compute_aei",
    "compute_mpi",
    "recover_utility",
    "construct_afriat_utility",
    # Risk analysis
    "compute_risk_profile",
    "check_expected_utility_axioms",
    "classify_risk_type",
    # Spatial analysis
    "find_ideal_point",
    "check_euclidean_rationality",
    "compute_preference_strength",
    "find_multiple_ideal_points",
    # Separability analysis
    "check_separability",
    "find_separable_partition",
    "compute_cannibalization",
]
