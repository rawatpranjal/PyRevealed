"""
EVAL: AEI binary search edge cases.

Tests for AEI (Afriat Efficiency Index) convergence and boundary conditions.
"""

import numpy as np
import pytest
from pyrevealed.core.session import BehaviorLog


class TestAEIBinarySearch:
    """EVAL: AEI binary search convergence issues."""

    def test_aei_max_iterations_exceeded(self):
        """EVAL: AEI may not converge within max_iterations=50.

        In aei.py around line 123:
            for iteration in range(max_iterations):
                ...
            return mid  # May not have converged

        With tight tolerance and complex data, 50 iterations may be insufficient.
        """
        from pyrevealed.algorithms.aei import compute_aei

        # Create data that requires many iterations
        np.random.seed(42)
        log = BehaviorLog(
            cost_vectors=np.random.rand(20, 5) + 0.1,
            action_vectors=np.random.rand(20, 5) + 0.1,
        )

        result = compute_aei(log, tolerance=1e-15)

        # Check if iteration limit was a concern
        # AEI should still be valid
        assert 0.0 <= result.efficiency_index <= 1.0

    def test_aei_convergence_at_extremes(self):
        """EVAL: AEI binary search at 0 and 1 boundaries."""
        from pyrevealed.algorithms.aei import compute_aei

        # Perfectly consistent data - AEI should be 1.0
        log = BehaviorLog(
            cost_vectors=np.array([[1.0, 2.0], [2.0, 1.0]]),
            action_vectors=np.array([[4.0, 1.0], [1.0, 4.0]]),
        )

        result = compute_aei(log)
        assert result.efficiency_index == 1.0, (
            f"Consistent data should have AEI=1.0, got {result.efficiency_index}"
        )

    def test_aei_boundary_violation(self):
        """EVAL: AEI just below 1.0 (near-consistent data)."""
        from pyrevealed.algorithms.aei import compute_aei

        # Data with tiny violation
        log = BehaviorLog(
            cost_vectors=np.array([[1.0, 1.0], [1.0, 1.0]]),
            action_vectors=np.array([[3.0, 1.0], [1.0, 3.0]]),
        )

        result = compute_aei(log)

        # AEI should be slightly below 1.0
        assert result.efficiency_index < 1.0


class TestAEIPrecision:
    """EVAL: AEI numerical precision issues."""

    def test_aei_very_tight_tolerance(self):
        """EVAL: AEI with tolerance=1e-15 (near machine precision)."""
        from pyrevealed.algorithms.aei import compute_aei

        log = BehaviorLog(
            cost_vectors=np.array([[1.0, 2.0], [2.0, 1.0]]),
            action_vectors=np.array([[3.0, 1.0], [1.0, 3.0]]),
        )

        result = compute_aei(log, tolerance=1e-15)

        # Should still produce valid result
        assert 0.0 <= result.efficiency_index <= 1.0

    def test_aei_precision_loss(self):
        """EVAL: AEI with ill-conditioned data may lose precision."""
        from pyrevealed.algorithms.aei import compute_aei

        # Data with extreme scale differences
        log = BehaviorLog(
            cost_vectors=np.array([[1e-10, 1e10], [1e10, 1e-10]]),
            action_vectors=np.array([[1e10, 1e-10], [1e-10, 1e10]]),
        )

        result = compute_aei(log)

        # AEI should still be valid
        assert 0.0 <= result.efficiency_index <= 1.0


class TestAEIEdgeCases:
    """EVAL: AEI edge cases."""

    def test_aei_identical_observations(self):
        """EVAL: AEI with all identical observations."""
        from pyrevealed.algorithms.aei import compute_aei

        log = BehaviorLog(
            cost_vectors=np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]),
            action_vectors=np.array([[3.0, 1.0], [3.0, 1.0], [3.0, 1.0]]),
        )

        result = compute_aei(log)

        # Identical observations are trivially consistent
        assert result.efficiency_index == 1.0

    def test_aei_single_good_consistency(self):
        """EVAL: AEI with N=1 single good."""
        from pyrevealed.algorithms.aei import compute_aei

        log = BehaviorLog(
            cost_vectors=np.array([[1.0], [2.0], [3.0]]),
            action_vectors=np.array([[5.0], [4.0], [3.0]]),
        )

        result = compute_aei(log)

        # Single good is always GARP-consistent
        assert result.efficiency_index == 1.0

    def test_aei_monotonicity(self):
        """EVAL: AEI should decrease as violations increase."""
        from pyrevealed.algorithms.aei import compute_aei

        # Consistent data
        consistent = BehaviorLog(
            cost_vectors=np.array([[1.0, 2.0], [2.0, 1.0]]),
            action_vectors=np.array([[4.0, 1.0], [1.0, 4.0]]),
        )

        # Violation data
        violation = BehaviorLog(
            cost_vectors=np.array([[1.0, 1.0], [1.0, 1.0]]),
            action_vectors=np.array([[3.0, 1.0], [1.0, 3.0]]),
        )

        aei_consistent = compute_aei(consistent).efficiency_index
        aei_violation = compute_aei(violation).efficiency_index

        assert aei_consistent >= aei_violation, (
            f"Consistent ({aei_consistent}) should have higher AEI than violation ({aei_violation})"
        )


class TestCCEI:
    """EVAL: CCEI (Critical Cost Efficiency Index) computation."""

    def test_ccei_consistent_data(self):
        """EVAL: CCEI should equal 1.0 for consistent data."""
        from pyrevealed.algorithms.aei import compute_ccei

        log = BehaviorLog(
            cost_vectors=np.array([[1.0, 2.0], [2.0, 1.0]]),
            action_vectors=np.array([[4.0, 1.0], [1.0, 4.0]]),
        )

        result = compute_ccei(log)

        assert result.efficiency_index == 1.0

    def test_ccei_equals_aei(self):
        """EVAL: CCEI should equal AEI (they're the same metric)."""
        from pyrevealed.algorithms.aei import compute_aei, compute_ccei

        log = BehaviorLog(
            cost_vectors=np.array([[1.0, 1.0], [1.0, 1.0]]),
            action_vectors=np.array([[3.0, 1.0], [1.0, 3.0]]),
        )

        aei = compute_aei(log)
        ccei = compute_ccei(log)

        assert np.isclose(aei.efficiency_index, ccei.efficiency_index), (
            f"AEI ({aei.efficiency_index}) should equal CCEI ({ccei.efficiency_index})"
        )


class TestAEIWithTolerance:
    """EVAL: AEI interaction with GARP tolerance."""

    def test_aei_tolerance_consistency_with_garp(self):
        """EVAL: AEI=1.0 should imply GARP consistency at same tolerance."""
        from pyrevealed.algorithms.aei import compute_aei
        from pyrevealed.algorithms.garp import check_garp

        log = BehaviorLog(
            cost_vectors=np.array([[1.0, 2.0], [2.0, 1.0]]),
            action_vectors=np.array([[4.0, 1.0], [1.0, 4.0]]),
        )

        tol = 1e-10
        aei = compute_aei(log, tolerance=tol)
        garp = check_garp(log, tolerance=tol)

        if aei.efficiency_index == 1.0:
            assert garp.is_consistent, "AEI=1.0 should imply GARP consistency"

    def test_aei_different_tolerances(self):
        """EVAL: AEI may differ with different tolerances."""
        from pyrevealed.algorithms.aei import compute_aei

        log = BehaviorLog(
            cost_vectors=np.array([[1.0, 1.0], [1.0, 1.0]]),
            action_vectors=np.array([
                [2.5, 2.5],
                [2.5 + 1e-9, 2.5 + 1e-9],  # Nearly equal
            ]),
        )

        aei_tight = compute_aei(log, tolerance=1e-12)
        aei_loose = compute_aei(log, tolerance=1e-6)

        # Results may differ
        print(f"Tight tolerance: AEI={aei_tight.efficiency_index}")
        print(f"Loose tolerance: AEI={aei_loose.efficiency_index}")
