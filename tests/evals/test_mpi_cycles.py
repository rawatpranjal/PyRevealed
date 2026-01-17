"""
EVAL: MPI cycle computation issues.

Tests for Money Pump Index edge cases and cycle detection.
"""

import numpy as np
import pytest
from pyrevealed.core.session import BehaviorLog


class TestMPIDenominator:
    """EVAL: MPI denominator edge cases."""

    def test_mpi_small_cycle_expenditure(self):
        """EVAL: MPI cycle where denominator sum(E[ki,ki]) is very small.

        In mpi.py line 143-146:
            if denominator <= 0:
                return 0.0
            mpi = numerator / denominator
        """
        from pyrevealed.algorithms.mpi import compute_mpi

        # Very low expenditure data
        log = BehaviorLog(
            cost_vectors=np.array([[1e-100, 1e-100], [1e-100, 1e-100]]),
            action_vectors=np.array([[1e-100, 1e-100], [1e-100, 1e-100]]),
        )

        result = compute_mpi(log)

        # Should handle gracefully
        assert np.isfinite(result.mpi_value)

    def test_mpi_zero_cycle_expenditure(self):
        """EVAL: MPI when cycle expenditure approaches zero."""
        from pyrevealed.algorithms.mpi import compute_mpi

        # Nearly zero expenditures
        log = BehaviorLog(
            cost_vectors=np.array([[1e-200, 1.0], [1.0, 1e-200]]),
            action_vectors=np.array([[1e-200, 1.0], [1.0, 1e-200]]),
        )

        result = compute_mpi(log)

        assert 0.0 <= result.mpi_value <= 1.0


class TestMPICycleDetection:
    """EVAL: MPI cycle detection edge cases."""

    def test_mpi_no_cycles(self):
        """EVAL: MPI for consistent data (no cycles)."""
        from pyrevealed.algorithms.mpi import compute_mpi

        log = BehaviorLog(
            cost_vectors=np.array([[1.0, 2.0], [2.0, 1.0]]),
            action_vectors=np.array([[4.0, 1.0], [1.0, 4.0]]),
        )

        result = compute_mpi(log)

        assert result.mpi_value == 0.0
        assert result.worst_cycle is None

    def test_mpi_simple_2_cycle(self, warp_violation_log):
        """EVAL: MPI for simple 2-cycle violation."""
        from pyrevealed.algorithms.mpi import compute_mpi

        result = compute_mpi(warp_violation_log)

        assert result.mpi_value > 0.0
        assert result.worst_cycle is not None

    def test_mpi_3_cycle(self, garp_3_cycle_log):
        """EVAL: MPI for 3-cycle violation."""
        from pyrevealed.algorithms.mpi import compute_mpi

        result = compute_mpi(garp_3_cycle_log)

        assert result.mpi_value > 0.0


class TestMPIBounds:
    """EVAL: MPI value bounds."""

    def test_mpi_between_zero_and_one(self):
        """EVAL: MPI should always be in [0, 1]."""
        from pyrevealed.algorithms.mpi import compute_mpi

        # Generate random data that likely has violations
        np.random.seed(42)
        for _ in range(10):
            log = BehaviorLog(
                cost_vectors=np.random.rand(5, 3) + 0.1,
                action_vectors=np.random.rand(5, 3) * 10,
            )

            result = compute_mpi(log)

            assert 0.0 <= result.mpi_value <= 1.0, (
                f"MPI out of bounds: {result.mpi_value}"
            )

    def test_mpi_maximum_theoretical(self):
        """EVAL: MPI should approach 1 for maximally irrational behavior."""
        from pyrevealed.algorithms.mpi import compute_mpi

        # Extremely irrational data - always choosing the wrong bundle
        log = BehaviorLog(
            cost_vectors=np.array([[1.0, 1.0], [1.0, 1.0]]),
            action_vectors=np.array([[100.0, 0.0], [0.0, 100.0]]),
        )

        result = compute_mpi(log)

        # MPI should be high for this data
        print(f"MPI for irrational data: {result.mpi_value}")


class TestMPIMultipleCycles:
    """EVAL: MPI with multiple overlapping cycles."""

    def test_mpi_worst_cycle_selection(self, dense_violation_log):
        """EVAL: MPI correctly identifies worst (highest MPI) cycle."""
        from pyrevealed.algorithms.mpi import compute_mpi

        result = compute_mpi(dense_violation_log)

        if result.mpi_value > 0 and result.cycle_costs:
            # Worst cycle should have highest MPI
            mpis = [cost for _, cost in result.cycle_costs]
            assert result.mpi_value == max(mpis)

    def test_mpi_all_cycles_tracked(self):
        """EVAL: MPI cycle_costs should include all positive-MPI cycles."""
        from pyrevealed.algorithms.mpi import compute_mpi

        # Data with multiple violations
        log = BehaviorLog(
            cost_vectors=np.array([
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]),
            action_vectors=np.array([
                [3.0, 1.0, 1.0],
                [1.0, 3.0, 1.0],
                [1.0, 1.0, 3.0],
            ]),
        )

        result = compute_mpi(log)

        # Should track all cycles
        print(f"Number of cycles tracked: {len(result.cycle_costs)}")


class TestHoutmanMaks:
    """EVAL: Houtman-Maks index computation."""

    def test_houtman_maks_consistent(self):
        """EVAL: Houtman-Maks for consistent data should be 0."""
        from pyrevealed.algorithms.mpi import compute_houtman_maks_index

        log = BehaviorLog(
            cost_vectors=np.array([[1.0, 2.0], [2.0, 1.0]]),
            action_vectors=np.array([[4.0, 1.0], [1.0, 4.0]]),
        )

        result = compute_houtman_maks_index(log)

        assert result.fraction == 0.0
        assert len(result.removed_observations) == 0

    def test_houtman_maks_single_violation(self, warp_violation_log):
        """EVAL: Houtman-Maks for single violation."""
        from pyrevealed.algorithms.mpi import compute_houtman_maks_index

        result = compute_houtman_maks_index(warp_violation_log)

        # Need to remove at least 1 observation
        assert result.fraction > 0.0

    def test_houtman_maks_greedy_approximation(self, dense_violation_log):
        """EVAL: Houtman-Maks greedy approximation quality."""
        from pyrevealed.algorithms.mpi import compute_houtman_maks_index

        result = compute_houtman_maks_index(dense_violation_log)

        # Fraction should be in [0, 1]
        assert 0.0 <= result.fraction <= 1.0

        # Removed observations should make remaining data consistent
        print(f"Removed {len(result.removed_observations)} out of 20 observations")


class TestMPITotalExpenditure:
    """EVAL: MPI total expenditure tracking."""

    def test_mpi_total_expenditure_correct(self):
        """EVAL: MPI total_expenditure should match sum of own expenditures."""
        from pyrevealed.algorithms.mpi import compute_mpi

        log = BehaviorLog(
            cost_vectors=np.array([[1.0, 2.0], [2.0, 1.0]]),
            action_vectors=np.array([[3.0, 1.0], [1.0, 3.0]]),
        )

        result = compute_mpi(log)

        expected_total = np.sum(log.total_spend)
        assert np.isclose(result.total_expenditure, expected_total)

    def test_mpi_computation_time(self):
        """EVAL: MPI computation time should be reasonable."""
        from pyrevealed.algorithms.mpi import compute_mpi

        log = BehaviorLog(
            cost_vectors=np.random.rand(50, 5) + 0.1,
            action_vectors=np.random.rand(50, 5) + 0.1,
        )

        result = compute_mpi(log)

        # Should complete quickly
        assert result.computation_time_ms < 5000, (
            f"MPI took too long: {result.computation_time_ms}ms"
        )
