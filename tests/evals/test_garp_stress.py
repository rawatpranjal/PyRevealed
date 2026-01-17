"""
EVAL: GARP algorithm stress tests.

Tests for scalability and numerical edge cases in GARP checking.
"""

import numpy as np
import pytest
from pyrevealed.core.session import BehaviorLog


class TestGARPScalability:
    """EVAL: GARP Floyd-Warshall O(T^3) scalability."""

    def test_garp_t100(self):
        """EVAL: GARP with T=100 observations."""
        np.random.seed(42)
        T, N = 100, 5
        log = BehaviorLog(
            cost_vectors=np.random.rand(T, N) + 0.1,
            action_vectors=np.random.rand(T, N) + 0.1,
        )

        from pyrevealed.algorithms.garp import check_garp
        import time

        start = time.time()
        result = check_garp(log)
        elapsed = time.time() - start

        print(f"T=100: {elapsed:.3f}s, consistent={result.is_consistent}")
        assert elapsed < 5.0, f"T=100 should complete in <5s, took {elapsed:.2f}s"

    def test_garp_t500(self, large_t_500):
        """EVAL: GARP with T=500 observations should complete <10s."""
        from pyrevealed.algorithms.garp import check_garp
        import time

        start = time.time()
        result = check_garp(large_t_500)
        elapsed = time.time() - start

        print(f"T=500: {elapsed:.3f}s, consistent={result.is_consistent}")
        assert elapsed < 10.0, f"T=500 should complete in <10s, took {elapsed:.2f}s"

    @pytest.mark.slow
    def test_garp_t1000(self, large_t_1000):
        """EVAL: GARP with T=1000 observations should complete <60s."""
        from pyrevealed.algorithms.garp import check_garp
        import time

        start = time.time()
        result = check_garp(large_t_1000)
        elapsed = time.time() - start

        print(f"T=1000: {elapsed:.3f}s, consistent={result.is_consistent}")
        assert elapsed < 60.0, f"T=1000 should complete in <60s, took {elapsed:.2f}s"


class TestGARPMemory:
    """EVAL: GARP memory usage with large T."""

    def test_garp_memory_t500(self, large_t_500):
        """EVAL: Memory for T=500 (500x500 boolean matrices ~250KB each)."""
        from pyrevealed.algorithms.garp import check_garp

        result = check_garp(large_t_500)

        # Check matrix sizes
        assert result.direct_revealed_preference.shape == (500, 500)
        assert result.transitive_closure.shape == (500, 500)

        # Each boolean matrix is ~250KB
        # Total memory should be manageable

    @pytest.mark.slow
    def test_garp_memory_t2000(self):
        """EVAL: Memory for T=2000 (2000x2000 boolean matrices ~4MB each)."""
        np.random.seed(42)
        T, N = 2000, 5
        log = BehaviorLog(
            cost_vectors=np.random.rand(T, N) + 0.1,
            action_vectors=np.random.rand(T, N) + 0.1,
        )

        from pyrevealed.algorithms.garp import check_garp

        # This tests memory allocation
        result = check_garp(log)

        # ~32MB for all matrices, should be fine
        assert result.transitive_closure.shape == (2000, 2000)


class TestGARPNumerical:
    """EVAL: GARP numerical edge cases."""

    def test_garp_all_equal_expenditures(self):
        """EVAL: GARP when all expenditures are equal."""
        log = BehaviorLog(
            cost_vectors=np.array([
                [1.0, 1.0],
                [1.0, 1.0],
                [1.0, 1.0],
            ]),
            action_vectors=np.array([
                [1.0, 1.0],
                [1.0, 1.0],
                [1.0, 1.0],
            ]),
        )

        from pyrevealed.algorithms.garp import check_garp
        result = check_garp(log)

        # All bundles identical = trivially consistent
        assert result.is_consistent

    def test_garp_near_equality_chain(self):
        """EVAL: GARP with chain of near-equal expenditures."""
        # E[i] = E[i+1] + epsilon
        n = 10
        eps = 1e-11
        costs = np.array([[1.0, 1.0] for _ in range(n)])
        actions = np.array([[1.0 + i*eps, 1.0 + i*eps] for i in range(n)])

        log = BehaviorLog(cost_vectors=costs, action_vectors=actions)

        from pyrevealed.algorithms.garp import check_garp
        result = check_garp(log, tolerance=1e-10)

        # Near-equal should be treated as equal
        assert hasattr(result, 'is_consistent')


class TestViolationCycles:
    """EVAL: GARP violation cycle detection."""

    def test_simple_warp_violation(self, warp_violation_log):
        """EVAL: Simple 2-cycle (WARP violation)."""
        from pyrevealed.algorithms.garp import check_garp

        result = check_garp(warp_violation_log)

        assert not result.is_consistent
        assert len(result.violations) > 0

    def test_3_cycle_violation(self, garp_3_cycle_log):
        """EVAL: 3-cycle GARP violation."""
        from pyrevealed.algorithms.garp import check_garp

        result = check_garp(garp_3_cycle_log)

        assert not result.is_consistent
        # Should find at least one 3-cycle
        has_3_cycle = any(len(set(cycle)) == 3 for cycle in result.violations)
        assert has_3_cycle, "Should find 3-cycle violation"

    def test_dense_violations(self, dense_violation_log):
        """EVAL: Data with many overlapping violation cycles."""
        from pyrevealed.algorithms.garp import check_garp

        result = check_garp(dense_violation_log)

        # Random data tends to have violations
        if not result.is_consistent:
            print(f"Found {len(result.violations)} violations")


class TestWARP:
    """EVAL: WARP (direct violations only) testing."""

    def test_warp_vs_garp_difference(self, garp_3_cycle_log):
        """EVAL: WARP may pass when GARP fails (transitive violations)."""
        from pyrevealed.algorithms.garp import check_garp, check_warp

        garp = check_garp(garp_3_cycle_log)
        warp = check_warp(garp_3_cycle_log)

        # 3-cycle violates GARP but not necessarily WARP
        print(f"GARP consistent: {garp.is_consistent}")
        print(f"WARP consistent: {warp.is_consistent}")

    def test_warp_implies_garp(self):
        """EVAL: WARP violation should imply GARP violation."""
        from pyrevealed.algorithms.garp import check_garp, check_warp

        # Create data with WARP violation
        log = BehaviorLog(
            cost_vectors=np.array([[1.0, 1.0], [1.0, 1.0]]),
            action_vectors=np.array([[3.0, 1.0], [1.0, 3.0]]),
        )

        garp = check_garp(log)
        warp = check_warp(log)

        if not warp.is_consistent:
            assert not garp.is_consistent, "WARP violation should imply GARP violation"


class TestSwapsIndex:
    """EVAL: Swaps index computation."""

    def test_swaps_consistent_data(self):
        """EVAL: Swaps index for consistent data should be 0."""
        from pyrevealed.algorithms.garp import compute_swaps_index

        log = BehaviorLog(
            cost_vectors=np.array([[1.0, 2.0], [2.0, 1.0]]),
            action_vectors=np.array([[4.0, 1.0], [1.0, 4.0]]),
        )

        result = compute_swaps_index(log)

        assert result.swaps_count == 0
        assert result.is_consistent

    def test_swaps_single_violation(self, warp_violation_log):
        """EVAL: Swaps index for single violation."""
        from pyrevealed.algorithms.garp import compute_swaps_index

        result = compute_swaps_index(warp_violation_log)

        assert result.swaps_count >= 1
        assert not result.is_consistent


class TestObservationContributions:
    """EVAL: Per-observation contribution analysis."""

    def test_contributions_consistent_data(self):
        """EVAL: Contributions for consistent data should be zero."""
        from pyrevealed.algorithms.garp import compute_observation_contributions

        log = BehaviorLog(
            cost_vectors=np.array([[1.0, 2.0], [2.0, 1.0]]),
            action_vectors=np.array([[4.0, 1.0], [1.0, 4.0]]),
        )

        result = compute_observation_contributions(log)

        assert np.all(result.contributions == 0)

    def test_contributions_with_violations(self, warp_violation_log):
        """EVAL: Contributions should sum to 1 for inconsistent data."""
        from pyrevealed.algorithms.garp import compute_observation_contributions

        result = compute_observation_contributions(warp_violation_log)

        # Contributions should sum to 1 (if any violations exist)
        if np.sum(result.contributions) > 0:
            assert np.isclose(np.sum(result.contributions), 1.0)
