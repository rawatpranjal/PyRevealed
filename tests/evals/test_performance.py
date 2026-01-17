"""
EVAL: Performance and scalability tests.

Tests for O(T^3) scaling, memory usage, and computational limits.
"""

import numpy as np
import pytest
import time
from pyrevealed.core.session import BehaviorLog, StochasticChoiceLog


class TestFloydWarshallScaling:
    """EVAL: Floyd-Warshall O(T^3) scaling in GARP."""

    def test_garp_scaling_t50(self):
        """EVAL: GARP timing for T=50."""
        from pyrevealed.algorithms.garp import check_garp

        np.random.seed(42)
        log = BehaviorLog(
            cost_vectors=np.random.rand(50, 5) + 0.1,
            action_vectors=np.random.rand(50, 5) + 0.1,
        )

        start = time.time()
        result = check_garp(log)
        elapsed = time.time() - start

        print(f"T=50: {elapsed:.3f}s")
        assert elapsed < 1.0, f"T=50 should complete in <1s, took {elapsed:.2f}s"

    def test_garp_scaling_t100(self):
        """EVAL: GARP timing for T=100."""
        from pyrevealed.algorithms.garp import check_garp

        np.random.seed(42)
        log = BehaviorLog(
            cost_vectors=np.random.rand(100, 5) + 0.1,
            action_vectors=np.random.rand(100, 5) + 0.1,
        )

        start = time.time()
        result = check_garp(log)
        elapsed = time.time() - start

        print(f"T=100: {elapsed:.3f}s")
        assert elapsed < 2.0, f"T=100 should complete in <2s, took {elapsed:.2f}s"

    def test_garp_scaling_t200(self):
        """EVAL: GARP timing for T=200."""
        from pyrevealed.algorithms.garp import check_garp

        np.random.seed(42)
        log = BehaviorLog(
            cost_vectors=np.random.rand(200, 5) + 0.1,
            action_vectors=np.random.rand(200, 5) + 0.1,
        )

        start = time.time()
        result = check_garp(log)
        elapsed = time.time() - start

        print(f"T=200: {elapsed:.3f}s")
        assert elapsed < 5.0, f"T=200 should complete in <5s, took {elapsed:.2f}s"

    @pytest.mark.slow
    def test_garp_scaling_t500(self, large_t_500):
        """EVAL: GARP timing for T=500 (target: <10s)."""
        from pyrevealed.algorithms.garp import check_garp

        start = time.time()
        result = check_garp(large_t_500)
        elapsed = time.time() - start

        print(f"T=500: {elapsed:.3f}s")
        assert elapsed < 10.0, f"T=500 should complete in <10s, took {elapsed:.2f}s"

    @pytest.mark.slow
    def test_garp_scaling_t1000(self, large_t_1000):
        """EVAL: GARP timing for T=1000 (target: <60s)."""
        from pyrevealed.algorithms.garp import check_garp

        start = time.time()
        result = check_garp(large_t_1000)
        elapsed = time.time() - start

        print(f"T=1000: {elapsed:.3f}s")
        assert elapsed < 60.0, f"T=1000 should complete in <60s, took {elapsed:.2f}s"


class TestMemoryUsage:
    """EVAL: Memory usage for large datasets."""

    def test_memory_t500(self, large_t_500):
        """EVAL: Memory for T=500 (~250KB per TxT boolean matrix)."""
        from pyrevealed.algorithms.garp import check_garp

        result = check_garp(large_t_500)

        # Check matrix sizes
        R = result.direct_revealed_preference
        R_star = result.transitive_closure
        P = result.strict_revealed_preference

        assert R.shape == (500, 500)
        assert R_star.shape == (500, 500)
        assert P.shape == (500, 500)

        # Memory estimate: 3 matrices * 500^2 bytes ≈ 750KB
        print(f"Matrix memory: ~{3 * 500 * 500 / 1024:.0f}KB")

    @pytest.mark.slow
    def test_memory_t2000(self):
        """EVAL: Memory for T=2000 (~4MB per TxT boolean matrix)."""
        np.random.seed(42)
        T = 2000
        log = BehaviorLog(
            cost_vectors=np.random.rand(T, 5) + 0.1,
            action_vectors=np.random.rand(T, 5) + 0.1,
        )

        from pyrevealed.algorithms.garp import check_garp

        result = check_garp(log)

        # Memory estimate: 3 matrices * 2000^2 bytes ≈ 12MB
        print(f"Matrix memory: ~{3 * T * T / 1024 / 1024:.1f}MB")


class TestAEIScaling:
    """EVAL: AEI binary search scaling."""

    def test_aei_scaling_t100(self):
        """EVAL: AEI timing for T=100."""
        from pyrevealed.algorithms.aei import compute_aei

        np.random.seed(42)
        log = BehaviorLog(
            cost_vectors=np.random.rand(100, 5) + 0.1,
            action_vectors=np.random.rand(100, 5) + 0.1,
        )

        start = time.time()
        result = compute_aei(log)
        elapsed = time.time() - start

        print(f"AEI T=100: {elapsed:.3f}s")
        # AEI requires multiple GARP checks
        assert elapsed < 30.0, f"AEI T=100 should complete in <30s, took {elapsed:.2f}s"


class TestMPIScaling:
    """EVAL: MPI cycle detection scaling."""

    def test_mpi_scaling_t100(self):
        """EVAL: MPI timing for T=100."""
        from pyrevealed.algorithms.mpi import compute_mpi

        np.random.seed(42)
        log = BehaviorLog(
            cost_vectors=np.random.rand(100, 5) + 0.1,
            action_vectors=np.random.rand(100, 5) + 0.1,
        )

        start = time.time()
        result = compute_mpi(log)
        elapsed = time.time() - start

        print(f"MPI T=100: {elapsed:.3f}s")
        assert elapsed < 10.0, f"MPI T=100 should complete in <10s, took {elapsed:.2f}s"


class TestStochasticScaling:
    """EVAL: Stochastic choice model scaling."""

    def test_rum_scaling_5_items(self):
        """EVAL: RUM timing for 5 items (120 orderings)."""
        from pyrevealed.algorithms.stochastic import test_rum_consistency

        log = StochasticChoiceLog(
            menus=[frozenset(range(5))],
            choice_frequencies=[{i: 20 for i in range(5)}],
        )

        start = time.time()
        result = test_rum_consistency(log)
        elapsed = time.time() - start

        print(f"RUM 5 items: {elapsed:.3f}s")
        assert elapsed < 5.0, f"RUM 5 items should complete in <5s, took {elapsed:.2f}s"

    def test_rum_scaling_6_items(self):
        """EVAL: RUM timing for 6 items (720 orderings) - boundary."""
        from pyrevealed.algorithms.stochastic import test_rum_consistency

        log = StochasticChoiceLog(
            menus=[frozenset(range(6))],
            choice_frequencies=[{i: 17 for i in range(6)}],
        )

        start = time.time()
        result = test_rum_consistency(log)
        elapsed = time.time() - start

        print(f"RUM 6 items: {elapsed:.3f}s")
        assert elapsed < 10.0, f"RUM 6 items should complete in <10s, took {elapsed:.2f}s"


class TestLPSolverScaling:
    """EVAL: LP solver scaling in utility recovery."""

    def test_utility_recovery_scaling_t50(self):
        """EVAL: Utility recovery LP timing for T=50."""
        from pyrevealed.algorithms.utility import recover_utility

        np.random.seed(42)
        log = BehaviorLog(
            cost_vectors=np.random.rand(50, 5) + 0.1,
            action_vectors=np.random.rand(50, 5) + 0.1,
        )

        start = time.time()
        result = recover_utility(log)
        elapsed = time.time() - start

        print(f"Utility T=50: {elapsed:.3f}s")
        assert elapsed < 5.0, f"Utility T=50 should complete in <5s, took {elapsed:.2f}s"


class TestSlutksyScaling:
    """EVAL: Slutsky matrix estimation scaling."""

    def test_slutsky_scaling_n10(self):
        """EVAL: Slutsky timing for N=10 goods."""
        from pyrevealed.algorithms.integrability import compute_slutsky_matrix

        np.random.seed(42)
        log = BehaviorLog(
            cost_vectors=np.random.rand(30, 10) + 0.1,
            action_vectors=np.random.rand(30, 10) + 0.1,
        )

        start = time.time()
        S = compute_slutsky_matrix(log)
        elapsed = time.time() - start

        print(f"Slutsky N=10: {elapsed:.3f}s")
        assert elapsed < 2.0, f"Slutsky N=10 should complete in <2s, took {elapsed:.2f}s"

    def test_slutsky_scaling_n50(self):
        """EVAL: Slutsky timing for N=50 goods."""
        from pyrevealed.algorithms.integrability import compute_slutsky_matrix

        np.random.seed(42)
        log = BehaviorLog(
            cost_vectors=np.random.rand(100, 50) + 0.1,
            action_vectors=np.random.rand(100, 50) + 0.1,
        )

        start = time.time()
        S = compute_slutsky_matrix(log)
        elapsed = time.time() - start

        print(f"Slutsky N=50: {elapsed:.3f}s")
        assert elapsed < 10.0, f"Slutsky N=50 should complete in <10s, took {elapsed:.2f}s"


class TestParallelization:
    """EVAL: Parallelization opportunities (currently sequential)."""

    def test_multiple_garp_calls_sequential(self):
        """EVAL: Multiple GARP calls (potential for parallelization)."""
        from pyrevealed.algorithms.garp import check_garp

        np.random.seed(42)
        logs = [
            BehaviorLog(
                cost_vectors=np.random.rand(50, 5) + 0.1,
                action_vectors=np.random.rand(50, 5) + 0.1,
            )
            for _ in range(5)
        ]

        start = time.time()
        results = [check_garp(log) for log in logs]
        elapsed = time.time() - start

        print(f"5 GARP calls (sequential): {elapsed:.3f}s")
