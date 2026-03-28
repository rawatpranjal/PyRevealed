"""
SCC optimization correctness and performance benchmarks.

Tests that the SCC-based transitive closure produces identical results
to full Floyd-Warshall, and that the optimized algorithms scale properly.
"""

import numpy as np
import pytest
import time
from prefgraph.core.session import BehaviorLog, MenuChoiceLog


# =============================================================================
# CORRECTNESS: SCC TC == Floyd-Warshall TC
# =============================================================================


class TestSCCTransitiveClosureCorrectness:
    """Verify SCC TC produces identical output to direct Floyd-Warshall."""

    def _compare_tc(self, adjacency):
        """Helper: compare SCC TC vs direct FW on the same adjacency matrix."""
        from prefgraph.graph.transitive_closure import (
            scc_transitive_closure,
            _floyd_warshall_direct,
        )

        adj = np.ascontiguousarray(adjacency, dtype=np.bool_)
        tc_scc = scc_transitive_closure(adj.copy())
        tc_fw = _floyd_warshall_direct(adj.copy())
        np.testing.assert_array_equal(tc_scc, tc_fw)

    def test_empty_graph(self):
        """No edges - TC is just the identity."""
        adj = np.zeros((5, 5), dtype=np.bool_)
        self._compare_tc(adj)

    def test_full_graph(self):
        """Complete graph - TC is all True."""
        adj = np.ones((5, 5), dtype=np.bool_)
        self._compare_tc(adj)

    def test_chain(self):
        """Linear chain A -> B -> C -> D."""
        adj = np.zeros((4, 4), dtype=np.bool_)
        for i in range(3):
            adj[i, i + 1] = True
        self._compare_tc(adj)

    def test_single_cycle(self):
        """A -> B -> C -> A (one SCC of size 3)."""
        adj = np.zeros((3, 3), dtype=np.bool_)
        adj[0, 1] = adj[1, 2] = adj[2, 0] = True
        self._compare_tc(adj)

    def test_two_sccs_with_bridge(self):
        """Two cycles connected by a bridge edge."""
        adj = np.zeros((6, 6), dtype=np.bool_)
        # SCC 1: 0 -> 1 -> 2 -> 0
        adj[0, 1] = adj[1, 2] = adj[2, 0] = True
        # SCC 2: 3 -> 4 -> 5 -> 3
        adj[3, 4] = adj[4, 5] = adj[5, 3] = True
        # Bridge: 2 -> 3
        adj[2, 3] = True
        self._compare_tc(adj)

    def test_all_singletons(self):
        """DAG with no cycles - all SCCs are size 1."""
        adj = np.zeros((5, 5), dtype=np.bool_)
        adj[0, 1] = adj[0, 2] = adj[1, 3] = adj[2, 4] = True
        self._compare_tc(adj)

    def test_random_sparse(self):
        """Random sparse graph at T=100."""
        np.random.seed(42)
        T = 100
        adj = np.random.rand(T, T) < 0.05  # ~5% density
        np.fill_diagonal(adj, False)
        self._compare_tc(adj)

    def test_random_dense(self):
        """Random dense graph at T=50."""
        np.random.seed(42)
        T = 50
        adj = np.random.rand(T, T) < 0.5  # ~50% density
        np.fill_diagonal(adj, False)
        self._compare_tc(adj)

    def test_random_t200(self):
        """Random graph at T=200 - larger scale correctness check."""
        np.random.seed(123)
        T = 200
        adj = np.random.rand(T, T) < 0.1
        np.fill_diagonal(adj, False)
        self._compare_tc(adj)


# =============================================================================
# CORRECTNESS: GARP with SCC optimization
# =============================================================================


class TestGARPSCCCorrectness:
    """Verify GARP results are unchanged by SCC optimization."""

    def test_consistent_data(self):
        """Consistent data still identified as consistent."""
        from prefgraph.algorithms.garp import check_garp

        log = BehaviorLog(
            cost_vectors=np.array([[1.0, 2.0], [2.0, 1.0]]),
            action_vectors=np.array([[4.0, 1.0], [1.0, 4.0]]),
        )
        result = check_garp(log)
        assert result.is_consistent

    def test_warp_violation(self):
        """Simple WARP violation still detected."""
        from prefgraph.algorithms.garp import check_garp

        # Obs 0 can afford obs 1's bundle (strictly), and vice versa
        log = BehaviorLog(
            cost_vectors=np.array([[2.0, 1.0], [1.0, 2.0]]),
            action_vectors=np.array([[3.0, 2.0], [2.0, 3.0]]),
        )
        result = check_garp(log)
        assert not result.is_consistent
        assert len(result.violations) > 0

    def test_3cycle_violation(self):
        """3-cycle GARP violation still detected."""
        from prefgraph.algorithms.garp import check_garp

        # Each obs spends 10 on own bundle, but others cost only 7
        # Creates full revealed preference graph with strict preferences
        log = BehaviorLog(
            cost_vectors=np.array([
                [4.0, 1.0, 1.0],
                [1.0, 4.0, 1.0],
                [1.0, 1.0, 4.0],
            ]),
            action_vectors=np.array([
                [2.0, 1.0, 1.0],
                [1.0, 2.0, 1.0],
                [1.0, 1.0, 2.0],
            ]),
        )
        result = check_garp(log)
        assert not result.is_consistent

    def test_random_t100(self):
        """Random T=100 produces a result (smoke test)."""
        from prefgraph.algorithms.garp import check_garp

        np.random.seed(42)
        log = BehaviorLog(
            cost_vectors=np.random.rand(100, 5) + 0.1,
            action_vectors=np.random.rand(100, 5) + 0.1,
        )
        result = check_garp(log)
        assert isinstance(result.is_consistent, (bool, np.bool_))


# =============================================================================
# CORRECTNESS: AEI with SCC optimization
# =============================================================================


class TestAEISCCCorrectness:
    """Verify AEI values are unchanged by SCC optimization."""

    def test_consistent_aei_is_one(self):
        """Consistent data has AEI = 1.0."""
        from prefgraph.algorithms.aei import compute_aei

        log = BehaviorLog(
            cost_vectors=np.array([[1.0, 2.0], [2.0, 1.0]]),
            action_vectors=np.array([[4.0, 1.0], [1.0, 4.0]]),
        )
        result = compute_aei(log)
        assert result.efficiency_index == 1.0
        assert result.is_perfectly_consistent

    def test_inconsistent_aei_less_than_one(self):
        """Inconsistent data has AEI < 1.0."""
        from prefgraph.algorithms.aei import compute_aei

        log = BehaviorLog(
            cost_vectors=np.array([[2.0, 1.0], [1.0, 2.0]]),
            action_vectors=np.array([[3.0, 2.0], [2.0, 3.0]]),
        )
        result = compute_aei(log)
        assert 0 < result.efficiency_index < 1.0


# =============================================================================
# CORRECTNESS: Houtman-Maks
# =============================================================================


class TestHoutmanMaksSCCCorrectness:
    """Verify Houtman-Maks results are reasonable with FVS optimization."""

    def test_consistent_data_no_removal(self):
        """Consistent data requires no removals."""
        from prefgraph.algorithms.mpi import compute_houtman_maks_index

        log = BehaviorLog(
            cost_vectors=np.array([[1.0, 2.0], [2.0, 1.0]]),
            action_vectors=np.array([[4.0, 1.0], [1.0, 4.0]]),
        )
        result = compute_houtman_maks_index(log)
        assert result.fraction == 0.0
        assert len(result.removed_observations) == 0

    def test_violation_requires_removal(self):
        """Inconsistent data requires at least one removal."""
        from prefgraph.algorithms.mpi import compute_houtman_maks_index

        log = BehaviorLog(
            cost_vectors=np.array([[2.0, 1.0], [1.0, 2.0]]),
            action_vectors=np.array([[3.0, 2.0], [2.0, 3.0]]),
        )
        result = compute_houtman_maks_index(log)
        assert result.fraction > 0.0
        assert len(result.removed_observations) >= 1

    def test_menu_consistent_no_removal(self):
        """Consistent menu data requires no removals."""
        from prefgraph.algorithms.abstract_choice import compute_menu_efficiency

        log = MenuChoiceLog(
            menus=[frozenset({0, 1, 2}), frozenset({1, 2})],
            choices=[0, 1],
        )
        result = compute_menu_efficiency(log)
        assert result.efficiency_index == 1.0

    def test_menu_violation_requires_removal(self):
        """Menu SARP violation requires removal."""
        from prefgraph.algorithms.abstract_choice import compute_menu_efficiency

        log = MenuChoiceLog(
            menus=[frozenset({0, 1}), frozenset({0, 1})],
            choices=[0, 1],  # WARP violation
        )
        result = compute_menu_efficiency(log)
        assert result.efficiency_index < 1.0


# =============================================================================
# PERFORMANCE BENCHMARKS
# =============================================================================


class TestGARPPerformance:
    """Benchmark GARP at various scales."""

    def test_garp_t100_under_1s(self):
        """GARP T=100 completes in under 1 second."""
        from prefgraph.algorithms.garp import check_garp

        np.random.seed(42)
        log = BehaviorLog(
            cost_vectors=np.random.rand(100, 5) + 0.1,
            action_vectors=np.random.rand(100, 5) + 0.1,
        )

        start = time.time()
        result = check_garp(log)
        elapsed = time.time() - start

        print(f"GARP T=100: {elapsed:.3f}s, consistent={result.is_consistent}")
        assert elapsed < 1.0

    def test_garp_t500_under_5s(self):
        """GARP T=500 completes in under 5 seconds (was 46s before SCC)."""
        from prefgraph.algorithms.garp import check_garp

        np.random.seed(42)
        log = BehaviorLog(
            cost_vectors=np.random.rand(500, 5) + 0.1,
            action_vectors=np.random.rand(500, 5) + 0.1,
        )

        start = time.time()
        result = check_garp(log)
        elapsed = time.time() - start

        print(f"GARP T=500: {elapsed:.3f}s, consistent={result.is_consistent}")
        assert elapsed < 5.0

    @pytest.mark.slow
    def test_garp_t1000_under_10s(self):
        """GARP T=1000 completes in under 10 seconds (previously timed out)."""
        from prefgraph.algorithms.garp import check_garp

        np.random.seed(42)
        log = BehaviorLog(
            cost_vectors=np.random.rand(1000, 5) + 0.1,
            action_vectors=np.random.rand(1000, 5) + 0.1,
        )

        start = time.time()
        result = check_garp(log)
        elapsed = time.time() - start

        print(f"GARP T=1000: {elapsed:.3f}s, consistent={result.is_consistent}")
        assert elapsed < 10.0

    @pytest.mark.slow
    def test_garp_t2000_under_60s(self):
        """GARP T=2000 completes in under 60 seconds."""
        from prefgraph.algorithms.garp import check_garp

        np.random.seed(42)
        log = BehaviorLog(
            cost_vectors=np.random.rand(2000, 5) + 0.1,
            action_vectors=np.random.rand(2000, 5) + 0.1,
        )

        start = time.time()
        result = check_garp(log)
        elapsed = time.time() - start

        print(f"GARP T=2000: {elapsed:.3f}s, consistent={result.is_consistent}")
        assert elapsed < 60.0


class TestGARPNearConsistentPerformance:
    """Benchmark GARP on near-consistent data (realistic scenario)."""

    @staticmethod
    def _make_near_consistent(T, N=5, noise=0.01):
        """Generate near-consistent Cobb-Douglas data with small noise."""
        np.random.seed(42)
        alpha = np.random.dirichlet(np.ones(N))
        prices = np.random.rand(T, N) + 0.1
        income = np.random.rand(T) * 10 + 5

        # Cobb-Douglas demand: q_i = alpha_i * m / p_i
        quantities = np.zeros((T, N))
        for t in range(T):
            for i in range(N):
                quantities[t, i] = alpha[i] * income[t] / prices[t, i]

        # Add small noise
        quantities += np.random.rand(T, N) * noise
        quantities = np.maximum(quantities, 0.01)

        return BehaviorLog(cost_vectors=prices, action_vectors=quantities)

    @pytest.mark.slow
    def test_near_consistent_t1000(self):
        """Near-consistent T=1000 should be very fast due to small SCCs."""
        from prefgraph.algorithms.garp import check_garp

        log = self._make_near_consistent(1000)
        start = time.time()
        result = check_garp(log)
        elapsed = time.time() - start

        print(f"Near-consistent T=1000: {elapsed:.3f}s, consistent={result.is_consistent}")
        assert elapsed < 5.0

    @pytest.mark.slow
    def test_near_consistent_t5000(self):
        """Near-consistent T=5000 should complete due to tiny SCCs."""
        from prefgraph.algorithms.garp import check_garp

        log = self._make_near_consistent(5000)
        start = time.time()
        result = check_garp(log)
        elapsed = time.time() - start

        print(f"Near-consistent T=5000: {elapsed:.3f}s, consistent={result.is_consistent}")
        assert elapsed < 30.0


class TestAEIPerformance:
    """Benchmark AEI at various scales."""

    def test_aei_t100_under_5s(self):
        """AEI T=100 completes in under 5 seconds."""
        from prefgraph.algorithms.aei import compute_aei

        np.random.seed(42)
        log = BehaviorLog(
            cost_vectors=np.random.rand(100, 5) + 0.1,
            action_vectors=np.random.rand(100, 5) + 0.1,
        )

        start = time.time()
        result = compute_aei(log)
        elapsed = time.time() - start

        print(f"AEI T=100: {elapsed:.3f}s, aei={result.efficiency_index:.4f}")
        assert elapsed < 5.0

    @pytest.mark.slow
    def test_aei_t500_under_30s(self):
        """AEI T=500 completes in under 30 seconds."""
        from prefgraph.algorithms.aei import compute_aei

        np.random.seed(42)
        log = BehaviorLog(
            cost_vectors=np.random.rand(500, 5) + 0.1,
            action_vectors=np.random.rand(500, 5) + 0.1,
        )

        start = time.time()
        result = compute_aei(log)
        elapsed = time.time() - start

        print(f"AEI T=500: {elapsed:.3f}s, aei={result.efficiency_index:.4f}")
        assert elapsed < 30.0


class TestHoutmanMaksPerformance:
    """Benchmark Houtman-Maks with FVS optimization."""

    def test_hm_t100_under_2s(self):
        """Houtman-Maks T=100 completes in under 2 seconds."""
        from prefgraph.algorithms.mpi import compute_houtman_maks_index

        np.random.seed(42)
        log = BehaviorLog(
            cost_vectors=np.random.rand(100, 5) + 0.1,
            action_vectors=np.random.rand(100, 5) + 0.1,
        )

        start = time.time()
        result = compute_houtman_maks_index(log)
        elapsed = time.time() - start

        print(f"HM T=100: {elapsed:.3f}s, fraction={result.fraction:.3f}")
        assert elapsed < 2.0

    @pytest.mark.slow
    def test_hm_t500_under_30s(self):
        """Houtman-Maks T=500 completes in under 30 seconds."""
        from prefgraph.algorithms.mpi import compute_houtman_maks_index

        np.random.seed(42)
        log = BehaviorLog(
            cost_vectors=np.random.rand(500, 5) + 0.1,
            action_vectors=np.random.rand(500, 5) + 0.1,
        )

        start = time.time()
        result = compute_houtman_maks_index(log)
        elapsed = time.time() - start

        print(f"HM T=500: {elapsed:.3f}s, fraction={result.fraction:.3f}")
        assert elapsed < 30.0
