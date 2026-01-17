"""
EVAL: Tolerance parameter sensitivity tests.

These tests expose how algorithms behave at tolerance boundaries.
"""

import numpy as np
import pytest
from pyrevealed.core.session import BehaviorLog


class TestGARPToleranceSensitivity:
    """EVAL: GARP results change dramatically with tolerance."""

    def test_garp_tolerance_boundary(self):
        """EVAL: GARP flip between consistent/inconsistent at tolerance boundary.

        Data is designed so that p1@q1 - p1@q2 = exactly the tolerance value.
        """
        from pyrevealed.algorithms.garp import check_garp

        # Create data where expenditure difference is exactly 1e-10
        log = BehaviorLog(
            cost_vectors=np.array([[1.0, 1.0], [1.0, 1.0]]),
            action_vectors=np.array([
                [2.5, 2.5],  # exp = 5.0
                [2.5 + 5e-11, 2.5 + 5e-11],  # exp = 5.0 + 1e-10
            ]),
        )

        # At tolerance=1e-10, this is on the boundary
        result_strict = check_garp(log, tolerance=1e-11)
        result_loose = check_garp(log, tolerance=1e-9)

        # Results might differ
        print(f"Strict (tol=1e-11): {result_strict.is_consistent}")
        print(f"Loose (tol=1e-9): {result_loose.is_consistent}")

    @pytest.mark.parametrize("tolerance", [1e-15, 1e-12, 1e-10, 1e-8, 1e-5])
    def test_garp_tolerance_sweep(self, tolerance):
        """EVAL: GARP consistency across tolerance values."""
        from pyrevealed.algorithms.garp import check_garp

        log = BehaviorLog(
            cost_vectors=np.array([[1.0, 2.0], [2.0, 1.0]]),
            action_vectors=np.array([[3.0, 1.0], [1.0, 3.0]]),
        )

        result = check_garp(log, tolerance=tolerance)

        # Should be deterministic at each tolerance
        assert hasattr(result, 'is_consistent')

    def test_garp_zero_tolerance(self):
        """EVAL: GARP with tolerance=0 (exact comparison)."""
        from pyrevealed.algorithms.garp import check_garp

        log = BehaviorLog(
            cost_vectors=np.array([[1.0, 2.0], [2.0, 1.0]]),
            action_vectors=np.array([[3.0, 1.0], [1.0, 3.0]]),
        )

        result = check_garp(log, tolerance=0.0)

        # Zero tolerance means exact floating point comparison
        assert hasattr(result, 'is_consistent')


class TestAEIToleranceSensitivity:
    """EVAL: AEI binary search tolerance affects precision."""

    def test_aei_tolerance_precision(self):
        """EVAL: AEI precision depends on tolerance."""
        from pyrevealed.algorithms.aei import compute_aei

        log = BehaviorLog(
            cost_vectors=np.array([[1.0, 1.0], [1.0, 1.0]]),
            action_vectors=np.array([[3.0, 1.0], [1.0, 3.0]]),
        )

        result_strict = compute_aei(log, tolerance=1e-12)
        result_loose = compute_aei(log, tolerance=1e-3)

        # Results should be similar but may differ slightly
        diff = abs(result_strict.efficiency_index - result_loose.efficiency_index)
        print(f"AEI strict: {result_strict.efficiency_index:.10f}")
        print(f"AEI loose: {result_loose.efficiency_index:.10f}")
        print(f"Difference: {diff:.10f}")

    def test_aei_binary_search_iterations(self):
        """EVAL: AEI binary search may not converge with tight tolerance."""
        from pyrevealed.algorithms.aei import compute_aei

        log = BehaviorLog(
            cost_vectors=np.array([[1.0, 1.0], [1.0, 1.0]]),
            action_vectors=np.array([[3.0, 1.0], [1.0, 3.0]]),
        )

        # Very tight tolerance may require more iterations
        result = compute_aei(log, tolerance=1e-15)

        # Should still complete within max_iterations
        assert hasattr(result, 'efficiency_index')


class TestSlutskyToleranceSensitivity:
    """EVAL: Slutsky test sensitivity to tolerance parameters."""

    def test_symmetry_tolerance(self):
        """EVAL: Slutsky symmetry test with different tolerances."""
        from pyrevealed.algorithms.integrability import check_slutsky_symmetry

        # Nearly symmetric matrix
        S = np.array([[1.0, 2.0], [2.0 + 1e-5, 1.0]])

        is_sym_strict, viol_strict, _ = check_slutsky_symmetry(S, tolerance=1e-6)
        is_sym_loose, viol_loose, _ = check_slutsky_symmetry(S, tolerance=1e-4)

        print(f"Strict (tol=1e-6): symmetric={is_sym_strict}, violations={viol_strict}")
        print(f"Loose (tol=1e-4): symmetric={is_sym_loose}, violations={viol_loose}")

    def test_nsd_tolerance(self):
        """EVAL: Slutsky NSD test with different eigenvalue tolerances."""
        from pyrevealed.algorithms.integrability import check_slutsky_nsd

        # Matrix with tiny positive eigenvalue
        S = np.array([[-1.0, 0.0], [0.0, 1e-8]])

        is_nsd_strict, eig_strict, _, _ = check_slutsky_nsd(S, tolerance=1e-10)
        is_nsd_loose, eig_loose, _, _ = check_slutsky_nsd(S, tolerance=1e-6)

        print(f"Strict (tol=1e-10): NSD={is_nsd_strict}")
        print(f"Loose (tol=1e-6): NSD={is_nsd_loose}")


class TestStochasticToleranceSensitivity:
    """EVAL: Stochastic choice test tolerance sensitivity."""

    def test_iia_tolerance(self):
        """EVAL: IIA test with different CV tolerances."""
        from pyrevealed.core.session import StochasticChoiceLog
        from pyrevealed.algorithms.stochastic import check_independence_irrelevant_alternatives

        # Data with slight IIA violation
        log = StochasticChoiceLog(
            menus=[frozenset({0, 1, 2}), frozenset({0, 1})],
            choice_frequencies=[
                {0: 40, 1: 35, 2: 25},
                {0: 55, 1: 45},  # Slight change in odds
            ],
        )

        iia_strict = check_independence_irrelevant_alternatives(log, tolerance=0.05)
        iia_loose = check_independence_irrelevant_alternatives(log, tolerance=0.2)

        print(f"Strict (tol=0.05): IIA={iia_strict}")
        print(f"Loose (tol=0.2): IIA={iia_loose}")

    def test_regularity_tolerance(self):
        """EVAL: Regularity test with different probability tolerances."""
        from pyrevealed.core.session import StochasticChoiceLog
        from pyrevealed.algorithms.stochastic import test_regularity

        # Slight regularity violation
        log = StochasticChoiceLog(
            menus=[frozenset({0, 1}), frozenset({0, 1, 2})],
            choice_frequencies=[
                {0: 59, 1: 41},  # P(0|{0,1}) = 0.59
                {0: 60, 1: 30, 2: 10},  # P(0|{0,1,2}) = 0.60 > 0.59 (slight violation)
            ],
        )

        result_strict = test_regularity(log, tolerance=0.005)
        result_loose = test_regularity(log, tolerance=0.05)

        print(f"Strict (tol=0.005): satisfies={result_strict.satisfies_regularity}")
        print(f"Loose (tol=0.05): satisfies={result_loose.satisfies_regularity}")


class TestIntertemporalToleranceSensitivity:
    """EVAL: Intertemporal choice tolerance sensitivity."""

    def test_discounting_tolerance(self):
        """EVAL: Exponential discounting test with different tolerances."""
        from pyrevealed.algorithms.intertemporal import (
            test_exponential_discounting,
            DatedChoice,
        )

        # Near-indifferent choices
        choices = [
            DatedChoice(
                amounts=np.array([100.0, 101.0]),  # Nearly equal
                dates=np.array([0, 1]),
                chosen=1,
            ),
        ]

        result_strict = test_exponential_discounting(choices, tolerance=1e-12)
        result_loose = test_exponential_discounting(choices, tolerance=1e-3)

        print(f"Strict: delta in [{result_strict.delta_lower:.6f}, {result_strict.delta_upper:.6f}]")
        print(f"Loose: delta in [{result_loose.delta_lower:.6f}, {result_loose.delta_upper:.6f}]")


class TestDefaultToleranceValues:
    """EVAL: Default tolerance values are appropriate."""

    def test_garp_default_tolerance(self):
        """EVAL: GARP default tolerance (1e-10) is reasonable."""
        from pyrevealed.algorithms.garp import check_garp

        # Data with very small differences
        log = BehaviorLog(
            cost_vectors=np.array([[1.0, 1.0], [1.0, 1.0]]),
            action_vectors=np.array([
                [1.0, 1.0],
                [1.0 + 1e-12, 1.0],  # Tiny difference
            ]),
        )

        # With default tolerance (1e-10), these should be treated as equal
        result = check_garp(log)
        assert result.is_consistent, "Tiny differences should not trigger GARP violation"

    def test_aei_default_tolerance(self):
        """EVAL: AEI default tolerance gives reasonable precision."""
        from pyrevealed.algorithms.aei import compute_aei

        log = BehaviorLog(
            cost_vectors=np.array([[1.0, 1.0], [1.0, 1.0]]),
            action_vectors=np.array([[3.0, 1.0], [1.0, 3.0]]),
        )

        result = compute_aei(log)  # Default tolerance

        # AEI should be precise to at least 4 decimal places
        assert result.efficiency_index == round(result.efficiency_index, 4)


class TestToleranceConsistency:
    """EVAL: Tolerance parameters should be consistent across functions."""

    def test_garp_and_aei_tolerance_consistency(self):
        """EVAL: GARP and AEI should use same tolerance for consistent results."""
        from pyrevealed.algorithms.garp import check_garp
        from pyrevealed.algorithms.aei import compute_aei

        log = BehaviorLog(
            cost_vectors=np.array([[1.0, 1.0], [1.0, 1.0]]),
            action_vectors=np.array([[3.0, 1.0], [1.0, 3.0]]),
        )

        garp = check_garp(log, tolerance=1e-10)
        aei = compute_aei(log, tolerance=1e-10)

        # If GARP is consistent, AEI should be 1.0
        if garp.is_consistent:
            assert aei.efficiency_index == 1.0, (
                f"GARP consistent but AEI != 1.0: {aei.efficiency_index}"
            )
