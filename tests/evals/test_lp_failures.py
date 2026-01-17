"""
EVAL: LP solver failure modes.

Tests for ill-conditioned constraint matrices and solver failures.
"""

import numpy as np
import pytest
from pyrevealed.core.session import BehaviorLog
from pyrevealed.core.exceptions import SolverError, OptimizationError


class TestIllConditionedLP:
    """EVAL: Ill-conditioned LP constraint matrices."""

    def test_afriat_lp_ill_conditioned(self, high_condition_number_log):
        """EVAL: Afriat LP with ill-conditioned constraint matrix."""
        from pyrevealed.algorithms.utility import recover_utility

        try:
            result = recover_utility(high_condition_number_log)
            if result.success:
                # Check for numerical issues in solution
                if result.utility_values is not None:
                    assert np.all(np.isfinite(result.utility_values)), (
                        f"Non-finite utilities: {result.utility_values}"
                    )
        except SolverError as e:
            pytest.xfail(f"LP solver failed on ill-conditioned data: {e}")

    def test_welfare_lp_ill_conditioned(self, high_condition_number_log):
        """EVAL: Welfare LP with ill-conditioned data."""
        from pyrevealed.algorithms.welfare import compute_cv_exact

        try:
            cv, success = compute_cv_exact(
                high_condition_number_log, high_condition_number_log
            )
            if success:
                assert np.isfinite(cv)
        except (SolverError, OptimizationError) as e:
            pytest.xfail(f"Welfare LP failed on ill-conditioned data: {e}")


class TestInfeasibleLP:
    """EVAL: Infeasible LP problems."""

    def test_afriat_infeasible_constraints(self):
        """EVAL: Afriat LP with infeasible constraints (severe GARP violations)."""
        from pyrevealed.algorithms.utility import recover_utility

        # Data with severe violations
        log = BehaviorLog(
            cost_vectors=np.array([
                [1.0, 1.0],
                [1.0, 1.0],
            ]),
            action_vectors=np.array([
                [10.0, 1.0],
                [1.0, 10.0],
            ]),
        )

        result = recover_utility(log)

        # Infeasible should be reported (not crash)
        assert hasattr(result, 'success')

    def test_rum_lp_infeasible(self):
        """EVAL: RUM LP with infeasible probability constraints."""
        from pyrevealed.algorithms.stochastic import test_rum_consistency
        from pyrevealed.core.session import StochasticChoiceLog

        # Intransitive cycle - no RUM can rationalize
        log = StochasticChoiceLog(
            menus=[frozenset({0, 1}), frozenset({1, 2}), frozenset({0, 2})],
            choice_frequencies=[
                {0: 100, 1: 0},
                {1: 100, 2: 0},
                {0: 0, 2: 100},  # Cycle!
            ],
        )

        result = test_rum_consistency(log)

        # Should report infeasibility
        assert not result.is_rum_consistent


class TestPoorScaling:
    """EVAL: LP with poor coefficient scaling."""

    def test_lp_extreme_coefficient_ratio(self):
        """EVAL: LP with coefficients varying by 1e15."""
        from pyrevealed.algorithms.utility import recover_utility

        log = BehaviorLog(
            cost_vectors=np.array([
                [1e-7, 1e8],
                [1e8, 1e-7],
            ]),
            action_vectors=np.array([
                [1e8, 1e-8],
                [1e-8, 1e8],
            ]),
        )

        try:
            result = recover_utility(log)
            # May succeed or fail - document behavior
            print(f"Extreme scaling: success={result.success}")
        except SolverError as e:
            pytest.xfail(f"LP failed with extreme scaling: {e}")

    def test_lp_near_zero_coefficients(self):
        """EVAL: LP with near-zero coefficients."""
        from pyrevealed.algorithms.utility import recover_utility

        log = BehaviorLog(
            cost_vectors=np.array([
                [1e-100, 1e-100],
                [1e-100, 1e-100],
            ]),
            action_vectors=np.array([
                [1.0, 2.0],
                [2.0, 1.0],
            ]),
        )

        try:
            result = recover_utility(log)
            print(f"Near-zero prices: success={result.success}")
        except SolverError as e:
            pytest.xfail(f"LP failed with near-zero coefficients: {e}")


class TestSolverMethods:
    """EVAL: Different LP solver methods."""

    def test_highs_solver(self):
        """EVAL: Using HiGHS solver (default)."""
        from pyrevealed.algorithms.utility import recover_utility

        log = BehaviorLog(
            cost_vectors=np.array([[1.0, 2.0], [2.0, 1.0]]),
            action_vectors=np.array([[3.0, 1.0], [1.0, 3.0]]),
        )

        result = recover_utility(log, solver="highs")

        assert hasattr(result, 'success')

    def test_solver_timeout(self):
        """EVAL: LP solver timeout behavior."""
        from pyrevealed.algorithms.utility import recover_utility

        # Large problem that might be slow
        np.random.seed(42)
        log = BehaviorLog(
            cost_vectors=np.random.rand(100, 10) + 0.1,
            action_vectors=np.random.rand(100, 10) + 0.1,
        )

        import time
        start = time.time()
        result = recover_utility(log)
        elapsed = time.time() - start

        print(f"Utility recovery (T=100, N=10): {elapsed:.2f}s")


class TestBoundaryConditions:
    """EVAL: LP boundary conditions."""

    def test_lp_tight_bounds(self):
        """EVAL: LP where solution is at bounds."""
        from pyrevealed.algorithms.utility import recover_utility

        # Consistent data where utilities should be at lower bounds (0)
        log = BehaviorLog(
            cost_vectors=np.array([[1.0, 2.0], [2.0, 1.0]]),
            action_vectors=np.array([[4.0, 1.0], [1.0, 4.0]]),
        )

        result = recover_utility(log)

        if result.success and result.utility_values is not None:
            # Utilities should be non-negative
            assert np.all(result.utility_values >= 0)

    def test_lp_unbounded_variables(self):
        """EVAL: LP with potentially unbounded variables."""
        from pyrevealed.algorithms.welfare import recover_expenditure_function

        log = BehaviorLog(
            cost_vectors=np.array([[1.0, 1.0], [1.0, 1.0]]),
            action_vectors=np.array([[1.0, 1.0], [1.0, 1.0]]),
        )

        result = recover_expenditure_function(log)

        # Should handle gracefully
        assert 'success' in result


class TestDegenerateSolutions:
    """EVAL: Degenerate LP solutions."""

    def test_lp_multiple_optimal_solutions(self):
        """EVAL: LP with multiple optimal solutions."""
        from pyrevealed.algorithms.utility import recover_utility

        # Data with symmetry - multiple equivalent utility assignments
        log = BehaviorLog(
            cost_vectors=np.array([
                [1.0, 1.0],
                [1.0, 1.0],
                [1.0, 1.0],
            ]),
            action_vectors=np.array([
                [2.0, 2.0],
                [2.0, 2.0],
                [2.0, 2.0],
            ]),
        )

        result = recover_utility(log)

        # Any optimal solution is acceptable
        assert hasattr(result, 'success')

    def test_lp_nearly_degenerate(self):
        """EVAL: LP with nearly degenerate constraints."""
        from pyrevealed.algorithms.utility import recover_utility

        # Very similar observations
        log = BehaviorLog(
            cost_vectors=np.array([
                [1.0, 1.0],
                [1.0 + 1e-10, 1.0],
                [1.0, 1.0 + 1e-10],
            ]),
            action_vectors=np.array([
                [2.0, 2.0],
                [2.0 + 1e-10, 2.0],
                [2.0, 2.0 + 1e-10],
            ]),
        )

        result = recover_utility(log)

        # Should handle near-degeneracy
        assert hasattr(result, 'success')


class TestSolverErrors:
    """EVAL: Error handling from LP solvers."""

    def test_solver_error_propagation(self):
        """EVAL: SolverError should propagate with useful information."""
        from pyrevealed.algorithms.welfare import _recover_afriat_utility

        # Try to trigger solver error with bad data
        log = BehaviorLog(
            cost_vectors=np.array([[1e-300, 1e300]]),
            action_vectors=np.array([[1e300, 1e-300]]),
        )

        try:
            U, lambdas, success = _recover_afriat_utility(log)
            # If it succeeds, check results
            if success:
                assert np.all(np.isfinite(U))
        except SolverError as e:
            # Error should have useful message
            assert str(e), "SolverError should have message"

    def test_optimization_error_handling(self):
        """EVAL: OptimizationError handling in welfare computation."""
        from pyrevealed.algorithms.welfare import compute_cv_exact

        # Data that might fail optimization
        baseline = BehaviorLog(
            cost_vectors=np.array([[1e-10, 1e10]]),
            action_vectors=np.array([[1e10, 1e-10]]),
        )

        try:
            cv, success = compute_cv_exact(baseline, baseline)
            print(f"CV computation: success={success}, cv={cv}")
        except (SolverError, OptimizationError) as e:
            # Should fail gracefully with informative error
            assert str(e), "Error should have message"
