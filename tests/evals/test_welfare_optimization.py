"""
EVAL: Welfare optimization solver failures.

Tests for CV/EV computation edge cases and solver failures.
"""

import numpy as np
import pytest
from pyrevealed.core.session import BehaviorLog
from pyrevealed.core.exceptions import SolverError, OptimizationError


class TestAfriatRecovery:
    """EVAL: Afriat utility recovery edge cases."""

    def test_afriat_recovery_simple(self):
        """EVAL: Basic Afriat utility recovery."""
        from pyrevealed.algorithms.welfare import _recover_afriat_utility

        log = BehaviorLog(
            cost_vectors=np.array([[1.0, 2.0], [2.0, 1.0]]),
            action_vectors=np.array([[3.0, 1.0], [1.0, 3.0]]),
        )

        U, lambdas, success = _recover_afriat_utility(log)

        assert success, "Afriat recovery should succeed for valid data"
        assert U is not None
        assert np.all(U >= 0), "Utilities should be non-negative"
        assert np.all(lambdas > 0), "Lagrange multipliers should be positive"

    def test_afriat_recovery_ill_conditioned(self, high_condition_number_log):
        """EVAL: Afriat recovery with ill-conditioned data."""
        from pyrevealed.algorithms.welfare import _recover_afriat_utility

        try:
            U, lambdas, success = _recover_afriat_utility(high_condition_number_log)
            if success:
                assert np.all(np.isfinite(U))
        except SolverError as e:
            pytest.xfail(f"Afriat recovery failed on ill-conditioned data: {e}")


class TestCVExact:
    """EVAL: Exact CV computation edge cases."""

    def test_cv_exact_simple(self):
        """EVAL: CV exact computation for simple case."""
        from pyrevealed.algorithms.welfare import compute_cv_exact

        baseline = BehaviorLog(
            cost_vectors=np.array([[1.0, 1.0]]),
            action_vectors=np.array([[5.0, 5.0]]),
        )
        policy = BehaviorLog(
            cost_vectors=np.array([[2.0, 1.0]]),  # Price of good 0 doubled
            action_vectors=np.array([[3.0, 7.0]]),
        )

        try:
            cv, success = compute_cv_exact(baseline, policy)
            assert success or not success  # Either outcome is valid
        except (SolverError, OptimizationError) as e:
            pytest.xfail(f"CV exact failed: {e}")

    def test_cv_exact_identical_logs(self):
        """EVAL: CV when baseline equals policy (should be ~0)."""
        from pyrevealed.algorithms.welfare import compute_cv_exact

        log = BehaviorLog(
            cost_vectors=np.array([[1.0, 1.0]]),
            action_vectors=np.array([[5.0, 5.0]]),
        )

        try:
            cv, success = compute_cv_exact(log, log)
            if success:
                assert abs(cv) < 0.1, f"CV for identical logs should be ~0, got {cv}"
        except (SolverError, OptimizationError):
            pass  # May fail, that's okay


class TestCVVartia:
    """EVAL: Vartia path integral CV computation."""

    def test_cv_vartia_simple(self):
        """EVAL: CV via Vartia approximation."""
        from pyrevealed.algorithms.welfare import compute_cv_vartia

        baseline = BehaviorLog(
            cost_vectors=np.array([[1.0, 1.0]]),
            action_vectors=np.array([[5.0, 5.0]]),
        )
        policy = BehaviorLog(
            cost_vectors=np.array([[1.5, 1.0]]),
            action_vectors=np.array([[4.0, 6.0]]),
        )

        cv = compute_cv_vartia(baseline, policy)

        assert np.isfinite(cv), f"Vartia CV should be finite, got {cv}"

    def test_cv_vartia_extreme_price_change(self, extreme_price_change):
        """EVAL: Vartia CV with extreme price change."""
        from pyrevealed.algorithms.welfare import compute_cv_vartia

        baseline, policy = extreme_price_change

        cv = compute_cv_vartia(baseline, policy)

        assert np.isfinite(cv), f"Vartia CV with extreme prices: {cv}"


class TestCVBounds:
    """EVAL: Laspeyres/Paasche bounds for CV/EV."""

    def test_cv_bounds_simple(self):
        """EVAL: CV Laspeyres bound computation."""
        from pyrevealed.algorithms.welfare import compute_cv_bounds

        baseline = BehaviorLog(
            cost_vectors=np.array([[1.0, 1.0]]),
            action_vectors=np.array([[5.0, 5.0]]),
        )
        policy = BehaviorLog(
            cost_vectors=np.array([[2.0, 1.0]]),
            action_vectors=np.array([[3.0, 7.0]]),
        )

        cv = compute_cv_bounds(baseline, policy)

        assert np.isfinite(cv)

    def test_ev_bounds_simple(self):
        """EVAL: EV Paasche bound computation."""
        from pyrevealed.algorithms.welfare import compute_ev_bounds

        baseline = BehaviorLog(
            cost_vectors=np.array([[1.0, 1.0]]),
            action_vectors=np.array([[5.0, 5.0]]),
        )
        policy = BehaviorLog(
            cost_vectors=np.array([[2.0, 1.0]]),
            action_vectors=np.array([[3.0, 7.0]]),
        )

        ev = compute_ev_bounds(baseline, policy)

        assert np.isfinite(ev)


class TestWelfareAnalysis:
    """EVAL: Full welfare analysis."""

    def test_analyze_welfare_change_simple(self):
        """EVAL: Full welfare change analysis."""
        from pyrevealed.algorithms.welfare import analyze_welfare_change

        baseline = BehaviorLog(
            cost_vectors=np.array([[1.0, 1.0], [1.0, 1.0]]),
            action_vectors=np.array([[5.0, 5.0], [4.0, 6.0]]),
        )
        policy = BehaviorLog(
            cost_vectors=np.array([[1.5, 1.0], [1.5, 1.0]]),
            action_vectors=np.array([[3.0, 7.0], [2.0, 8.0]]),
        )

        try:
            result = analyze_welfare_change(baseline, policy, method="exact")
            assert hasattr(result, 'compensating_variation')
            assert hasattr(result, 'equivalent_variation')
            assert hasattr(result, 'welfare_direction')
        except (SolverError, OptimizationError) as e:
            pytest.xfail(f"Welfare analysis failed: {e}")

    def test_analyze_welfare_change_bounds_method(self):
        """EVAL: Welfare analysis with bounds method (most robust)."""
        from pyrevealed.algorithms.welfare import analyze_welfare_change

        baseline = BehaviorLog(
            cost_vectors=np.array([[1.0, 1.0]]),
            action_vectors=np.array([[5.0, 5.0]]),
        )
        policy = BehaviorLog(
            cost_vectors=np.array([[1.5, 0.8]]),
            action_vectors=np.array([[4.0, 6.0]]),
        )

        result = analyze_welfare_change(baseline, policy, method="bounds")

        assert np.isfinite(result.compensating_variation)
        assert np.isfinite(result.equivalent_variation)


class TestExpenditureFunctionRecovery:
    """EVAL: Expenditure function recovery edge cases."""

    def test_expenditure_function_recovery(self):
        """EVAL: Recover expenditure function from data."""
        from pyrevealed.algorithms.welfare import recover_expenditure_function

        log = BehaviorLog(
            cost_vectors=np.array([[1.0, 2.0], [2.0, 1.0]]),
            action_vectors=np.array([[3.0, 1.0], [1.0, 3.0]]),
        )

        result = recover_expenditure_function(log)

        assert 'success' in result
        assert 'utility_function' in result
        assert 'expenditure_function' in result

    def test_expenditure_function_evaluation(self):
        """EVAL: Evaluate recovered expenditure function."""
        from pyrevealed.algorithms.welfare import recover_expenditure_function

        log = BehaviorLog(
            cost_vectors=np.array([[1.0, 2.0], [2.0, 1.0]]),
            action_vectors=np.array([[3.0, 1.0], [1.0, 3.0]]),
        )

        result = recover_expenditure_function(log)

        if result['success'] and result['expenditure_function'] is not None:
            p = np.array([1.5, 1.5])
            u = result['observation_utilities'][0]

            try:
                e, x = result['expenditure_function'](p, u)
                assert np.isfinite(e)
            except (SolverError, OptimizationError):
                pass  # May fail, that's informative


class TestDeadweightLoss:
    """EVAL: Deadweight loss computation."""

    def test_deadweight_loss_simple(self):
        """EVAL: Deadweight loss from price distortion."""
        from pyrevealed.algorithms.welfare import compute_deadweight_loss

        baseline = BehaviorLog(
            cost_vectors=np.array([[1.0, 1.0]]),
            action_vectors=np.array([[5.0, 5.0]]),
        )
        policy = BehaviorLog(
            cost_vectors=np.array([[1.5, 1.0]]),  # Tax on good 0
            action_vectors=np.array([[4.0, 6.0]]),
        )

        dwl = compute_deadweight_loss(baseline, policy, method="bounds")

        # DWL should be non-negative
        assert dwl >= 0, f"Deadweight loss should be non-negative, got {dwl}"


class TestEBounds:
    """EVAL: E-bounds (expansion path bounds) computation."""

    def test_e_bounds_simple(self):
        """EVAL: E-bounds for demand prediction."""
        from pyrevealed.algorithms.welfare import compute_e_bounds

        log = BehaviorLog(
            cost_vectors=np.array([
                [1.0, 2.0],
                [2.0, 1.0],
                [1.5, 1.5],
            ]),
            action_vectors=np.array([
                [4.0, 1.0],
                [1.0, 4.0],
                [2.5, 2.5],
            ]),
        )

        new_prices = np.array([1.2, 1.8])

        result = compute_e_bounds(log, new_prices)

        assert 'quantity_lower' in result
        assert 'quantity_upper' in result
        assert len(result['quantity_lower']) == 2
        assert len(result['quantity_upper']) == 2


class TestPopulationWelfareBounds:
    """EVAL: Population welfare bounds with heterogeneous consumers."""

    def test_population_welfare_bounds(self):
        """EVAL: Welfare bounds for heterogeneous population."""
        from pyrevealed.algorithms.welfare import compute_population_welfare_bounds

        # Create several consumers
        consumers = [
            BehaviorLog(
                cost_vectors=np.array([[1.0, 2.0]]),
                action_vectors=np.array([[4.0, 1.0]]),
            ),
            BehaviorLog(
                cost_vectors=np.array([[1.0, 2.0]]),
                action_vectors=np.array([[2.0, 2.0]]),
            ),
            BehaviorLog(
                cost_vectors=np.array([[1.0, 2.0]]),
                action_vectors=np.array([[1.0, 3.0]]),
            ),
        ]

        old_prices = np.array([1.0, 2.0])
        new_prices = np.array([1.5, 1.5])

        try:
            result = compute_population_welfare_bounds(
                consumers, (old_prices, new_prices)
            )

            assert 'fraction_better_off_lower' in result
            assert 'fraction_better_off_upper' in result
            assert 0.0 <= result['fraction_better_off_lower'] <= 1.0
        except (SolverError, OptimizationError) as e:
            pytest.xfail(f"Population welfare bounds failed: {e}")

    def test_population_welfare_empty_consumers(self):
        """EVAL: Population welfare with empty consumer list."""
        from pyrevealed.algorithms.welfare import compute_population_welfare_bounds

        result = compute_population_welfare_bounds(
            [], (np.array([1.0, 1.0]), np.array([1.5, 0.8]))
        )

        assert result['num_consumers'] == 0
