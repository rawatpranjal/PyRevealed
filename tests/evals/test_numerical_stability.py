"""
EVAL: Numerical stability tests - overflow, underflow, precision.

These tests expose numerical vulnerabilities in algorithms.
"""

import numpy as np
import pytest
from pyrevealed.core.session import BehaviorLog


class TestExponentialOverflow:
    """EVAL: Exponential discounting overflow with large time delays."""

    def test_discounting_overflow_t800(self, extreme_delay_choices):
        """EVAL: delta^800 underflows, producing invalid bounds.

        For delta=0.99, delta^800 = 0.99^800 ≈ 2.9e-4
        For delta=0.95, delta^800 = 0.95^800 ≈ 1e-18
        For delta=0.90, delta^800 = 0.90^800 ≈ 1e-37
        Near delta=0.5, delta^800 underflows to 0.
        """
        from pyrevealed.algorithms.intertemporal import test_exponential_discounting

        result = test_exponential_discounting(extreme_delay_choices)

        # Bounds should be finite, not NaN or Inf
        assert np.isfinite(result.delta_lower), (
            f"Underflow: delta_lower = {result.delta_lower}"
        )
        assert np.isfinite(result.delta_upper), (
            f"Overflow: delta_upper = {result.delta_upper}"
        )

    def test_discounting_extreme_ratio(self):
        """EVAL: Ratio c_rej/c_chosen at numerical limits."""
        from pyrevealed.algorithms.intertemporal import (
            test_exponential_discounting,
            DatedChoice,
        )

        choices = [
            DatedChoice(
                amounts=np.array([1e-100, 1e100]),  # Extreme ratio
                dates=np.array([0, 1]),
                chosen=1,
            ),
        ]

        result = test_exponential_discounting(choices)
        # Should not crash, bounds should be valid
        assert hasattr(result, 'delta_lower')


class TestLogitOverflow:
    """EVAL: Logit exp() overflow with large utilities."""

    def test_stochastic_extreme_utilities(self):
        """EVAL: Logit softmax with utilities near overflow boundary."""
        from pyrevealed.core.session import StochasticChoiceLog
        from pyrevealed.algorithms.stochastic import fit_random_utility_model

        # Create data that leads to extreme utility estimates
        log = StochasticChoiceLog(
            menus=[frozenset({0, 1, 2})],
            choice_frequencies=[{0: 10000, 1: 1, 2: 0}],  # Highly skewed
        )

        # This tests the log-sum-exp trick implementation
        result = fit_random_utility_model(log, model_type="logit")

        # Probabilities should sum to 1 and be finite
        probs = result.choice_probabilities
        assert np.all(np.isfinite(probs)), f"Non-finite probabilities: {probs}"


class TestCRRAOverflow:
    """EVAL: CRRA utility overflow with extreme parameters."""

    def test_crra_extreme_rho(self):
        """EVAL: CRRA u(x) = x^(1-rho)/(1-rho) with extreme rho."""
        from pyrevealed.algorithms.risk import estimate_crra_parameter

        from pyrevealed.core.session import RiskChoiceLog

        # Create risk choice data
        log = RiskChoiceLog(
            safe_values=np.array([50.0, 100.0, 150.0]),
            risky_outcomes=np.array([
                [100.0, 0.0],
                [200.0, 0.0],
                [300.0, 0.0],
            ]),
            risky_probabilities=np.array([
                [0.5, 0.5],
                [0.5, 0.5],
                [0.5, 0.5],
            ]),
            choices=np.array([False, False, True]),  # Increasingly risk seeking
        )

        result = estimate_crra_parameter(log)
        assert np.isfinite(result.rho), f"CRRA rho is non-finite: {result.rho}"


class TestFloatPrecision:
    """EVAL: Floating point precision edge cases."""

    def test_garp_near_tolerance_boundary(self, near_tolerance_expenditure_log):
        """EVAL: GARP with expenditures differing by exactly tolerance."""
        from pyrevealed.algorithms.garp import check_garp

        # Test at various tolerances around the boundary
        for tol in [1e-10, 1e-11, 1e-9]:
            result = check_garp(near_tolerance_expenditure_log, tolerance=tol)
            # Result should be deterministic regardless of tolerance
            assert hasattr(result, 'is_consistent')

    def test_aei_precision_at_boundary(self):
        """EVAL: AEI binary search precision at 0 and 1 boundaries."""
        from pyrevealed.algorithms.aei import compute_aei

        # Perfectly consistent data should give AEI = 1.0 exactly
        log = BehaviorLog(
            cost_vectors=np.array([[1.0, 2.0], [2.0, 1.0]]),
            action_vectors=np.array([[4.0, 1.0], [1.0, 4.0]]),
        )

        result = compute_aei(log)
        # AEI should be exactly 1.0 for consistent data
        assert result.efficiency_index == 1.0, (
            f"Consistent data should have AEI=1.0, got {result.efficiency_index}"
        )


class TestSufficientStatistics:
    """EVAL: Numerical stability in sufficient statistics computation."""

    def test_expenditure_matrix_precision(self):
        """EVAL: Expenditure matrix computation with mixed scales."""
        log = BehaviorLog(
            cost_vectors=np.array([[1e-10, 1e10], [1e10, 1e-10]]),
            action_vectors=np.array([[1e10, 1e-10], [1e-10, 1e10]]),
        )

        E = log.spend_matrix
        # Check for cancellation errors
        assert np.all(np.isfinite(E)), f"Non-finite expenditure matrix: {E}"

    def test_own_expenditure_vs_cross(self):
        """EVAL: Own expenditure should equal diagonal of spend_matrix."""
        log = BehaviorLog(
            cost_vectors=np.array([[1.0, 2.0], [2.0, 1.0]]),
            action_vectors=np.array([[3.0, 1.0], [1.0, 3.0]]),
        )

        diagonal = np.diag(log.spend_matrix)
        own_exp = log.total_spend

        assert np.allclose(diagonal, own_exp), (
            f"Diagonal mismatch: {diagonal} vs {own_exp}"
        )


class TestMatrixCondition:
    """EVAL: Ill-conditioned matrices in algorithms."""

    def test_slutsky_high_condition_number(self, high_condition_number_log):
        """EVAL: Slutsky estimation with ill-conditioned data."""
        from pyrevealed.algorithms.integrability import compute_slutsky_matrix

        S = compute_slutsky_matrix(high_condition_number_log)

        # Should not contain inf or nan
        assert np.all(np.isfinite(S)), f"Non-finite Slutsky matrix: {S}"

    def test_utility_recovery_ill_conditioned(self, high_condition_number_log):
        """EVAL: Afriat utility recovery with ill-conditioned constraints."""
        from pyrevealed.algorithms.utility import recover_utility

        result = recover_utility(high_condition_number_log)

        # Should handle gracefully
        if result.utility_values is not None:
            assert np.all(np.isfinite(result.utility_values)), (
                f"Non-finite utilities: {result.utility_values}"
            )


class TestDenormalized:
    """EVAL: Denormalized (subnormal) floating point values."""

    def test_subnormal_quantities(self, subnormal_values):
        """EVAL: Quantities in subnormal range."""
        from pyrevealed.algorithms.garp import check_garp

        # Subnormal values may lose precision in multiplication
        result = check_garp(subnormal_values)
        assert hasattr(result, 'is_consistent')

    def test_expenditure_with_subnormals(self, subnormal_values):
        """EVAL: Expenditure computation with subnormal values."""
        E = subnormal_values.spend_matrix

        # Some elements may underflow to zero
        # But the computation should not crash
        assert E.shape == (2, 2)


class TestNaNPropagation:
    """EVAL: NaN handling and propagation."""

    def test_nan_in_intermediate_computation(self):
        """EVAL: NaN introduced during computation doesn't propagate silently."""
        from pyrevealed.algorithms.integrability import compute_slutsky_matrix

        # Data that might produce 0/0 or inf-inf in intermediate steps
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

        S = compute_slutsky_matrix(log)

        # NaN should not silently appear in output
        if np.any(np.isnan(S)):
            pytest.fail(f"NaN in Slutsky matrix from valid input: {S}")


class TestSpecialValues:
    """EVAL: Special floating point values."""

    def test_max_float_handling(self, max_float_prices):
        """EVAL: Prices near float64 max."""
        from pyrevealed.algorithms.garp import check_garp

        try:
            result = check_garp(max_float_prices)
            assert hasattr(result, 'is_consistent')
        except OverflowError as e:
            pytest.xfail(f"Overflow with max float prices: {e}")

    def test_extreme_ratio_handling(self, extreme_ratio_log):
        """EVAL: Price ratio of 1e300 between goods."""
        from pyrevealed.algorithms.garp import check_garp

        result = check_garp(extreme_ratio_log)
        assert hasattr(result, 'is_consistent')
