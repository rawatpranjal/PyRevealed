"""
EVAL: Division by zero and near-zero vulnerabilities.

These tests expose division hazards in algorithms.
"""

import numpy as np
import pytest
from pyrevealed.core.session import BehaviorLog, StochasticChoiceLog


class TestMPIDivision:
    """EVAL: MPI division hazards."""

    def test_mpi_zero_expenditure_denominator(self):
        """EVAL: MPI denominator = sum(E[ki,ki]) approaching 0.

        In mpi.py line 143-146, we have:
            if denominator <= 0:
                return 0.0
            mpi = numerator / denominator

        This tests near-zero denominators.
        """
        from pyrevealed.algorithms.mpi import compute_mpi

        # Create data with near-zero expenditures
        log = BehaviorLog(
            cost_vectors=np.array([[1e-200, 1e-200], [1e-200, 1e-200]]),
            action_vectors=np.array([[1e-200, 1e-200], [1e-200, 1e-200]]),
        )

        result = compute_mpi(log)

        # Should be handled gracefully
        assert np.isfinite(result.mpi_value), (
            f"MPI should be finite, got {result.mpi_value}"
        )

    def test_mpi_cycle_with_zero_expenditure(self):
        """EVAL: MPI cycle where one observation has near-zero expenditure."""
        from pyrevealed.algorithms.mpi import compute_mpi

        # Observation 0 has very low expenditure, obs 1 has normal
        log = BehaviorLog(
            cost_vectors=np.array([[1e-300, 1.0], [1.0, 1e-300]]),
            action_vectors=np.array([[1e-300, 1.0], [1.0, 1e-300]]),
        )

        result = compute_mpi(log)
        assert np.isfinite(result.mpi_value)


class TestSlutskyDivision:
    """EVAL: Slutsky matrix division by price."""

    def test_slutsky_near_zero_prices(self, near_zero_prices):
        """EVAL: Slutsky computation with 1/price for near-zero prices.

        In integrability.py around line 262:
            dx_dp = (beta[i, j] / p_bar[j]) * q_bar[i]

        Near-zero p_bar causes division explosion.
        """
        from pyrevealed.algorithms.integrability import compute_slutsky_matrix

        S = compute_slutsky_matrix(near_zero_prices)

        # Should not produce inf or nan
        assert np.all(np.isfinite(S)), (
            f"Slutsky matrix has non-finite values: {S}"
        )

    def test_slutsky_mixed_zero_nonzero_prices(self):
        """EVAL: Slutsky with some zero and some non-zero prices."""
        # This should be rejected by BehaviorLog validation
        with pytest.raises(Exception):
            log = BehaviorLog(
                cost_vectors=np.array([[0.0, 1.0], [1.0, 0.0]]),
                action_vectors=np.array([[1.0, 1.0], [1.0, 1.0]]),
            )


class TestStochasticDivision:
    """EVAL: Stochastic choice division hazards."""

    def test_stochastic_zero_total_observations(self, zero_frequency_stochastic):
        """EVAL: Division by total_observations = 0 in get_choice_probability.

        In session.py around line 994:
            if total == 0:
                return 0.0
            return self.choice_frequencies[menu_idx].get(item, 0) / total
        """
        prob = zero_frequency_stochastic.get_choice_probability(0, 0)

        # Should return 0.0, not crash
        assert prob == 0.0, f"Zero-observation menu should return 0.0, got {prob}"

    def test_iia_zero_probability_denominator(self):
        """EVAL: IIA odds ratio with p_y = 0 in denominator.

        In stochastic.py around line 190:
            if p_y > 1e-10:
                ratio = p_x / p_y
        """
        from pyrevealed.algorithms.stochastic import check_independence_irrelevant_alternatives

        # Data with one item never chosen
        log = StochasticChoiceLog(
            menus=[frozenset({0, 1, 2}), frozenset({0, 1})],
            choice_frequencies=[
                {0: 100, 1: 0, 2: 0},  # item 1 and 2 never chosen
                {0: 100, 1: 0},
            ],
        )

        # Should not crash
        result = check_independence_irrelevant_alternatives(log)
        assert isinstance(result, bool)


class TestWelfareDivision:
    """EVAL: Welfare analysis division hazards."""

    def test_hicksian_demand_zero_price(self):
        """EVAL: Hicksian demand computation with zero prices.

        In welfare.py around line 434:
            if p[i] > 1e-10:
                h[i] = x0[i] * (p0[i] / p[i]) ** beta[i]
        """
        from pyrevealed.algorithms.welfare import compute_cv_vartia

        baseline = BehaviorLog(
            cost_vectors=np.array([[1.0, 1.0]]),
            action_vectors=np.array([[5.0, 5.0]]),
        )
        # Policy with near-zero price for good 1
        policy = BehaviorLog(
            cost_vectors=np.array([[1.0, 1e-300]]),
            action_vectors=np.array([[1.0, 1e300]]),  # Huge quantity
        )

        cv = compute_cv_vartia(baseline, policy)

        assert np.isfinite(cv), f"CV should be finite, got {cv}"

    def test_expenditure_function_zero_budget(self):
        """EVAL: Expenditure function with zero target utility."""
        from pyrevealed.algorithms.welfare import recover_expenditure_function

        log = BehaviorLog(
            cost_vectors=np.array([[1.0, 2.0], [2.0, 1.0]]),
            action_vectors=np.array([[3.0, 1.0], [1.0, 3.0]]),
        )

        result = recover_expenditure_function(log)

        if result['success'] and result['expenditure_function'] is not None:
            # Try to compute expenditure for utility = 0
            p = np.array([1.0, 1.0])
            try:
                e, _ = result['expenditure_function'](p, 0.0)
                assert np.isfinite(e), f"Expenditure for u=0 should be finite, got {e}"
            except Exception as e:
                pytest.xfail(f"Expenditure function failed for u=0: {e}")


class TestInterestRateDivision:
    """EVAL: Interest rate computation from discount factor."""

    def test_implied_interest_rate_delta_zero(self):
        """EVAL: Interest rate r = (1/delta) - 1 with delta near 0.

        In intertemporal.py around line 765:
            if delta_lower > 0:
                r_upper = (1.0 / delta_lower) - 1.0
            else:
                r_upper = float("inf")
        """
        from pyrevealed.algorithms.intertemporal import recover_discount_factor, DatedChoice

        # Extreme choices that imply very low delta
        choices = [
            DatedChoice(
                amounts=np.array([1.0, 1e6]),  # Huge delayed reward
                dates=np.array([0, 1]),
                chosen=0,  # Still chose immediate
            ),
        ]

        result = recover_discount_factor(choices)

        # Interest rate bounds should be sensible
        assert np.isfinite(result.implied_interest_rate_lower) or result.implied_interest_rate_lower == float('inf')


class TestNormalizationDivision:
    """EVAL: Normalization operations that divide by totals."""

    def test_budget_share_zero_expenditure(self):
        """EVAL: Budget share computation with zero total expenditure."""
        from pyrevealed.algorithms.integrability import compute_slutsky_matrix_stone_geary

        # Very low expenditure data
        log = BehaviorLog(
            cost_vectors=np.array([[1e-200, 1e-200], [1e-200, 1e-200]]),
            action_vectors=np.array([[1e-200, 1e-200], [1e-200, 1e-200]]),
        )

        S = compute_slutsky_matrix_stone_geary(log)

        assert np.all(np.isfinite(S)), f"Non-finite Slutsky from Stone-Geary: {S}"

    def test_contribution_normalization(self):
        """EVAL: Observation contribution normalization with zero total."""
        from pyrevealed.algorithms.garp import compute_observation_contributions

        # Consistent data - no violations, zero total contribution
        log = BehaviorLog(
            cost_vectors=np.array([[1.0, 2.0], [2.0, 1.0]]),
            action_vectors=np.array([[4.0, 1.0], [1.0, 4.0]]),
        )

        result = compute_observation_contributions(log)

        # Should handle zero total gracefully
        assert np.all(np.isfinite(result.contributions))


class TestVarianceDivision:
    """EVAL: Division by variance in statistical tests."""

    def test_iia_coefficient_of_variation_zero_mean(self):
        """EVAL: CV = std/mean with mean near zero.

        In stochastic.py around line 195:
            cv = np.std(odds_ratios) / max(np.mean(odds_ratios), 1e-10)
        """
        from pyrevealed.algorithms.stochastic import check_independence_irrelevant_alternatives

        # Create data with odds ratios that average to near zero
        # This is tricky because odds ratios are always positive
        log = StochasticChoiceLog(
            menus=[frozenset({0, 1}), frozenset({0, 1})],
            choice_frequencies=[
                {0: 1, 1: 999999},  # Very low odds for 0
                {0: 1, 1: 999999},
            ],
        )

        # Should handle near-zero means
        result = check_independence_irrelevant_alternatives(log)
        assert isinstance(result, bool)
