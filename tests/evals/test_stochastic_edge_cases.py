"""
EVAL: Stochastic choice edge cases.

These tests expose vulnerabilities in stochastic choice algorithms.
"""

import numpy as np
import pytest
from pyrevealed.core.session import StochasticChoiceLog


class TestZeroFrequencies:
    """EVAL: Zero and near-zero choice frequencies."""

    def test_all_zero_frequencies(self, zero_frequency_stochastic):
        """EVAL: Menu with all zero frequencies."""
        from pyrevealed.algorithms.stochastic import fit_random_utility_model

        result = fit_random_utility_model(zero_frequency_stochastic)

        # Should handle gracefully
        assert hasattr(result, 'model_type')

    def test_single_nonzero_frequency(self, single_choice_stochastic):
        """EVAL: Only one item ever chosen (100% probability)."""
        from pyrevealed.algorithms.stochastic import fit_random_utility_model

        result = fit_random_utility_model(single_choice_stochastic)

        # Utility of chosen item should be highest
        assert hasattr(result, 'parameters')


class TestFactorialExplosion:
    """EVAL: RUM testing with many items causes factorial explosion."""

    def test_rum_7_items_factorial_explosion(self, many_items_stochastic):
        """EVAL: RUM exact test with 7 items (7! = 5040 orderings).

        In stochastic.py around line 637:
            if n_items <= 6:
                result = _test_rum_exact(log, tolerance)
            else:
                result = _test_rum_column_generation(log, tolerance, max_iterations)
        """
        from pyrevealed.algorithms.stochastic import test_rum_consistency

        result = test_rum_consistency(many_items_stochastic, max_iterations=100)

        # Should use column generation, not exact enumeration
        assert result.num_iterations <= 100

    @pytest.mark.slow
    def test_rum_8_items_very_slow(self):
        """EVAL: RUM with 8 items (8! = 40320 orderings) is very slow."""
        items = frozenset(range(8))
        freqs = {i: 10 + i for i in range(8)}
        log = StochasticChoiceLog(
            menus=[items],
            choice_frequencies=[freqs],
        )

        from pyrevealed.algorithms.stochastic import test_rum_consistency

        # This should use column generation
        result = test_rum_consistency(log, max_iterations=50)
        assert hasattr(result, 'is_rum_consistent')


class TestRegularityViolations:
    """EVAL: Regularity violation detection."""

    def test_regularity_nested_menus(self, nested_menus_stochastic):
        """EVAL: Regularity with properly nested menus."""
        from pyrevealed.algorithms.stochastic import test_regularity

        result = test_regularity(nested_menus_stochastic)

        # Check if violations are detected correctly
        assert hasattr(result, 'satisfies_regularity')
        assert hasattr(result, 'violations')

    def test_regularity_slight_violation(self):
        """EVAL: Regularity violation just above tolerance."""
        log = StochasticChoiceLog(
            menus=[frozenset({0, 1}), frozenset({0, 1, 2})],
            choice_frequencies=[
                {0: 50, 1: 50},  # P(0|{0,1}) = 0.50
                {0: 52, 1: 28, 2: 20},  # P(0|{0,1,2}) = 0.52 > 0.50
            ],
        )

        from pyrevealed.algorithms.stochastic import test_regularity

        result = test_regularity(log, tolerance=0.01)

        # Should detect the 0.02 violation
        assert len(result.violations) > 0, "Should detect regularity violation"


class TestIIAEdgeCases:
    """EVAL: Independence of Irrelevant Alternatives edge cases."""

    def test_iia_zero_probability_item(self):
        """EVAL: IIA with item that has zero probability in some menus."""
        log = StochasticChoiceLog(
            menus=[frozenset({0, 1, 2}), frozenset({0, 1})],
            choice_frequencies=[
                {0: 50, 1: 50, 2: 0},  # Item 2 never chosen
                {0: 50, 1: 50},
            ],
        )

        from pyrevealed.algorithms.stochastic import check_independence_irrelevant_alternatives

        # Should handle zero probabilities
        result = check_independence_irrelevant_alternatives(log)
        assert isinstance(result, bool)

    def test_iia_single_overlapping_item(self):
        """EVAL: IIA with only one item in common between menus."""
        log = StochasticChoiceLog(
            menus=[frozenset({0, 1}), frozenset({0, 2})],
            choice_frequencies=[
                {0: 70, 1: 30},
                {0: 60, 2: 40},
            ],
        )

        from pyrevealed.algorithms.stochastic import check_independence_irrelevant_alternatives

        # Can't compute odds ratio with single common item
        result = check_independence_irrelevant_alternatives(log)
        assert isinstance(result, bool)


class TestStochasticTransitivity:
    """EVAL: Stochastic transitivity edge cases."""

    def test_transitivity_indifference(self):
        """EVAL: Transitivity with 50-50 probabilities (indifference)."""
        log = StochasticChoiceLog(
            menus=[frozenset({0, 1}), frozenset({1, 2}), frozenset({0, 2})],
            choice_frequencies=[
                {0: 50, 1: 50},  # P(0,1) = 0.5
                {1: 50, 2: 50},  # P(1,2) = 0.5
                {0: 50, 2: 50},  # P(0,2) = 0.5
            ],
        )

        from pyrevealed.algorithms.stochastic import test_stochastic_transitivity

        result = test_stochastic_transitivity(log)

        # With P=0.5 for all pairs, transitivity conditions are vacuously satisfied
        assert hasattr(result, 'satisfies_wst')

    def test_transitivity_perfect_chain(self):
        """EVAL: Transitivity with perfect a > b > c chain."""
        log = StochasticChoiceLog(
            menus=[frozenset({0, 1}), frozenset({1, 2}), frozenset({0, 2})],
            choice_frequencies=[
                {0: 90, 1: 10},  # P(0,1) = 0.9
                {1: 80, 2: 20},  # P(1,2) = 0.8
                {0: 85, 2: 15},  # P(0,2) = 0.85
            ],
        )

        from pyrevealed.algorithms.stochastic import test_stochastic_transitivity

        result = test_stochastic_transitivity(log)

        # Should satisfy WST: P(0,2) > 0.5 given P(0,1) > 0.5 and P(1,2) > 0.5
        assert result.satisfies_wst


class TestRUMConsistency:
    """EVAL: RUM consistency testing edge cases."""

    def test_rum_deterministic_choices(self):
        """EVAL: RUM with deterministic (100% probability) choices."""
        log = StochasticChoiceLog(
            menus=[frozenset({0, 1}), frozenset({1, 2}), frozenset({0, 2})],
            choice_frequencies=[
                {0: 100, 1: 0},
                {1: 100, 2: 0},
                {0: 100, 2: 0},
            ],
        )

        from pyrevealed.algorithms.stochastic import test_rum_consistency

        result = test_rum_consistency(log)

        # Deterministic choices form ordering: 0 > 1 > 2, should be RUM consistent
        assert result.is_rum_consistent

    def test_rum_intransitive_cycle(self):
        """EVAL: RUM with intransitive deterministic cycle."""
        log = StochasticChoiceLog(
            menus=[frozenset({0, 1}), frozenset({1, 2}), frozenset({0, 2})],
            choice_frequencies=[
                {0: 100, 1: 0},  # 0 > 1
                {1: 100, 2: 0},  # 1 > 2
                {0: 0, 2: 100},  # 2 > 0 - cycle!
            ],
        )

        from pyrevealed.algorithms.stochastic import test_rum_consistency

        result = test_rum_consistency(log)

        # Intransitive cycle cannot be rationalized by single ordering
        # But stochastic RUM with multiple orderings might work
        assert hasattr(result, 'is_rum_consistent')


class TestLogLikelihood:
    """EVAL: Log-likelihood computation edge cases."""

    def test_loglikelihood_zero_probability(self):
        """EVAL: Log-likelihood with zero probability observations."""
        log = StochasticChoiceLog(
            menus=[frozenset({0, 1, 2})],
            choice_frequencies=[{0: 100, 1: 0, 2: 0}],  # Only 0 chosen
        )

        from pyrevealed.algorithms.stochastic import fit_random_utility_model

        result = fit_random_utility_model(log)

        # Log-likelihood should be finite (using log(max(p, 1e-10)))
        assert np.isfinite(result.log_likelihood)

    def test_loglikelihood_all_equal_probabilities(self):
        """EVAL: Log-likelihood when all items equally likely."""
        log = StochasticChoiceLog(
            menus=[frozenset({0, 1, 2})],
            choice_frequencies=[{0: 33, 1: 33, 2: 34}],
        )

        from pyrevealed.algorithms.stochastic import fit_random_utility_model

        result = fit_random_utility_model(log)

        # All utilities should be similar
        assert np.isfinite(result.log_likelihood)


class TestModelFitting:
    """EVAL: Model fitting edge cases."""

    def test_logit_extreme_skew(self):
        """EVAL: Logit fitting with extremely skewed data."""
        log = StochasticChoiceLog(
            menus=[frozenset({0, 1})],
            choice_frequencies=[{0: 9999999, 1: 1}],
        )

        from pyrevealed.algorithms.stochastic import fit_random_utility_model

        result = fit_random_utility_model(log, model_type="logit")

        # Should converge despite extreme skew
        assert hasattr(result, 'parameters')

    def test_luce_model_fitting(self):
        """EVAL: Luce model with degenerate data."""
        log = StochasticChoiceLog(
            menus=[frozenset({0, 1})],
            choice_frequencies=[{0: 100, 1: 0}],
        )

        from pyrevealed.algorithms.stochastic import fit_random_utility_model

        result = fit_random_utility_model(log, model_type="luce")

        # Luce model should handle this
        assert hasattr(result, 'model_type')
        assert result.model_type == "luce"
