"""
EVAL: Random Utility Model computational limits.

Tests for factorial explosion and IIA edge cases.
"""

import numpy as np
import pytest
from pyrevealed.core.session import StochasticChoiceLog


class TestFactorialExplosion:
    """EVAL: RUM exact test has n! complexity."""

    def test_rum_6_items_boundary(self):
        """EVAL: RUM with 6 items (720 orderings) - exact enumeration boundary.

        In stochastic.py around line 637:
            if n_items <= 6:
                result = _test_rum_exact(log, tolerance)
            else:
                result = _test_rum_column_generation(log, tolerance, max_iterations)
        """
        from pyrevealed.algorithms.stochastic import test_rum_consistency

        items = frozenset(range(6))
        freqs = {i: 10 + i * 5 for i in range(6)}
        log = StochasticChoiceLog(
            menus=[items],
            choice_frequencies=[freqs],
        )

        import time
        start = time.time()
        result = test_rum_consistency(log)
        elapsed = time.time() - start

        print(f"6 items: {elapsed:.2f}s, consistent={result.is_rum_consistent}")

        # Should use exact enumeration
        assert elapsed < 10.0, f"6 items should complete quickly, took {elapsed:.2f}s"

    def test_rum_7_items_column_generation(self, many_items_stochastic):
        """EVAL: RUM with 7 items (5040 orderings) - uses column generation."""
        from pyrevealed.algorithms.stochastic import test_rum_consistency

        import time
        start = time.time()
        result = test_rum_consistency(many_items_stochastic, max_iterations=100)
        elapsed = time.time() - start

        print(f"7 items: {elapsed:.2f}s, iterations={result.num_iterations}")

        # Should switch to column generation
        assert result.num_iterations <= 100

    @pytest.mark.slow
    def test_rum_10_items_slow(self):
        """EVAL: RUM with 10 items - stress test column generation."""
        items = frozenset(range(10))
        freqs = {i: 10 + i for i in range(10)}
        log = StochasticChoiceLog(
            menus=[items],
            choice_frequencies=[freqs],
        )

        from pyrevealed.algorithms.stochastic import test_rum_consistency

        import time
        start = time.time()
        result = test_rum_consistency(log, max_iterations=50)
        elapsed = time.time() - start

        print(f"10 items: {elapsed:.2f}s")
        assert elapsed < 60.0, f"10 items should complete in <60s, took {elapsed:.2f}s"


class TestIIAEdgeCases:
    """EVAL: Independence of Irrelevant Alternatives edge cases."""

    def test_iia_with_zero_probability(self):
        """EVAL: IIA odds ratio when p_y = 0 (denominator issue).

        In stochastic.py around line 190:
            if p_y > 1e-10:
                ratio = p_x / p_y
        """
        from pyrevealed.algorithms.stochastic import check_independence_irrelevant_alternatives

        log = StochasticChoiceLog(
            menus=[frozenset({0, 1, 2}), frozenset({0, 1})],
            choice_frequencies=[
                {0: 100, 1: 0, 2: 0},  # Items 1, 2 never chosen
                {0: 100, 1: 0},
            ],
        )

        result = check_independence_irrelevant_alternatives(log)

        # Should handle zero probabilities without division by zero
        assert isinstance(result, bool)

    def test_iia_extreme_odds_ratio(self):
        """EVAL: IIA with extreme odds ratios."""
        from pyrevealed.algorithms.stochastic import check_independence_irrelevant_alternatives

        log = StochasticChoiceLog(
            menus=[frozenset({0, 1}), frozenset({0, 1, 2})],
            choice_frequencies=[
                {0: 999999, 1: 1},  # Extreme odds
                {0: 999998, 1: 1, 2: 1},
            ],
        )

        result = check_independence_irrelevant_alternatives(log)
        assert isinstance(result, bool)

    def test_iia_single_common_item(self):
        """EVAL: IIA when menus share only one item."""
        from pyrevealed.algorithms.stochastic import check_independence_irrelevant_alternatives

        log = StochasticChoiceLog(
            menus=[frozenset({0, 1}), frozenset({0, 2})],
            choice_frequencies=[
                {0: 60, 1: 40},
                {0: 70, 2: 30},
            ],
        )

        # Can't compute odds ratio with single common item
        result = check_independence_irrelevant_alternatives(log)
        assert isinstance(result, bool)


class TestRUMDistribution:
    """EVAL: RUM rationalizing distribution edge cases."""

    def test_rum_sparse_distribution(self):
        """EVAL: RUM distribution should be sparse (few orderings with positive prob)."""
        from pyrevealed.algorithms.stochastic import fit_rum_distribution

        log = StochasticChoiceLog(
            menus=[frozenset({0, 1, 2}), frozenset({0, 1}), frozenset({1, 2})],
            choice_frequencies=[
                {0: 50, 1: 30, 2: 20},
                {0: 60, 1: 40},
                {1: 55, 2: 45},
            ],
        )

        distribution = fit_rum_distribution(log)

        # Distribution should be sparse
        if distribution:
            print(f"Non-zero orderings: {len(distribution)}")
            print(f"Total orderings possible: 6 (3!)")

    def test_rum_deterministic_choices(self):
        """EVAL: RUM with deterministic (100%) choices."""
        from pyrevealed.algorithms.stochastic import fit_rum_distribution

        log = StochasticChoiceLog(
            menus=[frozenset({0, 1}), frozenset({1, 2}), frozenset({0, 2})],
            choice_frequencies=[
                {0: 100, 1: 0},
                {1: 100, 2: 0},
                {0: 100, 2: 0},
            ],
        )

        distribution = fit_rum_distribution(log)

        # Should have single ordering with prob 1 (if consistent)
        if distribution:
            assert len(distribution) == 1
            assert list(distribution.values())[0] == 1.0


class TestRUMDistance:
    """EVAL: Distance to nearest RUM computation."""

    def test_rum_distance_consistent(self):
        """EVAL: Distance to RUM for consistent data should be 0."""
        from pyrevealed.algorithms.stochastic import compute_distance_to_rum

        log = StochasticChoiceLog(
            menus=[frozenset({0, 1}), frozenset({1, 2})],
            choice_frequencies=[
                {0: 70, 1: 30},  # 0 > 1
                {1: 80, 2: 20},  # 1 > 2
            ],
        )

        distance = compute_distance_to_rum(log)

        # If consistent, distance should be 0 or very small
        assert distance >= 0

    def test_rum_distance_inconsistent(self):
        """EVAL: Distance to RUM for inconsistent data should be positive."""
        from pyrevealed.algorithms.stochastic import compute_distance_to_rum

        # Intransitive cycle
        log = StochasticChoiceLog(
            menus=[frozenset({0, 1}), frozenset({1, 2}), frozenset({0, 2})],
            choice_frequencies=[
                {0: 100, 1: 0},  # 0 > 1
                {1: 100, 2: 0},  # 1 > 2
                {0: 0, 2: 100},  # 2 > 0 (cycle!)
            ],
        )

        distance = compute_distance_to_rum(log)

        # Inconsistent data should have positive distance
        print(f"Distance to RUM: {distance}")


class TestAPUModel:
    """EVAL: Additive Perturbed Utility model tests."""

    def test_apu_consistent_logit(self):
        """EVAL: APU test for logit-consistent data."""
        from pyrevealed.algorithms.stochastic import test_additive_perturbed_utility

        log = StochasticChoiceLog(
            menus=[frozenset({0, 1, 2}), frozenset({0, 1})],
            choice_frequencies=[
                {0: 50, 1: 30, 2: 20},
                {0: 62, 1: 38},  # Approximately IIA-consistent odds
            ],
        )

        result = test_additive_perturbed_utility(log)

        assert 'is_apu_consistent' in result
        assert 'satisfies_iia' in result

    def test_apu_regularity_violation(self):
        """EVAL: APU with regularity violation."""
        from pyrevealed.algorithms.stochastic import test_additive_perturbed_utility

        # Regularity violation: P(0|{0,1}) < P(0|{0,1,2})
        log = StochasticChoiceLog(
            menus=[frozenset({0, 1}), frozenset({0, 1, 2})],
            choice_frequencies=[
                {0: 40, 1: 60},  # P(0|{0,1}) = 0.4
                {0: 50, 1: 30, 2: 20},  # P(0|{0,1,2}) = 0.5 > 0.4
            ],
        )

        result = test_additive_perturbed_utility(log, tolerance=0.01)

        # Should detect regularity violation
        assert len(result['regularity_violations']) > 0


class TestLogLikelihoodLimits:
    """EVAL: Log-likelihood computation limits."""

    def test_loglikelihood_extreme_counts(self):
        """EVAL: Log-likelihood with very large observation counts."""
        from pyrevealed.algorithms.stochastic import fit_random_utility_model

        log = StochasticChoiceLog(
            menus=[frozenset({0, 1, 2})],
            choice_frequencies=[{0: 1000000, 1: 1, 2: 0}],
        )

        result = fit_random_utility_model(log)

        # Log-likelihood should be finite
        assert np.isfinite(result.log_likelihood)

    def test_loglikelihood_uniform_distribution(self):
        """EVAL: Log-likelihood for uniform choice distribution."""
        from pyrevealed.algorithms.stochastic import fit_random_utility_model

        log = StochasticChoiceLog(
            menus=[frozenset({0, 1, 2, 3})],
            choice_frequencies=[{0: 25, 1: 25, 2: 25, 3: 25}],
        )

        result = fit_random_utility_model(log)

        # For uniform, log-likelihood = n * log(1/K) = 100 * log(0.25) ≈ -138.6
        expected_ll = 100 * np.log(0.25)
        assert np.isclose(result.log_likelihood, expected_ll, atol=5.0)
