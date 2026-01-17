"""Tests for RUM consistency testing functions."""

import numpy as np
import pytest

from pyrevealed import StochasticChoiceLog
from pyrevealed import (
    # Result types
    RUMConsistencyResult,
    # RUM functions - import with alias to avoid pytest collection
    compute_distance_to_rum,
    fit_rum_distribution,
    check_rum_consistency,
)
# Import test_rum_consistency with alias to avoid pytest collecting it as a test
from pyrevealed import test_rum_consistency as rum_consistency_test


class TestRUMConsistency:
    """Tests for Random Utility Model (RUM) consistency."""

    @pytest.fixture
    def rum_consistent_log(self):
        """Create stochastic choice data consistent with RUM.

        RUM consistency: choice probabilities can be rationalized by
        a probability distribution over preference orderings.
        """
        # Simple case: choices consistent with a mixture of orderings
        menus = [
            frozenset({0, 1}),
            frozenset({0, 1, 2}),
        ]
        # Choice frequencies that can be explained by RUM
        choice_frequencies = [
            {0: 0.7, 1: 0.3},           # Binary choice
            {0: 0.5, 1: 0.3, 2: 0.2},   # Ternary choice
        ]
        return StochasticChoiceLog(menus=menus, choice_frequencies=choice_frequencies)

    @pytest.fixture
    def rum_inconsistent_log(self):
        """Create stochastic choice data violating RUM (regularity violation).

        Regularity: adding more options should not increase choice probability.
        p(x|S) >= p(x|T) when S is subset of T
        """
        menus = [
            frozenset({0, 1}),
            frozenset({0, 1, 2}),
        ]
        # Violates regularity: item 0 is chosen MORE when menu is larger
        choice_frequencies = [
            {0: 0.3, 1: 0.7},           # 30% choose 0 from {0,1}
            {0: 0.6, 1: 0.3, 2: 0.1},   # 60% choose 0 from {0,1,2} - violation!
        ]
        return StochasticChoiceLog(menus=menus, choice_frequencies=choice_frequencies)

    def test_rum_returns_correct_type(self, rum_consistent_log):
        """Test that rum_consistency_test returns RUMConsistencyResult."""
        result = rum_consistency_test(rum_consistent_log)
        assert isinstance(result, RUMConsistencyResult)

    def test_rum_consistent_data(self, rum_consistent_log):
        """Test RUM consistency with consistent data."""
        result = rum_consistency_test(rum_consistent_log)
        assert hasattr(result, "is_rum_consistent")
        assert hasattr(result, "distance_to_rum")
        assert hasattr(result, "regularity_satisfied")

    def test_rum_has_distance(self, rum_consistent_log):
        """Test that distance to RUM is computed."""
        result = rum_consistency_test(rum_consistent_log)
        assert result.distance_to_rum >= 0
        if result.is_rum_consistent:
            assert result.distance_to_rum < 1e-4  # Should be nearly 0

    def test_rum_computation_time(self, rum_consistent_log):
        """Test that computation time is recorded."""
        result = rum_consistency_test(rum_consistent_log)
        assert result.computation_time_ms >= 0

    def test_legacy_alias(self, rum_consistent_log):
        """Test that check_rum_consistency is alias for rum_consistency_test."""
        assert check_rum_consistency == rum_consistency_test
        result = check_rum_consistency(rum_consistent_log)
        assert isinstance(result, RUMConsistencyResult)

    def test_rum_score(self, rum_consistent_log, rum_inconsistent_log):
        """Test RUMConsistencyResult score method."""
        result1 = rum_consistency_test(rum_consistent_log)
        result2 = rum_consistency_test(rum_inconsistent_log)

        # Scores should be in [0, 1]
        assert 0.0 <= result1.score() <= 1.0
        assert 0.0 <= result2.score() <= 1.0

    def test_rum_summary(self, rum_consistent_log):
        """Test RUMConsistencyResult summary method."""
        result = rum_consistency_test(rum_consistent_log)
        summary = result.summary()
        assert isinstance(summary, str)
        assert "RUM" in summary

    def test_rum_to_dict(self, rum_consistent_log):
        """Test RUMConsistencyResult to_dict method."""
        result = rum_consistency_test(rum_consistent_log)
        d = result.to_dict()
        assert "is_rum_consistent" in d
        assert "distance_to_rum" in d
        assert "regularity_satisfied" in d
        assert "num_orderings_used" in d


class TestDistanceToRUM:
    """Tests for compute_distance_to_rum function."""

    @pytest.fixture
    def simple_log(self):
        """Create simple stochastic choice log."""
        menus = [frozenset({0, 1})]
        choice_frequencies = [{0: 0.6, 1: 0.4}]
        return StochasticChoiceLog(menus=menus, choice_frequencies=choice_frequencies)

    @pytest.fixture
    def complex_log(self):
        """Create more complex stochastic choice log."""
        menus = [
            frozenset({0, 1}),
            frozenset({1, 2}),
            frozenset({0, 2}),
            frozenset({0, 1, 2}),
        ]
        choice_frequencies = [
            {0: 0.6, 1: 0.4},
            {1: 0.5, 2: 0.5},
            {0: 0.7, 2: 0.3},
            {0: 0.4, 1: 0.35, 2: 0.25},
        ]
        return StochasticChoiceLog(menus=menus, choice_frequencies=choice_frequencies)

    def test_distance_returns_float(self, simple_log):
        """Test that compute_distance_to_rum returns a float."""
        distance = compute_distance_to_rum(simple_log)
        assert isinstance(distance, float)

    def test_distance_non_negative(self, simple_log, complex_log):
        """Test that distance is non-negative."""
        d1 = compute_distance_to_rum(simple_log)
        d2 = compute_distance_to_rum(complex_log)
        assert d1 >= 0
        assert d2 >= 0

    def test_distance_l1_norm(self, simple_log):
        """Test distance with L1 norm."""
        distance = compute_distance_to_rum(simple_log, norm="l1")
        assert distance >= 0


class TestFitRUMDistribution:
    """Tests for fit_rum_distribution function."""

    @pytest.fixture
    def small_log(self):
        """Create log with small number of items (exact enumeration)."""
        menus = [
            frozenset({0, 1}),
            frozenset({0, 1, 2}),
        ]
        choice_frequencies = [
            {0: 0.7, 1: 0.3},
            {0: 0.5, 1: 0.35, 2: 0.15},
        ]
        return StochasticChoiceLog(menus=menus, choice_frequencies=choice_frequencies)

    def test_fit_returns_dict(self, small_log):
        """Test that fit_rum_distribution returns a dictionary."""
        distribution = fit_rum_distribution(small_log)
        assert isinstance(distribution, dict)

    def test_distribution_probabilities_valid(self, small_log):
        """Test that distribution probabilities are valid."""
        distribution = fit_rum_distribution(small_log)

        # All values should be probabilities (non-negative)
        for ordering, prob in distribution.items():
            assert prob >= 0
            assert isinstance(ordering, tuple)

    def test_distribution_sums_to_one(self, small_log):
        """Test that probabilities sum to approximately 1."""
        distribution = fit_rum_distribution(small_log)
        if distribution:  # If non-empty
            total = sum(distribution.values())
            assert abs(total - 1.0) < 0.01  # Allow some numerical tolerance

    def test_distribution_orderings_valid(self, small_log):
        """Test that orderings are valid permutations."""
        distribution = fit_rum_distribution(small_log)

        # Get all items
        items = set()
        for menu in small_log.menus:
            items.update(menu)

        for ordering, prob in distribution.items():
            if prob > 0:
                # Each ordering should be a permutation of items
                assert set(ordering) == items


class TestRUMRegularity:
    """Tests for regularity condition checking."""

    @pytest.fixture
    def irregular_log(self):
        """Create log clearly violating regularity."""
        menus = [
            frozenset({0, 1}),
            frozenset({0, 1, 2}),
        ]
        # Violates regularity: p(0|{0,1}) < p(0|{0,1,2})
        choice_frequencies = [
            {0: 0.3, 1: 0.7},           # p(0) = 0.3
            {0: 0.5, 1: 0.35, 2: 0.15}, # p(0) = 0.5 > 0.3 VIOLATION
        ]
        return StochasticChoiceLog(menus=menus, choice_frequencies=choice_frequencies)

    def test_irregular_data_violates_regularity(self, irregular_log):
        """Test that irregular data is detected as violating regularity."""
        result = rum_consistency_test(irregular_log)
        assert result.regularity_satisfied is False

    def test_regularity_field_exists(self, irregular_log):
        """Test that result has regularity_satisfied field."""
        result = rum_consistency_test(irregular_log)
        assert hasattr(result, "regularity_satisfied")
        assert isinstance(result.regularity_satisfied, bool)


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_single_menu(self):
        """Test with single menu."""
        menus = [frozenset({0, 1, 2})]
        choice_frequencies = [{0: 0.4, 1: 0.35, 2: 0.25}]
        log = StochasticChoiceLog(menus=menus, choice_frequencies=choice_frequencies)

        result = rum_consistency_test(log)
        assert isinstance(result, RUMConsistencyResult)
        # Single menu is always RUM consistent
        assert result.is_rum_consistent is True

    def test_binary_choices_only(self):
        """Test with only binary choices."""
        menus = [
            frozenset({0, 1}),
            frozenset({1, 2}),
            frozenset({0, 2}),
        ]
        choice_frequencies = [
            {0: 0.6, 1: 0.4},
            {1: 0.5, 2: 0.5},
            {0: 0.7, 2: 0.3},
        ]
        log = StochasticChoiceLog(menus=menus, choice_frequencies=choice_frequencies)

        result = rum_consistency_test(log)
        assert isinstance(result, RUMConsistencyResult)

    def test_deterministic_choices(self):
        """Test with deterministic choices (probability 1 for one item)."""
        menus = [
            frozenset({0, 1}),
            frozenset({0, 1, 2}),
        ]
        choice_frequencies = [
            {0: 1.0, 1: 0.0},        # Always choose 0
            {0: 1.0, 1: 0.0, 2: 0.0}, # Always choose 0
        ]
        log = StochasticChoiceLog(menus=menus, choice_frequencies=choice_frequencies)

        result = rum_consistency_test(log)
        assert isinstance(result, RUMConsistencyResult)
        # Deterministic rational choice is RUM consistent
        assert result.is_rum_consistent is True


class TestResultMethods:
    """Tests for RUMConsistencyResult methods."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample RUMConsistencyResult."""
        return RUMConsistencyResult(
            is_rum_consistent=True,
            distance_to_rum=0.0,
            regularity_satisfied=True,
            num_orderings_used=3,
            rationalizing_distribution={(0, 1, 2): 0.5, (1, 0, 2): 0.3, (0, 2, 1): 0.2},
            num_iterations=5,
            constraint_violations=[],
            computation_time_ms=10.5,
        )

    def test_score_consistent(self, sample_result):
        """Test score for consistent result."""
        score = sample_result.score()
        assert 0.0 <= score <= 1.0
        assert score == 1.0  # Consistent = perfect score

    def test_score_inconsistent(self):
        """Test score for inconsistent result."""
        result = RUMConsistencyResult(
            is_rum_consistent=False,
            distance_to_rum=0.15,
            regularity_satisfied=False,
            num_orderings_used=0,
            rationalizing_distribution=None,
            num_iterations=10,
            constraint_violations=["Regularity violation at menu 0"],
            computation_time_ms=15.0,
        )
        score = result.score()
        assert 0.0 <= score <= 1.0
        assert score < 1.0

    def test_summary_format(self, sample_result):
        """Test summary output format."""
        summary = sample_result.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_to_dict_completeness(self, sample_result):
        """Test to_dict includes all important fields."""
        d = sample_result.to_dict()

        required_keys = [
            "is_rum_consistent",
            "distance_to_rum",
            "regularity_satisfied",
            "num_orderings_used",
        ]
        for key in required_keys:
            assert key in d

    def test_repr(self, sample_result):
        """Test string representation."""
        r = repr(sample_result)
        assert "RUMConsistencyResult" in r


class TestColumnGeneration:
    """Tests for column generation algorithm behavior."""

    @pytest.fixture
    def six_item_log(self):
        """Create log with 6 items (boundary for exact enumeration)."""
        items = list(range(6))
        menus = [
            frozenset({0, 1, 2}),
            frozenset({3, 4, 5}),
            frozenset({0, 2, 4}),
            frozenset({1, 3, 5}),
            frozenset(items),  # Full menu
        ]
        choice_frequencies = [
            {0: 0.5, 1: 0.3, 2: 0.2},
            {3: 0.4, 4: 0.35, 5: 0.25},
            {0: 0.45, 2: 0.35, 4: 0.2},
            {1: 0.4, 3: 0.35, 5: 0.25},
            {i: 1/6 for i in items},
        ]
        return StochasticChoiceLog(menus=menus, choice_frequencies=choice_frequencies)

    def test_six_items_completes(self, six_item_log):
        """Test that algorithm completes for 6 items."""
        result = rum_consistency_test(six_item_log)
        assert isinstance(result, RUMConsistencyResult)

    def test_num_iterations_recorded(self, six_item_log):
        """Test that number of iterations is recorded."""
        result = rum_consistency_test(six_item_log)
        assert result.num_iterations >= 0
