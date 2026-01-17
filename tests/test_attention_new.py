"""Tests for WARP-LA and Random Attention Model (RAM) functions."""

import numpy as np
import pytest

from pyrevealed import MenuChoiceLog, StochasticChoiceLog
from pyrevealed import (
    # Result types
    WARPLAResult,
    RandomAttentionResult,
    # WARP-LA functions - import with alias to avoid pytest collection
    recover_preference_with_attention,
    validate_attention_filter_consistency,
    check_warp_la,
    # RAM functions
    fit_random_attention_model,
    estimate_attention_probabilities,
    compute_attention_bounds,
)
# Import test_* functions with aliases to avoid pytest collecting them as tests
from pyrevealed import test_warp_la as warp_la_test
from pyrevealed import test_ram_consistency as ram_consistency_test


class TestWARPLA:
    """Tests for WARP(LA) - Weak Axiom of Revealed Preference with Limited Attention."""

    @pytest.fixture
    def consistent_menu_log(self):
        """Create menu choice data consistent with WARP(LA)."""
        # Choices that can be explained by limited attention
        # Item 0 is always preferred when in attention set
        menus = [
            frozenset({0, 1, 2}),  # Chose 0
            frozenset({1, 2, 3}),  # Chose 1
            frozenset({0, 2, 3}),  # Chose 0
        ]
        choices = [0, 1, 0]
        return MenuChoiceLog(menus=menus, choices=choices)

    @pytest.fixture
    def warp_violation_log(self):
        """Create menu choice data with clear WARP violation (may or may not satisfy WARP-LA)."""
        # x chosen over y in one menu, y chosen over x in another
        menus = [
            frozenset({0, 1}),  # Chose 0
            frozenset({0, 1}),  # Chose 1 (same menu, different choice = violation)
        ]
        choices = [0, 1]
        return MenuChoiceLog(menus=menus, choices=choices)

    def warp_la_test_returns_correct_type(self, consistent_menu_log):
        """Test that warp_la_test returns WARPLAResult."""
        result = warp_la_test(consistent_menu_log)
        assert isinstance(result, WARPLAResult)

    def warp_la_test_consistent_data(self, consistent_menu_log):
        """Test WARP(LA) with consistent data."""
        result = warp_la_test(consistent_menu_log)
        # Should satisfy WARP(LA) - revealed preference is acyclic
        assert result.satisfies_warp_la is True
        assert len(result.violations) == 0
        assert result.num_observations == 3

    def warp_la_test_has_revealed_preference(self, consistent_menu_log):
        """Test that revealed preference is computed."""
        result = warp_la_test(consistent_menu_log)
        assert result.revealed_preference is not None
        assert isinstance(result.revealed_preference, list)

    def warp_la_test_computation_time(self, consistent_menu_log):
        """Test that computation time is recorded."""
        result = warp_la_test(consistent_menu_log)
        assert result.computation_time_ms >= 0

    def test_legacy_alias(self, consistent_menu_log):
        """Test that check_warp_la is alias for warp_la_test."""
        assert check_warp_la == warp_la_test
        result = check_warp_la(consistent_menu_log)
        assert isinstance(result, WARPLAResult)

    def warp_la_test_score(self, consistent_menu_log, warp_violation_log):
        """Test WARPLAResult score method."""
        result1 = warp_la_test(consistent_menu_log)
        result2 = warp_la_test(warp_violation_log)

        # Consistent should have high score
        assert 0.0 <= result1.score() <= 1.0
        assert 0.0 <= result2.score() <= 1.0

    def warp_la_test_summary(self, consistent_menu_log):
        """Test WARPLAResult summary method."""
        result = warp_la_test(consistent_menu_log)
        summary = result.summary()
        assert isinstance(summary, str)
        assert "WARP" in summary

    def warp_la_test_to_dict(self, consistent_menu_log):
        """Test WARPLAResult to_dict method."""
        result = warp_la_test(consistent_menu_log)
        d = result.to_dict()
        assert "satisfies_warp_la" in d
        assert "num_violations" in d
        assert "computation_time_ms" in d


class TestPreferenceRecovery:
    """Tests for preference recovery with attention filters."""

    @pytest.fixture
    def recoverable_log(self):
        """Create data where preference can be recovered."""
        menus = [
            frozenset({0, 1, 2}),
            frozenset({1, 2}),
            frozenset({0, 2}),
        ]
        choices = [0, 1, 0]
        return MenuChoiceLog(menus=menus, choices=choices)

    def test_recover_preference_returns_tuple(self, recoverable_log):
        """Test that recover_preference_with_attention returns proper tuple."""
        result = recover_preference_with_attention(recoverable_log)
        assert isinstance(result, tuple)
        assert len(result) == 2  # (preference, attention_filter)

    def test_recovered_preference_valid(self, recoverable_log):
        """Test that recovered preference is valid if successful."""
        preference, attention_filter = recover_preference_with_attention(recoverable_log)
        if preference is not None:
            assert isinstance(preference, tuple)
            # Preference should be a permutation of items
            items = set()
            for menu in recoverable_log.menus:
                items.update(menu)
            assert set(preference) == items


class TestAttentionFilterValidation:
    """Tests for attention filter validation."""

    @pytest.fixture
    def simple_log(self):
        """Create simple menu choice log."""
        menus = [frozenset({0, 1, 2}), frozenset({1, 2})]
        choices = [0, 1]
        return MenuChoiceLog(menus=menus, choices=choices)

    def test_validate_attention_filter_consistency(self, simple_log):
        """Test attention filter validation."""
        # Create a simple attention filter (consider all items)
        attention_filter = {
            frozenset({0, 1, 2}): {0, 1, 2},
            frozenset({1, 2}): {1, 2},
        }
        result = validate_attention_filter_consistency(simple_log, attention_filter)
        assert isinstance(result, dict)
        assert "is_valid" in result


class TestRandomAttentionModel:
    """Tests for Random Attention Model (RAM)."""

    @pytest.fixture
    def stochastic_log(self):
        """Create stochastic choice log for RAM testing."""
        menus = [
            frozenset({0, 1, 2}),
            frozenset({0, 1}),
            frozenset({1, 2}),
        ]
        # Choice frequencies: probability distribution over items for each menu
        choice_frequencies = [
            {0: 0.5, 1: 0.3, 2: 0.2},  # Menu {0,1,2}
            {0: 0.7, 1: 0.3},           # Menu {0,1}
            {1: 0.6, 2: 0.4},           # Menu {1,2}
        ]
        return StochasticChoiceLog(menus=menus, choice_frequencies=choice_frequencies)

    def test_fit_ram_returns_correct_type(self, stochastic_log):
        """Test that fit_random_attention_model returns RandomAttentionResult."""
        result = fit_random_attention_model(stochastic_log)
        assert isinstance(result, RandomAttentionResult)

    def test_ram_has_required_fields(self, stochastic_log):
        """Test RAM result has all required fields."""
        result = fit_random_attention_model(stochastic_log)
        assert hasattr(result, "is_ram_consistent")
        assert hasattr(result, "preference_ranking")
        assert hasattr(result, "attention_bounds")
        assert hasattr(result, "item_attention_scores")
        assert hasattr(result, "test_statistic")
        assert hasattr(result, "p_value")

    def test_ram_score_range(self, stochastic_log):
        """Test RAM score is in valid range."""
        result = fit_random_attention_model(stochastic_log)
        assert 0.0 <= result.score() <= 1.0

    def test_ram_summary(self, stochastic_log):
        """Test RAM summary method."""
        result = fit_random_attention_model(stochastic_log)
        summary = result.summary()
        assert isinstance(summary, str)
        assert "RAM" in summary or "Attention" in summary

    def test_ram_to_dict(self, stochastic_log):
        """Test RAM to_dict method."""
        result = fit_random_attention_model(stochastic_log)
        d = result.to_dict()
        assert "is_ram_consistent" in d
        assert "p_value" in d

    def test_ram_consistency_function(self, stochastic_log):
        """Test ram_consistency_test function."""
        result = ram_consistency_test(stochastic_log)
        assert isinstance(result, RandomAttentionResult)
        assert 0.0 <= result.p_value <= 1.0


class TestAttentionProbabilities:
    """Tests for attention probability estimation."""

    @pytest.fixture
    def stochastic_log(self):
        """Create stochastic choice log."""
        menus = [
            frozenset({0, 1}),
            frozenset({0, 1, 2}),
        ]
        choice_frequencies = [
            {0: 0.6, 1: 0.4},
            {0: 0.4, 1: 0.4, 2: 0.2},
        ]
        return StochasticChoiceLog(menus=menus, choice_frequencies=choice_frequencies)

    def test_estimate_attention_probabilities(self, stochastic_log):
        """Test attention probability estimation."""
        # Use preference ranking (0 > 1 > 2)
        preference = (0, 1, 2)
        probs = estimate_attention_probabilities(stochastic_log, preference)
        assert isinstance(probs, np.ndarray)
        # Probabilities should be non-negative
        assert np.all(probs >= 0)

    def test_compute_attention_bounds(self, stochastic_log):
        """Test attention bounds computation."""
        preference = (0, 1, 2)
        menu = frozenset({0, 1})
        item = 0
        bounds = compute_attention_bounds(stochastic_log, preference, item, menu)
        assert isinstance(bounds, tuple)
        assert len(bounds) == 2
        lower, upper = bounds
        assert lower <= upper
        assert 0.0 <= lower
        assert upper <= 1.0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_item_menu(self):
        """Test with single-item menus."""
        menus = [frozenset({0}), frozenset({1})]
        choices = [0, 1]
        log = MenuChoiceLog(menus=menus, choices=choices)
        result = warp_la_test(log)
        assert isinstance(result, WARPLAResult)

    def test_two_item_choice(self):
        """Test with two-item menus."""
        menus = [frozenset({0, 1}), frozenset({1, 2})]
        choices = [0, 1]
        log = MenuChoiceLog(menus=menus, choices=choices)
        result = warp_la_test(log)
        assert isinstance(result, WARPLAResult)

    def test_large_menu(self):
        """Test with larger menu."""
        menus = [frozenset(range(10))]
        choices = [5]
        log = MenuChoiceLog(menus=menus, choices=choices)
        result = warp_la_test(log)
        assert isinstance(result, WARPLAResult)
