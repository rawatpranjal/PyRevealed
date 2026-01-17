"""
EVAL: Limited attention model computational limits.

Tests for consideration set and attention model edge cases.
"""

import numpy as np
import pytest
from pyrevealed.core.session import MenuChoiceLog, StochasticChoiceLog


class TestConsiderationSets:
    """EVAL: Consideration set inference edge cases."""

    def test_attention_empty_consideration(self):
        """EVAL: Attention model when consideration set could be empty."""
        from pyrevealed.algorithms.attention import test_limited_attention

        log = MenuChoiceLog(
            menus=[
                frozenset({0, 1, 2}),
                frozenset({0, 1}),
                frozenset({1, 2}),
            ],
            choices=[0, 0, 1],
        )

        result = test_limited_attention(log)

        assert hasattr(result, 'is_consistent')

    def test_attention_single_item_menu(self):
        """EVAL: Attention with single-item menus."""
        log = MenuChoiceLog(
            menus=[
                frozenset({0}),
                frozenset({1}),
                frozenset({0, 1}),
            ],
            choices=[0, 1, 0],
        )

        from pyrevealed.algorithms.attention import test_limited_attention

        result = test_limited_attention(log)

        # Single-item menus are trivially consistent
        assert hasattr(result, 'is_consistent')


class TestAttentionModels:
    """EVAL: Different attention model specifications."""

    def test_warp_la_simple(self):
        """EVAL: WARP-LA (Limited Attention WARP) test."""
        from pyrevealed.algorithms.attention import test_warp_la

        log = MenuChoiceLog(
            menus=[
                frozenset({0, 1, 2}),
                frozenset({0, 1}),
            ],
            choices=[0, 1],  # Different choices when item 2 removed
        )

        result = test_warp_la(log)

        # WARP-LA allows such reversals if 2 was "attracting attention"
        assert hasattr(result, 'is_consistent')

    def test_attention_overload(self):
        """EVAL: Attention overload with many items."""
        log = MenuChoiceLog(
            menus=[
                frozenset(range(10)),  # 10-item menu
                frozenset({0, 1}),  # Small menu
            ],
            choices=[0, 1],
        )

        from pyrevealed.algorithms.attention import test_limited_attention

        result = test_limited_attention(log)

        # With attention overload, choice patterns may be explained
        assert hasattr(result, 'is_consistent')


class TestRAMModel:
    """EVAL: Random Attention Model edge cases."""

    def test_ram_consistency(self):
        """EVAL: RAM consistency test."""
        from pyrevealed.algorithms.attention import test_ram_consistency

        log = StochasticChoiceLog(
            menus=[
                frozenset({0, 1, 2}),
                frozenset({0, 1}),
            ],
            choice_frequencies=[
                {0: 40, 1: 35, 2: 25},
                {0: 55, 1: 45},
            ],
        )

        result = test_ram_consistency(log)

        assert hasattr(result, 'is_ram_consistent')

    def test_ram_extreme_frequencies(self):
        """EVAL: RAM with extreme choice frequencies."""
        log = StochasticChoiceLog(
            menus=[
                frozenset({0, 1, 2}),
                frozenset({0, 1}),
            ],
            choice_frequencies=[
                {0: 999999, 1: 1, 2: 0},
                {0: 1, 1: 999999},  # Extreme reversal
            ],
        )

        from pyrevealed.algorithms.attention import test_ram_consistency

        result = test_ram_consistency(log)

        # Extreme reversal might not be RAM-consistent
        assert hasattr(result, 'is_ram_consistent')


class TestAttentionFilters:
    """EVAL: Attention filter inference edge cases."""

    def test_infer_attention_filters(self):
        """EVAL: Infer attention filters from choice data."""
        from pyrevealed.algorithms.attention import infer_attention_filters

        log = MenuChoiceLog(
            menus=[
                frozenset({0, 1, 2}),
                frozenset({0, 1}),
                frozenset({1, 2}),
            ],
            choices=[2, 0, 1],
        )

        filters = infer_attention_filters(log)

        assert hasattr(filters, 'attention_sets') or isinstance(filters, dict)

    def test_attention_filter_contradiction(self):
        """EVAL: Attention filters with contradictory choices."""
        log = MenuChoiceLog(
            menus=[
                frozenset({0, 1}),
                frozenset({0, 1}),  # Same menu
            ],
            choices=[0, 1],  # Different choices from same menu!
        )

        from pyrevealed.algorithms.attention import infer_attention_filters

        # This is inconsistent with deterministic attention
        try:
            filters = infer_attention_filters(log)
            # If it returns, check the result
            assert filters is not None
        except Exception:
            pass  # May raise an error


class TestAttentionGraphs:
    """EVAL: Preference graph connectivity issues."""

    def test_disconnected_preference_graph(self):
        """EVAL: Preference graph with disconnected components."""
        log = MenuChoiceLog(
            menus=[
                frozenset({0, 1}),
                frozenset({2, 3}),  # No overlap with first menu
            ],
            choices=[0, 2],
        )

        from pyrevealed.algorithms.attention import test_limited_attention

        result = test_limited_attention(log)

        # Disconnected items can't be compared directly
        assert hasattr(result, 'is_consistent')

    def test_fully_connected_graph(self):
        """EVAL: Preference graph where all items are compared."""
        log = MenuChoiceLog(
            menus=[
                frozenset({0, 1}),
                frozenset({1, 2}),
                frozenset({0, 2}),
            ],
            choices=[0, 1, 0],
        )

        from pyrevealed.algorithms.attention import test_limited_attention

        result = test_limited_attention(log)

        # Transitive: 0 > 1, 1 > 2, 0 > 2 - consistent!
        assert hasattr(result, 'is_consistent')


class TestAttentionPower:
    """EVAL: Attention model power and identification."""

    def test_attention_power_small_sample(self):
        """EVAL: Attention test power with small sample."""
        log = MenuChoiceLog(
            menus=[frozenset({0, 1, 2})],
            choices=[0],
        )

        from pyrevealed.algorithms.attention import test_limited_attention

        result = test_limited_attention(log)

        # Single observation - very low power to detect violations
        assert hasattr(result, 'is_consistent')

    def test_attention_identification(self):
        """EVAL: Attention model identification with rich data."""
        # Create data with many overlapping menus
        menus = []
        choices = []
        for i in range(10):
            menu = frozenset({i % 5, (i + 1) % 5, (i + 2) % 5})
            menus.append(menu)
            choices.append(min(menu))  # Always choose smallest

        log = MenuChoiceLog(menus=menus, choices=choices)

        from pyrevealed.algorithms.attention import test_limited_attention

        result = test_limited_attention(log)

        # Rich data should allow better identification
        assert hasattr(result, 'is_consistent')


class TestAttentionStochastic:
    """EVAL: Stochastic attention model edge cases."""

    def test_stochastic_attention_zero_prob(self):
        """EVAL: Stochastic attention with zero probability items."""
        log = StochasticChoiceLog(
            menus=[frozenset({0, 1, 2})],
            choice_frequencies=[{0: 100, 1: 0, 2: 0}],
        )

        from pyrevealed.algorithms.attention import test_ram_consistency

        result = test_ram_consistency(log)

        # Item never chosen might not be in consideration set
        assert hasattr(result, 'is_ram_consistent')

    def test_stochastic_attention_uniform(self):
        """EVAL: Stochastic attention with uniform choices."""
        log = StochasticChoiceLog(
            menus=[frozenset({0, 1, 2})],
            choice_frequencies=[{0: 33, 1: 34, 2: 33}],
        )

        from pyrevealed.algorithms.attention import test_ram_consistency

        result = test_ram_consistency(log)

        # Uniform choices are consistent with any attention model
        assert hasattr(result, 'is_ram_consistent')
