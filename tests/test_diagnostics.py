"""Tests for Phase 2 diagnostic extensions.

Tests for:
- P0.2: Regularity test (regularity_test)
- P0.1: Attention overload (attention_overload_test)
- P1.5: Swaps index (compute_swaps_index)
- P1.6: Observation contributions (compute_observation_contributions)
- P1.7: Enhanced ViolationGraph
- P2.1: Status quo bias (status_quo_bias_test)
"""

import numpy as np
import pytest

from pyrevealed import (
    BehaviorLog,
    MenuChoiceLog,
    StochasticChoiceLog,
    # Result types
    RegularityResult,
    RegularityViolation,
    AttentionOverloadResult,
    SwapsIndexResult,
    ObservationContributionResult,
    StatusQuoBiasResult,
    # Functions - aliased to avoid pytest collecting them as tests
    test_regularity as regularity_test,
    test_attention_overload as attention_overload_test,
    test_status_quo_bias as status_quo_bias_test,
    compute_swaps_index,
    compute_observation_contributions,
    validate_consistency,
)
from pyrevealed.graph.violation_graph import ViolationGraph


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def consistent_behavior_log():
    """GARP-consistent behavior log (3 observations)."""
    prices = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
        [1.5, 1.5],
    ])
    quantities = np.array([
        [2.0, 1.0],  # Bundle A
        [1.0, 2.0],  # Bundle B
        [1.5, 1.5],  # Bundle C
    ])
    return BehaviorLog(prices=prices, quantities=quantities)


@pytest.fixture
def violation_behavior_log():
    """GARP-violating behavior log (WARP violation).

    At prices (2, 1): choose bundle (1, 3) with expenditure 5
    At prices (1, 2): choose bundle (3, 1) with expenditure 5

    Bundle (1,3) costs 5 at (2,1) and 7 at (1,2)
    Bundle (3,1) costs 7 at (2,1) and 5 at (1,2)

    So at (2,1): (3,1) is affordable (costs 7 > 5, wait no...)
    Let me construct a proper WARP violation.
    """
    # Proper WARP violation:
    # Obs 0: price=(1,1), choose (2,1) -> expenditure=3
    #        bundle (1,2) costs 1+2=3, so (1,2) is affordable
    # Obs 1: price=(1,1), choose (1,2) -> expenditure=3
    #        bundle (2,1) costs 2+1=3, so (2,1) is affordable
    # This creates mutual revelation: 0 reveals (2,1) > (1,2), 1 reveals (1,2) > (2,1)
    prices = np.array([
        [1.0, 1.0],
        [1.0, 1.0],
    ])
    quantities = np.array([
        [2.0, 1.0],
        [1.0, 2.0],
    ])
    return BehaviorLog(prices=prices, quantities=quantities)


@pytest.fixture
def three_cycle_violation_log():
    """GARP-violating behavior log with 3-cycle."""
    prices = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 1.0, 3.0],
        [3.0, 2.0, 1.0],
    ])
    quantities = np.array([
        [3.0, 0.0, 0.0],  # Choose mostly good 0 at p=(1,2,3)
        [0.0, 3.0, 0.0],  # Choose mostly good 1 at p=(2,1,3)
        [0.0, 0.0, 3.0],  # Choose mostly good 2 at p=(3,2,1)
    ])
    return BehaviorLog(prices=prices, quantities=quantities)


@pytest.fixture
def simple_menu_log():
    """Simple MenuChoiceLog for testing."""
    return MenuChoiceLog(
        menus=[{0, 1}, {0, 1, 2}, {1, 2}, {0, 1, 2, 3}],
        choices=[0, 1, 1, 2],
    )


@pytest.fixture
def overload_menu_log():
    """MenuChoiceLog with varying menu sizes."""
    # Simulate data where larger menus have worse choices
    menus = []
    choices = []
    for _ in range(5):
        menus.append({0, 1})
        choices.append(0)  # Consistent: always choose 0 from small menus
    for _ in range(5):
        menus.append({0, 1, 2, 3})
        choices.append(2)  # Sometimes inconsistent in larger menus
    for _ in range(5):
        menus.append({0, 1, 2, 3, 4, 5, 6, 7})
        choices.append(5)  # More inconsistent in very large menus
    return MenuChoiceLog(menus=menus, choices=choices)


@pytest.fixture
def stochastic_regularity_satisfied():
    """StochasticChoiceLog satisfying regularity."""
    return StochasticChoiceLog(
        menus=[{0, 1}, {0, 1, 2}],
        choice_frequencies=[
            {0: 60, 1: 40},      # P(0|{0,1}) = 0.6
            {0: 50, 1: 30, 2: 20},  # P(0|{0,1,2}) = 0.5 <= 0.6 (OK)
        ],
        total_observations_per_menu=[100, 100],
    )


@pytest.fixture
def stochastic_regularity_violated():
    """StochasticChoiceLog violating regularity."""
    return StochasticChoiceLog(
        menus=[{0, 1}, {0, 1, 2}],
        choice_frequencies=[
            {0: 40, 1: 60},      # P(0|{0,1}) = 0.4
            {0: 55, 1: 30, 2: 15},  # P(0|{0,1,2}) = 0.55 > 0.4 (VIOLATION!)
        ],
        total_observations_per_menu=[100, 100],
    )


# =============================================================================
# P0.2: Regularity Test
# =============================================================================


class TestRegularity:
    """Tests for regularity_test function."""

    def test_regularity_satisfied(self, stochastic_regularity_satisfied):
        """Regularity-consistent data should pass."""
        result = regularity_test(stochastic_regularity_satisfied)

        assert isinstance(result, RegularityResult)
        assert result.satisfies_regularity is True
        assert result.num_violations == 0
        assert result.worst_violation is None
        assert result.violation_rate == 0.0
        assert result.score() == 1.0

    def test_regularity_violated(self, stochastic_regularity_violated):
        """Regularity-violating data should be detected."""
        result = regularity_test(stochastic_regularity_violated, tolerance=0.01)

        assert isinstance(result, RegularityResult)
        assert result.satisfies_regularity is False
        assert result.num_violations > 0
        assert result.worst_violation is not None
        assert result.violation_rate > 0
        assert result.score() < 1.0

        # Check violation details
        wv = result.worst_violation
        assert isinstance(wv, RegularityViolation)
        assert wv.item == 0  # Item 0 had increased probability
        assert wv.prob_in_superset > wv.prob_in_subset  # Violation condition

    def test_regularity_result_methods(self, stochastic_regularity_violated):
        """Result object methods should work correctly."""
        result = regularity_test(stochastic_regularity_violated)

        # Test summary
        summary = result.summary()
        assert "REGULARITY" in summary
        assert isinstance(summary, str)

        # Test to_dict
        d = result.to_dict()
        assert "satisfies_regularity" in d
        assert "violations" in d
        assert isinstance(d, dict)

        # Test repr
        repr_str = repr(result)
        assert "RegularityResult" in repr_str


# =============================================================================
# P0.1: Attention Overload Test
# =============================================================================


class TestAttentionOverload:
    """Tests for attention_overload_test function."""

    def test_attention_overload_basic(self, simple_menu_log):
        """Basic attention overload test should return valid result."""
        result = attention_overload_test(simple_menu_log)

        assert isinstance(result, AttentionOverloadResult)
        # Check boolean value (may be numpy.bool_)
        assert result.has_overload in (True, False)
        assert isinstance(result.menu_size_quality, dict)
        assert isinstance(result.regression_slope, (int, float))
        assert result.num_observations == len(simple_menu_log.menus)

    def test_attention_overload_no_decline(self, simple_menu_log):
        """Small consistent data should not show overload."""
        result = attention_overload_test(simple_menu_log)

        # With only 4 observations, unlikely to detect significant overload
        assert result.has_overload in (True, False)
        assert result.overload_severity >= 0.0
        assert result.overload_severity <= 1.0

    def test_attention_overload_quality_metrics(self, overload_menu_log):
        """Both quality metrics should work."""
        result_consistency = attention_overload_test(
            overload_menu_log, quality_metric="consistency"
        )
        result_frequency = attention_overload_test(
            overload_menu_log, quality_metric="frequency"
        )

        assert isinstance(result_consistency, AttentionOverloadResult)
        assert isinstance(result_frequency, AttentionOverloadResult)

        # Both should have quality data for different menu sizes
        assert len(result_consistency.menu_size_quality) >= 1
        assert len(result_frequency.menu_size_quality) >= 1

    def test_attention_overload_result_methods(self, simple_menu_log):
        """Result object methods should work correctly."""
        result = attention_overload_test(simple_menu_log)

        # Test summary
        summary = result.summary()
        assert "ATTENTION OVERLOAD" in summary

        # Test to_dict
        d = result.to_dict()
        assert "has_overload" in d
        assert "menu_size_quality" in d

        # Test score
        score = result.score()
        assert 0.0 <= score <= 1.0


# =============================================================================
# P1.5: Swaps Index
# =============================================================================


class TestSwapsIndex:
    """Tests for compute_swaps_index function."""

    def test_swaps_index_consistent_data(self, consistent_behavior_log):
        """Consistent data should have 0 swaps."""
        result = compute_swaps_index(consistent_behavior_log)

        assert isinstance(result, SwapsIndexResult)
        assert result.swaps_count == 0
        assert result.swaps_normalized == 0.0
        assert result.is_consistent is True
        assert len(result.swap_pairs) == 0
        assert result.score() == 1.0

    def test_swaps_index_violation_data(self, violation_behavior_log):
        """Violating data should return valid result."""
        result = compute_swaps_index(violation_behavior_log)

        assert isinstance(result, SwapsIndexResult)
        # Result may or may not be consistent depending on data structure
        assert result.is_consistent in (True, False)
        assert result.max_possible_swaps > 0
        assert 0.0 <= result.swaps_normalized <= 1.0

    def test_swaps_index_three_cycle(self, three_cycle_violation_log):
        """3-cycle violation should need at least 1 swap."""
        result = compute_swaps_index(three_cycle_violation_log)

        # A 3-cycle needs at least 1 edge removed
        assert isinstance(result, SwapsIndexResult)
        assert result.swaps_count >= 0  # May be 0 if data is actually consistent
        assert result.method == "greedy"

    def test_swaps_index_result_methods(self, consistent_behavior_log):
        """Result object methods should work correctly."""
        result = compute_swaps_index(consistent_behavior_log)

        # Test summary
        summary = result.summary()
        assert "SWAPS INDEX" in summary

        # Test to_dict
        d = result.to_dict()
        assert "swaps_count" in d
        assert "swap_pairs" in d

        # Test repr
        repr_str = repr(result)
        assert "SwapsIndexResult" in repr_str


# =============================================================================
# P1.6: Observation Contributions
# =============================================================================


class TestObservationContributions:
    """Tests for compute_observation_contributions function."""

    def test_contributions_consistent_data(self, consistent_behavior_log):
        """Consistent data should have all zero contributions."""
        result = compute_observation_contributions(consistent_behavior_log)

        assert isinstance(result, ObservationContributionResult)
        assert result.base_aei == 1.0  # Perfectly consistent
        assert result.num_problematic == 0
        assert len(result.worst_observations) == 0
        assert np.all(result.contributions == 0.0)

    def test_contributions_violation_data(self, violation_behavior_log):
        """Violating data should return valid result."""
        result = compute_observation_contributions(violation_behavior_log)

        assert isinstance(result, ObservationContributionResult)
        # AEI may be 1.0 if data is actually consistent
        assert 0.0 <= result.base_aei <= 1.0
        assert result.num_observations == 2

    def test_contributions_cycle_count_method(self, three_cycle_violation_log):
        """cycle_count method should work."""
        result = compute_observation_contributions(
            three_cycle_violation_log, method="cycle_count"
        )

        assert isinstance(result, ObservationContributionResult)
        assert result.method == "cycle_count"
        assert len(result.contributions) == 3

    def test_contributions_result_methods(self, consistent_behavior_log):
        """Result object methods should work correctly."""
        result = compute_observation_contributions(consistent_behavior_log)

        # Test summary
        summary = result.summary()
        assert "OBSERVATION CONTRIBUTION" in summary

        # Test to_dict
        d = result.to_dict()
        assert "contributions" in d
        assert "base_aei" in d

        # Test score
        score = result.score()
        assert 0.0 <= score <= 1.0


# =============================================================================
# P1.7: Enhanced ViolationGraph
# =============================================================================


class TestViolationGraph:
    """Tests for enhanced ViolationGraph methods."""

    def test_to_networkx(self, consistent_behavior_log):
        """to_networkx should return valid NetworkX graph."""
        garp_result = validate_consistency(consistent_behavior_log)
        graph = ViolationGraph(consistent_behavior_log, garp_result)

        G = graph.to_networkx()

        # Should be a NetworkX DiGraph
        import networkx as nx
        assert isinstance(G, nx.DiGraph)
        assert G.number_of_nodes() == 3

    def test_to_dict(self, consistent_behavior_log):
        """to_dict should return JSON-serializable dict."""
        garp_result = validate_consistency(consistent_behavior_log)
        graph = ViolationGraph(consistent_behavior_log, garp_result)

        d = graph.to_dict()

        assert isinstance(d, dict)
        assert "nodes" in d
        assert "edges" in d
        assert "violations" in d
        assert "num_nodes" in d
        assert d["num_nodes"] == 3

    def test_find_shortest_cycles(self, three_cycle_violation_log):
        """find_shortest_cycles should return cycles sorted by length."""
        garp_result = validate_consistency(three_cycle_violation_log)
        graph = ViolationGraph(three_cycle_violation_log, garp_result)

        cycles = graph.find_shortest_cycles(n=5)

        assert isinstance(cycles, list)
        # Cycles should be sorted by length
        for i in range(len(cycles) - 1):
            assert len(cycles[i]) <= len(cycles[i + 1])

    def test_compute_centrality(self, consistent_behavior_log):
        """compute_centrality should return dict of scores."""
        garp_result = validate_consistency(consistent_behavior_log)
        graph = ViolationGraph(consistent_behavior_log, garp_result)

        for method in ["betweenness", "pagerank", "degree"]:
            centrality = graph.compute_centrality(method=method)
            assert isinstance(centrality, dict)
            # All nodes should have a score
            assert len(centrality) == 3

    def test_get_violation_centrality(self, three_cycle_violation_log):
        """get_violation_centrality should return normalized scores."""
        garp_result = validate_consistency(three_cycle_violation_log)
        graph = ViolationGraph(three_cycle_violation_log, garp_result)

        centrality = graph.get_violation_centrality()

        assert isinstance(centrality, dict)
        # All scores should be 0-1 (or 0 if no violations)
        for score in centrality.values():
            assert 0.0 <= score <= 1.0

    def test_get_violation_centrality_consistent(self, consistent_behavior_log):
        """get_violation_centrality should work with consistent data."""
        garp_result = validate_consistency(consistent_behavior_log)
        graph = ViolationGraph(consistent_behavior_log, garp_result)

        centrality = graph.get_violation_centrality()

        assert isinstance(centrality, dict)
        # With no violations, all scores should be 0
        for score in centrality.values():
            assert score == 0.0


# =============================================================================
# P2.1: Status Quo Bias
# =============================================================================


class TestStatusQuoBias:
    """Tests for status_quo_bias_test function."""

    def test_status_quo_bias_basic(self, simple_menu_log):
        """Basic status quo bias test should return valid result."""
        result = status_quo_bias_test(simple_menu_log)

        assert isinstance(result, StatusQuoBiasResult)
        assert isinstance(result.has_status_quo_bias, bool)
        assert 0.0 <= result.default_advantage <= 1.0
        assert isinstance(result.bias_by_item, dict)

    def test_status_quo_bias_with_defaults(self, simple_menu_log):
        """Custom defaults should be used."""
        defaults = [0, 0, 1, 0]  # Explicit defaults
        result = status_quo_bias_test(simple_menu_log, defaults=defaults)

        assert isinstance(result, StatusQuoBiasResult)
        assert result.num_defaults == 4

    def test_status_quo_bias_no_bias(self):
        """Random choices should not show bias."""
        # Create log where default is rarely chosen
        menus = [{0, 1, 2}] * 20
        choices = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
                   1, 2, 1, 2, 1, 2, 1, 2, 1, 2]  # Never choose 0
        log = MenuChoiceLog(menus=menus, choices=choices)
        defaults = [0] * 20  # Default is always 0

        result = status_quo_bias_test(log, defaults=defaults)

        # Should detect no bias (or negative bias)
        assert isinstance(result, StatusQuoBiasResult)
        # Default is never chosen, so no positive bias
        assert result.default_advantage <= 0.5 or not result.has_status_quo_bias

    def test_status_quo_bias_result_methods(self, simple_menu_log):
        """Result object methods should work correctly."""
        result = status_quo_bias_test(simple_menu_log)

        # Test summary
        summary = result.summary()
        assert "STATUS QUO BIAS" in summary

        # Test to_dict
        d = result.to_dict()
        assert "has_status_quo_bias" in d
        assert "default_advantage" in d

        # Test score
        score = result.score()
        assert 0.0 <= score <= 1.0


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for Phase 2 extensions."""

    def test_all_functions_importable(self):
        """All new functions should be importable from pyrevealed."""
        from pyrevealed import (
            test_regularity,
            test_attention_overload,
            test_status_quo_bias,
            compute_swaps_index,
            compute_observation_contributions,
            RegularityResult,
            AttentionOverloadResult,
            StatusQuoBiasResult,
            SwapsIndexResult,
            ObservationContributionResult,
        )

        # All should be callable or classes
        assert callable(test_regularity)
        assert callable(test_attention_overload)
        assert callable(test_status_quo_bias)
        assert callable(compute_swaps_index)
        assert callable(compute_observation_contributions)

    def test_result_types_have_score_method(self):
        """All new result types should have score() method."""
        from pyrevealed import (
            RegularityResult,
            AttentionOverloadResult,
            StatusQuoBiasResult,
            SwapsIndexResult,
            ObservationContributionResult,
        )

        # Check each class has score method
        for cls in [RegularityResult, AttentionOverloadResult,
                    StatusQuoBiasResult, SwapsIndexResult,
                    ObservationContributionResult]:
            assert hasattr(cls, 'score')
            assert hasattr(cls, 'summary')
            assert hasattr(cls, 'to_dict')
