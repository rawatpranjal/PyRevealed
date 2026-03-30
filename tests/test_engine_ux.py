"""Tests for Engine UX improvements: validation, repr, to_dict, load_demo."""

import numpy as np
import pytest

from prefgraph.engine import Engine, EngineResult, MenuResult, results_to_dataframe
from prefgraph.core.exceptions import (
    DataValidationError,
    DimensionError,
    NaNInfError,
    ValueRangeError,
)


# =============================================================================
# Engine input validation (budget)
# =============================================================================

class TestBudgetValidation:
    def test_invalid_metric_name(self):
        with pytest.raises(ValueError, match="Unknown metrics"):
            Engine(metrics=["bogus"])

    def test_multiple_invalid_metrics(self):
        with pytest.raises(ValueError, match="Unknown metrics"):
            Engine(metrics=["garp", "bogus", "fake"])

    def test_valid_metrics_accepted(self):
        e = Engine(metrics=["garp", "ccei", "mpi", "harp", "hm", "utility", "vei"])
        assert len(e.metrics) == 7

    def test_non_list_input(self):
        e = Engine()
        with pytest.raises(TypeError, match="list of .* tuples"):
            e.analyze_arrays(np.ones((3, 2)))

    def test_empty_users_list(self):
        e = Engine()
        with pytest.raises(DataValidationError, match="empty"):
            e.analyze_arrays([])

    def test_invalid_tuple_length(self):
        e = Engine()
        with pytest.raises(DataValidationError, match="length 2"):
            e.analyze_arrays([(np.ones((3, 2)),)])

    def test_three_element_tuple(self):
        e = Engine()
        with pytest.raises(DataValidationError, match="length 3"):
            e.analyze_arrays([(np.ones((3, 2)), np.ones((3, 2)), np.ones((3, 2)))])

    def test_non_array_in_tuple(self):
        e = Engine()
        with pytest.raises(TypeError, match="numpy arrays"):
            e.analyze_arrays([([1, 2, 3], [4, 5, 6])])

    def test_1d_prices(self):
        e = Engine()
        with pytest.raises(DimensionError, match="2D"):
            e.analyze_arrays([(np.ones(6), np.ones((3, 2)))])

    def test_1d_quantities(self):
        e = Engine()
        with pytest.raises(DimensionError, match="2D"):
            e.analyze_arrays([(np.ones((3, 2)), np.ones(6))])

    def test_shape_mismatch(self):
        e = Engine()
        with pytest.raises(DimensionError, match="!="):
            e.analyze_arrays([(np.ones((3, 2)), np.ones((3, 5)))])

    def test_nan_prices_rejected(self):
        e = Engine()
        p = np.array([[np.nan, 1.0], [2.0, 1.0]])
        q = np.array([[1.0, 2.0], [2.0, 1.0]])
        with pytest.raises(NaNInfError, match="prices contain NaN"):
            e.analyze_arrays([(p, q)])

    def test_inf_quantities_rejected(self):
        e = Engine()
        p = np.array([[1.0, 2.0], [2.0, 1.0]])
        q = np.array([[1.0, np.inf], [2.0, 1.0]])
        with pytest.raises(NaNInfError, match="quantities contain NaN"):
            e.analyze_arrays([(p, q)])

    def test_negative_prices_rejected(self):
        e = Engine()
        p = np.array([[-1.0, 2.0], [2.0, 1.0]])
        q = np.array([[1.0, 2.0], [2.0, 1.0]])
        with pytest.raises(ValueRangeError, match="strictly positive"):
            e.analyze_arrays([(p, q)])

    def test_zero_prices_rejected(self):
        e = Engine()
        p = np.array([[0.0, 2.0], [2.0, 1.0]])
        q = np.array([[1.0, 2.0], [2.0, 1.0]])
        with pytest.raises(ValueRangeError, match="strictly positive"):
            e.analyze_arrays([(p, q)])

    def test_negative_quantities_rejected(self):
        e = Engine()
        p = np.array([[1.0, 2.0], [2.0, 1.0]])
        q = np.array([[-1.0, 2.0], [2.0, 1.0]])
        with pytest.raises(ValueRangeError, match="non-negative"):
            e.analyze_arrays([(p, q)])

    def test_zero_quantities_accepted(self):
        e = Engine(metrics=["garp"])
        p = np.array([[1.0, 2.0], [2.0, 1.0]])
        q = np.array([[0.0, 2.0], [2.0, 0.0]])
        results = e.analyze_arrays([(p, q)])
        assert len(results) == 1

    def test_valid_input_passes(self):
        e = Engine(metrics=["garp"])
        p = np.array([[1.0, 2.0], [3.0, 4.0]])
        q = np.array([[5.0, 3.0], [2.0, 6.0]])
        results = e.analyze_arrays([(p, q)])
        assert len(results) == 1
        assert isinstance(results[0], EngineResult)


# =============================================================================
# Engine input validation (menus)
# =============================================================================

class TestMenuValidation:
    def test_non_list_input(self):
        e = Engine()
        with pytest.raises(TypeError, match="list of .* tuples"):
            e.analyze_menus("bad input")

    def test_empty_users_list(self):
        e = Engine()
        with pytest.raises(DataValidationError, match="empty"):
            e.analyze_menus([])

    def test_invalid_tuple_length(self):
        e = Engine()
        with pytest.raises(DataValidationError, match="length 3"):
            e.analyze_menus([([0, 1, 2], [0])])

    def test_menus_choices_length_mismatch(self):
        e = Engine()
        with pytest.raises(DataValidationError, match="len.*!="):
            e.analyze_menus([([[0, 1], [1, 2]], [0], 3)])

    def test_choice_not_in_menu(self):
        e = Engine()
        with pytest.raises(DataValidationError, match="not in menu"):
            e.analyze_menus([([[0, 1], [1, 2]], [0, 5], 6)])

    def test_invalid_n_items(self):
        e = Engine()
        with pytest.raises(DataValidationError, match="positive integer"):
            e.analyze_menus([([[0, 1]], [0], 0)])

    def test_duplicate_items_in_menu_rejected(self):
        e = Engine()
        with pytest.raises(DataValidationError, match="duplicate items"):
            e.analyze_menus([([[0, 1, 1], [1, 2]], [0, 1], 3)])

    def test_item_id_out_of_range_rejected(self):
        e = Engine()
        with pytest.raises(DataValidationError, match="out of range"):
            e.analyze_menus([([[0, 5], [1, 2]], [0, 1], 3)])

    def test_negative_item_id_in_menu_rejected(self):
        e = Engine()
        with pytest.raises(DataValidationError, match="out of range"):
            e.analyze_menus([([[0, -1], [1, 2]], [0, 1], 3)])

    def test_valid_menu_input(self):
        e = Engine()
        users = [
            ([[0, 1, 2], [1, 2]], [1, 2], 3),
        ]
        results = e.analyze_menus(users)
        assert len(results) == 1
        assert isinstance(results[0], MenuResult)


# =============================================================================
# EngineResult ergonomics
# =============================================================================

class TestEngineResultErgonomics:
    def test_to_dict_keys(self):
        r = EngineResult(is_garp=True, ccei=1.0, mpi=0.0)
        d = r.to_dict()
        assert "is_garp" in d
        assert "ccei" in d
        assert "mpi" in d
        assert "compute_time_us" in d
        assert d["is_garp"] is True
        assert d["ccei"] == 1.0

    def test_repr_consistent(self):
        r = EngineResult(is_garp=True)
        s = repr(r)
        assert "[+]" in s
        assert "GARP-consistent" in s

    def test_repr_violation(self):
        r = EngineResult(is_garp=False, n_violations=3, ccei=0.87, mpi=0.12)
        s = repr(r)
        assert "[-]" in s
        assert "3 violations" in s
        assert "ccei=0.8700" in s
        assert "mpi=0.1200" in s

    def test_repr_with_hm(self):
        r = EngineResult(is_garp=True, hm_consistent=8, hm_total=10)
        s = repr(r)
        assert "hm=8/10" in s


class TestMenuResultErgonomics:
    def test_to_dict_keys(self):
        r = MenuResult(is_sarp=True, is_warp=True)
        d = r.to_dict()
        assert "is_sarp" in d
        assert "is_warp" in d
        assert "compute_time_us" in d

    def test_repr_consistent(self):
        r = MenuResult(is_sarp=True, is_warp=True)
        s = repr(r)
        assert "[+]" in s
        assert "SARP-consistent" in s

    def test_repr_violation(self):
        r = MenuResult(is_sarp=False, is_warp=False, n_sarp_violations=2, hm_consistent=6, hm_total=10)
        s = repr(r)
        assert "[-]" in s
        assert "2 SARP violations" in s
        assert "hm=6/10" in s


# =============================================================================
# results_to_dataframe
# =============================================================================

class TestResultsToDataframe:
    def test_basic(self):
        pd = pytest.importorskip("pandas")
        results = [
            EngineResult(is_garp=True, ccei=1.0),
            EngineResult(is_garp=False, ccei=0.87, n_violations=3),
        ]
        df = results_to_dataframe(results)
        assert len(df) == 2
        assert "is_garp" in df.columns
        assert "ccei" in df.columns

    def test_with_user_ids(self):
        pd = pytest.importorskip("pandas")
        results = [
            EngineResult(is_garp=True),
            EngineResult(is_garp=False),
        ]
        df = results_to_dataframe(results, user_ids=["alice", "bob"])
        assert df.index.name == "user_id"
        assert list(df.index) == ["alice", "bob"]

    def test_menu_results(self):
        pd = pytest.importorskip("pandas")
        results = [MenuResult(is_sarp=True, is_warp=True)]
        df = results_to_dataframe(results)
        assert "is_sarp" in df.columns


# =============================================================================
# load_demo
# =============================================================================

class TestLoadDemo:
    def test_returns_list_of_tuples(self):
        from prefgraph.datasets import load_demo
        users = load_demo(n_users=10)
        assert isinstance(users, list)
        assert len(users) == 10
        p, q = users[0]
        assert isinstance(p, np.ndarray)
        assert p.ndim == 2
        assert p.shape == q.shape

    def test_deterministic(self):
        from prefgraph.datasets import load_demo
        u1 = load_demo(n_users=5, seed=42)
        u2 = load_demo(n_users=5, seed=42)
        for (p1, q1), (p2, q2) in zip(u1, u2):
            np.testing.assert_array_equal(p1, p2)
            np.testing.assert_array_equal(q1, q2)

    def test_engine_compatible(self):
        from prefgraph.datasets import load_demo
        users = load_demo(n_users=5)
        e = Engine(metrics=["garp", "ccei"])
        results = e.analyze_arrays(users)
        assert len(results) == 5
        for r in results:
            assert isinstance(r, EngineResult)

    def test_return_panel(self):
        from prefgraph.datasets import load_demo
        from prefgraph.core.panel import BehaviorPanel
        panel = load_demo(n_users=5, return_panel=True)
        assert isinstance(panel, BehaviorPanel)
        assert panel.num_users == 5

    def test_has_variety(self):
        """Some users should be rational, some not."""
        from prefgraph.datasets import load_demo
        users = load_demo(n_users=50)
        e = Engine(metrics=["garp"])
        results = e.analyze_arrays(users)
        garp_values = [r.is_garp for r in results]
        assert any(garp_values), "Expected at least some GARP-consistent users"
        assert not all(garp_values), "Expected some GARP-violating users"


# =============================================================================
# Panel.to_engine_tuples
# =============================================================================

class TestPanelBridge:
    def test_behavior_panel_to_engine_tuples(self):
        from prefgraph.core.session import BehaviorLog
        from prefgraph.core.panel import BehaviorPanel

        logs = [
            BehaviorLog(
                cost_vectors=np.random.rand(5, 3).astype(np.float64) + 0.1,
                action_vectors=np.random.rand(5, 3).astype(np.float64),
                user_id=f"u{i}",
            )
            for i in range(3)
        ]
        panel = BehaviorPanel.from_logs(logs)
        tuples = panel.to_engine_tuples()
        assert len(tuples) == 3
        for p, q in tuples:
            assert isinstance(p, np.ndarray)
            assert p.shape == (5, 3)
