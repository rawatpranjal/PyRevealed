"""Tests for pyrevealed.analyze() one-liner API."""

import numpy as np
import pandas as pd
import pytest

import pyrevealed as rp
from pyrevealed.analyze import analyze, _detect_format
from pyrevealed.engine import Engine, EngineResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _wide_df():
    """3 users, 2 goods, 3 observations each — wide format."""
    rows = []
    for uid in ["A", "B", "C"]:
        np.random.seed(hash(uid) % 2**31)
        for _ in range(3):
            rows.append({
                "user_id": uid,
                "price_x": round(np.random.uniform(1, 5), 2),
                "price_y": round(np.random.uniform(1, 5), 2),
                "qty_x": round(np.random.uniform(0, 10), 2),
                "qty_y": round(np.random.uniform(0, 10), 2),
            })
    return pd.DataFrame(rows)


def _long_df():
    """2 users, 3 items, 4 time periods — long format (transaction logs)."""
    rows = []
    for uid in ["U1", "U2"]:
        for t in range(1, 5):
            for item in ["apple", "bread", "cheese"]:
                rows.append({
                    "user_id": uid,
                    "week": t,
                    "product": item,
                    "price": round(np.random.uniform(1, 5), 2),
                    "quantity": round(np.random.uniform(0, 10), 2),
                })
    np.random.seed(42)  # reset after
    return pd.DataFrame(rows)


def _menu_df():
    """2 users, 5 observations each — menu choice data."""
    rows = []
    for uid in ["M1", "M2"]:
        for _ in range(5):
            menu = frozenset(np.random.choice(5, size=3, replace=False))
            choice = np.random.choice(list(menu))
            rows.append({
                "user_id": uid,
                "shown": menu,
                "clicked": int(choice),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

class TestFormatDetection:

    def test_wide_detected(self):
        fmt = _detect_format(
            item_col=None, cost_col=None, action_col=None, time_col=None,
            cost_cols=["p1"], action_cols=["q1"],
            menu_col=None, choice_col=None,
        )
        assert fmt == "wide"

    def test_long_detected(self):
        fmt = _detect_format(
            item_col="product", cost_col="price", action_col="qty", time_col="week",
            cost_cols=None, action_cols=None,
            menu_col=None, choice_col=None,
        )
        assert fmt == "long"

    def test_menu_detected(self):
        fmt = _detect_format(
            item_col=None, cost_col=None, action_col=None, time_col=None,
            cost_cols=None, action_cols=None,
            menu_col="menu", choice_col="choice",
        )
        assert fmt == "menu"

    def test_conflict_raises(self):
        with pytest.raises(ValueError, match="Conflicting"):
            _detect_format(
                item_col="product", cost_col=None, action_col=None, time_col=None,
                cost_cols=["p1"], action_cols=None,
                menu_col=None, choice_col=None,
            )

    def test_no_format_raises(self):
        with pytest.raises(ValueError, match="Cannot detect"):
            _detect_format(
                item_col=None, cost_col=None, action_col=None, time_col=None,
                cost_cols=None, action_cols=None,
                menu_col=None, choice_col=None,
            )


# ---------------------------------------------------------------------------
# Wide format
# ---------------------------------------------------------------------------

class TestAnalyzeWide:

    def test_basic(self):
        df = _wide_df()
        result = analyze(df, cost_cols=["price_x", "price_y"],
                         action_cols=["qty_x", "qty_y"])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3  # 3 users
        assert "is_garp" in result.columns
        assert "ccei" in result.columns
        assert "mpi" in result.columns

    def test_returns_dataframe_by_default(self):
        df = _wide_df()
        result = analyze(df, cost_cols=["price_x", "price_y"],
                         action_cols=["qty_x", "qty_y"])
        assert isinstance(result, pd.DataFrame)

    def test_output_objects(self):
        df = _wide_df()
        result = analyze(df, cost_cols=["price_x", "price_y"],
                         action_cols=["qty_x", "qty_y"],
                         output="objects")
        assert isinstance(result, list)
        assert len(result) == 3
        # Each element is (user_id, EngineResult)
        uid, obj = result[0]
        assert isinstance(uid, str)
        assert isinstance(obj, EngineResult)

    def test_custom_metrics(self):
        df = _wide_df()
        result = analyze(df, cost_cols=["price_x", "price_y"],
                         action_cols=["qty_x", "qty_y"],
                         metrics=["garp", "ccei"])
        assert isinstance(result, pd.DataFrame)
        assert "is_garp" in result.columns
        assert "ccei" in result.columns

    def test_legacy_aliases(self):
        df = _wide_df()
        result = analyze(df, price_cols=["price_x", "price_y"],
                         qty_cols=["qty_x", "qty_y"])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_single_user(self):
        df = _wide_df()
        df = df[df["user_id"] == "A"]
        result = analyze(df, cost_cols=["price_x", "price_y"],
                         action_cols=["qty_x", "qty_y"])
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Long format
# ---------------------------------------------------------------------------

class TestAnalyzeLong:

    def test_basic(self):
        df = _long_df()
        result = analyze(df, item_col="product", cost_col="price",
                         action_col="quantity", time_col="week")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # 2 users
        assert "is_garp" in result.columns

    def test_legacy_aliases(self):
        df = _long_df()
        result = analyze(df, item_col="product", price_col="price",
                         qty_col="quantity", time_col="week")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Menu format
# ---------------------------------------------------------------------------

class TestAnalyzeMenu:

    def test_basic(self):
        np.random.seed(99)
        df = _menu_df()
        result = analyze(df, menu_col="shown", choice_col="clicked")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # 2 users
        assert "is_sarp" in result.columns
        assert "is_warp" in result.columns

    def test_metrics_warning(self):
        np.random.seed(99)
        df = _menu_df()
        with pytest.warns(UserWarning, match="metrics parameter is ignored"):
            analyze(df, menu_col="shown", choice_col="clicked",
                    metrics=["garp", "ccei"])


# ---------------------------------------------------------------------------
# Error messages
# ---------------------------------------------------------------------------

class TestErrorMessages:

    def test_missing_user_col(self):
        df = _wide_df()
        with pytest.raises(ValueError, match="not found in DataFrame"):
            analyze(df, user_col="nonexistent",
                    cost_cols=["price_x", "price_y"],
                    action_cols=["qty_x", "qty_y"])

    def test_no_format_error_lists_options(self):
        df = _wide_df()
        with pytest.raises(ValueError, match="Cannot detect"):
            analyze(df)

    def test_conflict_error(self):
        df = _wide_df()
        with pytest.raises(ValueError, match="Conflicting"):
            analyze(df, item_col="product",
                    cost_cols=["price_x", "price_y"],
                    action_cols=["qty_x", "qty_y"])


# ---------------------------------------------------------------------------
# Regression: matches direct Engine usage
# ---------------------------------------------------------------------------

class TestMatchesEngine:

    def test_wide_matches_engine(self):
        df = _wide_df()
        # Via analyze()
        result_df = analyze(df, cost_cols=["price_x", "price_y"],
                            action_cols=["qty_x", "qty_y"],
                            metrics=["garp", "ccei", "mpi"])

        # Via Engine directly
        from pyrevealed.core.panel import BehaviorPanel
        panel = BehaviorPanel.from_dataframe(
            df, user_col="user_id",
            cost_cols=["price_x", "price_y"],
            action_cols=["qty_x", "qty_y"],
        )
        engine = Engine(metrics=["garp", "ccei", "mpi"])
        results = engine.analyze_arrays(panel.to_engine_tuples())

        for i, (uid, log) in enumerate(panel):
            row = result_df.loc[uid]
            assert row["is_garp"] == results[i].is_garp
            assert abs(row["ccei"] - results[i].ccei) < 1e-10
            assert abs(row["mpi"] - results[i].mpi) < 1e-10
