"""Tests for ConsumerSession data structure."""

import numpy as np
import pytest

from pyrevealed import ConsumerSession


class TestConsumerSessionCreation:
    """Tests for ConsumerSession initialization and validation."""

    def test_basic_creation(self):
        """Test basic session creation with valid data."""
        prices = np.array([[1.0, 2.0], [2.0, 1.0]])
        quantities = np.array([[3.0, 1.0], [1.0, 3.0]])

        session = ConsumerSession(prices=prices, quantities=quantities)

        assert session.num_observations == 2
        assert session.num_goods == 2
        assert session.session_id is None

    def test_creation_with_session_id(self):
        """Test session creation with optional session_id."""
        prices = np.array([[1.0, 2.0]])
        quantities = np.array([[3.0, 1.0]])

        session = ConsumerSession(
            prices=prices, quantities=quantities, session_id="user_123"
        )

        assert session.session_id == "user_123"

    def test_shape_mismatch_raises(self):
        """Test that mismatched shapes raise ValueError."""
        prices = np.array([[1.0, 2.0], [2.0, 1.0]])
        quantities = np.array([[3.0, 1.0]])  # Wrong shape

        with pytest.raises(ValueError, match="shape"):
            ConsumerSession(prices=prices, quantities=quantities)

    def test_negative_prices_raises(self):
        """Test that negative prices raise ValueError."""
        prices = np.array([[1.0, -2.0], [2.0, 1.0]])
        quantities = np.array([[3.0, 1.0], [1.0, 3.0]])

        with pytest.raises(ValueError, match="positive"):
            ConsumerSession(prices=prices, quantities=quantities)

    def test_zero_prices_raises(self):
        """Test that zero prices raise ValueError."""
        prices = np.array([[1.0, 0.0], [2.0, 1.0]])
        quantities = np.array([[3.0, 1.0], [1.0, 3.0]])

        with pytest.raises(ValueError, match="positive"):
            ConsumerSession(prices=prices, quantities=quantities)

    def test_negative_quantities_raises(self):
        """Test that negative quantities raise ValueError."""
        prices = np.array([[1.0, 2.0], [2.0, 1.0]])
        quantities = np.array([[3.0, -1.0], [1.0, 3.0]])

        with pytest.raises(ValueError, match="non-negative"):
            ConsumerSession(prices=prices, quantities=quantities)

    def test_1d_array_raises(self):
        """Test that 1D arrays raise ValueError."""
        prices = np.array([1.0, 2.0])
        quantities = np.array([3.0, 1.0])

        with pytest.raises(ValueError, match="2D"):
            ConsumerSession(prices=prices, quantities=quantities)


class TestExpenditureMatrix:
    """Tests for expenditure matrix computation."""

    def test_expenditure_matrix_shape(self, simple_consistent_session):
        """Test expenditure matrix has correct shape."""
        E = simple_consistent_session.expenditure_matrix
        T = simple_consistent_session.num_observations

        assert E.shape == (T, T)

    def test_expenditure_matrix_values(self):
        """Test expenditure matrix computation is correct."""
        prices = np.array([[1.0, 2.0], [3.0, 4.0]])
        quantities = np.array([[5.0, 6.0], [7.0, 8.0]])

        session = ConsumerSession(prices=prices, quantities=quantities)
        E = session.expenditure_matrix

        # E[0,0] = p_0 @ q_0 = 1*5 + 2*6 = 17
        assert E[0, 0] == pytest.approx(17.0)

        # E[0,1] = p_0 @ q_1 = 1*7 + 2*8 = 23
        assert E[0, 1] == pytest.approx(23.0)

        # E[1,0] = p_1 @ q_0 = 3*5 + 4*6 = 39
        assert E[1, 0] == pytest.approx(39.0)

        # E[1,1] = p_1 @ q_1 = 3*7 + 4*8 = 53
        assert E[1, 1] == pytest.approx(53.0)

    def test_own_expenditures(self):
        """Test own_expenditures returns diagonal correctly."""
        prices = np.array([[1.0, 2.0], [3.0, 4.0]])
        quantities = np.array([[5.0, 6.0], [7.0, 8.0]])

        session = ConsumerSession(prices=prices, quantities=quantities)

        assert session.own_expenditures[0] == pytest.approx(17.0)
        assert session.own_expenditures[1] == pytest.approx(53.0)


class TestFromDataFrame:
    """Tests for DataFrame conversion methods."""

    def test_from_dataframe(self):
        """Test creating session from pandas DataFrame."""
        pytest.importorskip("pandas")
        import pandas as pd

        df = pd.DataFrame({
            "p_A": [1.0, 2.0],
            "p_B": [2.0, 1.0],
            "q_A": [3.0, 1.0],
            "q_B": [1.0, 3.0],
        })

        session = ConsumerSession.from_dataframe(
            df, price_cols=["p_A", "p_B"], quantity_cols=["q_A", "q_B"]
        )

        assert session.num_observations == 2
        assert session.num_goods == 2
        np.testing.assert_array_equal(session.prices, [[1.0, 2.0], [2.0, 1.0]])


class TestSplitByWindow:
    """Tests for session splitting functionality."""

    def test_split_by_window(self, large_random_session):
        """Test splitting session into windows."""
        windows = large_random_session.split_by_window(window_size=25)

        assert len(windows) == 4  # 100 / 25 = 4 windows
        for w in windows:
            assert w.num_observations == 25

    def test_split_by_window_uneven(self):
        """Test splitting when observations don't divide evenly."""
        prices = np.ones((10, 2))
        quantities = np.ones((10, 2))
        session = ConsumerSession(prices=prices, quantities=quantities)

        windows = session.split_by_window(window_size=3)

        # Should get windows of size 3, 3, 3, and last window of 1 is dropped (< 2)
        assert len(windows) == 3
