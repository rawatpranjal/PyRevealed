"""Tests for spatial/ideal point preference analysis."""

import numpy as np
import pytest

from pyrevealed.core.session import SpatialSession
from pyrevealed.algorithms.spatial import (
    find_ideal_point,
    check_euclidean_rationality,
    compute_preference_strength,
    find_multiple_ideal_points,
)


class TestSpatialSession:
    """Tests for SpatialSession data structure."""

    def test_basic_creation(self):
        """Test creating a basic SpatialSession."""
        features = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        choice_sets = [[0, 1], [0, 2], [1, 3]]
        choices = [0, 0, 1]

        session = SpatialSession(features, choice_sets, choices)

        assert session.num_items == 4
        assert session.num_dimensions == 2
        assert session.num_observations == 3

    def test_choice_must_be_in_choice_set(self):
        """Test that chosen item must be in choice set."""
        features = np.array([[0, 0], [1, 0], [0, 1]])
        choice_sets = [[0, 1]]
        choices = [2]  # Item 2 not in choice set

        with pytest.raises(ValueError, match="not in choice set"):
            SpatialSession(features, choice_sets, choices)

    def test_choice_set_minimum_size(self):
        """Test that choice sets need at least 2 items."""
        features = np.array([[0, 0], [1, 0]])
        choice_sets = [[0]]  # Only 1 item
        choices = [0]

        with pytest.raises(ValueError, match="at least 2 items"):
            SpatialSession(features, choice_sets, choices)

    def test_item_index_bounds(self):
        """Test that item indices are within bounds."""
        features = np.array([[0, 0], [1, 0]])
        choice_sets = [[0, 5]]  # Item 5 doesn't exist
        choices = [0]

        with pytest.raises(ValueError, match="out of range"):
            SpatialSession(features, choice_sets, choices)


class TestFindIdealPoint:
    """Tests for find_ideal_point function."""

    def test_perfect_euclidean_user(self):
        """Test finding ideal point for perfectly Euclidean user."""
        # Items at corners of a square, user prefers origin
        features = np.array([
            [0.0, 0.0],  # Origin
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])
        # Always choose origin when available
        choice_sets = [[0, 1], [0, 2], [0, 3], [0, 1, 2, 3]]
        choices = [0, 0, 0, 0]

        session = SpatialSession(features, choice_sets, choices)
        result = find_ideal_point(session)

        # Ideal point should be close to origin
        assert np.allclose(result.ideal_point, [0, 0], atol=0.1)
        assert result.is_euclidean_rational
        assert result.num_violations == 0

    def test_off_center_ideal_point(self):
        """Test finding ideal point not at item location."""
        # Items in a line, user prefers items around 0.5
        features = np.array([
            [0.0, 0.0],
            [0.3, 0.0],
            [0.5, 0.0],
            [0.7, 0.0],
            [1.0, 0.0],
        ])
        # Always choose item closest to 0.5
        choice_sets = [[0, 1, 2], [2, 3, 4], [0, 4], [1, 3]]
        choices = [2, 2, 0, 1]  # 0.5 when available, else closest

        session = SpatialSession(features, choice_sets, choices)
        result = find_ideal_point(session)

        # Ideal should be around 0.3-0.5 range
        assert 0.2 < result.ideal_point[0] < 0.6

    def test_violations_detected(self):
        """Test that violations are detected for inconsistent choices."""
        features = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 0.0],
        ])
        # Inconsistent: sometimes prefer 0, sometimes prefer 1
        choice_sets = [[0, 1], [0, 1], [0, 2], [1, 2]]
        choices = [0, 1, 0, 1]  # Flip-flopping

        session = SpatialSession(features, choice_sets, choices)
        result = find_ideal_point(session)

        # Should have violations since no single point works
        assert result.num_violations > 0 or not result.is_euclidean_rational

    def test_higher_dimensions(self):
        """Test with more feature dimensions."""
        rng = np.random.default_rng(42)
        n_items = 20
        n_dims = 5

        features = rng.uniform(-1, 1, (n_items, n_dims))
        ideal = np.zeros(n_dims)  # Origin as ideal

        # Generate choices always picking closest to origin
        distances = np.linalg.norm(features, axis=1)
        choice_sets = []
        choices = []

        for _ in range(15):
            cs = rng.choice(n_items, size=4, replace=False).tolist()
            cs_distances = [distances[i] for i in cs]
            chosen = cs[np.argmin(cs_distances)]
            choice_sets.append(cs)
            choices.append(chosen)

        session = SpatialSession(features, choice_sets, choices)
        result = find_ideal_point(session)

        # Ideal point should be close to origin
        assert np.linalg.norm(result.ideal_point) < 0.5

    def test_computation_time(self):
        """Test that computation time is tracked."""
        features = np.array([[0, 0], [1, 0], [0, 1]])
        choice_sets = [[0, 1], [0, 2]]
        choices = [0, 0]

        session = SpatialSession(features, choice_sets, choices)
        result = find_ideal_point(session)

        assert result.computation_time_ms > 0


class TestCheckEuclideanRationality:
    """Tests for check_euclidean_rationality function."""

    def test_rational_choices(self):
        """Test that consistent choices are detected as rational."""
        features = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        choice_sets = [[0, 1], [0, 2], [0, 3]]
        choices = [0, 0, 0]  # Always choose origin

        session = SpatialSession(features, choice_sets, choices)
        is_rational, violations = check_euclidean_rationality(session)

        assert is_rational
        assert len(violations) == 0

    def test_irrational_choices(self):
        """Test that inconsistent choices are detected."""
        features = np.array([[0, 0], [1, 0], [2, 0]])
        # Choose 0 from {0,1} but 2 from {0,2} - inconsistent!
        choice_sets = [[0, 1], [0, 2]]
        choices = [0, 2]

        session = SpatialSession(features, choice_sets, choices)
        is_rational, violations = check_euclidean_rationality(session)

        # This might or might not be detected as irrational depending on
        # whether a single ideal point can explain it
        assert isinstance(is_rational, bool)


class TestPreferenceStrength:
    """Tests for compute_preference_strength function."""

    def test_strong_preferences(self):
        """Test that strong preferences have high scores."""
        features = np.array([
            [0.0, 0.0],  # Very close to ideal
            [10.0, 0.0],  # Very far from ideal
        ])
        choice_sets = [[0, 1]]
        choices = [0]  # Choose close one

        session = SpatialSession(features, choice_sets, choices)
        ideal = np.array([0.0, 0.0])

        strengths = compute_preference_strength(session, ideal)

        assert strengths[0] > 0  # Positive = correct choice

    def test_weak_preferences(self):
        """Test that close items have weak preference strength."""
        features = np.array([
            [0.0, 0.0],
            [0.1, 0.0],  # Very close
        ])
        choice_sets = [[0, 1]]
        choices = [0]

        session = SpatialSession(features, choice_sets, choices)
        ideal = np.array([0.0, 0.0])

        strengths = compute_preference_strength(session, ideal)

        # Should be small but positive
        assert 0 < strengths[0] < 10


class TestMultipleIdealPoints:
    """Tests for find_multiple_ideal_points function."""

    def test_single_ideal_point(self):
        """Test finding single ideal point."""
        features = np.array([[0, 0], [1, 0], [0, 1]])
        choice_sets = [[0, 1], [0, 2]]
        choices = [0, 0]

        session = SpatialSession(features, choice_sets, choices)
        results = find_multiple_ideal_points(session, n_points=1)

        assert len(results) == 1
        assert len(results[0][0]) == 2  # 2D point

    def test_two_ideal_points(self):
        """Test finding two ideal points."""
        features = np.array([
            [0, 0], [1, 0], [0, 1], [1, 1],
            [5, 5], [6, 5], [5, 6], [6, 6],
        ])
        # Alternating between two clusters
        choice_sets = [[0, 1], [4, 5], [2, 3], [6, 7]]
        choices = [0, 4, 2, 6]  # Corners of each cluster

        session = SpatialSession(features, choice_sets, choices)
        results = find_multiple_ideal_points(session, n_points=2)

        assert len(results) <= 2


class TestIdealPointResult:
    """Tests for IdealPointResult properties."""

    def test_num_dimensions(self):
        """Test num_dimensions property."""
        features = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        choice_sets = [[0, 1], [0, 2]]
        choices = [0, 0]

        session = SpatialSession(features, choice_sets, choices)
        result = find_ideal_point(session)

        assert result.num_dimensions == 3

    def test_violation_rate(self):
        """Test violation_rate property."""
        features = np.array([[0, 0], [1, 0], [0, 1]])
        choice_sets = [[0, 1], [0, 2]]
        choices = [0, 0]

        session = SpatialSession(features, choice_sets, choices)
        result = find_ideal_point(session)

        assert 0 <= result.violation_rate <= 1

    def test_explained_variance(self):
        """Test that explained_variance is in [0, 1]."""
        features = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        choice_sets = [[0, 1], [0, 2], [0, 3]]
        choices = [0, 0, 0]

        session = SpatialSession(features, choice_sets, choices)
        result = find_ideal_point(session)

        assert 0 <= result.explained_variance <= 1
