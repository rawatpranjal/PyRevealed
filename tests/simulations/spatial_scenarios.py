"""Spatial/ideal point preference scenario generators.

Generate synthetic data for testing ideal point estimation in feature spaces.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import sys
sys.path.insert(0, '/Users/pranjal/Code/revealed/src')

from pyrevealed.core.session import SpatialSession


def generate_euclidean_user(
    ideal_point: NDArray[np.float64] | None = None,
    n_items: int = 50,
    n_dimensions: int = 5,
    n_choices: int = 30,
    choice_set_size: int = 4,
    seed: int | None = None,
) -> SpatialSession:
    """
    Generate a user who always chooses the item closest to their ideal point.

    This is a perfectly Euclidean-rational user with no noise.

    Args:
        ideal_point: D-dimensional ideal point (random if None)
        n_items: Number of items in the catalog
        n_dimensions: Number of feature dimensions
        n_choices: Number of choice observations
        choice_set_size: Number of items in each choice set
        seed: Random seed

    Returns:
        SpatialSession with perfectly Euclidean-rational choices
    """
    rng = np.random.default_rng(seed)

    if ideal_point is None:
        ideal_point = rng.uniform(-1, 1, n_dimensions)

    # Generate random item features
    item_features = rng.uniform(-2, 2, (n_items, n_dimensions))

    # Compute distances from ideal point
    distances = np.linalg.norm(item_features - ideal_point, axis=1)

    # Generate choice sets and choices
    choice_sets = []
    choices = []

    for _ in range(n_choices):
        # Random choice set
        choice_set = rng.choice(n_items, size=choice_set_size, replace=False).tolist()

        # Choose closest item
        set_distances = [distances[i] for i in choice_set]
        best_idx = np.argmin(set_distances)
        chosen = choice_set[best_idx]

        choice_sets.append(choice_set)
        choices.append(chosen)

    return SpatialSession(
        item_features=item_features,
        choice_sets=choice_sets,
        choices=choices,
        session_id="euclidean_perfect",
        metadata={"true_ideal_point": ideal_point.tolist()},
    )


def generate_noisy_user(
    ideal_point: NDArray[np.float64] | None = None,
    noise_level: float = 0.2,
    n_items: int = 50,
    n_dimensions: int = 5,
    n_choices: int = 30,
    choice_set_size: int = 4,
    seed: int | None = None,
) -> SpatialSession:
    """
    Generate a user with Euclidean preferences plus noise.

    With probability `noise_level`, the user chooses randomly instead
    of the closest item.

    Args:
        ideal_point: D-dimensional ideal point (random if None)
        noise_level: Probability of random choice (0-1)
        n_items: Number of items in the catalog
        n_dimensions: Number of feature dimensions
        n_choices: Number of choice observations
        choice_set_size: Number of items in each choice set
        seed: Random seed

    Returns:
        SpatialSession with noisy Euclidean choices
    """
    rng = np.random.default_rng(seed)

    if ideal_point is None:
        ideal_point = rng.uniform(-1, 1, n_dimensions)

    # Generate random item features
    item_features = rng.uniform(-2, 2, (n_items, n_dimensions))

    # Compute distances from ideal point
    distances = np.linalg.norm(item_features - ideal_point, axis=1)

    # Generate choice sets and choices
    choice_sets = []
    choices = []

    for _ in range(n_choices):
        # Random choice set
        choice_set = rng.choice(n_items, size=choice_set_size, replace=False).tolist()

        if rng.random() < noise_level:
            # Random choice
            chosen = rng.choice(choice_set)
        else:
            # Choose closest item
            set_distances = [distances[i] for i in choice_set]
            best_idx = np.argmin(set_distances)
            chosen = choice_set[best_idx]

        choice_sets.append(choice_set)
        choices.append(chosen)

    return SpatialSession(
        item_features=item_features,
        choice_sets=choice_sets,
        choices=choices,
        session_id=f"euclidean_noisy_{noise_level:.0%}",
        metadata={
            "true_ideal_point": ideal_point.tolist(),
            "noise_level": noise_level,
        },
    )


def generate_multi_ideal_user(
    ideal_points: list[NDArray[np.float64]] | None = None,
    n_items: int = 50,
    n_dimensions: int = 5,
    n_choices: int = 40,
    choice_set_size: int = 4,
    seed: int | None = None,
) -> SpatialSession:
    """
    Generate a user with MULTIPLE ideal points (simulates shared account).

    Each choice randomly uses one of the ideal points. This should
    fail the single ideal point test.

    Args:
        ideal_points: List of D-dimensional ideal points (2 random if None)
        n_items: Number of items in the catalog
        n_dimensions: Number of feature dimensions
        n_choices: Number of choice observations
        choice_set_size: Number of items in each choice set
        seed: Random seed

    Returns:
        SpatialSession with choices from multiple ideal points
    """
    rng = np.random.default_rng(seed)

    if ideal_points is None:
        # Two very different ideal points
        ideal_points = [
            rng.uniform(-1, 0, n_dimensions),  # Prefer negative features
            rng.uniform(0, 1, n_dimensions),   # Prefer positive features
        ]

    # Generate random item features
    item_features = rng.uniform(-2, 2, (n_items, n_dimensions))

    # Generate choice sets and choices
    choice_sets = []
    choices = []

    for i in range(n_choices):
        # Random choice set
        choice_set = rng.choice(n_items, size=choice_set_size, replace=False).tolist()

        # Randomly select which ideal point to use
        active_ideal = ideal_points[i % len(ideal_points)]

        # Compute distances and choose closest
        set_features = item_features[choice_set]
        distances = np.linalg.norm(set_features - active_ideal, axis=1)
        best_idx = np.argmin(distances)
        chosen = choice_set[best_idx]

        choice_sets.append(choice_set)
        choices.append(chosen)

    return SpatialSession(
        item_features=item_features,
        choice_sets=choice_sets,
        choices=choices,
        session_id="multi_ideal_user",
        metadata={
            "true_ideal_points": [p.tolist() for p in ideal_points],
            "num_users": len(ideal_points),
        },
    )


def generate_recommendation_scenario(
    n_users: int = 50,
    n_items: int = 100,
    n_dimensions: int = 10,
    choices_per_user: int = 20,
    noise_level: float = 0.1,
    seed: int | None = None,
) -> tuple[list[SpatialSession], NDArray[np.float64]]:
    """
    Generate a recommendation system scenario with multiple users.

    All users share the same item catalog but have different ideal points.

    Args:
        n_users: Number of users
        n_items: Number of items in shared catalog
        n_dimensions: Number of feature dimensions
        choices_per_user: Choices per user
        noise_level: Noise level for each user
        seed: Random seed

    Returns:
        Tuple of (list of SpatialSessions, shared item_features matrix)
    """
    rng = np.random.default_rng(seed)

    # Shared item catalog
    item_features = rng.uniform(-2, 2, (n_items, n_dimensions))

    sessions = []
    for i in range(n_users):
        # Each user has a random ideal point
        ideal_point = rng.uniform(-1, 1, n_dimensions)

        # Generate their choices
        choice_sets = []
        choices = []

        distances = np.linalg.norm(item_features - ideal_point, axis=1)

        for _ in range(choices_per_user):
            choice_set = rng.choice(n_items, size=4, replace=False).tolist()

            if rng.random() < noise_level:
                chosen = rng.choice(choice_set)
            else:
                set_distances = [distances[idx] for idx in choice_set]
                chosen = choice_set[np.argmin(set_distances)]

            choice_sets.append(choice_set)
            choices.append(chosen)

        session = SpatialSession(
            item_features=item_features,
            choice_sets=choice_sets,
            choices=choices,
            session_id=f"user_{i}",
            metadata={"true_ideal_point": ideal_point.tolist()},
        )
        sessions.append(session)

    return sessions, item_features


def generate_dating_app_scenario(
    n_profiles: int = 100,
    n_choices: int = 30,
    seed: int | None = None,
) -> SpatialSession:
    """
    Generate a dating app scenario with profile features.

    Features: [attractiveness, humor, intelligence, ambition, kindness]
    Each user has preferences for different feature combinations.

    Args:
        n_profiles: Number of dating profiles
        n_choices: Number of swipe decisions
        seed: Random seed

    Returns:
        SpatialSession representing swipe choices
    """
    rng = np.random.default_rng(seed)

    # Feature names (for reference)
    feature_names = ["attractiveness", "humor", "intelligence", "ambition", "kindness"]
    n_features = len(feature_names)

    # Generate profile features (normalized 0-1)
    profiles = rng.uniform(0, 1, (n_profiles, n_features))

    # User's ideal partner (random preference weights)
    # Some features weighted more than others
    ideal = rng.dirichlet(np.ones(n_features) * 2)

    # Generate choices (swipe right = choose this profile over others)
    choice_sets = []
    choices = []

    for _ in range(n_choices):
        # Show 2-4 profiles
        n_shown = rng.integers(2, 5)
        shown = rng.choice(n_profiles, size=n_shown, replace=False).tolist()

        # Compute "fit" score for each profile
        # Higher score = closer to ideal weighted preferences
        scores = [np.sum(profiles[p] * ideal) for p in shown]

        # Add some noise (dating is noisy!)
        noise = rng.normal(0, 0.1, len(scores))
        noisy_scores = np.array(scores) + noise

        # Choose best match
        best_idx = np.argmax(noisy_scores)
        chosen = shown[best_idx]

        choice_sets.append(shown)
        choices.append(chosen)

    return SpatialSession(
        item_features=profiles,
        choice_sets=choice_sets,
        choices=choices,
        session_id="dating_app",
        metadata={
            "feature_names": feature_names,
            "true_preferences": ideal.tolist(),
        },
    )
