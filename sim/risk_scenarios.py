"""Risk preference scenario generators.

Generate synthetic data for testing risk profile classification.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import sys
sys.path.insert(0, '/Users/pranjal/Code/revealed/src')

from pyrevealed.core.session import RiskSession


def generate_gambler(
    n_choices: int = 50,
    seed: int | None = None,
) -> RiskSession:
    """
    Generate a risk-seeking user (gambler profile).

    Gamblers prefer risky options even when expected value is lower.
    They're attracted to high-variance, high-upside gambles.

    Args:
        n_choices: Number of choice observations
        seed: Random seed for reproducibility

    Returns:
        RiskSession with risk-seeking behavior
    """
    rng = np.random.default_rng(seed)

    safe_values = np.zeros(n_choices)
    risky_outcomes = np.zeros((n_choices, 2))
    risky_probabilities = np.zeros((n_choices, 2))
    choices = np.zeros(n_choices, dtype=bool)

    for i in range(n_choices):
        # Create lottery: high upside, low probability
        base_value = rng.uniform(50, 200)

        # Safe option: certain value
        safe_values[i] = base_value

        # Risky option: high potential win, significant loss
        # EV typically LOWER than safe to show risk-seeking
        win_prob = rng.uniform(0.2, 0.4)
        win_amount = base_value * rng.uniform(2.0, 4.0)
        lose_amount = base_value * rng.uniform(0.0, 0.3)

        risky_outcomes[i] = [win_amount, lose_amount]
        risky_probabilities[i] = [win_prob, 1 - win_prob]

        # Gambler chooses risky with high probability (80-95%)
        # Even when EV is lower
        choices[i] = rng.random() < rng.uniform(0.8, 0.95)

    return RiskSession(
        safe_values=safe_values,
        risky_outcomes=risky_outcomes,
        risky_probabilities=risky_probabilities,
        choices=choices,
        session_id="gambler",
    )


def generate_investor(
    n_choices: int = 50,
    seed: int | None = None,
) -> RiskSession:
    """
    Generate a risk-averse user (investor profile).

    Investors prefer certainty even when expected value is higher for risky.
    They value safety and predictability.

    Args:
        n_choices: Number of choice observations
        seed: Random seed for reproducibility

    Returns:
        RiskSession with risk-averse behavior
    """
    rng = np.random.default_rng(seed)

    safe_values = np.zeros(n_choices)
    risky_outcomes = np.zeros((n_choices, 2))
    risky_probabilities = np.zeros((n_choices, 2))
    choices = np.zeros(n_choices, dtype=bool)

    for i in range(n_choices):
        base_value = rng.uniform(50, 200)

        # Safe option
        safe_values[i] = base_value

        # Risky option with higher EV but variance
        # EV typically HIGHER than safe to show risk-aversion
        win_prob = rng.uniform(0.5, 0.7)
        # Make EV about 10-30% higher than safe
        ev_premium = rng.uniform(1.1, 1.3)
        target_ev = base_value * ev_premium

        # Win amount such that EV = win_prob * win + (1-win_prob) * lose
        lose_amount = base_value * rng.uniform(0.3, 0.6)
        win_amount = (target_ev - (1 - win_prob) * lose_amount) / win_prob

        risky_outcomes[i] = [win_amount, lose_amount]
        risky_probabilities[i] = [win_prob, 1 - win_prob]

        # Investor chooses safe with high probability (75-95%)
        # Even when EV of risky is higher
        choices[i] = rng.random() > rng.uniform(0.75, 0.95)

    return RiskSession(
        safe_values=safe_values,
        risky_outcomes=risky_outcomes,
        risky_probabilities=risky_probabilities,
        choices=choices,
        session_id="investor",
    )


def generate_risk_neutral(
    n_choices: int = 50,
    seed: int | None = None,
) -> RiskSession:
    """
    Generate a risk-neutral user.

    Chooses based purely on expected value comparison.

    Args:
        n_choices: Number of choice observations
        seed: Random seed for reproducibility

    Returns:
        RiskSession with risk-neutral behavior
    """
    rng = np.random.default_rng(seed)

    safe_values = np.zeros(n_choices)
    risky_outcomes = np.zeros((n_choices, 2))
    risky_probabilities = np.zeros((n_choices, 2))
    choices = np.zeros(n_choices, dtype=bool)

    for i in range(n_choices):
        base_value = rng.uniform(50, 200)
        safe_values[i] = base_value

        # Random lottery
        win_prob = rng.uniform(0.3, 0.7)
        win_amount = rng.uniform(base_value * 0.5, base_value * 3.0)
        lose_amount = rng.uniform(0, base_value * 0.8)

        risky_outcomes[i] = [win_amount, lose_amount]
        risky_probabilities[i] = [win_prob, 1 - win_prob]

        # Compute EV and choose based on that
        ev = win_prob * win_amount + (1 - win_prob) * lose_amount

        # Add small noise (5% chance of "mistake")
        if rng.random() < 0.05:
            choices[i] = rng.random() < 0.5
        else:
            choices[i] = ev > safe_values[i]

    return RiskSession(
        safe_values=safe_values,
        risky_outcomes=risky_outcomes,
        risky_probabilities=risky_probabilities,
        choices=choices,
        session_id="risk_neutral",
    )


def generate_mixed_risk_population(
    n_users: int = 100,
    gambler_ratio: float = 0.3,
    investor_ratio: float = 0.5,
    choices_per_user: int = 30,
    seed: int | None = None,
) -> list[RiskSession]:
    """
    Generate a mixed population of risk profiles.

    Args:
        n_users: Total number of users
        gambler_ratio: Fraction that are gamblers
        investor_ratio: Fraction that are investors (rest are neutral)
        choices_per_user: Number of choices per user
        seed: Random seed

    Returns:
        List of RiskSession objects
    """
    rng = np.random.default_rng(seed)

    sessions = []
    n_gamblers = int(n_users * gambler_ratio)
    n_investors = int(n_users * investor_ratio)
    n_neutral = n_users - n_gamblers - n_investors

    for i in range(n_gamblers):
        sessions.append(generate_gambler(choices_per_user, seed=rng.integers(0, 2**31)))

    for i in range(n_investors):
        sessions.append(generate_investor(choices_per_user, seed=rng.integers(0, 2**31)))

    for i in range(n_neutral):
        sessions.append(generate_risk_neutral(choices_per_user, seed=rng.integers(0, 2**31)))

    # Shuffle
    rng.shuffle(sessions)

    return sessions


def generate_lottery_choice_experiment(
    n_choices: int = 20,
    min_safe: float = 10.0,
    max_safe: float = 100.0,
    seed: int | None = None,
) -> tuple[RiskSession, str]:
    """
    Generate a standard lottery choice experiment (MPL-style).

    Creates a series of choices with varying safe amounts against
    a fixed lottery, used to identify switching points.

    Args:
        n_choices: Number of choices
        min_safe: Minimum safe amount
        max_safe: Maximum safe amount
        seed: Random seed

    Returns:
        Tuple of (RiskSession, true_type) where true_type is the
        generating process type
    """
    rng = np.random.default_rng(seed)

    # Fixed lottery: 50% chance of $100, 50% chance of $0
    lottery_outcomes = np.array([100.0, 0.0])
    lottery_probs = np.array([0.5, 0.5])

    # Varying safe amounts
    safe_values = np.linspace(min_safe, max_safe, n_choices)

    # Generate with random risk type
    risk_type = rng.choice(["gambler", "investor", "neutral"])

    if risk_type == "gambler":
        # Switch point very low (only take safe if very high)
        switch_point = rng.uniform(0.7, 0.9) * max_safe
    elif risk_type == "investor":
        # Switch point low (take safe early)
        switch_point = rng.uniform(0.2, 0.4) * max_safe
    else:
        # Switch at EV = $50
        switch_point = 50.0 + rng.normal(0, 5)

    # Generate choices
    choices = safe_values < switch_point  # True = chose risky

    # Add some noise
    noise_mask = rng.random(n_choices) < 0.05
    choices[noise_mask] = ~choices[noise_mask]

    risky_outcomes = np.tile(lottery_outcomes, (n_choices, 1))
    risky_probabilities = np.tile(lottery_probs, (n_choices, 1))

    session = RiskSession(
        safe_values=safe_values,
        risky_outcomes=risky_outcomes,
        risky_probabilities=risky_probabilities,
        choices=choices,
        session_id=f"mpl_{risk_type}",
    )

    return session, risk_type
