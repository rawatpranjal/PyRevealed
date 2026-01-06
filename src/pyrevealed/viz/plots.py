"""Plotting functions for revealed preference analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pyrevealed.core.session import ConsumerSession


def plot_budget_sets(
    session: ConsumerSession,
    goods: tuple[int, int] = (0, 1),
    figsize: tuple[int, int] = (8, 8),
    ax: Any = None,
) -> tuple[Any, Any]:
    """
    Plot budget sets and chosen bundles for two goods.

    Each observation is shown as:
    - A budget line (all affordable bundles at those prices)
    - A point marking the actual chosen bundle

    This visualization helps understand revealed preference intuitively:
    if bundle A was chosen when bundle B was affordable, A is revealed
    preferred to B.

    Args:
        session: ConsumerSession with price and quantity data
        goods: Tuple of two good indices to plot (default: first two goods)
        figsize: Figure size as (width, height)
        ax: Optional matplotlib axes to draw on

    Returns:
        Tuple of (figure, axes) matplotlib objects

    Example:
        >>> from pyrevealed import ConsumerSession
        >>> from pyrevealed.viz import plot_budget_sets
        >>> session = ConsumerSession(prices, quantities)
        >>> fig, ax = plot_budget_sets(session)
        >>> fig.savefig("budget_sets.png")
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    g0, g1 = goods
    T = session.num_observations

    # Color map for different observations
    colors = plt.cm.tab10(np.linspace(0, 1, min(T, 10)))

    # Find plot limits
    max_q0 = session.quantities[:, g0].max() * 1.5
    max_q1 = session.quantities[:, g1].max() * 1.5

    for t in range(T):
        p0 = session.prices[t, g0]
        p1 = session.prices[t, g1]
        budget = session.own_expenditures[t]

        # Budget line: p0 * x0 + p1 * x1 = budget
        # x1 = (budget - p0 * x0) / p1
        x0_line = np.linspace(0, budget / p0, 100)
        x1_line = (budget - p0 * x0_line) / p1

        # Only plot positive values
        mask = x1_line >= 0
        x0_line = x0_line[mask]
        x1_line = x1_line[mask]

        color = colors[t % len(colors)]

        # Plot budget line
        ax.plot(
            x0_line,
            x1_line,
            color=color,
            linestyle="--",
            alpha=0.7,
            label=f"Budget {t}",
        )

        # Plot chosen bundle
        ax.scatter(
            session.quantities[t, g0],
            session.quantities[t, g1],
            color=color,
            s=100,
            zorder=5,
            edgecolors="black",
            linewidths=1,
        )

        # Annotate the point
        ax.annotate(
            f"t={t}",
            (session.quantities[t, g0], session.quantities[t, g1]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9,
        )

    ax.set_xlim(0, max_q0)
    ax.set_ylim(0, max_q1)
    ax.set_xlabel(f"Good {g0}")
    ax.set_ylabel(f"Good {g1}")
    ax.set_title("Budget Sets and Chosen Bundles")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_aei_distribution(
    scores: list[float],
    bins: int = 50,
    figsize: tuple[int, int] = (10, 6),
    ax: Any = None,
) -> tuple[Any, Any]:
    """
    Plot distribution of AEI scores across a population.

    Useful for analyzing the consistency of a user population:
    - Peaks near 1.0 indicate mostly rational users
    - Spread toward 0 indicates inconsistent behavior
    - Bimodal distributions may indicate distinct user segments

    Args:
        scores: List of AEI scores (values between 0 and 1)
        bins: Number of histogram bins
        figsize: Figure size as (width, height)
        ax: Optional matplotlib axes to draw on

    Returns:
        Tuple of (figure, axes) matplotlib objects
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    scores_arr = np.array(scores)

    # Plot histogram
    ax.hist(scores_arr, bins=bins, edgecolor="black", alpha=0.7, color="steelblue")

    # Add statistics
    mean_score = np.mean(scores_arr)
    median_score = np.median(scores_arr)
    std_score = np.std(scores_arr)

    ax.axvline(mean_score, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_score:.3f}")
    ax.axvline(median_score, color="orange", linestyle="-.", linewidth=2, label=f"Median: {median_score:.3f}")

    # Add threshold lines for interpretation
    ax.axvline(0.85, color="green", linestyle=":", alpha=0.7, label="Bot threshold (0.85)")
    ax.axvline(0.95, color="purple", linestyle=":", alpha=0.7, label="High consistency (0.95)")

    ax.set_xlim(0, 1)
    ax.set_xlabel("Afriat Efficiency Index (AEI)")
    ax.set_ylabel("Count")
    ax.set_title(f"AEI Distribution (n={len(scores)}, std={std_score:.3f})")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_mpi_over_time(
    mpi_values: list[float],
    timestamps: list[Any] | None = None,
    figsize: tuple[int, int] = (12, 5),
    ax: Any = None,
) -> tuple[Any, Any]:
    """
    Plot Money Pump Index over time to detect behavioral changes.

    Useful for detecting:
    - Account takeovers (sudden MPI spike)
    - UI changes affecting user behavior
    - Seasonal patterns in consistency

    Args:
        mpi_values: List of MPI values over time
        timestamps: Optional list of timestamps/labels for x-axis
        figsize: Figure size
        ax: Optional matplotlib axes

    Returns:
        Tuple of (figure, axes)
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    x = timestamps if timestamps is not None else list(range(len(mpi_values)))

    ax.plot(x, mpi_values, marker="o", linestyle="-", color="crimson", markersize=4)
    ax.fill_between(x, 0, mpi_values, alpha=0.3, color="crimson")

    # Add threshold line
    ax.axhline(0.1, color="orange", linestyle="--", label="Warning threshold (10%)")
    ax.axhline(0.2, color="red", linestyle="--", label="Critical threshold (20%)")

    ax.set_ylim(0, max(max(mpi_values) * 1.1, 0.25))
    ax.set_xlabel("Time")
    ax.set_ylabel("Money Pump Index")
    ax.set_title("Behavioral Consistency Over Time")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_preference_heatmap(
    session: ConsumerSession,
    matrix_type: str = "direct",
    figsize: tuple[int, int] = (8, 8),
    ax: Any = None,
) -> tuple[Any, Any]:
    """
    Plot revealed preference matrix as a heatmap.

    Args:
        session: ConsumerSession
        matrix_type: Type of matrix to plot:
            - 'direct': Direct revealed preference R
            - 'strict': Strict revealed preference P
            - 'transitive': Transitive closure R*
        figsize: Figure size
        ax: Optional matplotlib axes

    Returns:
        Tuple of (figure, axes)
    """
    import matplotlib.pyplot as plt
    from pyrevealed.algorithms.garp import check_garp

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    result = check_garp(session)

    if matrix_type == "direct":
        matrix = result.direct_revealed_preference.astype(float)
        title = "Direct Revealed Preference (R)"
    elif matrix_type == "strict":
        matrix = result.strict_revealed_preference.astype(float)
        title = "Strict Revealed Preference (P)"
    elif matrix_type == "transitive":
        matrix = result.transitive_closure.astype(float)
        title = "Transitive Closure (R*)"
    else:
        raise ValueError(f"Unknown matrix_type: {matrix_type}")

    im = ax.imshow(matrix, cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax, label="Preference")

    T = session.num_observations
    ax.set_xticks(range(T))
    ax.set_yticks(range(T))
    ax.set_xticklabels([f"{i}" for i in range(T)])
    ax.set_yticklabels([f"{i}" for i in range(T)])

    ax.set_xlabel("Observation j")
    ax.set_ylabel("Observation i")
    ax.set_title(title)

    # Add text annotations
    for i in range(T):
        for j in range(T):
            text = "1" if matrix[i, j] > 0.5 else ""
            ax.text(j, i, text, ha="center", va="center", color="white" if matrix[i, j] > 0.5 else "black")

    return fig, ax
