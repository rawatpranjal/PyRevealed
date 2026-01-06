"""Visualization utilities for revealed preference analysis."""

from pyrevealed.graph.violation_graph import ViolationGraph

__all__ = [
    "ViolationGraph",
    "plot_budget_sets",
    "plot_aei_distribution",
]


def plot_budget_sets(
    session,
    goods: tuple[int, int] = (0, 1),
    figsize: tuple[int, int] = (8, 8),
    ax=None,
):
    """
    Plot budget sets and chosen bundles for two goods.

    Args:
        session: ConsumerSession
        goods: Tuple of two good indices to plot
        figsize: Figure size
        ax: Optional matplotlib axes

    Returns:
        Tuple of (figure, axes)
    """
    from pyrevealed.viz.plots import plot_budget_sets as _plot

    return _plot(session, goods, figsize, ax)


def plot_aei_distribution(
    scores: list[float],
    bins: int = 50,
    figsize: tuple[int, int] = (10, 6),
    ax=None,
):
    """
    Plot distribution of AEI scores across a population.

    Args:
        scores: List of AEI scores
        bins: Number of histogram bins
        figsize: Figure size
        ax: Optional matplotlib axes

    Returns:
        Tuple of (figure, axes)
    """
    from pyrevealed.viz.plots import plot_aei_distribution as _plot

    return _plot(scores, bins, figsize, ax)
