"""Visualization utilities for revealed preference analysis."""

from pyrevealed.graph.violation_graph import ViolationGraph

__all__ = [
    # Core visualization
    "ViolationGraph",
    "plot_budget_sets",
    "plot_aei_distribution",
    # New visualizations
    "plot_ccei_sensitivity",
    "plot_power_analysis",
    "plot_violation_severity",
    "plot_budget_intersections",
    # Attention plots
    "plot_attention_decay",
    "plot_consideration_sizes",
    "plot_attention_heatmap",
    "plot_attention_bounds",
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


def plot_ccei_sensitivity(
    session,
    max_remove: int | None = None,
    figsize: tuple[int, int] = (10, 6),
    ax=None,
):
    """
    Plot CCEI (AEI) vs observations removed (worst-first).

    Shows how efficiency improves as problematic observations are removed.

    Args:
        session: ConsumerSession
        max_remove: Maximum observations to remove (default: 20% of data)
        figsize: Figure size
        ax: Optional matplotlib axes

    Returns:
        Tuple of (figure, axes)
    """
    from pyrevealed.viz.plots import plot_ccei_sensitivity as _plot

    return _plot(session, max_remove, figsize, ax)


def plot_power_analysis(
    session,
    n_simulations: int = 500,
    figsize: tuple[int, int] = (10, 6),
    ax=None,
):
    """
    Plot observed CCEI against simulated random choice distribution.

    Visualizes Bronars power - how much better than random is the data.

    Args:
        session: ConsumerSession
        n_simulations: Number of random simulations
        figsize: Figure size
        ax: Optional matplotlib axes

    Returns:
        Tuple of (figure, axes)
    """
    from pyrevealed.viz.plots import plot_power_analysis as _plot

    return _plot(session, n_simulations, figsize, ax)


def plot_violation_severity(
    session,
    result=None,
    figsize: tuple[int, int] = (10, 6),
    ax=None,
):
    """
    Histogram of violation magnitudes (budget fraction wasted).

    Args:
        session: ConsumerSession
        result: Optional GARPResult
        figsize: Figure size
        ax: Optional matplotlib axes

    Returns:
        Tuple of (figure, axes)
    """
    from pyrevealed.viz.plots import plot_violation_severity as _plot

    return _plot(session, result, figsize, ax)


def plot_budget_intersections(
    session,
    result=None,
    figsize: tuple[int, int] = (10, 10),
    ax=None,
):
    """
    Heatmap showing which budget pairs intersect with WARP violations marked.

    Args:
        session: ConsumerSession
        result: Optional GARPResult
        figsize: Figure size
        ax: Optional matplotlib axes

    Returns:
        Tuple of (figure, axes)
    """
    from pyrevealed.viz.plots import plot_budget_intersections as _plot

    return _plot(session, result, figsize, ax)


def plot_attention_decay(
    result,
    figsize: tuple[int, int] = (10, 6),
    ax=None,
):
    """
    Plot attention probability by menu position.

    Args:
        result: AttentionResult from attention analysis
        figsize: Figure size
        ax: Optional matplotlib axes

    Returns:
        Tuple of (figure, axes)
    """
    from pyrevealed.viz.attention_plots import plot_attention_decay as _plot

    return _plot(result, figsize, ax)


def plot_consideration_sizes(
    result,
    figsize: tuple[int, int] = (10, 6),
    ax=None,
):
    """
    Plot distribution of consideration set sizes.

    Args:
        result: AttentionResult or RandomAttentionResult
        figsize: Figure size
        ax: Optional matplotlib axes

    Returns:
        Tuple of (figure, axes)
    """
    from pyrevealed.viz.attention_plots import plot_consideration_sizes as _plot

    return _plot(result, figsize, ax)


def plot_attention_heatmap(
    result,
    figsize: tuple[int, int] = (10, 8),
    ax=None,
):
    """
    Plot attention probability matrix as a heatmap.

    Args:
        result: AttentionResult or RandomAttentionResult
        figsize: Figure size
        ax: Optional matplotlib axes

    Returns:
        Tuple of (figure, axes)
    """
    from pyrevealed.viz.attention_plots import plot_attention_heatmap as _plot

    return _plot(result, figsize, ax)


def plot_attention_bounds(
    result,
    figsize: tuple[int, int] = (12, 6),
    ax=None,
):
    """
    Plot attention probability bounds for each alternative.

    Args:
        result: RandomAttentionResult from RAM analysis
        figsize: Figure size
        ax: Optional matplotlib axes

    Returns:
        Tuple of (figure, axes)
    """
    from pyrevealed.viz.attention_plots import plot_attention_bounds as _plot

    return _plot(result, figsize, ax)
