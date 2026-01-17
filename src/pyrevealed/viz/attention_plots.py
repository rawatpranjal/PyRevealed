"""Plotting functions for attention-based choice analysis.

This module provides visualization functions for limited attention models,
including attention decay by position and consideration set size distributions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pyrevealed.core.result import AttentionResult, RandomAttentionResult


def plot_attention_decay(
    result: AttentionResult,
    figsize: tuple[int, int] = (10, 6),
    ax: Any = None,
) -> tuple[Any, Any]:
    """
    Plot attention probability by menu position.

    Shows how attention/consideration probability decays with position
    in the choice set, revealing position bias effects.

    Args:
        result: AttentionResult from attention analysis
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

    # Extract attention weights or probabilities
    if hasattr(result, "salience_weights") and result.salience_weights is not None:
        weights = np.array(result.salience_weights)
        positions = np.arange(len(weights))

        # Normalize to probabilities
        if np.sum(weights) > 0:
            probs = weights / np.sum(weights)
        else:
            probs = weights

        # Plot
        bars = ax.bar(positions, probs, color="steelblue", edgecolor="black", alpha=0.7)

        # Add value labels
        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            ax.annotate(
                f"{prob:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_xlabel("Position in Menu")
        ax.set_ylabel("Attention Probability")
        ax.set_title("Attention Decay by Position")
        ax.set_xticks(positions)
        ax.set_xticklabels([f"Pos {i}" for i in positions])

    elif hasattr(result, "attention_filter") and result.attention_filter is not None:
        # Alternative: plot attention filter function
        filter_data = result.attention_filter
        ax.text(
            0.5,
            0.5,
            "Attention filter available but not position-based",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No attention data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.grid(True, alpha=0.3, axis="y")
    return fig, ax


def plot_consideration_sizes(
    result: AttentionResult | RandomAttentionResult,
    figsize: tuple[int, int] = (10, 6),
    ax: Any = None,
) -> tuple[Any, Any]:
    """
    Plot distribution of consideration set sizes.

    Shows how many alternatives are typically considered in each choice,
    revealing the extent of limited attention.

    Args:
        result: AttentionResult or RandomAttentionResult from attention analysis
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

    # Try to extract consideration set sizes
    sizes = None

    if hasattr(result, "consideration_sets") and result.consideration_sets is not None:
        # Extract sizes from consideration sets
        sizes = [len(cs) for cs in result.consideration_sets if cs is not None]
    elif hasattr(result, "estimated_set_sizes") and result.estimated_set_sizes is not None:
        sizes = list(result.estimated_set_sizes)

    if sizes is not None and len(sizes) > 0:
        sizes = np.array(sizes)

        # Histogram
        unique_sizes = np.unique(sizes)
        counts = [np.sum(sizes == s) for s in unique_sizes]

        ax.bar(unique_sizes, counts, color="coral", edgecolor="black", alpha=0.7)

        # Statistics
        mean_size = np.mean(sizes)
        ax.axvline(
            mean_size,
            color="darkred",
            linewidth=2,
            linestyle="--",
            label=f"Mean: {mean_size:.2f}",
        )

        # Add text annotation
        ax.text(
            0.98,
            0.98,
            f"Mean: {mean_size:.2f}\nMin: {np.min(sizes)}\nMax: {np.max(sizes)}",
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        ax.set_xlabel("Consideration Set Size")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Consideration Set Sizes")
        ax.set_xticks(unique_sizes)
        ax.legend(loc="upper left")
    else:
        ax.text(
            0.5,
            0.5,
            "No consideration set data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.grid(True, alpha=0.3, axis="y")
    return fig, ax


def plot_attention_heatmap(
    result: AttentionResult | RandomAttentionResult,
    figsize: tuple[int, int] = (10, 8),
    ax: Any = None,
) -> tuple[Any, Any]:
    """
    Plot attention probability matrix as a heatmap.

    Shows the estimated attention probability for each alternative
    across different choice observations.

    Args:
        result: AttentionResult or RandomAttentionResult from attention analysis
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

    # Try to extract attention matrix
    attention_matrix = None

    if hasattr(result, "attention_probabilities") and result.attention_probabilities is not None:
        attention_matrix = np.array(result.attention_probabilities)
    elif hasattr(result, "consideration_sets") and result.consideration_sets is not None:
        # Convert consideration sets to binary matrix
        sets = result.consideration_sets
        if sets and len(sets) > 0:
            n_obs = len(sets)
            max_alt = max(max(cs) for cs in sets if cs) + 1
            attention_matrix = np.zeros((n_obs, max_alt))
            for i, cs in enumerate(sets):
                if cs:
                    for alt in cs:
                        attention_matrix[i, alt] = 1.0

    if attention_matrix is not None:
        im = ax.imshow(attention_matrix, cmap="YlOrRd", aspect="auto")
        plt.colorbar(im, ax=ax, label="Attention Probability")

        ax.set_xlabel("Alternative")
        ax.set_ylabel("Observation")
        ax.set_title("Attention Probability Matrix")
    else:
        ax.text(
            0.5,
            0.5,
            "No attention matrix data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    return fig, ax


def plot_attention_bounds(
    result: RandomAttentionResult,
    figsize: tuple[int, int] = (12, 6),
    ax: Any = None,
) -> tuple[Any, Any]:
    """
    Plot attention probability bounds for each alternative.

    Shows the lower and upper bounds on attention probability
    that are consistent with the observed choice data.

    Args:
        result: RandomAttentionResult from RAM analysis
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

    # Check for bounds data
    if hasattr(result, "attention_bounds") and result.attention_bounds is not None:
        bounds = result.attention_bounds

        if isinstance(bounds, dict):
            alternatives = sorted(bounds.keys())
            lowers = [bounds[a][0] for a in alternatives]
            uppers = [bounds[a][1] for a in alternatives]
        else:
            # Assume numpy array [n_alternatives, 2]
            bounds = np.array(bounds)
            alternatives = list(range(len(bounds)))
            lowers = bounds[:, 0]
            uppers = bounds[:, 1]

        x = np.arange(len(alternatives))
        width = 0.35

        # Plot bounds as error bars
        midpoints = [(l + u) / 2 for l, u in zip(lowers, uppers)]
        errors = [(m - l, u - m) for l, m, u in zip(lowers, midpoints, uppers)]
        errors = np.array(errors).T

        ax.bar(x, midpoints, width, yerr=errors, color="steelblue", alpha=0.7,
               capsize=5, edgecolor="black", label="Attention bounds")

        # Add labels
        ax.set_xlabel("Alternative")
        ax.set_ylabel("Attention Probability")
        ax.set_title("Attention Probability Bounds (Random Attention Model)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Alt {a}" for a in alternatives])
        ax.legend()
        ax.set_ylim(0, 1.05)
    else:
        ax.text(
            0.5,
            0.5,
            "No attention bounds data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.grid(True, alpha=0.3, axis="y")
    return fig, ax
