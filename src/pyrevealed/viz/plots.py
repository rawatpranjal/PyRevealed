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

    ax.axvline(
        mean_score,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_score:.3f}",
    )
    ax.axvline(
        median_score,
        color="orange",
        linestyle="-.",
        linewidth=2,
        label=f"Median: {median_score:.3f}",
    )

    # Add threshold lines for interpretation
    ax.axvline(
        0.85, color="green", linestyle=":", alpha=0.7, label="Bot threshold (0.85)"
    )
    ax.axvline(
        0.95, color="purple", linestyle=":", alpha=0.7, label="High consistency (0.95)"
    )

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
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if matrix[i, j] > 0.5 else "black",
            )

    return fig, ax


def plot_ccei_sensitivity(
    session: ConsumerSession,
    max_remove: int | None = None,
    figsize: tuple[int, int] = (10, 6),
    ax: Any = None,
) -> tuple[Any, Any]:
    """
    Plot CCEI (AEI) vs observations removed (worst-first).

    Shows how efficiency improves as problematic observations are removed.
    Useful for understanding which observations are driving inconsistencies.

    Args:
        session: ConsumerSession with price and quantity data
        max_remove: Maximum number of observations to remove (default: 20% of data)
        figsize: Figure size as (width, height)
        ax: Optional matplotlib axes to draw on

    Returns:
        Tuple of (figure, axes) matplotlib objects
    """
    import matplotlib.pyplot as plt
    from pyrevealed.algorithms.aei import compute_integrity_score
    from pyrevealed.algorithms.vei import compute_granular_integrity

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    T = session.num_observations
    if max_remove is None:
        max_remove = max(1, int(T * 0.2))

    # Compute per-observation efficiency
    vei_result = compute_granular_integrity(session)
    efficiency_vector = vei_result.efficiency_vector

    # Sort by efficiency (worst first)
    sorted_indices = np.argsort(efficiency_vector)

    # Compute CCEI for each removal level
    ccei_values = [compute_integrity_score(session).efficiency_index]
    removed_counts = [0]

    current_mask = np.ones(T, dtype=bool)
    for i in range(min(max_remove, T - 2)):
        # Remove worst observation
        worst_idx = sorted_indices[i]
        current_mask[worst_idx] = False

        # Create filtered session
        from pyrevealed.core.session import BehaviorLog
        filtered_session = BehaviorLog(
            cost_vectors=session.prices[current_mask],
            action_vectors=session.quantities[current_mask],
        )

        if filtered_session.num_observations < 2:
            break

        ccei = compute_integrity_score(filtered_session).efficiency_index
        ccei_values.append(ccei)
        removed_counts.append(i + 1)

    # Plot
    ax.plot(removed_counts, ccei_values, "o-", color="steelblue", markersize=6)
    ax.fill_between(removed_counts, ccei_values, alpha=0.3, color="steelblue")

    # Reference lines
    ax.axhline(1.0, color="green", linestyle="--", alpha=0.7, label="Perfect (1.0)")
    ax.axhline(0.95, color="orange", linestyle=":", alpha=0.7, label="High (0.95)")

    ax.set_xlim(-0.5, max(removed_counts) + 0.5)
    ax.set_ylim(min(ccei_values) * 0.95, 1.02)
    ax.set_xlabel("Observations Removed (worst-first)")
    ax.set_ylabel("CCEI (Afriat Efficiency Index)")
    ax.set_title("CCEI Sensitivity to Observation Removal")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_power_analysis(
    session: ConsumerSession,
    n_simulations: int = 500,
    figsize: tuple[int, int] = (10, 6),
    ax: Any = None,
) -> tuple[Any, Any]:
    """
    Plot observed CCEI against simulated random choice distribution.

    Visualizes Bronars power - how much better than random is the data.
    A vertical line shows the observed CCEI; the histogram shows what
    random behavior would produce.

    Args:
        session: ConsumerSession with price and quantity data
        n_simulations: Number of random simulations (default: 500)
        figsize: Figure size as (width, height)
        ax: Optional matplotlib axes to draw on

    Returns:
        Tuple of (figure, axes) matplotlib objects
    """
    import matplotlib.pyplot as plt
    from pyrevealed.algorithms.aei import compute_integrity_score
    from pyrevealed.algorithms.bronars import compute_test_power

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Compute observed CCEI
    observed_ccei = compute_integrity_score(session).efficiency_index

    # Run power analysis
    power_result = compute_test_power(session, n_simulations=n_simulations)

    if power_result.simulation_integrity_values is not None:
        random_ccei = power_result.simulation_integrity_values

        # Plot histogram of random CCEIs
        ax.hist(
            random_ccei,
            bins=30,
            edgecolor="black",
            alpha=0.7,
            color="lightcoral",
            label=f"Random choice (n={n_simulations})",
        )

        # Plot observed CCEI
        ax.axvline(
            observed_ccei,
            color="darkblue",
            linewidth=3,
            linestyle="-",
            label=f"Observed CCEI: {observed_ccei:.4f}",
        )

        # Statistics
        mean_random = np.mean(random_ccei)
        ax.axvline(
            mean_random,
            color="red",
            linewidth=2,
            linestyle="--",
            label=f"Random mean: {mean_random:.4f}",
        )

        # Percentile of observed
        percentile = np.mean(random_ccei <= observed_ccei) * 100
        ax.text(
            0.02,
            0.98,
            f"Observed is better than {percentile:.1f}% of random",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    ax.set_xlim(0, 1.05)
    ax.set_xlabel("CCEI (Afriat Efficiency Index)")
    ax.set_ylabel("Count")
    ax.set_title(f"Power Analysis (Bronars Power = {power_result.power_index:.4f})")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_violation_severity(
    session: ConsumerSession,
    result: Any = None,
    figsize: tuple[int, int] = (10, 6),
    ax: Any = None,
) -> tuple[Any, Any]:
    """
    Histogram of violation magnitudes (budget fraction wasted).

    Shows the distribution of how severe each GARP violation is,
    measured by the fraction of budget that could be saved.

    Args:
        session: ConsumerSession with price and quantity data
        result: Optional GARPResult (computed if not provided)
        figsize: Figure size as (width, height)
        ax: Optional matplotlib axes to draw on

    Returns:
        Tuple of (figure, axes) matplotlib objects
    """
    import matplotlib.pyplot as plt
    from pyrevealed.algorithms.garp import check_garp
    from pyrevealed.algorithms.vei import compute_granular_integrity

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if result is None:
        result = check_garp(session)

    # Compute per-observation efficiency (inverse of severity)
    vei_result = compute_granular_integrity(session)
    severities = 1.0 - vei_result.efficiency_vector

    # Plot histogram
    ax.hist(severities, bins=20, edgecolor="black", alpha=0.7, color="salmon")

    # Add mean line
    mean_severity = np.mean(severities)
    ax.axvline(
        mean_severity,
        color="darkred",
        linewidth=2,
        linestyle="--",
        label=f"Mean severity: {mean_severity:.4f}",
    )

    # Thresholds
    ax.axvline(0.05, color="green", linestyle=":", alpha=0.7, label="Low (5%)")
    ax.axvline(0.15, color="orange", linestyle=":", alpha=0.7, label="High (15%)")

    ax.set_xlim(-0.01, max(severities) * 1.1 + 0.01)
    ax.set_xlabel("Violation Severity (1 - efficiency)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Violation Severities")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_budget_intersections(
    session: ConsumerSession,
    result: Any = None,
    figsize: tuple[int, int] = (10, 10),
    ax: Any = None,
) -> tuple[Any, Any]:
    """
    Heatmap showing which budget pairs intersect with WARP violations marked.

    An intersection at (i, j) means bundle j was affordable at budget i.
    Red markers highlight pairs with WARP violations.

    Args:
        session: ConsumerSession with price and quantity data
        result: Optional GARPResult (computed if not provided)
        figsize: Figure size as (width, height)
        ax: Optional matplotlib axes to draw on

    Returns:
        Tuple of (figure, axes) matplotlib objects
    """
    import matplotlib.pyplot as plt
    from pyrevealed.algorithms.garp import check_garp

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if result is None:
        result = check_garp(session)

    T = session.num_observations

    # Compute intersection matrix: can bundle j be afforded at budget i?
    intersection_matrix = np.zeros((T, T))
    for i in range(T):
        for j in range(T):
            cost_j_at_i = np.dot(session.prices[i], session.quantities[j])
            budget_i = session.own_expenditures[i]
            if cost_j_at_i <= budget_i * 1.0001:  # Small tolerance
                intersection_matrix[i, j] = 1

    # Create heatmap
    im = ax.imshow(intersection_matrix, cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax, label="Affordable (1) / Not Affordable (0)")

    # Mark WARP violations in red
    R = result.direct_revealed_preference
    for i in range(T):
        for j in range(T):
            if i != j and R[i, j] and R[j, i]:  # Mutual preference = WARP violation
                ax.scatter(j, i, color="red", s=100, marker="x", linewidths=2)

    ax.set_xticks(range(T))
    ax.set_yticks(range(T))
    ax.set_xticklabels([f"{i}" for i in range(T)])
    ax.set_yticklabels([f"{i}" for i in range(T)])
    ax.set_xlabel("Bundle j")
    ax.set_ylabel("Budget i")
    ax.set_title("Budget Intersections (red X = WARP violation)")

    return fig, ax
