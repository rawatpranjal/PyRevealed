"""Display mixins for rich result presentation.

This module provides mixins that add HTML rendering and plotting capabilities
to result dataclasses, following the statsmodels pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyrevealed.core.session import BehaviorLog


class ResultDisplayMixin:
    """Mixin providing _repr_html_() and enhanced display for Jupyter notebooks.

    This mixin adds rich HTML rendering support for result dataclasses,
    enabling pretty display in Jupyter notebooks and HTML-capable environments.

    Methods:
        _repr_html_: Returns HTML representation for Jupyter display
        short_summary: Returns one-liner with [+]/[-] indicator
    """

    def _repr_html_(self) -> str:
        """Return HTML representation for Jupyter notebook display.

        Returns a styled HTML table that renders nicely in Jupyter environments.
        Falls back to summary() for the content.
        """
        from pyrevealed.viz.html_templates import render_result_html

        return render_result_html(self)

    def short_summary(self) -> str:
        """Return a one-line summary with [+]/[-] indicator.

        Returns:
            Single line string with pass/fail indicator and key metric.
        """
        # Default implementation - subclasses should override
        return self._default_short_summary()

    def _default_short_summary(self) -> str:
        """Default short summary implementation."""
        class_name = self.__class__.__name__
        if hasattr(self, "is_consistent"):
            indicator = "[+]" if self.is_consistent else "[-]"
            status = "PASS" if self.is_consistent else "FAIL"
            return f"{class_name}: {indicator} {status}"
        elif hasattr(self, "success"):
            indicator = "[+]" if self.success else "[-]"
            status = "SUCCESS" if self.success else "FAILED"
            return f"{class_name}: {indicator} {status}"
        elif hasattr(self, "score"):
            score_val = self.score() if callable(self.score) else self.score
            indicator = "[+]" if score_val >= 0.9 else "[-]"
            return f"{class_name}: {indicator} score={score_val:.4f}"
        else:
            return f"{class_name}: (no summary available)"

    def _get_display_name(self) -> str:
        """Get human-readable display name for the result type."""
        name = self.__class__.__name__
        # Convert CamelCase to Title Case with spaces
        import re

        name = re.sub(r"Result$", "", name)
        name = re.sub(r"([A-Z])", r" \1", name).strip()
        return name

    def _get_status_indicator(self) -> tuple[str, str, str]:
        """Get status indicator, status text, and CSS class.

        Returns:
            Tuple of (indicator, status_text, css_class)
        """
        if hasattr(self, "is_consistent"):
            if self.is_consistent:
                return "[+]", "PASS", "status-pass"
            else:
                return "[-]", "FAIL", "status-fail"
        elif hasattr(self, "is_perfectly_consistent"):
            if self.is_perfectly_consistent:
                return "[+]", "PERFECT", "status-pass"
            else:
                return "[~]", "IMPERFECT", "status-warn"
        elif hasattr(self, "success"):
            if self.success:
                return "[+]", "SUCCESS", "status-pass"
            else:
                return "[-]", "FAILED", "status-fail"
        elif hasattr(self, "is_rationalizable"):
            if self.is_rationalizable:
                return "[+]", "RATIONALIZABLE", "status-pass"
            else:
                return "[-]", "NOT RATIONALIZABLE", "status-fail"
        else:
            return "[?]", "UNKNOWN", "status-neutral"

    def _get_key_metrics(self) -> list[tuple[str, Any]]:
        """Get key metrics for display as list of (name, value) tuples."""
        metrics = []

        # Common attributes to extract
        attr_names = [
            ("efficiency_index", "Efficiency Index"),
            ("mpi_value", "MPI Value"),
            ("power_index", "Power Index"),
            ("consistency_score", "Consistency Score"),
            ("num_violations", "Violations"),
            ("waste_fraction", "Waste Fraction"),
        ]

        for attr, display_name in attr_names:
            if hasattr(self, attr):
                value = getattr(self, attr)
                if callable(value):
                    value = value()
                metrics.append((display_name, value))

        # Add score if available and not already included
        if hasattr(self, "score") and "score" not in [m[0].lower() for m in metrics]:
            score_val = self.score() if callable(self.score) else self.score
            metrics.append(("Score", score_val))

        return metrics


class ResultPlotMixin:
    """Mixin providing plot() method with lazy matplotlib import.

    This mixin adds plotting capabilities to result dataclasses. It uses
    lazy importing of matplotlib to ensure the core package works without
    visualization dependencies.

    Methods:
        plot: Generic plot method that dispatches to specific visualizations
        to_graph: Returns a ViolationGraph for GARP-related results
    """

    def plot(self, kind: str = "auto", **kwargs) -> tuple[Any, Any]:
        """Create a visualization of this result.

        Args:
            kind: Type of plot to create:
                - 'auto': Automatically select appropriate plot
                - 'graph': Preference graph (for GARP results)
                - 'heatmap': Preference matrix heatmap
                - 'distribution': Distribution plot
            **kwargs: Additional arguments passed to the plotting function

        Returns:
            Tuple of (figure, axes) matplotlib objects

        Raises:
            ImportError: If matplotlib is not installed
            NotImplementedError: If no plot is available for this result type
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "Plotting requires matplotlib. Install with: pip install pyrevealed[viz]"
            )

        # Dispatch based on result type and kind
        class_name = self.__class__.__name__

        if kind == "auto":
            kind = self._get_default_plot_kind()

        if kind == "graph":
            return self._plot_graph(**kwargs)
        elif kind == "heatmap":
            return self._plot_heatmap(**kwargs)
        elif kind == "distribution":
            return self._plot_distribution(**kwargs)
        else:
            raise NotImplementedError(
                f"Plot kind '{kind}' not available for {class_name}"
            )

    def _get_default_plot_kind(self) -> str:
        """Get the default plot kind for this result type."""
        class_name = self.__class__.__name__

        if "GARP" in class_name or "Consistency" in class_name:
            return "graph"
        elif "AEI" in class_name or "Integrity" in class_name:
            return "distribution"
        else:
            return "heatmap"

    def _plot_graph(self, **kwargs) -> tuple[Any, Any]:
        """Plot preference graph. Requires session parameter."""
        raise NotImplementedError(
            "Graph plotting requires a BehaviorLog. Use to_graph(session).plot() instead."
        )

    def _plot_heatmap(self, **kwargs) -> tuple[Any, Any]:
        """Plot heatmap visualization."""
        import matplotlib.pyplot as plt
        import numpy as np

        figsize = kwargs.get("figsize", (8, 8))
        fig, ax = plt.subplots(figsize=figsize)

        # Try to find a matrix to plot
        matrix = None
        title = "Result Heatmap"

        if hasattr(self, "direct_revealed_preference"):
            matrix = self.direct_revealed_preference.astype(float)
            title = "Direct Revealed Preference Matrix"
        elif hasattr(self, "transitive_closure"):
            matrix = self.transitive_closure.astype(float)
            title = "Transitive Closure Matrix"

        if matrix is None:
            ax.text(
                0.5,
                0.5,
                "No matrix data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return fig, ax

        im = ax.imshow(matrix, cmap="Blues", aspect="auto")
        plt.colorbar(im, ax=ax, label="Value")
        ax.set_title(title)
        ax.set_xlabel("Observation j")
        ax.set_ylabel("Observation i")

        return fig, ax

    def _plot_distribution(self, **kwargs) -> tuple[Any, Any]:
        """Plot distribution visualization."""
        import matplotlib.pyplot as plt
        import numpy as np

        figsize = kwargs.get("figsize", (10, 6))
        fig, ax = plt.subplots(figsize=figsize)

        # Try to find values to plot
        values = None
        title = "Distribution"

        if hasattr(self, "utility_values") and self.utility_values is not None:
            values = np.array(self.utility_values)
            title = "Utility Values Distribution"
        elif hasattr(self, "efficiency_vector") and self.efficiency_vector is not None:
            values = np.array(self.efficiency_vector)
            title = "Efficiency Distribution"

        if values is None:
            ax.text(
                0.5,
                0.5,
                "No distribution data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return fig, ax

        ax.hist(values, bins=kwargs.get("bins", 30), edgecolor="black", alpha=0.7)
        ax.axvline(np.mean(values), color="red", linestyle="--", label=f"Mean: {np.mean(values):.3f}")
        ax.set_title(title)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.legend()

        return fig, ax

    def to_graph(self, session: BehaviorLog) -> Any:
        """Convert this result to a ViolationGraph for visualization.

        Args:
            session: The BehaviorLog used to generate this result

        Returns:
            ViolationGraph instance for visualization and analysis

        Raises:
            NotImplementedError: If result type doesn't support graph conversion
        """
        class_name = self.__class__.__name__

        # Only works for GARP-related results
        if not (
            hasattr(self, "direct_revealed_preference")
            and hasattr(self, "violations")
        ):
            raise NotImplementedError(
                f"{class_name} does not support graph conversion. "
                "Only GARP-related results can be converted to graphs."
            )

        from pyrevealed.graph.violation_graph import ViolationGraph

        return ViolationGraph(session, self)
