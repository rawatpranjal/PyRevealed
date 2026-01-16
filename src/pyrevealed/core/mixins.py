"""Mixin classes for result dataclasses.

This module provides common formatting utilities for result summaries.
"""

from __future__ import annotations

from typing import Any


class ResultSummaryMixin:
    """Common formatting utilities for result summaries.

    Provides helper methods for generating human-readable summary reports
    with consistent formatting across all result types.
    """

    @staticmethod
    def _format_header(title: str, width: int = 80) -> str:
        """Format a section header.

        Args:
            title: Header title text
            width: Total width of the header

        Returns:
            Formatted header string with border
        """
        border = "=" * width
        # Center the title
        padding = (width - len(title)) // 2
        centered_title = " " * padding + title
        return f"{border}\n{centered_title}\n{border}"

    @staticmethod
    def _format_metric(label: str, value: Any, width: int = 40) -> str:
        """Format a metric label-value pair.

        Args:
            label: Metric name
            value: Metric value
            width: Total width for alignment

        Returns:
            Formatted metric string
        """
        # Format the value appropriately
        if isinstance(value, float):
            if abs(value) < 0.0001 and value != 0:
                formatted_value = f"{value:.4e}"
            elif abs(value) >= 1000:
                formatted_value = f"{value:,.2f}"
            else:
                formatted_value = f"{value:.4f}"
        elif isinstance(value, bool):
            formatted_value = "Yes" if value else "No"
        elif value is None:
            formatted_value = "N/A"
        else:
            formatted_value = str(value)

        # Right-align the label with dots for visual tracking
        dots = "." * max(1, width - len(label) - len(formatted_value) - 2)
        return f"  {label} {dots} {formatted_value}"

    @staticmethod
    def _format_status(passed: bool, pass_text: str = "PASSED",
                       fail_text: str = "FAILED") -> str:
        """Format a pass/fail status indicator.

        Args:
            passed: Whether the test passed
            pass_text: Text to show if passed
            fail_text: Text to show if failed

        Returns:
            Formatted status string
        """
        return pass_text if passed else fail_text

    @staticmethod
    def _format_interpretation(score: float, metric_type: str = "efficiency") -> str:
        """Generate interpretation text based on score thresholds.

        Args:
            score: Score value in [0, 1]
            metric_type: Type of metric for context-specific interpretation

        Returns:
            Interpretation text string
        """
        if metric_type == "efficiency":
            # AEI/integrity style thresholds
            if score >= 1.0 - 1e-6:
                return "Perfect consistency - behavior fully rationalized by utility maximization"
            elif score >= 0.95:
                return "Excellent consistency - minor noise or measurement error"
            elif score >= 0.9:
                return "Good consistency - behavior largely rational"
            elif score >= 0.7:
                return "Moderate consistency - some behavioral anomalies present"
            else:
                return "Low consistency - significant departures from rational behavior"

        elif metric_type == "mpi":
            # MPI (confusion/exploitability) - lower is better, but we get 1-mpi
            if score >= 1.0 - 1e-6:
                return "No exploitability - choices are fully consistent"
            elif score >= 0.95:
                return "Very low exploitability (<5% vulnerable)"
            elif score >= 0.85:
                return "Low exploitability (5-15% vulnerable)"
            else:
                return "High exploitability (>15% vulnerable) - significant preference cycles"

        elif metric_type == "power":
            # Bronars power
            if score >= 0.9:
                return "High discriminatory power - test is very informative"
            elif score >= 0.5:
                return "Moderate power - test has reasonable discriminatory ability"
            else:
                return "Low power - passing may not be meaningful"

        else:
            # Generic interpretation
            if score >= 0.95:
                return "Excellent"
            elif score >= 0.8:
                return "Good"
            elif score >= 0.5:
                return "Moderate"
            else:
                return "Poor"

    @staticmethod
    def _format_footer(computation_time_ms: float, width: int = 80) -> str:
        """Format the report footer with computation time.

        Args:
            computation_time_ms: Time in milliseconds
            width: Total width of the footer

        Returns:
            Formatted footer string
        """
        border = "=" * width
        if computation_time_ms < 1000:
            time_str = f"{computation_time_ms:.2f} ms"
        else:
            time_str = f"{computation_time_ms / 1000:.2f} s"
        return f"\nComputation Time: {time_str}\n{border}"

    @staticmethod
    def _format_section(title: str) -> str:
        """Format a section subheader.

        Args:
            title: Section title

        Returns:
            Formatted section header
        """
        return f"\n{title}:\n{'-' * len(title)}"

    @staticmethod
    def _format_list(items: list, max_items: int = 5, item_name: str = "item") -> str:
        """Format a list with optional truncation.

        Args:
            items: List of items to format
            max_items: Maximum items to show before truncating
            item_name: Name for items (singular form)

        Returns:
            Formatted list string
        """
        if not items:
            return "  (none)"

        result = []
        for i, item in enumerate(items[:max_items]):
            result.append(f"  {i + 1}. {item}")

        if len(items) > max_items:
            remaining = len(items) - max_items
            result.append(f"  ... and {remaining} more {item_name}(s)")

        return "\n".join(result)
