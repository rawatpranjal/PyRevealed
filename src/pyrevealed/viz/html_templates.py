"""HTML templates and CSS styles for Jupyter notebook display.

This module provides HTML rendering functions for result dataclasses,
enabling rich display in Jupyter notebooks and HTML-capable environments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


# CSS styles for Jupyter notebook display
RESULT_CSS = """
<style>
.pyrevealed-result {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    font-size: 14px;
    line-height: 1.5;
    max-width: 600px;
    border: 1px solid #e1e4e8;
    border-radius: 6px;
    overflow: hidden;
    margin: 10px 0;
}

.pyrevealed-result .header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 12px 16px;
    font-weight: 600;
    font-size: 15px;
}

.pyrevealed-result .header.status-pass {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
}

.pyrevealed-result .header.status-fail {
    background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
}

.pyrevealed-result .header.status-warn {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
}

.pyrevealed-result .content {
    padding: 16px;
    background: #fafbfc;
}

.pyrevealed-result .metric-row {
    display: flex;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid #e1e4e8;
}

.pyrevealed-result .metric-row:last-child {
    border-bottom: none;
}

.pyrevealed-result .metric-label {
    color: #586069;
}

.pyrevealed-result .metric-value {
    font-weight: 600;
    color: #24292e;
}

.pyrevealed-result .metric-value.good {
    color: #22863a;
}

.pyrevealed-result .metric-value.bad {
    color: #cb2431;
}

.pyrevealed-result .metric-value.neutral {
    color: #6f42c1;
}

.pyrevealed-result .status-indicator {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 600;
    font-size: 12px;
    margin-left: 8px;
}

.pyrevealed-result .status-indicator.pass {
    background: #dcffe4;
    color: #22863a;
}

.pyrevealed-result .status-indicator.fail {
    background: #ffeef0;
    color: #cb2431;
}

.pyrevealed-result .footer {
    padding: 8px 16px;
    background: #f6f8fa;
    color: #586069;
    font-size: 12px;
    border-top: 1px solid #e1e4e8;
}

.pyrevealed-result .interpretation {
    margin-top: 12px;
    padding: 12px;
    background: #fff;
    border-left: 3px solid #0366d6;
    color: #24292e;
    font-size: 13px;
}

.pyrevealed-summary-table {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    border-collapse: collapse;
    width: 100%;
    max-width: 800px;
    margin: 10px 0;
}

.pyrevealed-summary-table th {
    background: #f6f8fa;
    padding: 12px;
    text-align: left;
    border-bottom: 2px solid #e1e4e8;
    font-weight: 600;
}

.pyrevealed-summary-table td {
    padding: 10px 12px;
    border-bottom: 1px solid #e1e4e8;
}

.pyrevealed-summary-table tr:hover {
    background: #f6f8fa;
}

.pyrevealed-summary-table .pass {
    color: #22863a;
    font-weight: 600;
}

.pyrevealed-summary-table .fail {
    color: #cb2431;
    font-weight: 600;
}
</style>
"""


def render_result_html(result: Any) -> str:
    """Render a result object as styled HTML for Jupyter display.

    Args:
        result: A result dataclass with _get_display_name, _get_status_indicator,
                and _get_key_metrics methods (from ResultDisplayMixin)

    Returns:
        HTML string for Jupyter notebook display
    """
    # Get display information from the mixin methods
    if hasattr(result, "_get_display_name"):
        display_name = result._get_display_name()
    else:
        display_name = result.__class__.__name__

    if hasattr(result, "_get_status_indicator"):
        indicator, status_text, css_class = result._get_status_indicator()
    else:
        indicator, status_text, css_class = "[?]", "UNKNOWN", "status-neutral"

    if hasattr(result, "_get_key_metrics"):
        metrics = result._get_key_metrics()
    else:
        metrics = []

    # Build metrics HTML
    metrics_html = ""
    for label, value in metrics:
        value_class = _get_value_class(label, value)
        formatted_value = _format_value(value)
        metrics_html += f"""
        <div class="metric-row">
            <span class="metric-label">{label}</span>
            <span class="metric-value {value_class}">{formatted_value}</span>
        </div>
        """

    # Get interpretation if available
    interpretation_html = ""
    if hasattr(result, "summary"):
        # Extract a brief interpretation from the summary
        summary_text = result.summary()
        interpretation = _extract_interpretation(summary_text)
        if interpretation:
            interpretation_html = f'<div class="interpretation">{interpretation}</div>'

    # Get computation time if available
    footer_html = ""
    if hasattr(result, "computation_time_ms"):
        time_ms = result.computation_time_ms
        if time_ms < 1000:
            time_str = f"{time_ms:.2f} ms"
        else:
            time_str = f"{time_ms / 1000:.2f} s"
        footer_html = f'<div class="footer">Computed in {time_str}</div>'

    # Build the complete HTML
    html = f"""
{RESULT_CSS}
<div class="pyrevealed-result">
    <div class="header {css_class}">
        {display_name}
        <span class="status-indicator {'pass' if 'pass' in css_class else 'fail' if 'fail' in css_class else ''}">{indicator} {status_text}</span>
    </div>
    <div class="content">
        {metrics_html}
        {interpretation_html}
    </div>
    {footer_html}
</div>
"""
    return html


def render_summary_table_html(results: list[tuple[str, Any]]) -> str:
    """Render multiple results as a summary table for Jupyter display.

    Args:
        results: List of (name, result) tuples

    Returns:
        HTML string with a summary table
    """
    rows_html = ""
    for name, result in results:
        # Get status
        if hasattr(result, "_get_status_indicator"):
            indicator, status_text, css_class = result._get_status_indicator()
        elif hasattr(result, "is_consistent"):
            if result.is_consistent:
                indicator, status_text, css_class = "[+]", "PASS", "pass"
            else:
                indicator, status_text, css_class = "[-]", "FAIL", "fail"
        elif hasattr(result, "success"):
            if result.success:
                indicator, status_text, css_class = "[+]", "SUCCESS", "pass"
            else:
                indicator, status_text, css_class = "[-]", "FAILED", "fail"
        else:
            indicator, status_text, css_class = "[?]", "N/A", ""

        # Get score
        if hasattr(result, "score"):
            score_val = result.score() if callable(result.score) else result.score
            score_str = f"{score_val:.4f}"
        else:
            score_str = "N/A"

        # Get computation time
        if hasattr(result, "computation_time_ms"):
            time_ms = result.computation_time_ms
            if time_ms < 1000:
                time_str = f"{time_ms:.1f}ms"
            else:
                time_str = f"{time_ms / 1000:.2f}s"
        else:
            time_str = "N/A"

        status_class = "pass" if "pass" in css_class.lower() else "fail" if "fail" in css_class.lower() else ""

        rows_html += f"""
        <tr>
            <td>{name}</td>
            <td class="{status_class}">{indicator} {status_text}</td>
            <td>{score_str}</td>
            <td>{time_str}</td>
        </tr>
        """

    html = f"""
{RESULT_CSS}
<table class="pyrevealed-summary-table">
    <thead>
        <tr>
            <th>Test</th>
            <th>Status</th>
            <th>Score</th>
            <th>Time</th>
        </tr>
    </thead>
    <tbody>
        {rows_html}
    </tbody>
</table>
"""
    return html


def render_behavioral_summary_html(
    num_observations: int,
    num_goods: int,
    consistency_tests: list[tuple[str, bool]],
    goodness_metrics: list[tuple[str, float]],
    computation_time_ms: float,
) -> str:
    """Render a comprehensive behavioral summary as HTML.

    Args:
        num_observations: Number of observations in the dataset
        num_goods: Number of goods/dimensions
        consistency_tests: List of (test_name, passed) tuples
        goodness_metrics: List of (metric_name, value) tuples
        computation_time_ms: Total computation time

    Returns:
        HTML string for Jupyter display
    """
    # Build consistency tests HTML
    tests_html = ""
    for test_name, passed in consistency_tests:
        indicator = "[+]" if passed else "[-]"
        status = "PASS" if passed else "FAIL"
        status_class = "pass" if passed else "fail"
        tests_html += f"""
        <div class="metric-row">
            <span class="metric-label">{test_name}</span>
            <span class="metric-value {status_class}">{indicator} {status}</span>
        </div>
        """

    # Build goodness metrics HTML
    metrics_html = ""
    for metric_name, value in goodness_metrics:
        value_class = "good" if value >= 0.9 else "neutral" if value >= 0.7 else "bad"
        metrics_html += f"""
        <div class="metric-row">
            <span class="metric-label">{metric_name}</span>
            <span class="metric-value {value_class}">{value:.4f}</span>
        </div>
        """

    # Format time
    if computation_time_ms < 1000:
        time_str = f"{computation_time_ms:.2f} ms"
    else:
        time_str = f"{computation_time_ms / 1000:.2f} s"

    # Determine overall status
    all_passed = all(passed for _, passed in consistency_tests)
    overall_css = "status-pass" if all_passed else "status-fail"
    overall_status = "ALL TESTS PASS" if all_passed else "SOME TESTS FAIL"

    html = f"""
{RESULT_CSS}
<div class="pyrevealed-result" style="max-width: 700px;">
    <div class="header {overall_css}">
        Behavioral Summary
        <span class="status-indicator {'pass' if all_passed else 'fail'}">{overall_status}</span>
    </div>
    <div class="content">
        <div style="margin-bottom: 16px;">
            <strong>Data:</strong>
            <div class="metric-row">
                <span class="metric-label">Observations</span>
                <span class="metric-value">{num_observations}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Goods/Dimensions</span>
                <span class="metric-value">{num_goods}</span>
            </div>
        </div>
        <div style="margin-bottom: 16px;">
            <strong>Consistency Tests:</strong>
            {tests_html}
        </div>
        <div>
            <strong>Goodness-of-Fit:</strong>
            {metrics_html}
        </div>
    </div>
    <div class="footer">Total computation time: {time_str}</div>
</div>
"""
    return html


def _get_value_class(label: str, value: Any) -> str:
    """Determine CSS class for a metric value."""
    label_lower = label.lower()

    if isinstance(value, bool):
        return "good" if value else "bad"

    if isinstance(value, (int, float)):
        # For efficiency/score metrics (higher is better)
        if any(
            term in label_lower
            for term in ["efficiency", "score", "index", "power"]
        ):
            if "waste" in label_lower or "mpi" in label_lower:
                # Lower is better for these
                if value <= 0.05:
                    return "good"
                elif value <= 0.15:
                    return "neutral"
                else:
                    return "bad"
            else:
                # Higher is better
                if value >= 0.9:
                    return "good"
                elif value >= 0.7:
                    return "neutral"
                else:
                    return "bad"

        # For violations (lower is better)
        if "violation" in label_lower:
            return "good" if value == 0 else "bad"

    return "neutral"


def _format_value(value: Any) -> str:
    """Format a value for display."""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    elif isinstance(value, float):
        if abs(value) < 0.0001 and value != 0:
            return f"{value:.4e}"
        elif abs(value) >= 1000:
            return f"{value:,.2f}"
        else:
            return f"{value:.4f}"
    elif value is None:
        return "N/A"
    else:
        return str(value)


def _extract_interpretation(summary_text: str) -> str:
    """Extract interpretation section from a summary string."""
    lines = summary_text.split("\n")

    # Look for interpretation section
    in_interpretation = False
    interpretation_lines = []

    for line in lines:
        if "Interpretation:" in line:
            in_interpretation = True
            continue
        elif in_interpretation:
            if line.strip().startswith("=") or (
                line.strip() and not line.startswith(" ")
            ):
                break
            if line.strip():
                interpretation_lines.append(line.strip())

    if interpretation_lines:
        return " ".join(interpretation_lines[:2])  # First 2 lines

    return ""
