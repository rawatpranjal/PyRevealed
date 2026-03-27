"""One-liner DataFrame-to-scores API.

Provides ``analyze()`` — feed a pandas DataFrame, get rationality scores back.

Example::

    import pyrevealed as rp

    results = rp.analyze(df, user_col="user_id",
                         cost_cols=["price_A", "price_B"],
                         action_cols=["qty_A", "qty_B"])
"""

from __future__ import annotations

import warnings
from typing import Any, Literal


_DEFAULT_BUDGET_METRICS = ["garp", "ccei", "mpi"]


def _detect_format(
    *,
    item_col: str | None,
    cost_col: str | None,
    action_col: str | None,
    time_col: str | None,
    cost_cols: list[str] | None,
    action_cols: list[str] | None,
    menu_col: str | None,
    choice_col: str | None,
) -> Literal["long", "wide", "menu"]:
    """Detect input format from provided parameters."""
    has_long = item_col is not None
    has_wide = cost_cols is not None or action_cols is not None
    has_menu = menu_col is not None or choice_col is not None

    active = sum([has_long, has_wide, has_menu])

    if active > 1:
        parts = []
        if has_long:
            parts.append("long-format (item_col)")
        if has_wide:
            parts.append("wide-format (cost_cols/action_cols)")
        if has_menu:
            parts.append("menu (menu_col/choice_col)")
        raise ValueError(
            f"Conflicting format parameters: {', '.join(parts)}. "
            f"Provide parameters for exactly one format."
        )

    if has_long:
        return "long"
    if has_wide:
        return "wide"
    if has_menu:
        return "menu"

    raise ValueError(
        "Cannot detect data format. Provide parameters for one of:\n\n"
        "  Wide format (one row per observation, items as columns):\n"
        "    rp.analyze(df, cost_cols=['p1','p2'], action_cols=['q1','q2'])\n\n"
        "  Long format (one row per item per time):\n"
        "    rp.analyze(df, item_col='product', cost_col='price', "
        "action_col='quantity', time_col='week')\n\n"
        "  Menu choice (one row per observation):\n"
        "    rp.analyze(df, menu_col='shown_items', choice_col='clicked')"
    )


def analyze(
    df: Any,
    *,
    user_col: str = "user_id",
    # Long format (transaction logs)
    item_col: str | None = None,
    cost_col: str | None = None,
    action_col: str | None = None,
    time_col: str | None = None,
    # Wide format (pivoted)
    cost_cols: list[str] | None = None,
    action_cols: list[str] | None = None,
    # Menu choice
    menu_col: str | None = None,
    choice_col: str | None = None,
    # Options
    metrics: list[str] | None = None,
    output: Literal["dataframe", "objects"] = "dataframe",
    # Legacy aliases
    price_col: str | None = None,
    qty_col: str | None = None,
    price_cols: list[str] | None = None,
    qty_cols: list[str] | None = None,
) -> Any:
    """Score rationality of choices in a pandas DataFrame.

    Auto-detects whether your data is wide-format, long-format (transaction
    logs), or menu-choice based on which parameters you provide.

    Args:
        df: pandas DataFrame containing choice data.
        user_col: Column name for user/household IDs (default ``"user_id"``).
        item_col: (Long format) Column for item/product identifiers.
        cost_col: (Long format) Column for prices/costs.
        action_col: (Long format) Column for quantities/actions.
        time_col: (Long format) Column for time/observation identifiers.
        cost_cols: (Wide format) List of column names for cost vectors.
        action_cols: (Wide format) List of column names for action vectors.
        menu_col: (Menu) Column containing sets/lists of available items.
        choice_col: (Menu) Column containing the chosen item.
        metrics: Engine metrics to compute. Default ``["garp", "ccei", "mpi"]``
            for budget data. Ignored for menu data (always SARP/WARP/HM).
        output: ``"dataframe"`` (default) returns a pandas DataFrame with one
            row per user. ``"objects"`` returns a list of EngineResult/MenuResult.
        price_col: Alias for ``cost_col``.
        qty_col: Alias for ``action_col``.
        price_cols: Alias for ``cost_cols``.
        qty_cols: Alias for ``action_cols``.

    Returns:
        pandas DataFrame (default) or list of result objects.

    Examples:
        Wide format::

            results = rp.analyze(df,
                cost_cols=["price_A", "price_B"],
                action_cols=["qty_A", "qty_B"])

        Long format (transaction logs)::

            results = rp.analyze(df,
                item_col="product", cost_col="price",
                action_col="quantity", time_col="week")

        Menu choice::

            results = rp.analyze(df,
                menu_col="shown_items", choice_col="clicked")
    """
    # --- Resolve legacy aliases ---
    cost_col = cost_col or price_col
    action_col = action_col or qty_col
    cost_cols = cost_cols or price_cols
    action_cols = action_cols or qty_cols

    # --- Detect format ---
    fmt = _detect_format(
        item_col=item_col,
        cost_col=cost_col,
        action_col=action_col,
        time_col=time_col,
        cost_cols=cost_cols,
        action_cols=action_cols,
        menu_col=menu_col,
        choice_col=choice_col,
    )

    # --- Validate user_col exists ---
    available = list(df.columns)
    if user_col not in available:
        raise ValueError(
            f"Column '{user_col}' not found in DataFrame. "
            f"Available columns: {available}. "
            f"Set user_col= to the column containing user/household IDs."
        )

    # --- Dispatch by format ---
    if fmt == "wide":
        user_ids, results = _analyze_wide(
            df, user_col=user_col,
            cost_cols=cost_cols, action_cols=action_cols,
            metrics=metrics,
        )
    elif fmt == "long":
        user_ids, results = _analyze_long(
            df, user_col=user_col,
            item_col=item_col, cost_col=cost_col,
            action_col=action_col, time_col=time_col,
            metrics=metrics,
        )
    else:  # menu
        if metrics is not None:
            warnings.warn(
                "metrics parameter is ignored for menu choice data. "
                "Engine.analyze_menus() always computes SARP/WARP/HM.",
                stacklevel=2,
            )
        user_ids, results = _analyze_menu(
            df, user_col=user_col,
            menu_col=menu_col, choice_col=choice_col,
        )

    # --- Return ---
    if output == "objects":
        return list(zip(user_ids, results))

    from pyrevealed.engine import results_to_dataframe
    return results_to_dataframe(results, user_ids=user_ids)


def _analyze_wide(
    df: Any,
    *,
    user_col: str,
    cost_cols: list[str] | None,
    action_cols: list[str] | None,
    metrics: list[str] | None,
) -> tuple[list[str], list]:
    """Wide format: one row per observation, items as columns."""
    from pyrevealed.core.panel import BehaviorPanel
    from pyrevealed.engine import Engine

    if cost_cols is None:
        raise ValueError("Wide format requires cost_cols (list of column names for prices).")
    if action_cols is None:
        raise ValueError("Wide format requires action_cols (list of column names for quantities).")

    panel = BehaviorPanel.from_dataframe(
        df, user_col=user_col, cost_cols=cost_cols, action_cols=action_cols
    )
    engine = Engine(metrics=metrics or _DEFAULT_BUDGET_METRICS)
    results = engine.analyze_arrays(panel.to_engine_tuples())
    return panel.user_ids, results


def _analyze_long(
    df: Any,
    *,
    user_col: str,
    item_col: str | None,
    cost_col: str | None,
    action_col: str | None,
    time_col: str | None,
    metrics: list[str] | None,
) -> tuple[list[str], list]:
    """Long format: one row per item per time per user."""
    from pyrevealed.core.session import BehaviorLog
    from pyrevealed.engine import Engine

    if item_col is None:
        raise ValueError("Long format requires item_col.")
    _cost = cost_col or "price"
    _action = action_col or "quantity"
    _time = time_col or "time"

    user_ids: list[str] = []
    tuples: list[tuple] = []

    for uid, group in df.groupby(user_col, sort=True):
        uid_str = str(uid)
        log = BehaviorLog.from_long_format(
            group,
            time_col=_time,
            item_col=item_col,
            cost_col=_cost,
            action_col=_action,
            user_id=uid_str,
        )
        user_ids.append(uid_str)
        tuples.append(log.to_engine_tuple())

    engine = Engine(metrics=metrics or _DEFAULT_BUDGET_METRICS)
    results = engine.analyze_arrays(tuples)
    return user_ids, results


def _analyze_menu(
    df: Any,
    *,
    user_col: str,
    menu_col: str | None,
    choice_col: str | None,
) -> tuple[list[str], list]:
    """Menu choice: sets of items and which was chosen."""
    from pyrevealed.core.panel import MenuChoicePanel
    from pyrevealed.engine import Engine

    _menu = menu_col or "menu"
    _choice = choice_col or "choice"

    panel = MenuChoicePanel.from_dataframe(
        df, user_col=user_col, menu_col=_menu, choice_col=_choice
    )

    tuples = [log.to_engine_tuple() for _, log in panel]
    engine = Engine()  # metrics param only applies to budget data
    results = engine.analyze_menus(tuples)
    return panel.user_ids, results
