"""One-liner DataFrame-to-scores API.

Provides ``analyze()`` - feed a pandas DataFrame, get rationality scores back.

Example::

    import prefgraph as rp

    results = rp.analyze(df, user_col="user_id",
                         cost_cols=["price_A", "price_B"],
                         action_cols=["qty_A", "qty_B"])
"""

from __future__ import annotations

import warnings
from pathlib import Path
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


def _check_columns(
    df: Any,
    fmt: str,
    *,
    user_col: str,
    item_col: str | None,
    cost_col: str | None,
    action_col: str | None,
    time_col: str | None,
    cost_cols: list[str] | None,
    action_cols: list[str] | None,
    menu_col: str | None,
    choice_col: str | None,
) -> None:
    """Validate that all referenced columns exist in the DataFrame."""
    available = set(df.columns)

    def _check(col_name: str | None, param: str, is_default: bool = False) -> None:
        if col_name is not None and col_name not in available:
            default_hint = (
                f" (defaulting to '{col_name}' - set {param}= explicitly)"
                if is_default else ""
            )
            # Suggest close matches
            close = [c for c in sorted(available)
                     if col_name.lower() in c.lower() or c.lower() in col_name.lower()]
            suggestion = f" Similar: {close}." if close else ""
            raise ValueError(
                f"Column '{col_name}'{default_hint} not found. "
                f"Available columns: {sorted(available)}.{suggestion}"
            )

    def _check_list(cols: list[str] | None, param: str) -> None:
        if cols is not None:
            missing = [c for c in cols if c not in available]
            if missing:
                raise ValueError(
                    f"Columns {missing} (from {param}=) not found. "
                    f"Available columns: {sorted(available)}"
                )

    if fmt == "wide":
        _check_list(cost_cols, "cost_cols")
        _check_list(action_cols, "action_cols")
    elif fmt == "long":
        _check(item_col, "item_col")
        _check(cost_col or "price", "cost_col", is_default=cost_col is None)
        _check(action_col or "quantity", "action_col", is_default=action_col is None)
        _check(time_col or "time", "time_col", is_default=time_col is None)
    elif fmt == "menu":
        _check(menu_col or "menu", "menu_col", is_default=menu_col is None)
        _check(choice_col or "choice", "choice_col", is_default=choice_col is None)


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
    nan_policy: Literal["raise", "warn", "drop"] = "raise",
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
        nan_policy: How to handle NaN/Inf values. ``"raise"`` (default) raises
            an error. ``"drop"`` silently removes affected rows. ``"warn"``
            drops with a warning.
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
    # --- Parquet file path shortcut ---
    if isinstance(df, (str, Path)):
        path = Path(df)
        if path.suffix == ".parquet" or path.is_dir():
            from prefgraph.engine import Engine
            engine = Engine(metrics=metrics or _DEFAULT_BUDGET_METRICS)
            result = engine.analyze_parquet(
                path,
                user_col=user_col,
                cost_cols=cost_cols or price_cols,
                action_cols=action_cols or qty_cols,
                item_col=item_col,
                cost_col=cost_col or price_col,
                action_col=action_col or qty_col,
                time_col=time_col,
            )
            if output == "objects":
                return list(result.iterrows())
            return result

    # --- Validate input type ---
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for analyze(). "
            "Install with: pip install pandas"
        ) from None

    if not isinstance(df, pd.DataFrame):
        hint = ""
        if isinstance(df, pd.Series):
            hint = " To convert a Series: pd.DataFrame(series)."
        elif isinstance(df, dict):
            hint = " To convert a dict: pd.DataFrame(your_dict)."
        elif hasattr(df, 'shape'):  # numpy array
            hint = " To convert a numpy array: pd.DataFrame(array, columns=[...])."
        raise TypeError(
            f"First argument must be a pandas DataFrame, got {type(df).__name__}.{hint}"
        )
    if len(df) == 0:
        raise ValueError("DataFrame is empty (0 rows). Nothing to analyze.")

    # --- Validate output parameter ---
    if output not in ("dataframe", "objects"):
        raise ValueError(
            f"output must be 'dataframe' or 'objects', got '{output}'."
        )

    # --- Resolve legacy aliases ---
    cost_col = cost_col or price_col
    action_col = action_col or qty_col
    cost_cols = cost_cols or price_cols
    action_cols = action_cols or qty_cols

    # --- Catch string-instead-of-list mistake ---
    for param_name, param_val in [("cost_cols", cost_cols), ("action_cols", action_cols),
                                   ("price_cols", price_cols), ("qty_cols", qty_cols)]:
        if isinstance(param_val, str):
            raise TypeError(
                f"{param_name} must be a list of column names, got a string '{param_val}'. "
                f"Use {param_name}=['{param_val}'] instead."
            )

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

    # --- Validate columns exist ---
    available = list(df.columns)
    if user_col not in available:
        # Try to suggest the closest match
        close = [c for c in available if "user" in c.lower() or "id" in c.lower()
                 or "customer" in c.lower() or "household" in c.lower()]
        suggestion = f" Did you mean: {close}?" if close else ""
        raise ValueError(
            f"Column '{user_col}' not found in DataFrame. "
            f"Available columns: {available}.{suggestion}"
        )
    _check_columns(df, fmt, user_col=user_col, item_col=item_col,
                    cost_col=cost_col, action_col=action_col, time_col=time_col,
                    cost_cols=cost_cols, action_cols=action_cols,
                    menu_col=menu_col, choice_col=choice_col)

    # --- Dispatch by format ---
    if fmt == "wide":
        user_ids, results = _analyze_wide(
            df, user_col=user_col,
            cost_cols=cost_cols, action_cols=action_cols,
            metrics=metrics, nan_policy=nan_policy,
        )
    elif fmt == "long":
        user_ids, results = _analyze_long(
            df, user_col=user_col,
            item_col=item_col, cost_col=cost_col,
            action_col=action_col, time_col=time_col,
            metrics=metrics, nan_policy=nan_policy,
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

    from prefgraph.engine import results_to_dataframe
    return results_to_dataframe(results, user_ids=user_ids)


def _handle_nan(
    df: Any,
    data_cols: list[str],
    nan_policy: str,
) -> Any:
    """Handle NaN/Inf values in data columns before analysis."""
    import numpy as np

    subset = df[data_cols]
    has_problem = subset.isnull().any(axis=1) | subset.isin([np.inf, -np.inf]).any(axis=1)

    if not has_problem.any():
        return df

    n_bad = has_problem.sum()
    if nan_policy == "raise":
        raise ValueError(
            f"Found {n_bad} rows with NaN or Inf values in columns {data_cols}. "
            f"Options: set nan_policy='drop' to remove them, "
            f"nan_policy='warn' to drop with a warning, "
            f"or clean your data first with df.dropna(subset={data_cols})."
        )
    elif nan_policy == "warn":
        warnings.warn(
            f"Dropping {n_bad} rows with NaN/Inf values in {data_cols}.",
            stacklevel=4,
        )
    # drop or warn: filter out bad rows
    return df[~has_problem].copy()


def _analyze_wide(
    df: Any,
    *,
    user_col: str,
    cost_cols: list[str] | None,
    action_cols: list[str] | None,
    metrics: list[str] | None,
    nan_policy: str = "raise",
) -> tuple[list[str], list]:
    """Wide format: one row per observation, items as columns."""
    from prefgraph.core.panel import BehaviorPanel
    from prefgraph.engine import Engine

    if cost_cols is None:
        raise ValueError("Wide format requires cost_cols (list of column names for prices).")
    if action_cols is None:
        raise ValueError("Wide format requires action_cols (list of column names for quantities).")

    df = _handle_nan(df, cost_cols + action_cols, nan_policy)

    try:
        panel = BehaviorPanel.from_dataframe(
            df, user_col=user_col, cost_cols=cost_cols, action_cols=action_cols
        )
    except (ValueError, TypeError) as e:
        if "could not convert" in str(e).lower():
            raise ValueError(
                f"Non-numeric data in cost or action columns. "
                f"Ensure all values in {cost_cols} and {action_cols} are numeric. "
                f"Tip: df[cols].dtypes to check types, pd.to_numeric(df[col], errors='coerce') to convert."
            ) from None
        raise
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
    nan_policy: str = "raise",
) -> tuple[list[str], list]:
    """Long format: one row per item per time per user."""
    from prefgraph.core.session import BehaviorLog
    from prefgraph.engine import Engine

    if item_col is None:
        raise ValueError("Long format requires item_col.")
    _cost = cost_col or "price"
    _action = action_col or "quantity"
    _time = time_col or "time"

    df = _handle_nan(df, [_cost, _action], nan_policy)

    user_ids: list[str] = []
    tuples: list[tuple] = []

    for uid, group in df.groupby(user_col, sort=True):
        uid_str = str(uid)
        try:
            log = BehaviorLog.from_long_format(
                group,
                time_col=_time,
                item_col=item_col,
                cost_col=_cost,
                action_col=_action,
                user_id=uid_str,
            )
        except ValueError as e:
            if "duplicate" in str(e).lower() or "reshape" in str(e).lower():
                raise ValueError(
                    f"User '{uid_str}': duplicate (time, item) pairs found. "
                    f"Each ({_time}, {item_col}) combination must be unique per user. "
                    f"Aggregate duplicates first: "
                    f"df.groupby(['{user_col}','{_time}','{item_col}']).agg(...)."
                ) from None
            raise
        except Exception as e:
            if "could not convert" in str(e).lower():
                raise ValueError(
                    f"User '{uid_str}': non-numeric data in cost or action columns. "
                    f"Ensure '{_cost}' and '{_action}' contain numeric values."
                ) from None
            raise
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
    from prefgraph.core.panel import MenuChoicePanel
    from prefgraph.engine import Engine

    _menu = menu_col or "menu"
    _choice = choice_col or "choice"

    try:
        panel = MenuChoicePanel.from_dataframe(
            df, user_col=user_col, menu_col=_menu, choice_col=_choice
        )
    except TypeError as e:
        raise TypeError(
            f"Menu choice data requires integer item indices. "
            f"If your items are strings, map them to integers first: "
            f"item_map = {{v: i for i, v in enumerate(all_items)}}; "
            f"df['{_menu}'] = df['{_menu}'].apply(lambda m: [item_map[x] for x in m])"
        ) from None

    tuples = [log.to_engine_tuple() for _, log in panel]
    engine = Engine()  # metrics param only applies to budget data
    results = engine.analyze_menus(tuples)
    return panel.user_ids, results
