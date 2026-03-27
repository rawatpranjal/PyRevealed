"""Panel containers for multi-user behavioral analysis.

Provides BehaviorPanel and MenuChoicePanel for managing collections of
BehaviorLog/MenuChoiceLog objects indexed by user ID, with optional
time-period support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pyrevealed.core.session import BehaviorLog, MenuChoiceLog
    from pyrevealed.core.summary import BehavioralSummary, PanelSummary


@dataclass
class BehaviorPanel:
    """Multi-user panel of BehaviorLog objects.

    Holds a collection of BehaviorLog objects indexed by user_id,
    supporting iteration, filtering, and aggregate analysis.

    Attributes:
        _logs: Internal dict mapping user_id -> BehaviorLog
        metadata: Optional metadata for the panel (e.g. dataset name)

    Example:
        >>> import numpy as np
        >>> from pyrevealed import BehaviorLog, BehaviorPanel
        >>> logs = [
        ...     BehaviorLog(np.random.rand(20,5), np.random.rand(20,5), user_id=f"u{i}")
        ...     for i in range(10)
        ... ]
        >>> panel = BehaviorPanel.from_logs(logs)
        >>> print(f"{panel.num_users} users, {sum(l.num_observations for _, l in panel)} obs")
        >>> print(panel.summary())
    """

    _logs: dict[str, "BehaviorLog"] = field(repr=False)
    metadata: dict[str, Any] = field(default_factory=dict)
    _period_map: dict[str, tuple[str, str]] | None = field(default=None, repr=False)
    """Maps panel key -> (user_id, period). None if no period structure."""

    # --- Construction ---

    @classmethod
    def from_logs(cls, logs: list["BehaviorLog"]) -> "BehaviorPanel":
        """Create panel from a list of BehaviorLog objects.

        Uses each log's user_id as the key. Logs without user_id get
        auto-assigned "user_0", "user_1", etc.
        """
        log_dict: dict[str, "BehaviorLog"] = {}
        auto_idx = 0
        for log in logs:
            uid = log.user_id
            if uid is None:
                uid = f"user_{auto_idx}"
                auto_idx += 1
            if uid in log_dict:
                raise ValueError(f"Duplicate user_id: {uid}")
            log_dict[uid] = log
        return cls(_logs=log_dict)

    @classmethod
    def from_dict(cls, logs: dict[str, "BehaviorLog"]) -> "BehaviorPanel":
        """Create panel from a dict of user_id -> BehaviorLog."""
        return cls(_logs=dict(logs))

    @classmethod
    def from_dataframe(
        cls,
        df: Any,
        user_col: str,
        cost_cols: list[str] | None = None,
        action_cols: list[str] | None = None,
        price_cols: list[str] | None = None,
        qty_cols: list[str] | None = None,
        period_col: str | None = None,
    ) -> "BehaviorPanel":
        """Create panel from a pandas DataFrame.

        Groups by user_col (and optionally period_col) and creates
        one BehaviorLog per group.

        Args:
            df: pandas DataFrame
            user_col: Column name for user/household IDs
            cost_cols: Column names for cost/price vectors
            action_cols: Column names for action/quantity vectors
            price_cols: Alias for cost_cols (backward compat)
            qty_cols: Alias for action_cols (backward compat)
            period_col: Optional column to group by time period.
                If provided, user_ids become "user_id__period".

        Returns:
            BehaviorPanel with one BehaviorLog per user (or user-period).
        """
        try:
            import pandas as pd  # noqa: F401
        except ImportError:
            raise ImportError(
                "pandas is required for from_dataframe(). "
                "Install with: pip install 'pyrevealed[datasets]'"
            ) from None

        from pyrevealed.core.session import BehaviorLog

        c_cols = cost_cols or price_cols
        a_cols = action_cols or qty_cols
        if c_cols is None or a_cols is None:
            raise ValueError("Must provide cost_cols/action_cols (or price_cols/qty_cols)")

        group_cols = [user_col]
        if period_col is not None:
            group_cols.append(period_col)

        log_dict: dict[str, BehaviorLog] = {}
        period_map: dict[str, tuple[str, str]] | None = None
        if period_col is not None:
            period_map = {}

        for keys, group in df.groupby(group_cols, sort=True):
            if isinstance(keys, tuple):
                uid = "__".join(str(k) for k in keys)
                if period_map is not None:
                    period_map[uid] = (str(keys[0]), str(keys[1]))
            else:
                uid = str(keys)

            costs = group[c_cols].values.astype(np.float64)
            actions = group[a_cols].values.astype(np.float64)
            log_dict[uid] = BehaviorLog(
                cost_vectors=costs,
                action_vectors=actions,
                user_id=uid,
            )

        return cls(_logs=log_dict, _period_map=period_map)

    # --- Access ---

    @property
    def user_ids(self) -> list[str]:
        """List of user IDs in the panel."""
        return list(self._logs.keys())

    @property
    def num_users(self) -> int:
        """Number of unique users in the panel."""
        if self._period_map is not None:
            return len(set(u for u, _ in self._period_map.values()))
        return len(self._logs)

    @property
    def has_periods(self) -> bool:
        """True if this panel has period structure."""
        return self._period_map is not None and len(self._period_map) > 0

    @property
    def periods(self) -> list[str]:
        """List of unique periods (empty if no period structure)."""
        if self._period_map is None:
            return []
        return sorted(set(p for _, p in self._period_map.values()))

    @property
    def num_entries(self) -> int:
        """Number of entries (user-period combinations or just users)."""
        return len(self._logs)

    def get_period(self, period: str) -> "BehaviorPanel":
        """Return panel filtered to a single period."""
        if self._period_map is None:
            raise ValueError("Panel has no period structure")
        filtered = {
            k: log for k, log in self._logs.items()
            if self._period_map.get(k, ("", ""))[1] == period
        }
        return BehaviorPanel(_logs=filtered, metadata=dict(self.metadata))

    def __getitem__(self, user_id: str) -> "BehaviorLog":
        """Get a BehaviorLog by user_id."""
        return self._logs[user_id]

    def __contains__(self, user_id: str) -> bool:
        return user_id in self._logs

    def __iter__(self) -> Iterator[tuple[str, "BehaviorLog"]]:
        """Iterate over (user_id, BehaviorLog) pairs."""
        return iter(self._logs.items())

    def __len__(self) -> int:
        return len(self._logs)

    def __repr__(self) -> str:
        total_obs = sum(log.num_observations for log in self._logs.values())
        return f"BehaviorPanel(users={self.num_users}, total_obs={total_obs})"

    # --- Analysis ---

    def analyze_user(self, user_id: str) -> "BehavioralSummary":
        """Run full behavioral analysis on a single user."""
        from pyrevealed.core.summary import BehavioralSummary
        return BehavioralSummary.from_log(self._logs[user_id])

    def summary(
        self,
        include_warp: bool = True,
        include_sarp: bool = True,
        include_power: bool = False,
    ) -> "PanelSummary":
        """Run analysis on all users and return aggregate PanelSummary."""
        from pyrevealed.core.summary import BehavioralSummary, PanelSummary

        user_summaries: dict[str, BehavioralSummary] = {}
        for uid, log in self._logs.items():
            user_summaries[uid] = BehavioralSummary.from_log(
                log,
                include_warp=include_warp,
                include_sarp=include_sarp,
                include_power=include_power,
            )

        return PanelSummary.from_summaries(user_summaries, period_map=self._period_map)

    def filter(self, predicate: Callable[["BehaviorLog"], bool]) -> "BehaviorPanel":
        """Return a new panel containing only logs that satisfy predicate."""
        filtered = {uid: log for uid, log in self._logs.items() if predicate(log)}
        return BehaviorPanel(_logs=filtered, metadata=dict(self.metadata))

    def to_engine_tuples(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Convert all logs to Engine-compatible format.

        Returns:
            List of (prices, quantities) tuples for Engine.analyze_arrays().

        Example:
            >>> engine = Engine(metrics=["garp", "ccei"])
            >>> results = engine.analyze_arrays(panel.to_engine_tuples())
        """
        return [log.to_engine_tuple() for log in self._logs.values()]


@dataclass
class MenuChoicePanel:
    """Multi-user panel of MenuChoiceLog objects.

    Same pattern as BehaviorPanel but for menu-based choice data.
    """

    _logs: dict[str, "MenuChoiceLog"] = field(repr=False)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_logs(cls, logs: list["MenuChoiceLog"]) -> "MenuChoicePanel":
        """Create panel from a list of MenuChoiceLog objects."""
        log_dict: dict[str, "MenuChoiceLog"] = {}
        auto_idx = 0
        for log in logs:
            uid = log.user_id if hasattr(log, 'user_id') and log.user_id else f"user_{auto_idx}"
            if uid == f"user_{auto_idx}" or log.user_id is None:
                auto_idx += 1
            if uid in log_dict:
                raise ValueError(f"Duplicate user_id: {uid}")
            log_dict[uid] = log
        return cls(_logs=log_dict)

    @classmethod
    def from_dict(cls, logs: dict[str, "MenuChoiceLog"]) -> "MenuChoicePanel":
        """Create panel from a dict of user_id -> MenuChoiceLog."""
        return cls(_logs=dict(logs))

    @property
    def user_ids(self) -> list[str]:
        return list(self._logs.keys())

    @property
    def num_users(self) -> int:
        return len(self._logs)

    def __getitem__(self, user_id: str) -> "MenuChoiceLog":
        return self._logs[user_id]

    def __contains__(self, user_id: str) -> bool:
        return user_id in self._logs

    def __iter__(self) -> Iterator[tuple[str, "MenuChoiceLog"]]:
        return iter(self._logs.items())

    def __len__(self) -> int:
        return len(self._logs)

    def __repr__(self) -> str:
        total_obs = sum(log.num_observations for log in self._logs.values())
        return f"MenuChoicePanel(users={self.num_users}, total_obs={total_obs})"

    def to_engine_tuples(self) -> list[tuple[list[list[int]], list[int], int]]:
        """Convert all logs to Engine-compatible format.

        Returns:
            List of (menus, choices, n_items) tuples for Engine.analyze_menus().
        """
        return [log.to_engine_tuple() for log in self._logs.values()]

    def filter(self, predicate: Callable[["MenuChoiceLog"], bool]) -> "MenuChoicePanel":
        filtered = {uid: log for uid, log in self._logs.items() if predicate(log)}
        return MenuChoicePanel(_logs=filtered, metadata=dict(self.metadata))
