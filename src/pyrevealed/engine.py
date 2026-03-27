"""Engine: batch revealed preference analysis for millions of users.

Automatically uses the Rust backend (Rayon parallel, thread-local scratchpads)
if installed, otherwise falls back to Python with ProcessPoolExecutor.

Usage:
    from pyrevealed.engine import Engine

    engine = Engine(metrics=["garp", "ccei", "mpi", "harp", "hm", "utility", "vei"])
    results = engine.analyze_arrays(user_data)

    # Get the full preference graph (for Tier 2 deep dives)
    graph = engine.build_graph(prices, quantities)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, fields
from typing import Any, Optional

import numpy as np

from pyrevealed._rust_backend import (
    HAS_RUST, _rust_analyze_batch, _rust_analyze_menu_batch, _rust_build_preference_graph,
)


@dataclass
class EngineResult:
    """Result for one user from the Engine (budget data).

    Contains all metrics requested via ``Engine(metrics=[...])``. Fields
    for unrequested metrics retain their defaults (e.g. ``ccei=1.0``).

    Attributes:
        is_garp: True if choices satisfy GARP (no revealed-preference cycles).
        n_violations: Number of GARP violation pairs. 0 when consistent.
        ccei: Critical Cost Efficiency Index (Afriat 1967). 1.0 = perfectly
            rational; lower values indicate wasted budget. Range: (0, 1].
        mpi: Money Pump Index (Echenique, Lee & Shum 2011). Average
            exploitability per dollar. 0.0 = unexploitable. Range: [0, 1).
        is_harp: True if choices satisfy HARP (homothetic preferences).
        hm_consistent: Houtman-Maks: size of largest GARP-consistent subset.
        hm_total: Total observations. ``hm_consistent / hm_total`` = noise fraction.
        utility_success: True if Afriat's LP recovered a rationalizing utility.
        vei_mean: Mean Varian Efficiency Index across observations. Range: [0, 1].
        vei_min: Worst-observation VEI. Range: [0, 1].
        vei_exact_mean: VEI via exact LP (vs binary-search approximation).
        vei_exact_min: Exact VEI, worst observation.
        max_scc: Largest strongly connected component in preference graph.
            1 = acyclic (no entangled violations).
        compute_time_us: Wall-clock computation time in microseconds.
    """

    is_garp: bool
    n_violations: int = 0
    ccei: float = 1.0
    mpi: float = 0.0
    is_harp: bool = False
    hm_consistent: int = 0
    hm_total: int = 0
    utility_success: bool = False
    vei_mean: float = 1.0
    vei_min: float = 1.0
    vei_exact_mean: float = 1.0
    vei_exact_min: float = 1.0
    max_scc: int = 0
    compute_time_us: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def summary(self) -> str:
        """Return human-readable summary report."""
        indicator = "[+]" if self.is_garp else "[-]"
        status = "GARP-consistent" if self.is_garp else f"{self.n_violations} GARP violations"
        lines = [f"Engine Budget Report: {indicator} {status}"]
        lines.append(f"  CCEI:  {self.ccei:.4f}")
        if self.mpi > 0.0:
            lines.append(f"  MPI:   {self.mpi:.4f}")
        if self.is_harp:
            lines.append("  HARP:  yes (homothetic)")
        if self.hm_total > 0:
            frac = self.hm_consistent / self.hm_total
            lines.append(f"  HM:    {self.hm_consistent}/{self.hm_total} ({frac:.0%} consistent)")
        if self.utility_success:
            lines.append("  Utility: recovered")
        if self.vei_mean < 1.0:
            lines.append(f"  VEI:   mean={self.vei_mean:.4f}  min={self.vei_min:.4f}")
        if self.max_scc > 1:
            lines.append(f"  SCC:   {self.max_scc} (entangled violations)")
        lines.append(f"  Time:  {self.compute_time_us}us")
        return "\n".join(lines)

    def __repr__(self) -> str:
        indicator = "[+]" if self.is_garp else "[-]"
        status = "GARP-consistent" if self.is_garp else f"{self.n_violations} violations"
        parts = [f"EngineResult: {indicator} {status}"]
        parts.append(f"ccei={self.ccei:.4f}")
        if self.mpi > 0.0:
            parts.append(f"mpi={self.mpi:.4f}")
        if self.hm_total > 0:
            parts.append(f"hm={self.hm_consistent}/{self.hm_total}")
        parts.append(f"({self.compute_time_us}us)")
        return "  ".join(parts)


@dataclass
class MenuResult:
    """Result for one user from menu/discrete choice analysis.

    Contains SARP, WARP, and optionally WARP-LA consistency tests plus
    Houtman-Maks noise fraction for discrete (menu-based) choice data.

    Attributes:
        is_sarp: True if choices satisfy SARP (no preference cycles of any length).
        is_warp: True if choices satisfy WARP (no direct preference reversals).
        is_warp_la: True if choices satisfy WARP under limited attention
            (Masatlioglu, Nakajima & Ozbay 2012). Only computed when
            ``compute_warp_la=True``.
        n_sarp_violations: Number of SARP violation cycles found.
        n_warp_violations: Number of WARP violation pairs found.
        hm_consistent: Houtman-Maks: size of largest SARP-consistent subset.
        hm_total: Total observations. ``hm_consistent / hm_total`` = consistency fraction.
        max_scc: Largest SCC in the item preference graph. 1 = acyclic.
        compute_time_us: Wall-clock computation time in microseconds.
    """

    is_sarp: bool
    is_warp: bool
    is_warp_la: bool = False
    n_sarp_violations: int = 0
    n_warp_violations: int = 0
    hm_consistent: int = 0
    hm_total: int = 0
    max_scc: int = 0
    compute_time_us: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def summary(self) -> str:
        """Return human-readable summary report."""
        indicator = "[+]" if self.is_sarp else "[-]"
        status = "SARP-consistent" if self.is_sarp else f"{self.n_sarp_violations} SARP violations"
        lines = [f"Engine Menu Report: {indicator} {status}"]
        if not self.is_warp:
            lines.append(f"  WARP:  {self.n_warp_violations} violations")
        else:
            lines.append("  WARP:  consistent")
        if self.is_warp_la:
            lines.append("  WARP-LA: consistent (limited attention)")
        if self.hm_total > 0:
            frac = self.hm_consistent / self.hm_total
            lines.append(f"  HM:    {self.hm_consistent}/{self.hm_total} ({frac:.0%} consistent)")
        if self.max_scc > 1:
            lines.append(f"  SCC:   {self.max_scc}")
        lines.append(f"  Time:  {self.compute_time_us}us")
        return "\n".join(lines)

    def __repr__(self) -> str:
        indicator = "[+]" if self.is_sarp else "[-]"
        status = "SARP-consistent" if self.is_sarp else f"{self.n_sarp_violations} SARP violations"
        parts = [f"MenuResult: {indicator} {status}"]
        if self.hm_total > 0:
            parts.append(f"hm={self.hm_consistent}/{self.hm_total}")
        parts.append(f"({self.compute_time_us}us)")
        return "  ".join(parts)


class Engine:
    """Analyzes revealed preference for millions of users.

    Automatically routes to Rust (if available) or Python backend.

    Args:
        metrics: Which metrics to compute. "garp" is always included.
            Supported: "garp", "ccei", "mpi", "harp", "hm", "utility", "vei".
        chunk_size: Number of users per batch (for streaming / memory bounding).
        tolerance: Numerical tolerance for GARP comparisons.
    """

    SUPPORTED_METRICS = {"garp", "ccei", "mpi", "harp", "hm", "utility", "vei", "vei_exact"}

    def __init__(
        self,
        metrics: tuple[str, ...] | list[str] = ("garp", "ccei"),
        chunk_size: int = 50_000,
        tolerance: float = 1e-10,
    ):
        unknown = set(metrics) - self.SUPPORTED_METRICS
        if unknown:
            raise ValueError(
                f"Unknown metrics: {sorted(unknown)}. "
                f"Supported: {sorted(self.SUPPORTED_METRICS)}"
            )
        self.metrics = list(metrics)
        self.chunk_size = chunk_size
        self.tolerance = tolerance
        self.backend = "rust" if HAS_RUST else "python"

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_budget_input(users: Any) -> None:
        """Validate input for analyze_arrays()."""
        from pyrevealed.core.exceptions import DataValidationError, DimensionError

        if not isinstance(users, (list, tuple)):
            raise TypeError(
                f"users must be a list of (prices, quantities) tuples, "
                f"got {type(users).__name__}. "
                f"Hint: Wrap a single user as [(prices, quantities)]. "
                f"If you have a pandas DataFrame, use pyrevealed.analyze(df, ...) instead."
            )
        if len(users) == 0:
            raise DataValidationError("users list is empty. Provide at least one (prices, quantities) tuple.")

        for i, item in enumerate(users):
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                length_hint = f" of length {len(item)}" if hasattr(item, '__len__') else ""
                raise DataValidationError(
                    f"users[{i}] must be a (prices, quantities) tuple of length 2, "
                    f"got {type(item).__name__}{length_hint}. "
                    f"Hint: Each element is (np.ndarray T*K, np.ndarray T*K)."
                )
            p, q = item
            if not isinstance(p, np.ndarray) or not isinstance(q, np.ndarray):
                raise TypeError(
                    f"users[{i}]: prices and quantities must be numpy arrays, "
                    f"got ({type(p).__name__}, {type(q).__name__}). "
                    f"Hint: Convert with np.asarray(data)."
                )
            if p.ndim != 2:
                raise DimensionError(
                    f"users[{i}]: prices must be 2D (T x K), got {p.ndim}D with shape {p.shape}. "
                    f"Hint: Use prices.reshape(-1, K) for 1D arrays, "
                    f"or pyrevealed.analyze(df, ...) for DataFrames."
                )
            if q.ndim != 2:
                raise DimensionError(
                    f"users[{i}]: quantities must be 2D (T x K), got {q.ndim}D with shape {q.shape}. "
                    f"Hint: Use quantities.reshape(-1, K) for 1D arrays, "
                    f"or pyrevealed.analyze(df, ...) for DataFrames."
                )
            if p.shape != q.shape:
                raise DimensionError(
                    f"users[{i}]: prices shape {p.shape} != quantities shape {q.shape}. "
                    f"Both must be (T, K) with matching dimensions."
                )

    @staticmethod
    def _validate_menu_input(users: Any) -> None:
        """Validate input for analyze_menus()."""
        from pyrevealed.core.exceptions import DataValidationError

        if not isinstance(users, (list, tuple)):
            raise TypeError(
                f"users must be a list of (menus, choices, n_items) tuples, "
                f"got {type(users).__name__}."
            )
        if len(users) == 0:
            raise DataValidationError("users list is empty. Provide at least one (menus, choices, n_items) tuple.")

        for i, item in enumerate(users):
            if not isinstance(item, (list, tuple)) or len(item) != 3:
                length_hint = f" of length {len(item)}" if hasattr(item, '__len__') else ""
                raise DataValidationError(
                    f"users[{i}] must be a (menus, choices, n_items) tuple of length 3, "
                    f"got {type(item).__name__}{length_hint}. "
                    f"Hint: Each element is (list[list[int]], list[int], int)."
                )
            menus, choices, n_items = item
            if not isinstance(menus, (list, tuple)):
                raise TypeError(f"users[{i}]: menus must be a list of lists, got {type(menus).__name__}.")
            if not isinstance(choices, (list, tuple)):
                raise TypeError(f"users[{i}]: choices must be a list, got {type(choices).__name__}.")
            if not isinstance(n_items, int) or n_items < 1:
                raise DataValidationError(f"users[{i}]: n_items must be a positive integer, got {n_items!r}.")
            if len(menus) != len(choices):
                raise DataValidationError(
                    f"users[{i}]: len(menus)={len(menus)} != len(choices)={len(choices)}. "
                    f"Each menu observation must have exactly one choice."
                )
            for j, (menu, choice) in enumerate(zip(menus, choices)):
                if choice not in menu:
                    raise DataValidationError(
                        f"users[{i}], observation {j}: choice {choice} not in menu {menu}."
                    )

    # ------------------------------------------------------------------
    # Budget analysis
    # ------------------------------------------------------------------

    def analyze_arrays(
        self,
        users: list[tuple[np.ndarray, np.ndarray]],
        data_type: str = "budget",
    ) -> list[EngineResult]:
        """Analyze users from a list of array pairs.

        Args:
            users: For budget data: list of (prices T*K, quantities T*K).
            data_type: "budget" (default). "menu" and "production" not yet implemented.

        Returns list of EngineResult, one per user.
        """
        if data_type != "budget":
            raise NotImplementedError(
                f"data_type='{data_type}' not yet implemented. "
                "Only 'budget' is currently supported."
            )

        self._validate_budget_input(users)

        n = len(users)
        all_results: list[EngineResult] = []
        flags = {
            "ccei": "ccei" in self.metrics,
            "mpi": "mpi" in self.metrics,
            "harp": "harp" in self.metrics,
            "hm": "hm" in self.metrics,
            "utility": "utility" in self.metrics,
            "vei": "vei" in self.metrics,
            "vei_exact": "vei_exact" in self.metrics,
        }

        for start in range(0, n, self.chunk_size):
            end = min(start + self.chunk_size, n)
            chunk = users[start:end]

            if self.backend == "rust":
                chunk_results = self._analyze_chunk_rust(chunk, flags)
            else:
                chunk_results = self._analyze_chunk_python(chunk, flags)

            all_results.extend(chunk_results)

        return all_results

    def build_graph(
        self,
        prices: np.ndarray,
        quantities: np.ndarray,
        tolerance: float | None = None,
    ) -> dict:
        """Build a preference graph and return it as numpy arrays.

        Tier 2 entry point for deep per-user analysis. Python modules
        (utility.py, welfare.py, etc.) can consume the Rust-computed graph.

        Returns dict with keys:
            r, p, r_star: T*T uint8 arrays (boolean preference matrices)
            expenditure: T*T float64 (expenditure matrix E)
            edge_weights: T*T float64 (log-ratios for HARP)
            own_expenditure: T float64 (diagonal of E)
            scc_labels: T uint32 (SCC component IDs)
            is_garp, n_violations, max_scc, n_components, t: scalars
        """
        if not HAS_RUST:
            raise RuntimeError(
                "build_graph requires the Rust backend. "
                "Install with: pip install rpt-python"
            )
        tol = tolerance if tolerance is not None else self.tolerance
        return _rust_build_preference_graph(
            np.ascontiguousarray(prices, dtype=np.float64),
            np.ascontiguousarray(quantities, dtype=np.float64),
            tol,
        )

    def _analyze_chunk_rust(
        self,
        chunk: list[tuple[np.ndarray, np.ndarray]],
        flags: dict[str, bool],
    ) -> list[EngineResult]:
        """Analyze a chunk using Rust Rayon backend."""
        prices_list = [
            np.ascontiguousarray(p, dtype=np.float64) for p, _ in chunk
        ]
        quantities_list = [
            np.ascontiguousarray(q, dtype=np.float64) for _, q in chunk
        ]

        raw_results = _rust_analyze_batch(
            prices_list, quantities_list,
            flags.get("ccei", False),
            flags.get("mpi", False),
            flags.get("harp", False),
            flags.get("hm", False),
            flags.get("utility", False),
            flags.get("vei", False),
            flags.get("vei_exact", False),
            self.tolerance,
        )

        return [
            EngineResult(
                is_garp=r["is_garp"],
                n_violations=r["n_violations"],
                ccei=r["ccei"],
                mpi=r.get("mpi", 0.0),
                is_harp=r.get("is_harp", False),
                hm_consistent=r.get("hm_consistent", 0),
                hm_total=r.get("hm_total", 0),
                utility_success=r.get("utility_success", False),
                vei_mean=r.get("vei_mean", 1.0),
                vei_min=r.get("vei_min", 1.0),
                vei_exact_mean=r.get("vei_exact_mean", 1.0),
                vei_exact_min=r.get("vei_exact_min", 1.0),
                max_scc=r["max_scc"],
                compute_time_us=r["compute_time_us"],
            )
            for r in raw_results
        ]

    def _analyze_chunk_python(
        self,
        chunk: list[tuple[np.ndarray, np.ndarray]],
        flags: dict[str, bool],
    ) -> list[EngineResult]:
        """Fallback: analyze using Python backend."""
        from pyrevealed import BehaviorLog, check_garp, compute_aei, compute_mpi

        results = []
        for prices, quantities in chunk:
            log = BehaviorLog(cost_vectors=prices, action_vectors=quantities)

            garp = check_garp(log, self.tolerance)
            ccei_val = 1.0
            mpi_val = 0.0

            if flags.get("ccei") and not garp.is_consistent:
                aei = compute_aei(log, method="discrete")
                ccei_val = aei.efficiency_index

            if flags.get("mpi") and not garp.is_consistent:
                mpi_result = compute_mpi(log)
                mpi_val = mpi_result.mpi_value

            results.append(EngineResult(
                is_garp=garp.is_consistent,
                n_violations=len(garp.violations),
                ccei=ccei_val,
                mpi=mpi_val,
            ))
        return results

    # ------------------------------------------------------------------
    # Menu analysis
    # ------------------------------------------------------------------

    def analyze_menus(
        self,
        users: list[tuple[list[list[int]], list[int], int]],
        compute_warp_la: bool = False,
    ) -> list[MenuResult]:
        """Analyze discrete/menu choice data for multiple users.

        Each user tuple: (menus, choices, n_items) where:
        - menus: list of menus, each a list of item indices shown
        - choices: list of chosen item index per menu
        - n_items: total number of distinct items for this user

        Returns list of MenuResult with SARP, WARP, HM scores.

        Example (rec/search click data):
            users = [
                ([[0,1,2,3], [1,2,4], [0,3,4]], [2, 1, 0], 5),  # user 0
                ([[0,1], [1,2], [0,2]], [0, 1, 2], 3),            # user 1
            ]
            results = engine.analyze_menus(users)
        """
        self._validate_menu_input(users)

        n = len(users)
        all_results: list[MenuResult] = []

        for start in range(0, n, self.chunk_size):
            end = min(start + self.chunk_size, n)
            chunk = users[start:end]

            if self.backend == "rust" and _rust_analyze_menu_batch is not None:
                menus_list = [u[0] for u in chunk]
                choices_list = [u[1] for u in chunk]
                n_items_list = [u[2] for u in chunk]

                raw = _rust_analyze_menu_batch(
                    menus_list, choices_list, n_items_list, compute_warp_la,
                )
                all_results.extend(
                    MenuResult(
                        is_sarp=r["is_sarp"],
                        is_warp=r["is_warp"],
                        is_warp_la=r.get("is_warp_la", False),
                        n_sarp_violations=r["n_sarp_violations"],
                        n_warp_violations=r["n_warp_violations"],
                        hm_consistent=r["hm_consistent"],
                        hm_total=r["hm_total"],
                        max_scc=r["max_scc"],
                        compute_time_us=r["compute_time_us"],
                    )
                    for r in raw
                )
            else:
                # Python fallback
                from pyrevealed import MenuChoiceLog
                from pyrevealed.algorithms.abstract_choice import (
                    validate_menu_sarp, validate_menu_warp, compute_menu_efficiency,
                )
                for menus, choices, n_items in chunk:
                    log = MenuChoiceLog(
                        menus=[frozenset(m) for m in menus],
                        choices=choices,
                    )
                    sarp = validate_menu_sarp(log)
                    warp = validate_menu_warp(log)
                    hm = compute_menu_efficiency(log)
                    all_results.append(MenuResult(
                        is_sarp=sarp.is_consistent,
                        is_warp=warp.is_consistent,
                        n_sarp_violations=len(sarp.violations),
                        n_warp_violations=len(warp.violations),
                        hm_consistent=len(hm.remaining_observations),
                        hm_total=hm.num_total,
                    ))

        return all_results

    def __repr__(self) -> str:
        return (f"Engine(backend={self.backend!r}, "
                f"metrics={self.metrics}, chunk_size={self.chunk_size})")


# ------------------------------------------------------------------
# DataFrame conversion
# ------------------------------------------------------------------

def results_to_dataframe(
    results: list[EngineResult] | list[MenuResult],
    user_ids: list[str] | None = None,
) -> Any:
    """Convert Engine results to a pandas DataFrame.

    Args:
        results: List of EngineResult or MenuResult from Engine.
        user_ids: Optional user ID labels for the index.

    Returns:
        pandas.DataFrame with one row per user.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for results_to_dataframe(). "
            "Install with: pip install pandas"
        ) from None

    rows = [r.to_dict() for r in results]
    df = pd.DataFrame(rows)
    if user_ids is not None:
        df.index = user_ids
        df.index.name = "user_id"
    return df
