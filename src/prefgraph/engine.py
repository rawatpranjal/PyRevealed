"""Engine: batch revealed preference analysis for millions of users.

Automatically uses the Rust backend (Rayon parallel, thread-local scratchpads)
if installed, otherwise falls back to Python with ProcessPoolExecutor.

Usage:
    from prefgraph.engine import Engine

    engine = Engine(metrics=["garp", "ccei", "mpi", "harp", "hm", "utility", "vei"])
    results = engine.analyze_arrays(user_data)

    # Get the full observation graph (for Tier 2 deep dives)
    graph = engine.build_graph(prices, quantities)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, fields, replace
from typing import Any, Optional

import numpy as np

from prefgraph._rust_backend import (
    HAS_RUST, _rust_analyze_batch, _rust_analyze_menu_batch, _rust_build_preference_graph,
    HAS_PARQUET_RUST, _rust_analyze_parquet_file,
)


@dataclass
class EngineResult:
    """Result for one user from the Engine (budget data).

    Contains all metrics requested via ``Engine(metrics=[...])``.
    Unrequested numeric metrics retain mathematically correct defaults
    (``ccei=1.0``, ``mpi=0.0``). Unrequested boolean/count metrics
    default to ``None`` (not ``False``/``0``) so they render as NaN
    in DataFrames — unambiguously "not computed" rather than "failed".

    Attributes:
        is_garp: True if choices satisfy GARP (no revealed-preference cycles).
        n_violations: Number of GARP violation pairs. 0 when consistent.
        ccei: Critical Cost Efficiency Index (Afriat 1967). 1.0 = perfectly
            rational; lower values indicate wasted budget. Range: (0, 1].
        mpi: Money Pump Index (Echenique, Lee & Shum 2011). Average
            exploitability per dollar. 0.0 = unexploitable. Range: [0, 1).
        is_harp: True if choices satisfy HARP (homothetic preferences).
        hm_consistent: Houtman-Maks: number of consistent observations (budget) or items (menu).
        hm_total: Total observations (budget) or items (menu).
        utility_success: True if Afriat's LP recovered a rationalizing utility.
        vei_mean: Mean Varian Efficiency Index across observations. Range: [0, 1].
        vei_min: Worst-observation VEI. Range: [0, 1].
        vei_exact_mean: VEI via exact LP (vs binary-search approximation).
        vei_exact_min: Exact VEI, worst observation.
        max_scc: Largest strongly connected component in observation graph.
            1 = acyclic (no entangled violations).
        compute_time_us: Wall-clock computation time in microseconds.
    """

    # is_garp is always computed — it gates every other metric.
    is_garp: bool
    n_violations: int = 0
    # ccei=1.0 and mpi=0.0 are mathematically correct defaults:
    # GARP-consistent data has CCEI=1.0 (Afriat 1967) and MPI=0.0
    # (Echenique, Lee & Shum 2011). These are only overwritten when
    # GARP fails AND the metric was requested.
    ccei: float = 1.0
    mpi: float = 0.0
    # Optional metrics default to None (= "not computed"), not False/0.
    # False would be indistinguishable from "computed and failed" in output
    # DataFrames, causing users to misinterpret uncomputed HARP as
    # "not homothetic" or uncomputed utility as "LP failed".
    # None serializes to NaN in pandas, which is unambiguously "missing".
    is_harp: Optional[bool] = None
    hm_consistent: Optional[int] = None
    hm_total: Optional[int] = None
    utility_success: Optional[bool] = None
    vei_mean: float = 1.0
    vei_min: float = 1.0
    vei_exact_mean: float = 1.0
    vei_exact_min: float = 1.0
    max_scc: int = 0
    compute_time_us: int = 0
    vei_std: float = 0.0
    vei_q25: float = 1.0
    vei_q75: float = 1.0
    vei_exact_std: float = 0.0
    vei_exact_q25: float = 1.0
    vei_exact_q75: float = 1.0
    n_scc: int = 0
    harp_severity: float = 1.0
    scc_mean_size: float = 0.0
    r_density: float = 0.0
    r_out_degree_std: float = 0.0
    degree_gini: float = 0.0
    ew_mean: float = 0.0
    ew_std: float = 0.0
    ew_skew: float = 0.0

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
        if self.is_harp is True:
            lines.append("  HARP:  yes (homothetic)")
        if self.hm_total is not None and self.hm_total > 0:
            frac = self.hm_consistent / self.hm_total
            lines.append(f"  HM:    {self.hm_consistent}/{self.hm_total} ({frac:.0%} consistent)")
        if self.utility_success is True:
            lines.append("  Utility: recovered")
        if self.vei_mean < 1.0:
            lines.append(f"  VEI:   mean={self.vei_mean:.4f}  min={self.vei_min:.4f}  std={self.vei_std:.4f}  IQR=[{self.vei_q25:.4f}, {self.vei_q75:.4f}]")
        if self.is_harp is False and self.harp_severity > 1.0:
            lines.append(f"  HARP:  violated (severity={self.harp_severity:.4f})")
        if self.max_scc > 1:
            lines.append(f"  SCC:   {self.n_scc} components, max={self.max_scc}, mean={self.scc_mean_size:.1f}")
        if self.r_density > 0:
            parts = [f"density={self.r_density:.3f}", f"deg_std={self.r_out_degree_std:.2f}", f"gini={self.degree_gini:.3f}"]
            if self.ew_std > 0:
                parts.append(f"ew_std={self.ew_std:.3f}")
            lines.append(f"  Graph: {', '.join(parts)}")
        lines.append(f"  Time:  {self.compute_time_us}us")
        return "\n".join(lines)

    def __repr__(self) -> str:
        indicator = "[+]" if self.is_garp else "[-]"
        status = "GARP-consistent" if self.is_garp else f"{self.n_violations} violations"
        parts = [f"EngineResult: {indicator} {status}"]
        parts.append(f"ccei={self.ccei:.4f}")
        if self.mpi > 0.0:
            parts.append(f"mpi={self.mpi:.4f}")
        if self.hm_total is not None and self.hm_total > 0:
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
        hm_consistent: Houtman-Maks: number of consistent observations (budget) or items (menu).
        hm_total: Total observations (budget) or items (menu).
        max_scc: Largest SCC in the item graph. 1 = acyclic.
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
    n_scc: int = 0
    r_density: float = 0.0
    pref_entropy: float = 0.0
    choice_diversity: float = 0.0
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
            lines.append(f"  SCC:   {self.n_scc} components, max={self.max_scc}")
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

    SUPPORTED_METRICS = {"garp", "ccei", "mpi", "harp", "hm", "utility", "vei", "vei_exact", "network"}

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
        from prefgraph.core.exceptions import DataValidationError, DimensionError

        if not isinstance(users, (list, tuple)):
            raise TypeError(
                f"users must be a list of (prices, quantities) tuples, "
                f"got {type(users).__name__}. "
                f"Hint: Wrap a single user as [(prices, quantities)]. "
                f"If you have a pandas DataFrame, use prefgraph.analyze(df, ...) instead."
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
                    f"or prefgraph.analyze(df, ...) for DataFrames."
                )
            if q.ndim != 2:
                raise DimensionError(
                    f"users[{i}]: quantities must be 2D (T x K), got {q.ndim}D with shape {q.shape}. "
                    f"Hint: Use quantities.reshape(-1, K) for 1D arrays, "
                    f"or prefgraph.analyze(df, ...) for DataFrames."
                )
            if p.shape != q.shape:
                raise DimensionError(
                    f"users[{i}]: prices shape {p.shape} != quantities shape {q.shape}. "
                    f"Both must be (T, K) with matching dimensions."
                )

    @staticmethod
    def _validate_menu_input(users: Any) -> None:
        """Validate input for analyze_menus()."""
        from prefgraph.core.exceptions import DataValidationError

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
            "network": "network" in self.metrics,
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
        """Build an observation graph and return it as numpy arrays.

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
            flags.get("network", False),
            self.tolerance,
        )

        # Rust backend always returns all keys (with False/0 defaults) even
        # when a metric wasn't requested. Use flags to emit None for unrequested
        # optional metrics, so DataFrames show NaN instead of misleading False/0.
        engine_results = [
            EngineResult(
                is_garp=r["is_garp"],
                n_violations=r["n_violations"],
                ccei=r["ccei"],
                mpi=r.get("mpi", 0.0),
                is_harp=r["is_harp"] if flags.get("harp") else None,
                hm_consistent=r["hm_consistent"] if flags.get("hm") else None,
                hm_total=r["hm_total"] if flags.get("hm") else None,
                utility_success=r["utility_success"] if flags.get("utility") else None,
                vei_mean=r.get("vei_mean", 1.0),
                vei_min=r.get("vei_min", 1.0),
                vei_exact_mean=r.get("vei_exact_mean", 1.0),
                vei_exact_min=r.get("vei_exact_min", 1.0),
                max_scc=r["max_scc"],
                compute_time_us=r["compute_time_us"],
                vei_std=r.get("vei_std", 0.0),
                vei_q25=r.get("vei_q25", 1.0),
                vei_q75=r.get("vei_q75", 1.0),
                vei_exact_std=r.get("vei_exact_std", 0.0),
                vei_exact_q25=r.get("vei_exact_q25", 1.0),
                vei_exact_q75=r.get("vei_exact_q75", 1.0),
                n_scc=r.get("n_scc", 0),
                harp_severity=r.get("harp_severity", 1.0),
                scc_mean_size=r.get("scc_mean_size", 0.0),
                r_density=r.get("r_density", 0.0),
                r_out_degree_std=r.get("r_out_degree_std", 0.0),
                degree_gini=r.get("degree_gini", 0.0),
                ew_mean=r.get("ew_mean", 0.0),
                ew_std=r.get("ew_std", 0.0),
                ew_skew=r.get("ew_skew", 0.0),
            )
            for r in raw_results
        ]
        # Post-process VEI: if vei was requested but Rust returned the 1.0 default
        # on GARP-violating data, compute in Python using compute_vei().
        # The Rust backend may not implement VEI or may silently return 1.0
        # (the EngineResult default) when it can't compute.
        if flags.get("vei"):
            from prefgraph.core.session import BehaviorLog as _BL
            from prefgraph.algorithms.vei import compute_vei as _compute_vei
            fixed = []
            for er, (prices, quantities) in zip(engine_results, chunk):
                if not er.is_garp and er.vei_mean == 1.0:
                    try:
                        log = _BL(cost_vectors=prices, action_vectors=quantities)
                        vr = _compute_vei(log)
                        er = replace(er, vei_mean=vr.mean_efficiency, vei_min=vr.min_efficiency)
                    except Exception:
                        pass
                fixed.append(er)
            engine_results = fixed
        return engine_results

    def _analyze_chunk_python(
        self,
        chunk: list[tuple[np.ndarray, np.ndarray]],
        flags: dict[str, bool],
    ) -> list[EngineResult]:
        """Python fallback when Rust backend is unavailable (HAS_RUST=False).

        Supports: GARP, CCEI, MPI, HM, HARP, utility.
        Does NOT support: VEI (requires Rust LP solver).

        Each metric matches the Rust Engine output. Algorithm references:
        - GARP: Varian (1982), SCC + Floyd-Warshall
        - CCEI: Afriat (1967), binary search
        - MPI: Echenique, Lee & Shum (2011), Karp's max-mean cycle
        - HM: Houtman & Maks (1985), exact ILP for T<=100, greedy FVS otherwise
        - HARP: Varian (1983), FW on log-ratios (binary test only, no severity)
        - Utility: Afriat (1967), LP via scipy/HiGHS
        """
        from prefgraph import BehaviorLog, check_garp, compute_aei, compute_mpi
        from prefgraph.algorithms.mpi import compute_houtman_maks_index
        from prefgraph.algorithms.harp import check_harp
        from prefgraph.algorithms.utility import recover_utility

        results = []
        for prices, quantities in chunk:
            log = BehaviorLog(cost_vectors=prices, action_vectors=quantities)

            # GARP: always computed — it's the foundation for all other metrics.
            # Varian (1982): SCC decomposition + Floyd-Warshall transitive closure.
            garp = check_garp(log, self.tolerance)
            ccei_val = 1.0
            mpi_val = 0.0
            # None = "not computed" (distinct from 0 = "computed, zero removals").
            hm_consistent = None
            hm_total = None
            is_harp = None
            utility_success = None

            # CCEI (Afriat Efficiency Index): only computed when GARP fails.
            # Consistent data has CCEI=1.0 by definition — no search needed.
            # Afriat (1967): binary search over e ∈ (0,1] for max e where e-GARP holds.
            if flags.get("ccei") and not garp.is_consistent:
                aei = compute_aei(log, method="discrete")
                ccei_val = aei.efficiency_index

            # MPI (Money Pump Index): only computed when GARP fails.
            # Consistent data has MPI=0.0 — unexploitable.
            # Echenique, Lee & Shum (2011): Karp's O(T^3) max-mean-weight cycle.
            if flags.get("mpi") and not garp.is_consistent:
                mpi_result = compute_mpi(log)
                mpi_val = mpi_result.mpi_value

            # HM (Houtman-Maks): minimum observations to remove for GARP consistency.
            # Houtman & Maks (1985). NP-hard (Smeulders et al. 2014).
            # Uses exact ILP (Demuynck & Rehbeck 2023) for T<=100, greedy FVS above.
            if flags.get("hm"):
                hm_total = prices.shape[0]
                hm_result = compute_houtman_maks_index(log, self.tolerance)
                hm_consistent = hm_total - len(hm_result.removed_observations)

            # HARP: binary test for homothetic preferences.
            # Varian (1983), C&E (2016) Theorem 4.2: (>=^H, >^H) is acyclic.
            # No severity metric exists in the literature — only pass/fail.
            if flags.get("harp"):
                harp_result = check_harp(log, self.tolerance)
                is_harp = harp_result.is_consistent

            # Utility recovery: Afriat LP — find U_t, lambda_t satisfying
            # Afriat's inequalities. Success = data is rationalizable.
            if flags.get("utility"):
                try:
                    util_result = recover_utility(log)
                    utility_success = util_result.success
                except Exception:
                    utility_success = False

            # VEI: per-observation efficiency — Varian (1990) "Goodness-of-fit
            # in optimizing models", J. Econometrics 46(1-2), 125-140.
            # papers/EcheniqueLeeShum2011_MoneyPump.pdf p.7 footnote 3:
            #   "Varian modifies AEI by allowing e to vary across the different
            #    price vectors, looking at a vector (e_k). Varian's measure is
            #    the closest distance to the unit vector (e_k = 1) of a (e_k)
            #    with no violations of GARP."
            # Smeulders et al. (2014), Section 2.2:
            #   "VI equals the vector e that is closest to one, for some given
            #    norm, such that the data satisfies the revealed-preference
            #    axiom under study."
            # Mononen (2023), Section 2.2:
            #   "Varian's index is the least average of adjustments required
            #    to rationalize the data."
            vei_mean_val = 1.0
            vei_min_val = 1.0
            if flags.get("vei") and not garp.is_consistent:
                try:
                    from prefgraph.algorithms.vei import compute_vei
                    vei_result = compute_vei(log)
                    vei_mean_val = vei_result.mean_efficiency
                    vei_min_val = vei_result.min_efficiency
                except Exception:
                    pass  # keep defaults on solver failure

            results.append(EngineResult(
                is_garp=garp.is_consistent,
                n_violations=len(garp.violations),
                ccei=ccei_val,
                mpi=mpi_val,
                hm_consistent=hm_consistent,
                hm_total=hm_total,
                is_harp=is_harp,
                utility_success=utility_success,
                vei_mean=vei_mean_val,
                vei_min=vei_min_val,
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
                    "network" in self.metrics,
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
                        n_scc=r.get("n_scc", 0),
                        r_density=r.get("r_density", 0.0),
                        pref_entropy=r.get("pref_entropy", 0.0),
                        choice_diversity=r.get("choice_diversity", 0.0),
                        compute_time_us=r["compute_time_us"],
                    )
                    for r in raw
                )
            else:
                # Python fallback
                from prefgraph import MenuChoiceLog
                from prefgraph.algorithms.abstract_choice import (
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

    # ------------------------------------------------------------------
    # Parquet streaming
    # ------------------------------------------------------------------

    def analyze_parquet(
        self,
        path: str | Any,
        *,
        user_col: str = "user_id",
        cost_cols: list[str] | None = None,
        action_cols: list[str] | None = None,
        item_col: str | None = None,
        cost_col: str | None = None,
        action_col: str | None = None,
        time_col: str | None = None,
        output_path: str | None = None,
    ) -> Any:
        """Stream-analyze a Parquet file without loading it all into memory.

        Reads row groups incrementally, groups by user, and feeds chunks
        to the Rust engine. Memory stays bounded at O(chunk_size) regardless
        of total dataset size.

        Args:
            path: Path to Parquet file.
            user_col: Column for user identifiers.
            cost_cols: (Wide format) Price column names.
            action_cols: (Wide format) Quantity column names.
            item_col: (Long format) Item identifier column.
            cost_col: (Long format) Price column.
            action_col: (Long format) Quantity column.
            time_col: (Long format) Time/period column.
            output_path: If set, write results incrementally to this Parquet
                file instead of accumulating in memory.

        Returns:
            pandas DataFrame with one row per user (or path to output Parquet
            if ``output_path`` is set).
        """
        # Fast path: wide-format + Rust parquet feature compiled
        if (
            HAS_PARQUET_RUST
            and cost_cols is not None
            and action_cols is not None
            and item_col is None
            and output_path is None
        ):
            return self._analyze_parquet_rust(
                str(path), user_col, cost_cols, action_cols
            )

        # Standard path: PyArrow streaming
        from prefgraph.io.parquet import ParquetUserIterator

        iterator = ParquetUserIterator(
            path,
            user_col=user_col,
            cost_cols=cost_cols,
            action_cols=action_cols,
            item_col=item_col,
            cost_col=cost_col,
            action_col=action_col,
            time_col=time_col,
            chunk_size=self.chunk_size,
        )

        flags = {
            "ccei": "ccei" in self.metrics,
            "mpi": "mpi" in self.metrics,
            "harp": "harp" in self.metrics,
            "hm": "hm" in self.metrics,
            "utility": "utility" in self.metrics,
            "vei": "vei" in self.metrics,
            "vei_exact": "vei_exact" in self.metrics,
            "network": "network" in self.metrics,
        }

        if output_path is not None:
            return self._analyze_parquet_to_file(iterator, flags, output_path)

        all_user_ids: list[str] = []
        all_results: list[EngineResult] = []

        for user_ids, user_tuples in iterator:
            if self.backend == "rust":
                chunk_results = self._analyze_chunk_rust(user_tuples, flags)
            else:
                chunk_results = self._analyze_chunk_python(user_tuples, flags)
            all_user_ids.extend(user_ids)
            all_results.extend(chunk_results)

        return results_to_dataframe(all_results, user_ids=all_user_ids)

    def _analyze_parquet_rust(
        self,
        path: str,
        user_col: str,
        cost_cols: list[str],
        action_cols: list[str],
    ) -> Any:
        """Full Rust pipeline: Parquet I/O + Rayon analysis, no Python overhead."""
        raw_results = _rust_analyze_parquet_file(
            path,
            user_col,
            cost_cols,
            action_cols,
            "ccei" in self.metrics,
            "mpi" in self.metrics,
            "harp" in self.metrics,
            "hm" in self.metrics,
            "utility" in self.metrics,
            "vei" in self.metrics,
            "vei_exact" in self.metrics,
            "network" in self.metrics,
            self.tolerance,
            self.chunk_size,
        )

        user_ids = [uid for uid, _ in raw_results]
        engine_results = [
            EngineResult(
                is_garp=r["is_garp"],
                n_violations=r["n_violations"],
                ccei=r["ccei"],
                mpi=r.get("mpi", 0.0),
                is_harp=r["is_harp"] if "harp" in self.metrics else None,
                hm_consistent=r["hm_consistent"] if "hm" in self.metrics else None,
                hm_total=r["hm_total"] if "hm" in self.metrics else None,
                utility_success=r["utility_success"] if "utility" in self.metrics else None,
                vei_mean=r.get("vei_mean", 1.0),
                vei_min=r.get("vei_min", 1.0),
                vei_exact_mean=r.get("vei_exact_mean", 1.0),
                vei_exact_min=r.get("vei_exact_min", 1.0),
                max_scc=r["max_scc"],
                compute_time_us=r["compute_time_us"],
                vei_std=r.get("vei_std", 0.0),
                vei_q25=r.get("vei_q25", 1.0),
                vei_q75=r.get("vei_q75", 1.0),
                vei_exact_std=r.get("vei_exact_std", 0.0),
                vei_exact_q25=r.get("vei_exact_q25", 1.0),
                vei_exact_q75=r.get("vei_exact_q75", 1.0),
                n_scc=r.get("n_scc", 0),
                harp_severity=r.get("harp_severity", 1.0),
                scc_mean_size=r.get("scc_mean_size", 0.0),
                r_density=r.get("r_density", 0.0),
                r_out_degree_std=r.get("r_out_degree_std", 0.0),
                degree_gini=r.get("degree_gini", 0.0),
                ew_mean=r.get("ew_mean", 0.0),
                ew_std=r.get("ew_std", 0.0),
                ew_skew=r.get("ew_skew", 0.0),
            )
            for _, r in raw_results
        ]

        return results_to_dataframe(engine_results, user_ids=user_ids)

    def _analyze_parquet_to_file(
        self,
        iterator: Any,
        flags: dict[str, bool],
        output_path: str,
    ) -> str:
        """Analyze streaming and write results to Parquet incrementally."""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError(
                "pyarrow is required for Parquet output. "
                "Install with: pip install prefgraph[parquet]"
            ) from None

        writer = None
        total_users = 0

        for user_ids, user_tuples in iterator:
            if self.backend == "rust":
                chunk_results = self._analyze_chunk_rust(user_tuples, flags)
            else:
                chunk_results = self._analyze_chunk_python(user_tuples, flags)

            result_df = results_to_dataframe(chunk_results, user_ids=user_ids)
            result_table = pa.Table.from_pandas(result_df)

            if writer is None:
                writer = pq.ParquetWriter(output_path, result_table.schema,
                                          compression="zstd")
            writer.write_table(result_table)
            total_users += len(user_ids)

        if writer is not None:
            writer.close()

        return output_path

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
