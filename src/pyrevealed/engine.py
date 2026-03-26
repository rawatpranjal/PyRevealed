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
from dataclasses import dataclass
from typing import Optional

import numpy as np

from pyrevealed._rust_backend import (
    HAS_RUST, _rust_analyze_batch, _rust_analyze_menu_batch, _rust_build_preference_graph,
)


@dataclass
class EngineResult:
    """Result for one user from the Engine (budget data)."""
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


@dataclass
class MenuResult:
    """Result for one user from menu/discrete choice analysis."""
    is_sarp: bool
    is_warp: bool
    is_warp_la: bool = False
    n_sarp_violations: int = 0
    n_warp_violations: int = 0
    hm_consistent: int = 0
    hm_total: int = 0
    max_scc: int = 0
    compute_time_us: int = 0


class Engine:
    """Analyzes revealed preference for millions of users.

    Automatically routes to Rust (if available) or Python backend.

    Args:
        metrics: Which metrics to compute. "garp" is always included.
            Supported: "garp", "ccei", "mpi", "harp", "hm", "utility", "vei".
        chunk_size: Number of users per batch (for streaming / memory bounding).
        tolerance: Numerical tolerance for GARP comparisons.
    """

    SUPPORTED_METRICS = {"garp", "ccei", "mpi", "harp", "hm", "utility", "vei"}

    def __init__(
        self,
        metrics: tuple[str, ...] | list[str] = ("garp", "ccei"),
        chunk_size: int = 50_000,
        tolerance: float = 1e-10,
    ):
        self.metrics = list(metrics)
        self.chunk_size = chunk_size
        self.tolerance = tolerance
        self.backend = "rust" if HAS_RUST else "python"

    def analyze_arrays(
        self,
        users: list[tuple[np.ndarray, np.ndarray]],
        data_type: str = "budget",
    ) -> list[EngineResult]:
        """Analyze users from a list of array pairs.

        Args:
            users: For budget data: list of (prices T×K, quantities T×K).
            data_type: "budget" (default). "menu" and "production" not yet implemented.

        Returns list of EngineResult, one per user.
        """
        if data_type != "budget":
            raise NotImplementedError(
                f"data_type='{data_type}' not yet implemented. "
                "Only 'budget' is currently supported."
            )

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
