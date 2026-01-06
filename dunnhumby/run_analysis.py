"""Run PyRevealed algorithms on Dunnhumby data (Phase 4).

This module executes the core revealed preference algorithms on all
qualifying households:

1. GARP Check - Binary rationality test (is the user consistent?)
2. Afriat Efficiency Index (AEI) - Efficiency score (0.0 to 1.0)
3. Optional: Utility recovery for consistent households
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pyrevealed import (
    check_garp,
    compute_aei,
    recover_utility,
    GARPResult,
    AEIResult,
    UtilityRecoveryResult,
)

from config import AEI_TOLERANCE, GARP_TOLERANCE, PROGRESS_INTERVAL
from session_builder import HouseholdData


@dataclass
class HouseholdResult:
    """Results for a single household analysis."""

    household_key: int
    num_observations: int
    num_goods: int
    total_spend: float

    # GARP results
    is_garp_consistent: bool
    num_violations: int
    garp_time_ms: float

    # AEI results
    efficiency_index: float
    aei_time_ms: float

    # Utility recovery (only for consistent households)
    utility_recovered: bool = False
    utility_values: Optional[NDArray[np.float64]] = None
    utility_time_ms: float = 0.0


@dataclass
class AnalysisResults:
    """Aggregated results for all households."""

    household_results: Dict[int, HouseholdResult] = field(default_factory=dict)
    total_households: int = 0
    processed_households: int = 0
    consistent_households: int = 0
    total_time_seconds: float = 0.0

    def add_result(self, result: HouseholdResult) -> None:
        """Add a household result and update counters."""
        self.household_results[result.household_key] = result
        self.processed_households += 1
        if result.is_garp_consistent:
            self.consistent_households += 1

    def get_efficiency_scores(self) -> List[float]:
        """Get list of all efficiency scores."""
        return [r.efficiency_index for r in self.household_results.values()]

    def get_summary_stats(self) -> dict:
        """Get summary statistics for the analysis."""
        scores = self.get_efficiency_scores()
        if not scores:
            return {
                "total_households": self.total_households,
                "processed_households": 0,
                "consistent_households": 0,
                "consistency_rate": 0.0,
                "mean_aei": 0.0,
                "median_aei": 0.0,
                "std_aei": 0.0,
                "min_aei": 0.0,
                "max_aei": 0.0,
                "total_time_seconds": self.total_time_seconds,
            }

        return {
            "total_households": self.total_households,
            "processed_households": self.processed_households,
            "consistent_households": self.consistent_households,
            "consistency_rate": self.consistent_households
            / max(self.processed_households, 1),
            "mean_aei": float(np.mean(scores)),
            "median_aei": float(np.median(scores)),
            "std_aei": float(np.std(scores)),
            "min_aei": float(np.min(scores)),
            "max_aei": float(np.max(scores)),
            "total_time_seconds": self.total_time_seconds,
            "households_below_0.7": sum(1 for s in scores if s < 0.7),
            "households_below_0.9": sum(1 for s in scores if s < 0.9),
            "households_perfect_1.0": sum(1 for s in scores if s == 1.0),
        }

    def to_dataframe(self):
        """Convert results to pandas DataFrame for analysis."""
        import pandas as pd

        records = []
        for hh_key, result in self.household_results.items():
            records.append(
                {
                    "household_key": hh_key,
                    "num_observations": result.num_observations,
                    "num_goods": result.num_goods,
                    "total_spend": result.total_spend,
                    "is_garp_consistent": result.is_garp_consistent,
                    "num_violations": result.num_violations,
                    "efficiency_index": result.efficiency_index,
                    "garp_time_ms": result.garp_time_ms,
                    "aei_time_ms": result.aei_time_ms,
                    "utility_recovered": result.utility_recovered,
                }
            )
        return pd.DataFrame(records)


def analyze_household(
    hh_data: HouseholdData,
    recover_utilities: bool = False,
    aei_tolerance: float = AEI_TOLERANCE,
) -> HouseholdResult:
    """
    Run full analysis on a single household.

    Args:
        hh_data: HouseholdData with BehaviorLog
        recover_utilities: Whether to attempt utility recovery for consistent households
        aei_tolerance: Tolerance for AEI computation (looser = faster)

    Returns:
        HouseholdResult with all analysis outcomes
    """
    log = hh_data.behavior_log

    # GARP check (binary consistency test)
    start = time.perf_counter()
    garp_result: GARPResult = check_garp(log, tolerance=GARP_TOLERANCE)
    garp_time = (time.perf_counter() - start) * 1000

    # AEI computation (efficiency score)
    start = time.perf_counter()
    aei_result: AEIResult = compute_aei(log, tolerance=aei_tolerance)
    aei_time = (time.perf_counter() - start) * 1000

    result = HouseholdResult(
        household_key=hh_data.household_key,
        num_observations=log.num_records,
        num_goods=log.num_features,
        total_spend=hh_data.total_spend,
        is_garp_consistent=garp_result.is_consistent,
        num_violations=garp_result.num_violations,
        garp_time_ms=garp_time,
        efficiency_index=aei_result.efficiency_index,
        aei_time_ms=aei_time,
    )

    # Utility recovery for consistent households (optional)
    if recover_utilities and garp_result.is_consistent:
        start = time.perf_counter()
        utility_result: UtilityRecoveryResult = recover_utility(log)
        utility_time = (time.perf_counter() - start) * 1000

        result.utility_recovered = utility_result.success
        if utility_result.success:
            result.utility_values = utility_result.utility_values
        result.utility_time_ms = utility_time

    return result


def run_full_analysis(
    households: Dict[int, HouseholdData],
    recover_utilities: bool = False,
    progress_interval: int = PROGRESS_INTERVAL,
) -> AnalysisResults:
    """
    Run analysis on all households.

    Args:
        households: Dict of household data
        recover_utilities: Whether to recover utilities for consistent households
        progress_interval: Print progress every N households

    Returns:
        AnalysisResults with all outcomes
    """
    results = AnalysisResults()
    results.total_households = len(households)

    overall_start = time.perf_counter()

    for i, (hh_key, hh_data) in enumerate(households.items()):
        try:
            result = analyze_household(hh_data, recover_utilities)
            results.add_result(result)
        except Exception as e:
            # Log error but continue with other households
            print(f"  Warning: Failed to analyze household {hh_key}: {e}")

        # Progress reporting
        if (i + 1) % progress_interval == 0:
            elapsed = time.perf_counter() - overall_start
            rate = (i + 1) / elapsed
            eta = (results.total_households - i - 1) / rate if rate > 0 else 0
            consistent_pct = (
                100 * results.consistent_households / (i + 1) if i > 0 else 0
            )
            print(
                f"  Processed {i + 1}/{results.total_households} "
                f"({elapsed:.1f}s, {consistent_pct:.1f}% consistent, ETA {eta:.1f}s)"
            )

    results.total_time_seconds = time.perf_counter() - overall_start

    return results


def run_quick_analysis(
    households: Dict[int, HouseholdData],
    sample_size: int = 100,
) -> AnalysisResults:
    """
    Run quick analysis on a sample of households (for testing).

    Args:
        households: Dict of household data
        sample_size: Number of households to sample

    Returns:
        AnalysisResults for the sample
    """
    import random

    keys = list(households.keys())
    sample_keys = random.sample(keys, min(sample_size, len(keys)))
    sample_households = {k: households[k] for k in sample_keys}

    print(f"  Running quick analysis on {len(sample_households)} households...")
    return run_full_analysis(sample_households, recover_utilities=False)
