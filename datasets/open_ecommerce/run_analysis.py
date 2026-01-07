"""Algorithm execution for Open E-Commerce 1.0 dataset (Phase 4).

This module runs PyRevealed's revealed preference algorithms on all users
and collects comprehensive statistics.

Algorithms:
- GARP: Binary consistency check (is user rational?)
- AEI: Afriat Efficiency Index (0.0-1.0 rationality score)
- MPI: Money Pump Index (exploitability measure)
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Add parent directories for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pyrevealed import (
    validate_consistency,
    compute_integrity_score,
    compute_confusion_metric,
)

sys.path.insert(0, str(Path(__file__).parent))
from config import CACHE_DIR
from session_builder import UserData


@dataclass
class UserResult:
    """Results for a single user."""

    user_id: str
    is_consistent: bool
    num_violations: int
    aei: float
    mpi: Optional[float]
    total_spend: float
    num_observations: int
    processing_time: float


@dataclass
class AnalysisResults:
    """Aggregated analysis results for all users."""

    user_results: List[UserResult] = field(default_factory=list)
    total_processing_time: float = 0.0

    def add_result(self, result: UserResult) -> None:
        """Add a user result."""
        self.user_results.append(result)

    @property
    def n_users(self) -> int:
        """Number of users analyzed."""
        return len(self.user_results)

    @property
    def n_consistent(self) -> int:
        """Number of GARP-consistent users."""
        return sum(1 for r in self.user_results if r.is_consistent)

    @property
    def n_violations(self) -> int:
        """Number of users with GARP violations."""
        return self.n_users - self.n_consistent

    @property
    def consistency_rate(self) -> float:
        """Percentage of GARP-consistent users."""
        if self.n_users == 0:
            return 0.0
        return 100.0 * self.n_consistent / self.n_users

    @property
    def aei_scores(self) -> List[float]:
        """All AEI scores."""
        return [r.aei for r in self.user_results]

    @property
    def aei_mean(self) -> float:
        """Mean AEI score."""
        return float(np.mean(self.aei_scores)) if self.aei_scores else 0.0

    @property
    def aei_median(self) -> float:
        """Median AEI score."""
        return float(np.median(self.aei_scores)) if self.aei_scores else 0.0

    @property
    def aei_std(self) -> float:
        """Standard deviation of AEI scores."""
        return float(np.std(self.aei_scores)) if self.aei_scores else 0.0

    def count_by_tier(
        self,
        high_threshold: float = 0.95,
        low_threshold: float = 0.70,
    ) -> Dict[str, int]:
        """Count users by rationality tier."""
        high = sum(1 for r in self.user_results if r.aei >= high_threshold)
        low = sum(1 for r in self.user_results if r.aei < low_threshold)
        medium = self.n_users - high - low
        return {
            "high": high,
            "medium": medium,
            "low": low,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        return pd.DataFrame([
            {
                "user_id": r.user_id,
                "is_consistent": r.is_consistent,
                "num_violations": r.num_violations,
                "aei": r.aei,
                "mpi": r.mpi,
                "total_spend": r.total_spend,
                "num_observations": r.num_observations,
                "processing_time": r.processing_time,
            }
            for r in self.user_results
        ])

    def save_to_csv(self, path: Path) -> None:
        """Save results to CSV file."""
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        print(f"  Saved results to: {path}")


def analyze_user(
    user: UserData,
    compute_mpi: bool = False,
) -> UserResult:
    """
    Run all algorithms on a single user.

    Args:
        user: UserData object with BehaviorLog
        compute_mpi: Whether to compute Money Pump Index (slower)

    Returns:
        UserResult with all metrics
    """
    start_time = time.time()

    log = user.behavior_log

    # GARP check
    garp_result = validate_consistency(log)
    is_consistent = garp_result.is_consistent
    num_violations = garp_result.num_violations if hasattr(garp_result, "num_violations") else 0

    # AEI computation
    if is_consistent:
        aei = 1.0
    else:
        aei_result = compute_integrity_score(log)
        aei = aei_result.efficiency_index

    # MPI computation (optional, slower)
    mpi = None
    if compute_mpi:
        try:
            mpi_result = compute_confusion_metric(log)
            mpi = mpi_result.money_pump_index
        except Exception:
            mpi = None

    processing_time = time.time() - start_time

    return UserResult(
        user_id=user.user_id,
        is_consistent=is_consistent,
        num_violations=num_violations,
        aei=aei,
        mpi=mpi,
        total_spend=user.total_spend,
        num_observations=user.num_observations,
        processing_time=processing_time,
    )


def run_analysis(
    users: Dict[str, UserData],
    compute_mpi: bool = False,
    progress_interval: int = 100,
    max_users: Optional[int] = None,
) -> AnalysisResults:
    """
    Run analysis on all users.

    Args:
        users: Dict mapping user_id -> UserData
        compute_mpi: Whether to compute MPI (slower)
        progress_interval: Print progress every N users
        max_users: Maximum number of users to analyze (None = all)

    Returns:
        AnalysisResults with all user results
    """
    results = AnalysisResults()
    start_time = time.time()

    user_ids = list(users.keys())
    if max_users is not None:
        user_ids = user_ids[:max_users]

    total_users = len(user_ids)

    for i, user_id in enumerate(user_ids):
        user = users[user_id]
        result = analyze_user(user, compute_mpi=compute_mpi)
        results.add_result(result)

        if (i + 1) % progress_interval == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (total_users - i - 1) / rate if rate > 0 else 0
            print(f"  Processed {i + 1}/{total_users} users... (ETA: {eta:.0f}s)")

    results.total_processing_time = time.time() - start_time

    return results


def print_analysis_summary(
    results: AnalysisResults,
    high_threshold: float = 0.95,
    low_threshold: float = 0.70,
) -> None:
    """Print formatted analysis summary."""
    print(f"\n  GARP Results:")
    print(f"    Consistent: {results.n_consistent} ({results.consistency_rate:.1f}%)")
    print(f"    Violations: {results.n_violations} ({100 - results.consistency_rate:.1f}%)")

    print(f"\n  AEI Distribution:")
    print(f"    Mean: {results.aei_mean:.4f}")
    print(f"    Median: {results.aei_median:.4f}")
    print(f"    Std: {results.aei_std:.4f}")
    print(f"    Min: {min(results.aei_scores):.4f}")
    print(f"    Max: {max(results.aei_scores):.4f}")

    tiers = results.count_by_tier(high_threshold, low_threshold)
    print(f"\n  Users by tier:")
    print(f"    High (≥{high_threshold}): {tiers['high']} ({100*tiers['high']/results.n_users:.1f}%)")
    print(f"    Medium ({low_threshold}-{high_threshold}): {tiers['medium']} ({100*tiers['medium']/results.n_users:.1f}%)")
    print(f"    Low (<{low_threshold}): {tiers['low']} ({100*tiers['low']/results.n_users:.1f}%)")

    print(f"\n  Processing time: {results.total_processing_time:.1f}s")
    print(f"  Rate: {results.n_users / results.total_processing_time:.1f} users/s")


if __name__ == "__main__":
    from session_builder import load_sessions

    print("Running analysis on Open E-Commerce 1.0 users...")
    users = load_sessions(use_cache=True)

    # Quick test with 100 users
    results = run_analysis(users, max_users=100, progress_interval=25)
    print_analysis_summary(results)
