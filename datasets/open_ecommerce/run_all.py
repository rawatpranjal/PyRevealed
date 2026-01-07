#!/usr/bin/env python3
"""Open E-Commerce 1.0 - PyRevealed Validation Suite.

This is the master orchestrator that runs the complete 5-phase validation
pipeline on the Open E-Commerce 1.0 dataset from Harvard Dataverse.

Usage:
    python run_all.py              # Full run (all users)
    python run_all.py --quick      # Quick run (500 users sample)
    python run_all.py --skip-viz   # Skip visualization generation

Phases:
    1. Data Ingestion & Filtering
    2. Price Oracle Construction
    3. Building BehaviorLog Objects
    4. Running PyRevealed Algorithms
    5. Generating Visualizations
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add parent directories for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATA_FILE,
    HIGH_RATIONALITY_THRESHOLD,
    LOW_RATIONALITY_THRESHOLD,
    MAX_PROCESSING_TIME_SECONDS,
    OUTPUT_DIR,
)


def print_banner(text: str, char: str = "=", width: int = 60) -> None:
    """Print a banner with the given text."""
    print(char * width)
    print(text)
    print(char * width)


def print_phase(phase_num: int, title: str) -> None:
    """Print a phase header."""
    print(f"\n[{phase_num}/5] {title}")
    print("-" * 50)


def run_suite(
    quick_mode: bool = False,
    skip_viz: bool = False,
    max_users: int | None = None,
) -> bool:
    """
    Run the complete validation suite.

    Args:
        quick_mode: If True, run on a sample of 500 users
        skip_viz: If True, skip visualization generation
        max_users: Maximum users to analyze (overrides quick_mode)

    Returns:
        True if all phases passed, False otherwise
    """
    suite_start = time.time()

    print_banner("Open E-Commerce 1.0 - PyRevealed Validation Suite")

    if quick_mode:
        print("\n[Quick mode: analyzing 500 users sample]")
        if max_users is None:
            max_users = 500

    # =========================================================================
    # Phase 1: Data Ingestion & Filtering
    # =========================================================================
    print_phase(1, "DATA INGESTION & FILTERING")

    try:
        from data_loader import load_filtered_data, get_data_summary

        if not DATA_FILE.exists():
            print(f"  Error: Data file not found: {DATA_FILE}")
            print("  Please run download.py first.")
            return False

        filtered_data = load_filtered_data(use_cache=True)
        summary = get_data_summary(filtered_data)

        print(f"\n  Loaded: {summary['n_transactions']:,} transactions")
        print(f"  Users: {summary['n_users']:,}")
        print(f"  Periods: {summary['n_periods']} (months)")
        print(f"  Categories: {summary['n_categories']}")
        print(f"  Date range: {summary['date_range']}")
        print(f"  Total spend: ${summary['total_spend']:,.2f}")

    except Exception as e:
        print(f"  Error in data loading: {e}")
        return False

    # =========================================================================
    # Phase 2: Price Oracle Construction
    # =========================================================================
    print_phase(2, "PRICE ORACLE CONSTRUCTION")

    try:
        from price_oracle import build_price_grid, print_price_grid_summary

        price_grid, periods, categories = build_price_grid(filtered_data, use_cache=True)
        print_price_grid_summary(price_grid, periods, categories)

    except Exception as e:
        print(f"  Error in price oracle: {e}")
        return False

    # =========================================================================
    # Phase 3: Building BehaviorLog Objects
    # =========================================================================
    print_phase(3, "BUILDING BEHAVIORLOG OBJECTS")

    try:
        from session_builder import build_all_sessions, get_session_summary
        import numpy as np

        users = build_all_sessions(filtered_data, price_grid, categories, periods)
        session_summary = get_session_summary(users)

        print(f"\n  Qualifying users: {session_summary['n_users']:,}")
        print(f"  Total observations: {session_summary['total_observations']:,}")
        print(f"  Mean per user: {session_summary['mean_observations']:.1f}")
        print(f"  Range: {session_summary['min_observations']} - {session_summary['max_observations']}")

        if session_summary['n_users'] == 0:
            print("  Error: No qualifying users found!")
            return False

    except Exception as e:
        print(f"  Error in session building: {e}")
        import traceback
        traceback.print_exc()
        return False

    # =========================================================================
    # Phase 4: Running PyRevealed Algorithms
    # =========================================================================
    print_phase(4, "RUNNING PYREVEALED ALGORITHMS")

    try:
        from run_analysis import run_analysis, print_analysis_summary

        analysis_start = time.time()
        results = run_analysis(
            users,
            compute_mpi=False,
            progress_interval=250,
            max_users=max_users,
        )
        analysis_time = time.time() - analysis_start

        print_analysis_summary(results, HIGH_RATIONALITY_THRESHOLD, LOW_RATIONALITY_THRESHOLD)

        # Performance check
        if analysis_time > MAX_PROCESSING_TIME_SECONDS:
            print(f"\n  Warning: Processing took {analysis_time:.0f}s (target: {MAX_PROCESSING_TIME_SECONDS}s)")
        else:
            print(f"\n  Performance: {analysis_time:.1f}s (target: <{MAX_PROCESSING_TIME_SECONDS}s)")

        # Save results to CSV
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        results.save_to_csv(OUTPUT_DIR / "user_results.csv")

    except Exception as e:
        print(f"  Error in analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

    # =========================================================================
    # Phase 5: Generating Visualizations
    # =========================================================================
    print_phase(5, "GENERATING VISUALIZATIONS")

    if skip_viz:
        print("  Skipped (--skip-viz flag)")
    else:
        try:
            from visualize_results import generate_all_visualizations

            n_viz = generate_all_visualizations(results, OUTPUT_DIR)

            if n_viz > 0:
                print(f"\n  Generated {n_viz} visualizations in {OUTPUT_DIR}")
            else:
                print("  No visualizations generated (matplotlib may not be installed)")

        except Exception as e:
            print(f"  Error in visualization: {e}")
            # Don't fail the suite for visualization errors

    # =========================================================================
    # Summary
    # =========================================================================
    suite_time = time.time() - suite_start

    print_banner("Validation Complete!")
    print(f"\nTotal time: {suite_time:.1f}s")
    print(f"Users analyzed: {results.n_users:,}")
    print(f"GARP consistency rate: {results.consistency_rate:.1f}%")
    print(f"Mean AEI: {results.aei_mean:.4f}")
    print(f"\nResults saved to: {OUTPUT_DIR}")

    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Open E-Commerce 1.0 PyRevealed Validation Suite"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: analyze 500 users sample",
    )
    parser.add_argument(
        "--skip-viz",
        action="store_true",
        help="Skip visualization generation",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=None,
        help="Maximum number of users to analyze",
    )
    args = parser.parse_args()

    success = run_suite(
        quick_mode=args.quick,
        skip_viz=args.skip_viz,
        max_users=args.max_users,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
