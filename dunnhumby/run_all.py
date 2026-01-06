#!/usr/bin/env python3
"""
Dunnhumby Integration Test Suite for PyRevealed

Master script that runs the full pipeline:
1. Load and filter Dunnhumby transaction data
2. Build master price grid
3. Create BehaviorLog objects for qualifying households
4. Run GARP and AEI analysis
5. Generate visualizations

Usage:
    python dunnhumby/run_all.py
    python -m dunnhumby.run_all

Requirements:
    - Download dataset first: ./dunnhumby/download_data.sh
    - Install dependencies: pip install pandas pyarrow matplotlib

Success Criteria:
    1. Data pipeline produces expected shapes
    2. All qualifying households produce valid BehaviorLog objects
    3. GARP/AEI complete without errors
    4. Performance meets 5-minute target
    5. Visualizations generate successfully
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    MAX_PROCESSING_TIME_SECONDS,
    MIN_SHOPPING_WEEKS,
    OUTPUT_DIR,
)
from data_loader import load_filtered_data, get_data_summary
from price_oracle import get_master_price_grid, get_price_grid_summary, validate_price_grid
from session_builder import build_all_sessions, get_session_summary
from run_analysis import run_full_analysis
from visualize_results import generate_all_visualizations, save_results_csv
from test_results import SimulationResults


def print_banner(text: str, char: str = "=", width: int = 70) -> None:
    """Print a formatted banner."""
    print()
    print(char * width)
    print(f" {text}")
    print(char * width)


def run_dunnhumby_test_suite(
    skip_viz: bool = False,
    quick_mode: bool = False,
) -> bool:
    """
    Run the complete Dunnhumby integration test suite.

    Args:
        skip_viz: If True, skip visualization generation
        quick_mode: If True, only process a sample of households

    Returns:
        True if all tests passed, False otherwise
    """
    print_banner(
        "DUNNHUMBY INTEGRATION TEST SUITE FOR PYREVEALED",
        "=",
        70,
    )
    print(" Dataset: The Complete Journey (~2,500 households, 2 years)")
    print(" Goal: Validate revealed preference algorithms on real consumer data")
    print("=" * 70)

    overall_start = time.perf_counter()
    results = SimulationResults("Dunnhumby Integration Tests")

    # =========================================================================
    # PHASE 1: Data Loading & Filtering
    # =========================================================================
    print_banner("[1/5] DATA INGESTION & FILTERING", "-", 60)

    try:
        filtered_data = load_filtered_data(use_cache=True)
        summary = get_data_summary(filtered_data)

        print(f"\n  Filtered dataset summary:")
        print(f"    Transactions: {summary['n_transactions']:,}")
        print(f"    Households: {summary['n_households']:,}")
        print(f"    Weeks: {summary['n_weeks']}")
        print(f"    Commodities: {summary['n_commodities']}")
        print(f"    Total quantity: {summary['total_quantity']:,.0f}")
        print(f"    Total spend: ${summary['total_spend']:,.2f}")

        results.record("data_loading", True)
    except FileNotFoundError as e:
        results.record("data_loading", False, str(e))
        print("\n  ERROR: Dataset not found. Please run download_data.sh first.")
        return False
    except Exception as e:
        results.record("data_loading", False, str(e))
        return False

    # =========================================================================
    # PHASE 2: Price Oracle Construction
    # =========================================================================
    print_banner("[2/5] PRICE ORACLE CONSTRUCTION", "-", 60)

    try:
        price_grid = get_master_price_grid(filtered_data, use_cache=True)
        pg_summary = get_price_grid_summary(price_grid)

        print(f"\n  Price grid: {pg_summary['shape']}")
        print(f"    Price range: ${pg_summary['min_price']:.2f} - ${pg_summary['max_price']:.2f}")
        print(f"    Mean price: ${pg_summary['mean_price']:.2f}")

        validate_price_grid(price_grid)
        results.record("price_oracle", True)
    except Exception as e:
        results.record("price_oracle", False, str(e))
        return False

    # =========================================================================
    # PHASE 3: Session Building
    # =========================================================================
    print_banner("[3/5] BUILDING BEHAVIORLOG OBJECTS", "-", 60)

    try:
        households = build_all_sessions(
            filtered_data,
            price_grid,
            min_weeks=MIN_SHOPPING_WEEKS,
        )

        session_summary = get_session_summary(households)
        print(f"\n  Session summary:")
        print(f"    Qualifying households: {session_summary['n_households']}")
        print(f"    Total observations: {session_summary['total_observations']:,}")
        print(f"    Observations per household: {session_summary['mean_observations']:.1f} (mean)")
        print(f"    Range: {session_summary['min_observations']} - {session_summary['max_observations']}")

        has_households = session_summary["n_households"] > 0
        results.record(
            "session_building",
            has_households,
            f"Only {session_summary['n_households']} households qualified",
        )

        if not has_households:
            return False

    except Exception as e:
        results.record("session_building", False, str(e))
        return False

    # =========================================================================
    # PHASE 4: Algorithm Execution
    # =========================================================================
    print_banner("[4/5] RUNNING PYREVEALED ALGORITHMS", "-", 60)

    try:
        if quick_mode:
            from run_analysis import run_quick_analysis
            analysis_results = run_quick_analysis(households, sample_size=100)
        else:
            analysis_results = run_full_analysis(
                households,
                recover_utilities=False,  # Skip utility recovery for speed
            )

        stats = analysis_results.get_summary_stats()

        print(f"\n  Analysis complete!")
        print(f"    Processed: {stats['processed_households']:,} households")
        print(f"    GARP-consistent: {stats['consistent_households']:,} ({stats['consistency_rate']:.1%})")
        print(f"    Mean AEI: {stats['mean_aei']:.4f}")
        print(f"    Median AEI: {stats['median_aei']:.4f}")
        print(f"    Std Dev: {stats['std_aei']:.4f}")
        print(f"    Below 0.7 (erratic): {stats['households_below_0.7']:,}")
        print(f"    Perfect 1.0: {stats['households_perfect_1.0']:,}")
        print(f"    Time: {stats['total_time_seconds']:.2f}s")

        results.record("algorithm_execution", True)

        # Performance check
        performance_ok = stats["total_time_seconds"] < MAX_PROCESSING_TIME_SECONDS
        results.record(
            "performance_target",
            performance_ok,
            f"Took {stats['total_time_seconds']:.1f}s (target: <{MAX_PROCESSING_TIME_SECONDS}s)",
        )

    except Exception as e:
        results.record("algorithm_execution", False, str(e))
        return False

    # =========================================================================
    # PHASE 5: Visualization
    # =========================================================================
    print_banner("[5/5] GENERATING VISUALIZATIONS", "-", 60)

    if skip_viz:
        print("  Skipping visualizations (--skip-viz flag)")
        results.record("visualizations", True, "Skipped")
    else:
        try:
            generate_all_visualizations(analysis_results, households, OUTPUT_DIR)
            save_results_csv(analysis_results, OUTPUT_DIR)
            results.record("visualizations", True)
        except ImportError as e:
            results.record("visualizations", False, f"Missing dependency: {e}")
            print(f"  Warning: Could not generate visualizations: {e}")
            print("  Install matplotlib: pip install matplotlib")
        except Exception as e:
            results.record("visualizations", False, str(e))
            print(f"  Warning: Visualization error: {e}")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    overall_time = time.perf_counter() - overall_start

    print_banner("FINAL SUMMARY", "=", 70)

    passed = results.summary()

    print(f"\n  Total time: {overall_time:.2f} seconds")

    if passed:
        print_banner("ALL TESTS PASSED - DUNNHUMBY INTEGRATION VALIDATED", "=", 70)
    else:
        print_banner("SOME TESTS FAILED - CHECK ABOVE FOR DETAILS", "=", 70)

    return passed


def main() -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Dunnhumby Integration Test Suite for PyRevealed"
    )
    parser.add_argument(
        "--skip-viz",
        action="store_true",
        help="Skip visualization generation",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: only process a sample of households",
    )
    args = parser.parse_args()

    success = run_dunnhumby_test_suite(
        skip_viz=args.skip_viz,
        quick_mode=args.quick,
    )
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
