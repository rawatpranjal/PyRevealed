#!/usr/bin/env python3
"""
Run All Simulation Studies

Master script that runs all validation simulations for pyrevealed.

Validates:
1. GARP consistency detection
2. Afriat Efficiency Index (AEI) computation
3. Money Pump Index (MPI) calculation
4. Utility recovery via Afriat inequalities

Reference: Chambers & Echenique, "Revealed Preference Theory"
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_all_simulations() -> bool:
    """Run all simulation studies and report results."""

    print("=" * 80)
    print(" PYREVEALED VALIDATION SIMULATION SUITE")
    print(" Reference: Chambers & Echenique, 'Revealed Preference Theory'")
    print("=" * 80)
    print()

    start_time = time.time()

    results = {}

    # Test 1: GARP Consistency
    print("\n" + "=" * 80)
    print(" [1/4] GARP CONSISTENCY TESTS")
    print("=" * 80)
    try:
        from sim.test_garp_consistency import run_all_tests as run_garp_tests
        results["GARP"] = run_garp_tests()
    except Exception as e:
        print(f"ERROR: {e}")
        results["GARP"] = False

    # Test 2: AEI Accuracy
    print("\n" + "=" * 80)
    print(" [2/4] AFRIAT EFFICIENCY INDEX (AEI) TESTS")
    print("=" * 80)
    try:
        from sim.test_aei_accuracy import run_all_tests as run_aei_tests
        results["AEI"] = run_aei_tests()
    except Exception as e:
        print(f"ERROR: {e}")
        results["AEI"] = False

    # Test 3: MPI Calculation
    print("\n" + "=" * 80)
    print(" [3/4] MONEY PUMP INDEX (MPI) TESTS")
    print("=" * 80)
    try:
        from sim.test_mpi_calculation import run_all_tests as run_mpi_tests
        results["MPI"] = run_mpi_tests()
    except Exception as e:
        print(f"ERROR: {e}")
        results["MPI"] = False

    # Test 4: Utility Recovery
    print("\n" + "=" * 80)
    print(" [4/4] UTILITY RECOVERY TESTS")
    print("=" * 80)
    try:
        from sim.test_utility_recovery import run_all_tests as run_utility_tests
        results["Utility"] = run_utility_tests()
    except Exception as e:
        print(f"ERROR: {e}")
        results["Utility"] = False

    # Final Summary
    elapsed = time.time() - start_time

    print("\n")
    print("=" * 80)
    print(" FINAL SUMMARY")
    print("=" * 80)
    print()

    all_passed = True
    for module, passed in results.items():
        status = "PASS" if passed else "FAIL"
        symbol = "[OK]" if passed else "[X]"
        print(f"  {symbol} {module}: {status}")
        if not passed:
            all_passed = False

    print()
    print(f"  Total time: {elapsed:.2f} seconds")
    print()

    if all_passed:
        print("=" * 80)
        print(" ALL SIMULATIONS PASSED - ALGORITHMS VALIDATED")
        print("=" * 80)
    else:
        print("=" * 80)
        print(" SOME SIMULATIONS FAILED - CHECK ABOVE FOR DETAILS")
        print("=" * 80)

    return all_passed


if __name__ == "__main__":
    success = run_all_simulations()
    sys.exit(0 if success else 1)
