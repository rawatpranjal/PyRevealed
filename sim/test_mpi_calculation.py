"""
Simulation Study: Money Pump Index (MPI) Calculation

Tests that:
1. MPI = 0 for GARP-consistent data
2. MPI > 0 for GARP-violating data
3. MPI matches theoretical formula from the book
4. MPI correlates with violation severity
5. Cycle MPI calculation is correct
6. Houtman-Maks index works correctly

Reference: Chambers & Echenique, Chapter 5, Section 5.1.3, Equation 5.1
"""

from __future__ import annotations

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sim.generators import (
    generate_rational_data,
    generate_irrational_data,
    generate_garp_violation_cycle,
    compute_theoretical_mpi,
)
from src.pyrevealed import ConsumerSession
from src.pyrevealed.algorithms.garp import check_garp
from src.pyrevealed.algorithms.mpi import compute_mpi, compute_houtman_maks_index


class SimulationResults:
    def __init__(self, name: str):
        self.name = name
        self.tests_run = 0
        self.tests_passed = 0
        self.failures: list[str] = []

    def record(self, test_name: str, passed: bool, message: str = ""):
        self.tests_run += 1
        if passed:
            self.tests_passed += 1
            print(f"  [PASS] {test_name}")
        else:
            self.failures.append(f"{test_name}: {message}")
            print(f"  [FAIL] {test_name}: {message}")

    def summary(self) -> bool:
        print(f"\n{self.name}: {self.tests_passed}/{self.tests_run} tests passed")
        if self.failures:
            for f in self.failures:
                print(f"  - {f}")
        return len(self.failures) == 0


def test_mpi_zero_for_consistent() -> SimulationResults:
    """
    Test 1: MPI = 0 for GARP-consistent data.

    No violations means no money can be pumped.
    """
    results = SimulationResults("Test 1: MPI = 0 for Consistent Data")
    print(f"\n{'='*60}")
    print(results.name)
    print("="*60)

    for trial in range(15):
        seed = 1000 + trial
        n_obs = np.random.default_rng(seed).integers(5, 12)
        n_goods = np.random.default_rng(seed).integers(2, 5)

        prices, quantities, _ = generate_rational_data(n_obs, n_goods, "cobb_douglas", seed)
        session = ConsumerSession(prices=prices, quantities=quantities)

        mpi_result = compute_mpi(session)

        results.record(
            f"mpi_zero_seed{seed}",
            mpi_result.mpi_value == 0.0,
            f"MPI = {mpi_result.mpi_value:.6f}, expected 0.0"
        )

    return results


def test_mpi_positive_for_violations() -> SimulationResults:
    """
    Test 2: MPI > 0 for GARP-violating data.

    Violations mean money can be pumped.
    """
    results = SimulationResults("Test 2: MPI > 0 for Violations")
    print(f"\n{'='*60}")
    print(results.name)
    print("="*60)

    violations_checked = 0

    for trial in range(40):
        seed = 2000 + trial
        prices, quantities = generate_irrational_data(10, 3, seed)
        session = ConsumerSession(prices=prices, quantities=quantities)

        garp_result = check_garp(session)

        if not garp_result.is_consistent:
            violations_checked += 1
            mpi_result = compute_mpi(session)

            results.record(
                f"mpi_positive_seed{seed}",
                mpi_result.mpi_value > 0.0,
                f"MPI = {mpi_result.mpi_value:.6f}, expected > 0"
            )

    print(f"  (Checked {violations_checked} violation cases)")
    return results


def test_mpi_formula_correctness() -> SimulationResults:
    """
    Test 3: MPI matches the book's formula exactly.

    MPI = sum(p_k @ (x_k - x_{k+1})) / sum(p_k @ x_k)

    We construct data with known cycles and verify MPI calculation.
    """
    results = SimulationResults("Test 3: MPI Formula Correctness")
    print(f"\n{'='*60}")
    print(results.name)
    print("="*60)

    for trial in range(10):
        seed = 3000 + trial

        # Generate data with guaranteed violation
        prices, quantities = generate_garp_violation_cycle(n_goods=3, cycle_length=3, seed=seed)
        session = ConsumerSession(prices=prices, quantities=quantities)

        garp_result = check_garp(session)

        if not garp_result.is_consistent and garp_result.violations:
            mpi_result = compute_mpi(session)

            # Compute theoretical MPI for the detected cycle
            if mpi_result.worst_cycle:
                theoretical = compute_theoretical_mpi(prices, quantities, mpi_result.worst_cycle)

                # Allow small numerical tolerance
                diff = abs(mpi_result.mpi_value - theoretical)
                results.record(
                    f"formula_match_seed{seed}",
                    diff < 1e-6,
                    f"Computed={mpi_result.mpi_value:.8f}, Theoretical={theoretical:.8f}, Diff={diff:.8f}"
                )
                print(f"  Seed {seed}: Computed={mpi_result.mpi_value:.6f}, Theoretical={theoretical:.6f}")

    return results


def test_mpi_manual_calculation() -> SimulationResults:
    """
    Test 4: Manual MPI calculation for simple cases.

    Construct explicit examples and verify MPI by hand.
    """
    results = SimulationResults("Test 4: Manual MPI Calculation")
    print(f"\n{'='*60}")
    print(results.name)
    print("="*60)

    # Test 1: Equal expenditure data should be consistent (MPI = 0)
    prices = np.array([
        [1.0, 1.0],
        [1.0, 1.0],
    ])
    quantities = np.array([
        [3.0, 2.0],  # exp = 5
        [2.0, 3.0],  # exp = 5
    ])
    # p0 @ x0 = 5, p0 @ x1 = 5 -> weak preference only
    # No STRICT preference, so GARP consistent

    session = ConsumerSession(prices=prices, quantities=quantities)
    garp_result = check_garp(session)
    mpi_result = compute_mpi(session)

    results.record(
        "equal_expenditure_consistent",
        garp_result.is_consistent,
        "Equal expenditure data should be GARP consistent"
    )

    results.record(
        "equal_expenditure_mpi_zero",
        mpi_result.mpi_value == 0.0,
        f"Equal expenditure data: MPI = {mpi_result.mpi_value}"
    )

    # Test 2: Verified 2-cycle GARP violation
    # Key insight: need DIFFERENT prices to create strict preferences on BOTH sides
    prices = np.array([
        [1.5, 1.0],
        [1.0, 1.5],
    ])
    quantities = np.array([
        [4.0, 3.0],  # x0: p0@x0 = 9, p1@x0 = 8.5
        [3.0, 4.0],  # x1: p0@x1 = 8.5, p1@x1 = 9
    ])
    # p0@x0 = 9 > p0@x1 = 8.5 -> x0 STRICTLY revealed preferred to x1
    # p1@x1 = 9 > p1@x0 = 8.5 -> x1 STRICTLY revealed preferred to x0
    # This creates a 2-cycle GARP violation!

    session = ConsumerSession(prices=prices, quantities=quantities)
    garp_result = check_garp(session)
    mpi_result = compute_mpi(session)

    results.record(
        "2cycle_is_violation",
        not garp_result.is_consistent,
        "2-cycle with crossing budgets should be GARP violation"
    )

    results.record(
        "2cycle_mpi_positive",
        mpi_result.mpi_value > 0.0,
        f"2-cycle MPI = {mpi_result.mpi_value} (should be > 0)"
    )

    # Test 3: Verify MPI calculation manually for the 2-cycle
    # MPI = sum(p_k @ x_k - p_k @ x_{k+1}) / sum(p_k @ x_k)
    # Cycle: 0 -> 1 -> 0
    # Numerator: (p0@x0 - p0@x1) + (p1@x1 - p1@x0) = (9-8.5) + (9-8.5) = 1.0
    # Denominator: p0@x0 + p1@x1 = 9 + 9 = 18
    # Expected MPI = 1.0 / 18 ≈ 0.0556
    expected_mpi = 1.0 / 18.0

    results.record(
        "2cycle_mpi_value",
        abs(mpi_result.mpi_value - expected_mpi) < 0.01,
        f"MPI = {mpi_result.mpi_value:.4f}, expected ≈ {expected_mpi:.4f}"
    )

    return results


def test_mpi_bounds() -> SimulationResults:
    """
    Test 5: MPI is bounded between 0 and 1.

    MPI represents fraction of expenditure "wasted".
    """
    results = SimulationResults("Test 5: MPI Bounds [0, 1]")
    print(f"\n{'='*60}")
    print(results.name)
    print("="*60)

    for trial in range(30):
        seed = 5000 + trial
        prices, quantities = generate_irrational_data(10, 3, seed)
        session = ConsumerSession(prices=prices, quantities=quantities)

        mpi_result = compute_mpi(session)

        in_bounds = 0.0 <= mpi_result.mpi_value <= 1.0
        results.record(
            f"bounds_seed{seed}",
            in_bounds,
            f"MPI = {mpi_result.mpi_value:.6f} out of [0, 1]"
        )

    return results


def test_houtman_maks_index() -> SimulationResults:
    """
    Test 6: Houtman-Maks index correctly identifies observations to remove.

    After removing identified observations, data should satisfy GARP.
    """
    results = SimulationResults("Test 6: Houtman-Maks Index")
    print(f"\n{'='*60}")
    print(results.name)
    print("="*60)

    for trial in range(15):
        seed = 6000 + trial
        prices, quantities = generate_irrational_data(8, 3, seed)
        session = ConsumerSession(prices=prices, quantities=quantities)

        garp_result = check_garp(session)

        if not garp_result.is_consistent:
            hm_result = compute_houtman_maks_index(session)

            # Verify: removing these observations makes data consistent
            if hm_result.removed_observations:
                remaining = [i for i in range(len(prices)) if i not in hm_result.removed_observations]
                if len(remaining) >= 2:
                    sub_prices = prices[remaining]
                    sub_quantities = quantities[remaining]
                    sub_session = ConsumerSession(prices=sub_prices, quantities=sub_quantities)
                    sub_result = check_garp(sub_session)

                    results.record(
                        f"hm_consistent_after_removal_seed{seed}",
                        sub_result.is_consistent,
                        f"Still violated after removing {hm_result.removed_observations}"
                    )
                    print(f"  Seed {seed}: Removed {hm_result.num_removed} obs, HM index = {hm_result.fraction:.3f}")

    return results


def test_houtman_maks_bounds() -> SimulationResults:
    """
    Test 7: Houtman-Maks index is bounded correctly.
    """
    results = SimulationResults("Test 7: Houtman-Maks Index Bounds")
    print(f"\n{'='*60}")
    print(results.name)
    print("="*60)

    # Consistent data: HM = 0
    prices, quantities, _ = generate_rational_data(10, 3, "cobb_douglas", seed=7001)
    session = ConsumerSession(prices=prices, quantities=quantities)
    hm_result = compute_houtman_maks_index(session)

    results.record(
        "hm_zero_for_consistent",
        hm_result.fraction == 0.0 and len(hm_result.removed_observations) == 0,
        f"HM = {hm_result.fraction}, removed = {hm_result.removed_observations}"
    )

    # Single observation: HM = 0
    prices = np.array([[1.0, 2.0]])
    quantities = np.array([[3.0, 1.0]])
    session = ConsumerSession(prices=prices, quantities=quantities)
    hm_result = compute_houtman_maks_index(session)

    results.record(
        "hm_single_observation",
        hm_result.fraction == 0.0,
        f"Single obs: HM = {hm_result.fraction}"
    )

    # HM index should be in [0, 1]
    for trial in range(20):
        seed = 7100 + trial
        prices, quantities = generate_irrational_data(10, 3, seed)
        session = ConsumerSession(prices=prices, quantities=quantities)
        hm_result = compute_houtman_maks_index(session)

        results.record(
            f"hm_bounds_seed{seed}",
            0.0 <= hm_result.fraction <= 1.0,
            f"HM = {hm_result.fraction} out of [0, 1]"
        )

    return results


def run_all_tests() -> bool:
    """Run all MPI simulation tests."""
    print("\n" + "="*70)
    print("MONEY PUMP INDEX (MPI) SIMULATION STUDY")
    print("="*70)

    all_results = [
        test_mpi_zero_for_consistent(),
        test_mpi_positive_for_violations(),
        test_mpi_formula_correctness(),
        test_mpi_manual_calculation(),
        test_mpi_bounds(),
        test_houtman_maks_index(),
        test_houtman_maks_bounds(),
    ]

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    all_passed = True
    for result in all_results:
        if not result.summary():
            all_passed = False

    print("\n" + "="*70)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("="*70)

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
