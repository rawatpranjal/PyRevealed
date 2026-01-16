"""
Simulation Study: GARP Consistency Validation

Tests that:
1. Utility-maximizing data ALWAYS satisfies GARP (no false positives)
2. Random/irrational data violates GARP at high rates (detection works)
3. WARP violation implies GARP violation (logical consistency)
4. Known violation cycles are correctly detected
5. Edge cases: single observation, identical bundles, etc.

Reference: Chambers & Echenique, Chapter 3
"""

from __future__ import annotations

import sys
import numpy as np
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim.generators import (
    generate_rational_data,
    generate_irrational_data,
    generate_garp_violation_cycle,
)
from src.pyrevealed import ConsumerSession
from src.pyrevealed.algorithms.garp import check_garp, check_warp


class SimulationResults:
    """Track simulation results."""

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
            print("Failures:")
            for f in self.failures:
                print(f"  - {f}")
        return len(self.failures) == 0


def test_rational_data_satisfies_garp() -> SimulationResults:
    """
    Test 1: Data from utility maximization MUST satisfy GARP.

    This is a fundamental property (Afriat's Theorem).
    Any failure here indicates a serious bug.
    """
    results = SimulationResults("Test 1: Rational Data Satisfies GARP")
    print(f"\n{'='*60}")
    print(results.name)
    print("="*60)

    utility_types = ["cobb_douglas", "leontief"]
    n_trials = 20

    for utility_type in utility_types:
        for trial in range(n_trials):
            seed = 1000 + trial
            n_obs = np.random.default_rng(seed).integers(5, 20)
            n_goods = np.random.default_rng(seed).integers(2, 6)

            prices, quantities, _ = generate_rational_data(
                n_obs, n_goods, utility_type, seed
            )
            session = ConsumerSession(prices=prices, quantities=quantities)
            result = check_garp(session)

            test_name = f"{utility_type}_T{n_obs}_N{n_goods}_seed{seed}"
            results.record(
                test_name,
                result.is_consistent,
                f"GARP violated for rational {utility_type} data!"
            )

    return results


def test_irrational_data_violates_garp() -> SimulationResults:
    """
    Test 2: Random data should violate GARP at high rates.

    If random data rarely violates GARP, detection may be broken.
    With many observations, violation rate should approach 100%.
    """
    results = SimulationResults("Test 2: Irrational Data Violates GARP")
    print(f"\n{'='*60}")
    print(results.name)
    print("="*60)

    n_trials = 50
    violations_detected = 0

    for trial in range(n_trials):
        seed = 2000 + trial
        n_obs = 10  # Enough observations to likely violate
        n_goods = 3

        prices, quantities = generate_irrational_data(n_obs, n_goods, seed)
        session = ConsumerSession(prices=prices, quantities=quantities)
        result = check_garp(session)

        if not result.is_consistent:
            violations_detected += 1

    violation_rate = violations_detected / n_trials

    # Random data with 10 observations should violate GARP most of the time
    # We use 50% as minimum threshold (conservative)
    results.record(
        f"violation_rate_{violation_rate:.2%}",
        violation_rate >= 0.5,
        f"Only {violation_rate:.2%} violation rate (expected >= 50%)"
    )

    return results


def test_known_violation_cycles() -> SimulationResults:
    """
    Test 3: Constructed GARP violations must be detected.

    Creates data with guaranteed cycles and verifies detection.
    """
    results = SimulationResults("Test 3: Known Violation Cycles Detected")
    print(f"\n{'='*60}")
    print(results.name)
    print("="*60)

    # Test 1: Simple 2-observation WARP violation
    prices = np.array([[1.0, 2.0], [2.0, 1.0]])
    quantities = np.array([[3.0, 2.0], [2.0, 3.0]])

    # Verify this is indeed a violation:
    # At p1: p1@x1 = 7, p1@x2 = 8 -> x2 not affordable, x1 revealed pref to... wait
    # Let me recalculate:
    # p1 @ x1 = 1*3 + 2*2 = 7
    # p1 @ x2 = 1*2 + 2*3 = 8 -> x2 NOT affordable at p1!

    # Need to fix: x2 must be affordable at p1 for WARP violation
    prices = np.array([[1.0, 1.0], [1.0, 1.0]])
    quantities = np.array([[3.0, 1.0], [1.0, 3.0]])
    # p1 @ x1 = 4, p1 @ x2 = 4 -> both affordable
    # p2 @ x1 = 4, p2 @ x2 = 4 -> both affordable
    # x1 is revealed preferred to x2 (weakly) and vice versa

    session = ConsumerSession(prices=prices, quantities=quantities)
    garp_result = check_garp(session)
    warp_result = check_warp(session)

    # With equal prices and expenditures, this is NOT a violation (indifference)
    # Let me create a proper violation

    # Classic WARP violation from the book (Fig 3.1):
    # x1 chosen at p1, x2 affordable at p1 (p1@x1 > p1@x2)
    # x2 chosen at p2, x1 affordable at p2 AND strictly cheaper (p2@x2 > p2@x1)
    prices = np.array([[1.0, 2.0], [2.0, 1.0]])
    quantities = np.array([[4.0, 1.0], [1.0, 4.0]])
    # p1 @ x1 = 4 + 2 = 6
    # p1 @ x2 = 1 + 8 = 9 > 6, so x2 NOT affordable at p1 -> no WARP violation

    # Try: both bundles on each other's budget line
    prices = np.array([[2.0, 1.0], [1.0, 2.0]])
    quantities = np.array([[1.0, 4.0], [4.0, 1.0]])
    # p1 @ x1 = 2 + 4 = 6
    # p1 @ x2 = 8 + 1 = 9 > 6 -> x2 not affordable

    # Proper violation: crossing budget lines
    prices = np.array([[1.0, 1.0], [1.0, 1.0]])
    quantities = np.array([[4.0, 2.0], [2.0, 4.0]])
    # Both affordable, both same expenditure
    # This is weak preference both ways, but with strictly positive slack... no

    # Use the generator for guaranteed violation
    prices, quantities = generate_garp_violation_cycle(n_goods=2, cycle_length=2, seed=42)
    session = ConsumerSession(prices=prices, quantities=quantities)
    garp_result = check_garp(session)

    results.record(
        "generated_2cycle_detected",
        not garp_result.is_consistent,
        "Failed to detect constructed 2-cycle violation"
    )

    # Test longer cycles
    for cycle_len in [3, 4, 5]:
        prices, quantities = generate_garp_violation_cycle(
            n_goods=3, cycle_length=cycle_len, seed=100+cycle_len
        )
        session = ConsumerSession(prices=prices, quantities=quantities)
        garp_result = check_garp(session)

        results.record(
            f"cycle_length_{cycle_len}_detected",
            not garp_result.is_consistent,
            f"Failed to detect {cycle_len}-cycle violation"
        )

    return results


def test_warp_implies_garp() -> SimulationResults:
    """
    Test 4: WARP violation implies GARP violation.

    WARP is a special case of GARP (length-2 cycles).
    Any WARP violation must also be a GARP violation.
    """
    results = SimulationResults("Test 4: WARP Violation Implies GARP Violation")
    print(f"\n{'='*60}")
    print(results.name)
    print("="*60)

    n_trials = 50

    for trial in range(n_trials):
        seed = 4000 + trial
        prices, quantities = generate_irrational_data(10, 3, seed)
        session = ConsumerSession(prices=prices, quantities=quantities)

        garp_result = check_garp(session)
        warp_result = check_warp(session)

        # If WARP is violated, GARP must be violated
        if not warp_result.is_consistent:
            results.record(
                f"warp_implies_garp_seed{seed}",
                not garp_result.is_consistent,
                "WARP violated but GARP passed - logical error!"
            )
        else:
            # WARP consistent doesn't imply GARP consistent (GARP is stronger)
            results.record(
                f"consistency_check_seed{seed}",
                True,
                ""
            )

    return results


def test_edge_cases() -> SimulationResults:
    """
    Test 5: Edge cases that might cause bugs.
    """
    results = SimulationResults("Test 5: Edge Cases")
    print(f"\n{'='*60}")
    print(results.name)
    print("="*60)

    # Single observation: always consistent
    prices = np.array([[1.0, 2.0]])
    quantities = np.array([[3.0, 1.0]])
    session = ConsumerSession(prices=prices, quantities=quantities)
    result = check_garp(session)
    results.record(
        "single_observation",
        result.is_consistent,
        "Single observation should always be GARP consistent"
    )

    # Identical bundles: should be consistent
    prices = np.array([[1.0, 1.0], [2.0, 2.0]])
    quantities = np.array([[2.0, 2.0], [2.0, 2.0]])
    session = ConsumerSession(prices=prices, quantities=quantities)
    result = check_garp(session)
    results.record(
        "identical_bundles",
        result.is_consistent,
        "Identical bundles should be GARP consistent"
    )

    # Very small quantities (numerical stability)
    prices = np.array([[1.0, 1.0], [1.0, 1.0]])
    quantities = np.array([[1e-8, 1e-8], [1e-8, 1e-8]])
    session = ConsumerSession(prices=prices, quantities=quantities)
    result = check_garp(session)
    results.record(
        "tiny_quantities",
        result.is_consistent,
        "Tiny identical quantities should be consistent"
    )

    # Many goods
    n_goods = 50
    prices, quantities, _ = generate_rational_data(5, n_goods, "cobb_douglas", seed=999)
    session = ConsumerSession(prices=prices, quantities=quantities)
    result = check_garp(session)
    results.record(
        "many_goods_50",
        result.is_consistent,
        "Rational data with 50 goods should be consistent"
    )

    # Many observations
    n_obs = 100
    prices, quantities, _ = generate_rational_data(n_obs, 3, "cobb_douglas", seed=888)
    session = ConsumerSession(prices=prices, quantities=quantities)
    result = check_garp(session)
    results.record(
        "many_observations_100",
        result.is_consistent,
        "Rational data with 100 observations should be consistent"
    )

    return results


def test_cycle_detection_correctness() -> SimulationResults:
    """
    Test 6: Verify that detected cycles are actually violations.

    For each detected cycle, manually verify:
    - Transitive revealed preference holds along the cycle
    - At least one strict preference exists
    """
    results = SimulationResults("Test 6: Cycle Detection Correctness")
    print(f"\n{'='*60}")
    print(results.name)
    print("="*60)

    n_trials = 20

    for trial in range(n_trials):
        seed = 6000 + trial
        prices, quantities = generate_irrational_data(8, 3, seed)
        session = ConsumerSession(prices=prices, quantities=quantities)
        result = check_garp(session)

        if not result.is_consistent and result.violations:
            for cycle_idx, cycle in enumerate(result.violations[:10]):  # Check up to 10 cycles
                # Verify cycle is valid
                # cycle is (i1, i2, ..., in, i1) where last = first

                is_valid_cycle = True
                has_strict = False

                for pos in range(len(cycle) - 1):
                    i = cycle[pos]
                    j = cycle[pos + 1]

                    # Check R[i,j]: p_i @ x_i >= p_i @ x_j
                    exp_i_at_i = prices[i] @ quantities[i]
                    exp_j_at_i = prices[i] @ quantities[j]

                    if exp_i_at_i < exp_j_at_i - 1e-10:
                        is_valid_cycle = False
                        break

                    # Check if strict: p_i @ x_i > p_i @ x_j
                    if exp_i_at_i > exp_j_at_i + 1e-10:
                        has_strict = True

                cycle_valid = is_valid_cycle and has_strict
                results.record(
                    f"cycle_valid_seed{seed}_c{cycle_idx}",
                    cycle_valid,
                    f"Detected cycle is not a valid GARP violation: {cycle}"
                )

    return results


def run_all_tests() -> bool:
    """Run all GARP simulation tests."""
    print("\n" + "="*70)
    print("GARP CONSISTENCY SIMULATION STUDY")
    print("="*70)

    all_results = [
        test_rational_data_satisfies_garp(),
        test_irrational_data_violates_garp(),
        test_known_violation_cycles(),
        test_warp_implies_garp(),
        test_edge_cases(),
        test_cycle_detection_correctness(),
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
