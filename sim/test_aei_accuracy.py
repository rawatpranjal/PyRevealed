"""
Simulation Study: Afriat Efficiency Index (AEI) Accuracy

Tests that:
1. AEI = 1.0 for perfectly rational data
2. AEI < 1.0 for data with violations
3. AEI is monotonic: more severe violations -> lower AEI
4. AEI respects efficiency thresholds: GARP holds at computed AEI
5. Binary search converges correctly
6. Edge cases: boundary values, numerical stability

Reference: Chambers & Echenique, Chapter 5, Section 5.1.1
"""

from __future__ import annotations

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sim.generators import generate_rational_data, generate_irrational_data
from src.pyrevealed import ConsumerSession
from src.pyrevealed.algorithms.garp import check_garp
from src.pyrevealed.algorithms.aei import compute_aei, _check_garp_at_efficiency


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


def test_aei_perfect_rationality() -> SimulationResults:
    """
    Test 1: AEI = 1.0 for utility-maximizing data.

    Rational data satisfies GARP, so AEI must be exactly 1.0.
    """
    results = SimulationResults("Test 1: AEI = 1.0 for Rational Data")
    print(f"\n{'='*60}")
    print(results.name)
    print("="*60)

    for utility_type in ["cobb_douglas", "leontief"]:
        for trial in range(10):
            seed = 1000 + trial
            n_obs = np.random.default_rng(seed).integers(5, 15)
            n_goods = np.random.default_rng(seed).integers(2, 5)

            prices, quantities, _ = generate_rational_data(
                n_obs, n_goods, utility_type, seed
            )
            session = ConsumerSession(prices=prices, quantities=quantities)
            aei_result = compute_aei(session)

            results.record(
                f"{utility_type}_seed{seed}",
                aei_result.efficiency_index == 1.0,
                f"AEI = {aei_result.efficiency_index:.6f}, expected 1.0"
            )

    return results


def test_aei_less_than_one_for_violations() -> SimulationResults:
    """
    Test 2: AEI < 1.0 when GARP is violated.

    If GARP is violated, AEI must be strictly less than 1.
    """
    results = SimulationResults("Test 2: AEI < 1.0 for Violations")
    print(f"\n{'='*60}")
    print(results.name)
    print("="*60)

    n_trials = 30
    violations_found = 0

    for trial in range(n_trials):
        seed = 2000 + trial
        prices, quantities = generate_irrational_data(10, 3, seed)
        session = ConsumerSession(prices=prices, quantities=quantities)

        garp_result = check_garp(session)

        if not garp_result.is_consistent:
            violations_found += 1
            aei_result = compute_aei(session)

            results.record(
                f"violation_seed{seed}",
                aei_result.efficiency_index < 1.0,
                f"AEI = {aei_result.efficiency_index:.6f} for violated data"
            )

    print(f"  (Found {violations_found} violations out of {n_trials} trials)")
    return results


def test_aei_monotonicity() -> SimulationResults:
    """
    Test 3: More severe perturbations -> lower AEI.

    Adding more noise to rational data should decrease AEI.
    """
    results = SimulationResults("Test 3: AEI Monotonicity")
    print(f"\n{'='*60}")
    print(results.name)
    print("="*60)

    base_seed = 3000
    prices, quantities, _ = generate_rational_data(10, 3, "cobb_douglas", base_seed)

    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5]
    aei_values = []

    rng = np.random.default_rng(base_seed + 1)

    for noise in noise_levels:
        perturbed_q = quantities * (1 + noise * rng.standard_normal(quantities.shape))
        perturbed_q = np.maximum(perturbed_q, 0.01)

        session = ConsumerSession(prices=prices, quantities=perturbed_q)
        aei_result = compute_aei(session)
        aei_values.append(aei_result.efficiency_index)
        print(f"  Noise {noise}: AEI = {aei_result.efficiency_index:.4f}")

    # Check monotonicity (allowing small tolerance for noise)
    is_monotonic = True
    for i in range(len(aei_values) - 1):
        if aei_values[i] < aei_values[i+1] - 0.05:  # Allow 5% tolerance
            is_monotonic = False
            break

    results.record(
        "monotonicity",
        is_monotonic,
        f"AEI not monotonically decreasing: {aei_values}"
    )

    return results


def test_aei_threshold_property() -> SimulationResults:
    """
    Test 4: GARP holds at computed AEI, fails above it.

    The defining property of AEI: sup{e : GARP holds at efficiency e}.
    """
    results = SimulationResults("Test 4: AEI Threshold Property")
    print(f"\n{'='*60}")
    print(results.name)
    print("="*60)

    n_trials = 20

    for trial in range(n_trials):
        seed = 4000 + trial
        prices, quantities = generate_irrational_data(8, 3, seed)
        session = ConsumerSession(prices=prices, quantities=quantities)

        aei_result = compute_aei(session, tolerance=1e-6)
        aei = aei_result.efficiency_index

        if aei >= 1.0:
            # Perfect rationality, skip
            continue

        # Check: GARP should hold at AEI (or just below)
        consistent_at_aei, _ = _check_garp_at_efficiency(session, aei - 1e-5)
        results.record(
            f"holds_at_aei_seed{seed}",
            consistent_at_aei,
            f"GARP fails at AEI={aei:.6f}"
        )

        # Check: GARP should fail slightly above AEI
        consistent_above, _ = _check_garp_at_efficiency(session, min(1.0, aei + 0.01))

        # If AEI < 1.0, there should be failure above
        if aei < 0.99:
            results.record(
                f"fails_above_aei_seed{seed}",
                not consistent_above,
                f"GARP still holds above AEI={aei:.6f}"
            )

    return results


def test_aei_boundary_values() -> SimulationResults:
    """
    Test 5: Boundary cases for AEI computation.
    """
    results = SimulationResults("Test 5: AEI Boundary Values")
    print(f"\n{'='*60}")
    print(results.name)
    print("="*60)

    # AEI = 1.0 boundary
    prices, quantities, _ = generate_rational_data(5, 2, "cobb_douglas", seed=5001)
    session = ConsumerSession(prices=prices, quantities=quantities)
    result = compute_aei(session)
    results.record(
        "aei_equals_one",
        result.efficiency_index == 1.0,
        f"Expected AEI=1.0, got {result.efficiency_index}"
    )

    # Single observation: AEI = 1.0
    prices = np.array([[1.0, 2.0]])
    quantities = np.array([[3.0, 1.0]])
    session = ConsumerSession(prices=prices, quantities=quantities)
    result = compute_aei(session)
    results.record(
        "single_observation",
        result.efficiency_index == 1.0,
        f"Single observation should have AEI=1.0, got {result.efficiency_index}"
    )

    # Two observations, same bundle: AEI = 1.0
    prices = np.array([[1.0, 1.0], [2.0, 2.0]])
    quantities = np.array([[2.0, 2.0], [2.0, 2.0]])
    session = ConsumerSession(prices=prices, quantities=quantities)
    result = compute_aei(session)
    results.record(
        "identical_bundles",
        result.efficiency_index == 1.0,
        f"Identical bundles should have AEI=1.0, got {result.efficiency_index}"
    )

    return results


def test_aei_numerical_precision() -> SimulationResults:
    """
    Test 6: AEI computation has proper numerical precision.
    """
    results = SimulationResults("Test 6: AEI Numerical Precision")
    print(f"\n{'='*60}")
    print(results.name)
    print("="*60)

    # Test with different tolerance levels
    seed = 6000
    prices, quantities = generate_irrational_data(10, 3, seed)
    session = ConsumerSession(prices=prices, quantities=quantities)

    tolerances = [1e-4, 1e-6, 1e-8]
    aei_values = []

    for tol in tolerances:
        result = compute_aei(session, tolerance=tol)
        aei_values.append(result.efficiency_index)
        print(f"  Tolerance {tol}: AEI = {result.efficiency_index:.10f}")

    # Results should converge as tolerance decreases
    # Check that higher precision gives similar result
    for i in range(len(aei_values) - 1):
        diff = abs(aei_values[i] - aei_values[i+1])
        results.record(
            f"convergence_tol_{tolerances[i]}",
            diff < tolerances[i] * 100,
            f"AEI difference {diff} too large for tolerance {tolerances[i]}"
        )

    return results


def test_aei_binary_search_iterations() -> SimulationResults:
    """
    Test 7: Binary search should converge in reasonable iterations.
    """
    results = SimulationResults("Test 7: Binary Search Convergence")
    print(f"\n{'='*60}")
    print(results.name)
    print("="*60)

    for trial in range(10):
        seed = 7000 + trial
        prices, quantities = generate_irrational_data(10, 3, seed)
        session = ConsumerSession(prices=prices, quantities=quantities)

        result = compute_aei(session, tolerance=1e-6, max_iterations=50)

        # For tolerance 1e-6, should converge in ~20 iterations (log2(1/1e-6) ≈ 20)
        if not result.is_perfectly_consistent:
            results.record(
                f"iterations_seed{seed}",
                result.binary_search_iterations <= 25,
                f"Too many iterations: {result.binary_search_iterations}"
            )
            print(f"  Seed {seed}: {result.binary_search_iterations} iterations, AEI={result.efficiency_index:.6f}")

    return results


def run_all_tests() -> bool:
    """Run all AEI simulation tests."""
    print("\n" + "="*70)
    print("AFRIAT EFFICIENCY INDEX (AEI) SIMULATION STUDY")
    print("="*70)

    all_results = [
        test_aei_perfect_rationality(),
        test_aei_less_than_one_for_violations(),
        test_aei_monotonicity(),
        test_aei_threshold_property(),
        test_aei_boundary_values(),
        test_aei_numerical_precision(),
        test_aei_binary_search_iterations(),
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
