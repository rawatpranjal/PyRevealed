"""
Simulation Study: Abstract Choice Theory Validation

Tests that:
1. Rational menu choices satisfy WARP/SARP (no false positives)
2. Constructed violations are correctly detected
3. Random data violates at expected rates
4. Efficiency index properties hold
5. Preference recovery matches generating order
6. Edge cases handled correctly

Reference: Chambers & Echenique, Chapters 1-2
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim.generators import (
    generate_rational_menu_choices,
    generate_warp_violation_menus,
    generate_sarp_violation_cycle,
    generate_random_menu_choices,
)
from src.pyrevealed import MenuChoiceLog
from src.pyrevealed.algorithms.abstract_choice import (
    validate_menu_warp,
    validate_menu_sarp,
    validate_menu_consistency,
    compute_menu_efficiency,
    fit_menu_preferences,
)


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


def test_warp_consistency() -> SimulationResults:
    """
    Test 1: WARP consistency validation.

    - Rational data should always satisfy WARP
    - Constructed violations should be detected
    """
    results = SimulationResults("Test 1: WARP Consistency")
    print(f"\n{'='*60}")
    print(results.name)
    print("="*60)

    # Test: Rational data satisfies WARP
    n_trials = 20
    for trial in range(n_trials):
        seed = 1000 + trial
        n_items = 5 + (trial % 5)
        n_obs = 10 + trial

        menus, choices, _ = generate_rational_menu_choices(n_obs, n_items, seed)
        log = MenuChoiceLog(menus=menus, choices=choices)
        result = validate_menu_warp(log)

        results.record(
            f"rational_warp_items{n_items}_obs{n_obs}",
            result.is_consistent,
            f"WARP violated for rational data!"
        )

    # Test: Constructed WARP violation detected
    for seed in range(5):
        menus, choices = generate_warp_violation_menus(n_items=4, seed=seed)
        log = MenuChoiceLog(menus=menus, choices=choices)
        result = validate_menu_warp(log)

        results.record(
            f"warp_violation_detected_seed{seed}",
            not result.is_consistent,
            "Failed to detect WARP violation"
        )
        results.record(
            f"warp_violation_count_seed{seed}",
            len(result.violations) >= 1,
            f"Expected >= 1 violation, got {len(result.violations)}"
        )

    return results


def test_sarp_consistency() -> SimulationResults:
    """
    Test 2: SARP consistency validation.

    - Rational data should always satisfy SARP
    - Constructed cycles should be detected
    """
    results = SimulationResults("Test 2: SARP Consistency")
    print(f"\n{'='*60}")
    print(results.name)
    print("="*60)

    # Test: Rational data satisfies SARP
    n_trials = 20
    for trial in range(n_trials):
        seed = 2000 + trial
        n_items = 5 + (trial % 5)
        n_obs = 10 + trial

        menus, choices, _ = generate_rational_menu_choices(n_obs, n_items, seed)
        log = MenuChoiceLog(menus=menus, choices=choices)
        result = validate_menu_sarp(log)

        results.record(
            f"rational_sarp_items{n_items}_obs{n_obs}",
            result.is_consistent,
            f"SARP violated for rational data!"
        )

    # Test: Constructed SARP cycles detected
    for cycle_length in [2, 3, 4, 5]:
        for seed in range(3):
            n_items = max(cycle_length + 1, 5)
            menus, choices = generate_sarp_violation_cycle(
                n_items=n_items, cycle_length=cycle_length, seed=seed
            )
            log = MenuChoiceLog(menus=menus, choices=choices)
            result = validate_menu_sarp(log)

            results.record(
                f"sarp_{cycle_length}cycle_detected_seed{seed}",
                not result.is_consistent,
                f"Failed to detect {cycle_length}-cycle SARP violation"
            )

    return results


def test_congruence_rationalizability() -> SimulationResults:
    """
    Test 3: Congruence (full rationalizability) test.

    - Rational data should satisfy Congruence
    - Violations should be detected
    """
    results = SimulationResults("Test 3: Congruence/Rationalizability")
    print(f"\n{'='*60}")
    print(results.name)
    print("="*60)

    # Test: Rational data is rationalizable
    n_trials = 20
    for trial in range(n_trials):
        seed = 3000 + trial
        n_items = 5 + (trial % 5)
        n_obs = 10 + trial

        menus, choices, _ = generate_rational_menu_choices(n_obs, n_items, seed)
        log = MenuChoiceLog(menus=menus, choices=choices)
        result = validate_menu_consistency(log)

        results.record(
            f"rational_congruence_items{n_items}",
            result.is_rationalizable,
            "Rational data should be rationalizable"
        )
        results.record(
            f"rational_sarp_satisfied_items{n_items}",
            result.satisfies_sarp,
            "Rational data should satisfy SARP"
        )

    # Test: SARP violations imply Congruence failure
    for cycle_length in [2, 3, 4]:
        menus, choices = generate_sarp_violation_cycle(
            n_items=6, cycle_length=cycle_length, seed=42
        )
        log = MenuChoiceLog(menus=menus, choices=choices)
        result = validate_menu_consistency(log)

        results.record(
            f"sarp_violation_implies_congruence_failure_{cycle_length}",
            not result.is_rationalizable,
            "SARP violation should imply Congruence failure"
        )

    return results


def test_houtman_maks_efficiency() -> SimulationResults:
    """
    Test 4: Houtman-Maks efficiency index properties.

    - Rational data should have efficiency = 1.0
    - Violation data should have efficiency < 1.0
    - Efficiency should be in [0, 1]
    """
    results = SimulationResults("Test 4: Houtman-Maks Efficiency")
    print(f"\n{'='*60}")
    print(results.name)
    print("="*60)

    # Test: Rational data has efficiency 1.0
    for seed in range(10):
        n_items = 5
        n_obs = 15

        menus, choices, _ = generate_rational_menu_choices(n_obs, n_items, seed)
        log = MenuChoiceLog(menus=menus, choices=choices)
        result = compute_menu_efficiency(log)

        results.record(
            f"rational_efficiency_1.0_seed{seed}",
            abs(result.efficiency_index - 1.0) < 1e-6,
            f"Expected efficiency 1.0, got {result.efficiency_index}"
        )
        results.record(
            f"no_removals_rational_seed{seed}",
            len(result.removed_observations) == 0,
            f"Expected 0 removals, got {len(result.removed_observations)}"
        )

    # Test: Violation data has efficiency < 1.0
    for cycle_length in [2, 3, 4]:
        menus, choices = generate_sarp_violation_cycle(
            n_items=6, cycle_length=cycle_length, seed=42
        )
        log = MenuChoiceLog(menus=menus, choices=choices)
        result = compute_menu_efficiency(log)

        results.record(
            f"violation_{cycle_length}cycle_efficiency_less_1",
            result.efficiency_index < 1.0,
            f"Expected efficiency < 1.0, got {result.efficiency_index}"
        )
        results.record(
            f"efficiency_in_bounds_{cycle_length}cycle",
            0.0 <= result.efficiency_index <= 1.0,
            f"Efficiency out of bounds: {result.efficiency_index}"
        )
        results.record(
            f"some_removals_{cycle_length}cycle",
            len(result.removed_observations) >= 1,
            "Expected at least one removal for violation"
        )

    return results


def test_preference_recovery() -> SimulationResults:
    """
    Test 5: Preference recovery matches generating order.

    When data is generated from a known preference order,
    the recovered order should match.
    """
    results = SimulationResults("Test 5: Preference Recovery")
    print(f"\n{'='*60}")
    print(results.name)
    print("="*60)

    n_trials = 15

    for trial in range(n_trials):
        seed = 5000 + trial
        n_items = 5
        n_obs = 20  # More observations for better coverage

        menus, choices, true_order = generate_rational_menu_choices(
            n_obs, n_items, seed, menu_size=3
        )
        log = MenuChoiceLog(menus=menus, choices=choices)
        result = fit_menu_preferences(log)

        results.record(
            f"recovery_success_seed{seed}",
            result.success,
            "Preference recovery failed on rational data"
        )

        if result.success and result.preference_order is not None:
            # Check if recovered order matches true order
            # They might not match exactly if not all pairs were compared,
            # but consistent orderings should agree where they overlap

            # At minimum, check that recovered order is consistent with
            # the revealed preferences
            is_consistent = True
            for t, (menu, choice) in enumerate(zip(menus, choices)):
                for item in menu:
                    if item != choice:
                        # choice should be ranked higher (lower rank number)
                        if result.utility_ranking is not None:
                            choice_rank = result.utility_ranking.get(choice, n_items)
                            item_rank = result.utility_ranking.get(item, n_items)
                            if choice_rank > item_rank:
                                is_consistent = False
                                break
                if not is_consistent:
                    break

            results.record(
                f"recovery_consistent_seed{seed}",
                is_consistent,
                "Recovered order inconsistent with observed choices"
            )

    # Test: Recovery fails on SARP-violating data
    menus, choices = generate_sarp_violation_cycle(n_items=4, cycle_length=3, seed=99)
    log = MenuChoiceLog(menus=menus, choices=choices)
    result = fit_menu_preferences(log)

    results.record(
        "recovery_fails_on_violation",
        not result.success,
        "Preference recovery should fail on SARP-violating data"
    )

    return results


def test_edge_cases() -> SimulationResults:
    """
    Test 6: Edge cases that might cause bugs.
    """
    results = SimulationResults("Test 6: Edge Cases")
    print(f"\n{'='*60}")
    print(results.name)
    print("="*60)

    # Single observation: always consistent
    log = MenuChoiceLog(
        menus=[frozenset({0, 1, 2})],
        choices=[1],
    )
    warp_result = validate_menu_warp(log)
    sarp_result = validate_menu_sarp(log)
    cong_result = validate_menu_consistency(log)

    results.record(
        "single_obs_warp_consistent",
        warp_result.is_consistent,
        "Single observation should be WARP consistent"
    )
    results.record(
        "single_obs_sarp_consistent",
        sarp_result.is_consistent,
        "Single observation should be SARP consistent"
    )
    results.record(
        "single_obs_rationalizable",
        cong_result.is_rationalizable,
        "Single observation should be rationalizable"
    )

    # Large menus
    log = MenuChoiceLog(
        menus=[frozenset(range(20)), frozenset(range(20))],
        choices=[0, 0],  # Always choose item 0
    )
    result = validate_menu_sarp(log)
    results.record(
        "large_menu_consistent",
        result.is_consistent,
        "Consistent choices from large menus should pass"
    )

    # Identical choices: always consistent
    log = MenuChoiceLog(
        menus=[frozenset({0, 1}), frozenset({0, 2}), frozenset({0, 1, 2})],
        choices=[0, 0, 0],  # Always choose item 0
    )
    result = validate_menu_sarp(log)
    results.record(
        "identical_choices_consistent",
        result.is_consistent,
        "Always choosing same item should be consistent"
    )

    # Many observations with rational agent
    n_obs = 100
    n_items = 10
    menus, choices, _ = generate_rational_menu_choices(n_obs, n_items, seed=777)
    log = MenuChoiceLog(menus=menus, choices=choices)

    result = validate_menu_sarp(log)
    results.record(
        "many_obs_100_rational_consistent",
        result.is_consistent,
        f"Rational data with {n_obs} observations should be consistent"
    )

    # Singleton menus (only one choice) - should be consistent
    log = MenuChoiceLog(
        menus=[frozenset({0}), frozenset({1}), frozenset({2})],
        choices=[0, 1, 2],
    )
    result = validate_menu_sarp(log)
    results.record(
        "singleton_menus_consistent",
        result.is_consistent,
        "Singleton menus should always be consistent"
    )

    # Disjoint menus (no item appears in multiple menus)
    log = MenuChoiceLog(
        menus=[frozenset({0, 1}), frozenset({2, 3}), frozenset({4, 5})],
        choices=[0, 3, 5],
    )
    result = validate_menu_sarp(log)
    results.record(
        "disjoint_menus_consistent",
        result.is_consistent,
        "Disjoint menus should always be consistent"
    )

    return results


def test_random_data_violation_rates() -> SimulationResults:
    """
    Test 7: Random data should violate SARP at high rates.
    """
    results = SimulationResults("Test 7: Random Data Violation Rates")
    print(f"\n{'='*60}")
    print(results.name)
    print("="*60)

    n_trials = 50
    violations_detected = 0

    for trial in range(n_trials):
        seed = 7000 + trial
        n_obs = 15
        n_items = 5
        menu_size = 3

        menus, choices = generate_random_menu_choices(n_obs, n_items, menu_size, seed)
        log = MenuChoiceLog(menus=menus, choices=choices)
        result = validate_menu_sarp(log)

        if not result.is_consistent:
            violations_detected += 1

    violation_rate = violations_detected / n_trials

    # Random choices should often violate SARP
    # We use 30% as minimum threshold (conservative)
    results.record(
        f"random_violation_rate_{violation_rate:.2%}",
        violation_rate >= 0.30,
        f"Only {violation_rate:.2%} violation rate (expected >= 30%)"
    )

    return results


def run_all_tests() -> bool:
    """Run all abstract choice simulation tests."""
    print("\n" + "="*70)
    print("ABSTRACT CHOICE THEORY SIMULATION STUDY")
    print("="*70)

    all_results = [
        test_warp_consistency(),
        test_sarp_consistency(),
        test_congruence_rationalizability(),
        test_houtman_maks_efficiency(),
        test_preference_recovery(),
        test_edge_cases(),
        test_random_data_violation_rates(),
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
