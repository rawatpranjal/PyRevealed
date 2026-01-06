"""
Simulation Study: Utility Recovery via Afriat Inequalities

Tests that:
1. Utility recovery succeeds for GARP-consistent data
2. Utility recovery fails for GARP-violating data
3. Recovered utility rationalizes the data (u(x_k) = max over budget)
4. Afriat inequalities are satisfied by recovered (U, lambda)
5. Recovered utility function ranks bundles correctly
6. Lagrange multipliers (marginal utility of money) are positive

Reference: Chambers & Echenique, Chapter 3, Afriat's Theorem (Statement III)
"""

from __future__ import annotations

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sim.generators import generate_rational_data, generate_irrational_data
from src.pyrevealed import ConsumerSession
from src.pyrevealed.algorithms.garp import check_garp
from src.pyrevealed.algorithms.utility import (
    recover_utility,
    construct_afriat_utility,
)


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


def test_recovery_success_for_consistent() -> SimulationResults:
    """
    Test 1: Utility recovery succeeds for GARP-consistent data.

    By Afriat's Theorem, GARP implies existence of solution.
    """
    results = SimulationResults("Test 1: Recovery Success for Consistent Data")
    print(f"\n{'='*60}")
    print(results.name)
    print("="*60)

    for trial in range(15):
        seed = 1000 + trial
        n_obs = np.random.default_rng(seed).integers(5, 12)
        n_goods = np.random.default_rng(seed).integers(2, 5)

        prices, quantities, _ = generate_rational_data(n_obs, n_goods, "cobb_douglas", seed)
        session = ConsumerSession(prices=prices, quantities=quantities)

        utility_result = recover_utility(session)

        results.record(
            f"recovery_success_seed{seed}",
            utility_result.success,
            f"LP status: {utility_result.lp_status}"
        )

    return results


def test_recovery_failure_for_violations() -> SimulationResults:
    """
    Test 2: Utility recovery fails for GARP-violating data.

    Afriat inequalities have no solution when GARP is violated.
    """
    results = SimulationResults("Test 2: Recovery Fails for Violations")
    print(f"\n{'='*60}")
    print(results.name)
    print("="*60)

    violations_tested = 0

    for trial in range(40):
        seed = 2000 + trial
        prices, quantities = generate_irrational_data(10, 3, seed)
        session = ConsumerSession(prices=prices, quantities=quantities)

        garp_result = check_garp(session)

        if not garp_result.is_consistent:
            violations_tested += 1
            utility_result = recover_utility(session)

            results.record(
                f"recovery_fails_seed{seed}",
                not utility_result.success,
                f"Recovery succeeded for violated data!"
            )

    print(f"  (Tested {violations_tested} violation cases)")
    return results


def test_afriat_inequalities_satisfied() -> SimulationResults:
    """
    Test 3: Recovered (U, lambda) satisfy Afriat inequalities.

    U_k <= U_l + lambda_l * p_l @ (x_k - x_l) for all k, l
    """
    results = SimulationResults("Test 3: Afriat Inequalities Satisfied")
    print(f"\n{'='*60}")
    print(results.name)
    print("="*60)

    for trial in range(15):
        seed = 3000 + trial
        n_obs = np.random.default_rng(seed).integers(5, 10)
        n_goods = np.random.default_rng(seed).integers(2, 4)

        prices, quantities, _ = generate_rational_data(n_obs, n_goods, "cobb_douglas", seed)
        session = ConsumerSession(prices=prices, quantities=quantities)

        utility_result = recover_utility(session)

        if utility_result.success:
            U = utility_result.utility_values
            lambdas = utility_result.lagrange_multipliers

            # Check all Afriat inequalities
            all_satisfied = True
            max_violation = 0.0

            for k in range(n_obs):
                for l in range(n_obs):
                    if k == l:
                        continue

                    # U_k <= U_l + lambda_l * p_l @ (x_k - x_l)
                    rhs = U[l] + lambdas[l] * (prices[l] @ (quantities[k] - quantities[l]))
                    violation = U[k] - rhs

                    if violation > 1e-6:  # Allow numerical tolerance
                        all_satisfied = False
                        max_violation = max(max_violation, violation)

            results.record(
                f"afriat_ineq_seed{seed}",
                all_satisfied,
                f"Max violation = {max_violation:.8f}"
            )

    return results


def test_utility_rationalizes_data() -> SimulationResults:
    """
    Test 4: Recovered utility rationalizes the data.

    For each observation k, x_k maximizes u(x) over the budget set:
    u(x_k) >= u(y) for all y with p_k @ y <= p_k @ x_k
    """
    results = SimulationResults("Test 4: Utility Rationalizes Data")
    print(f"\n{'='*60}")
    print(results.name)
    print("="*60)

    for trial in range(10):
        seed = 4000 + trial
        n_obs = 5
        n_goods = 2

        prices, quantities, _ = generate_rational_data(n_obs, n_goods, "cobb_douglas", seed)
        session = ConsumerSession(prices=prices, quantities=quantities)

        utility_result = recover_utility(session)

        if utility_result.success:
            u = construct_afriat_utility(session, utility_result)

            # For each observation, check that x_k maximizes utility
            all_rationalized = True

            for k in range(n_obs):
                budget_k = prices[k] @ quantities[k]
                u_at_xk = u(quantities[k])

                # Check against other observed bundles that are affordable
                for l in range(n_obs):
                    if l == k:
                        continue

                    cost_l_at_k = prices[k] @ quantities[l]
                    if cost_l_at_k <= budget_k + 1e-10:  # x_l affordable at k
                        u_at_xl = u(quantities[l])
                        if u_at_xl > u_at_xk + 1e-6:  # x_l gives higher utility
                            all_rationalized = False
                            print(f"    Violation: u(x_{l})={u_at_xl:.4f} > u(x_{k})={u_at_xk:.4f}")

            results.record(
                f"rationalization_seed{seed}",
                all_rationalized,
                "Recovered utility doesn't rationalize data"
            )

    return results


def test_lagrange_multipliers_positive() -> SimulationResults:
    """
    Test 5: Lagrange multipliers (marginal utility of money) are positive.

    lambda_k > 0 for all k (strict positivity required by theory).
    """
    results = SimulationResults("Test 5: Lagrange Multipliers Positive")
    print(f"\n{'='*60}")
    print(results.name)
    print("="*60)

    for trial in range(15):
        seed = 5000 + trial
        n_obs = np.random.default_rng(seed).integers(5, 12)
        n_goods = np.random.default_rng(seed).integers(2, 5)

        prices, quantities, _ = generate_rational_data(n_obs, n_goods, "cobb_douglas", seed)
        session = ConsumerSession(prices=prices, quantities=quantities)

        utility_result = recover_utility(session)

        if utility_result.success:
            lambdas = utility_result.lagrange_multipliers
            all_positive = np.all(lambdas > 0)
            min_lambda = np.min(lambdas)

            results.record(
                f"lambdas_positive_seed{seed}",
                all_positive,
                f"Min lambda = {min_lambda:.8f}"
            )

    return results


def test_utility_concave() -> SimulationResults:
    """
    Test 6: Afriat utility is concave (minimum of linear functions).

    u(alpha*x + (1-alpha)*y) >= alpha*u(x) + (1-alpha)*u(y)
    """
    results = SimulationResults("Test 6: Utility is Concave")
    print(f"\n{'='*60}")
    print(results.name)
    print("="*60)

    for trial in range(10):
        seed = 6000 + trial
        n_obs = 5
        n_goods = 2

        prices, quantities, _ = generate_rational_data(n_obs, n_goods, "cobb_douglas", seed)
        session = ConsumerSession(prices=prices, quantities=quantities)

        utility_result = recover_utility(session)

        if utility_result.success:
            u = construct_afriat_utility(session, utility_result)

            # Test concavity at random points
            rng = np.random.default_rng(seed)
            is_concave = True

            for _ in range(20):
                x = rng.uniform(0.1, 5.0, n_goods)
                y = rng.uniform(0.1, 5.0, n_goods)
                alpha = rng.uniform(0.0, 1.0)

                convex_comb = alpha * x + (1 - alpha) * y
                u_convex = u(convex_comb)
                u_linear = alpha * u(x) + (1 - alpha) * u(y)

                if u_convex < u_linear - 1e-6:  # Violation of concavity
                    is_concave = False
                    break

            results.record(
                f"concavity_seed{seed}",
                is_concave,
                "Utility function is not concave"
            )

    return results


def test_utility_monotonic() -> SimulationResults:
    """
    Test 7: Afriat utility is monotonically increasing.

    x >= y (component-wise) implies u(x) >= u(y)
    """
    results = SimulationResults("Test 7: Utility is Monotonic")
    print(f"\n{'='*60}")
    print(results.name)
    print("="*60)

    for trial in range(10):
        seed = 7000 + trial
        n_obs = 5
        n_goods = 2

        prices, quantities, _ = generate_rational_data(n_obs, n_goods, "cobb_douglas", seed)
        session = ConsumerSession(prices=prices, quantities=quantities)

        utility_result = recover_utility(session)

        if utility_result.success:
            u = construct_afriat_utility(session, utility_result)

            # Test monotonicity at random points
            rng = np.random.default_rng(seed)
            is_monotonic = True

            for _ in range(20):
                x = rng.uniform(0.1, 5.0, n_goods)
                delta = rng.uniform(0.0, 0.5, n_goods)  # Positive increment
                y = x + delta

                if u(y) < u(x) - 1e-6:  # More is worse?
                    is_monotonic = False
                    print(f"    Violation: u({y}) = {u(y):.4f} < u({x}) = {u(x):.4f}")
                    break

            results.record(
                f"monotonicity_seed{seed}",
                is_monotonic,
                "Utility function is not monotonic"
            )

    return results


def test_residuals_nonnegative() -> SimulationResults:
    """
    Test 8: Afriat residuals are non-negative when successful.

    Residual[k,l] = U_l + lambda_l * p_l @ (x_k - x_l) - U_k >= 0
    """
    results = SimulationResults("Test 8: Afriat Residuals Non-negative")
    print(f"\n{'='*60}")
    print(results.name)
    print("="*60)

    for trial in range(15):
        seed = 8000 + trial
        n_obs = np.random.default_rng(seed).integers(5, 10)
        n_goods = np.random.default_rng(seed).integers(2, 4)

        prices, quantities, _ = generate_rational_data(n_obs, n_goods, "cobb_douglas", seed)
        session = ConsumerSession(prices=prices, quantities=quantities)

        utility_result = recover_utility(session)

        if utility_result.success and utility_result.residuals is not None:
            min_residual = np.min(utility_result.residuals)

            results.record(
                f"residuals_nonneg_seed{seed}",
                min_residual >= -1e-6,  # Allow small numerical error
                f"Min residual = {min_residual:.8f}"
            )

    return results


def run_all_tests() -> bool:
    """Run all utility recovery simulation tests."""
    print("\n" + "="*70)
    print("UTILITY RECOVERY SIMULATION STUDY")
    print("="*70)

    all_results = [
        test_recovery_success_for_consistent(),
        test_recovery_failure_for_violations(),
        test_afriat_inequalities_satisfied(),
        test_utility_rationalizes_data(),
        test_lagrange_multipliers_positive(),
        test_utility_concave(),
        test_utility_monotonic(),
        test_residuals_nonnegative(),
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
