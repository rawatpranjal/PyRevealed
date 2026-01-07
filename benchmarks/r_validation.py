#!/usr/bin/env python3
"""
Cross-validation against R's revealedPrefs package.

Validates that PyRevealed produces identical results to the established
R implementation and benchmarks performance differences.

Requirements:
    - R (>= 4.0) with revealedPrefs: install.packages("revealedPrefs")
    - rpy2: pip install rpy2

Usage:
    python benchmarks/r_validation.py           # Full validation
    python benchmarks/r_validation.py --quick   # Quick test (skip large scale)
    python benchmarks/r_validation.py --correctness-only  # Skip performance
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyrevealed import ConsumerSession
from pyrevealed.algorithms.garp import check_garp, check_warp
from pyrevealed.algorithms.differentiable import check_sarp
from pyrevealed.algorithms.aei import compute_aei


# =============================================================================
# TEST DATA
# =============================================================================


def get_consistent_data() -> tuple[np.ndarray, np.ndarray]:
    """Data that satisfies GARP - rational behavior."""
    prices = np.array([
        [1.0, 2.0],  # Good A cheap
        [2.0, 1.0],  # Good B cheap
        [1.5, 1.5],  # Equal prices
    ])
    quantities = np.array([
        [4.0, 1.0],  # Bought more A (cheap)
        [1.0, 4.0],  # Bought more B (cheap)
        [2.0, 2.0],  # Balanced
    ])
    return prices, quantities


def get_warp_violation_data() -> tuple[np.ndarray, np.ndarray]:
    """Data with WARP violation (2-cycle)."""
    prices = np.array([
        [1.0, 0.1],  # A expensive, B cheap
        [0.1, 1.0],  # A cheap, B expensive
    ])
    quantities = np.array([
        [1.0, 0.0],  # Chose A (expensive!)
        [0.0, 1.0],  # Chose B (expensive!)
    ])
    return prices, quantities


def get_garp_3cycle_data() -> tuple[np.ndarray, np.ndarray]:
    """Data with GARP violation (3-cycle) but no WARP violation.

    Each observation can afford the next in cycle, but not both directions
    directly, so no 2-cycle exists (WARP passes).
    """
    prices = np.array([
        [1.0, 0.1, 0.1],  # A expensive
        [0.1, 1.0, 0.1],  # B expensive
        [0.1, 0.1, 1.0],  # C expensive
    ])
    quantities = np.array([
        [1.0, 0.0, 0.0],  # Chose A
        [0.0, 1.0, 0.0],  # Chose B
        [0.0, 0.0, 1.0],  # Chose C
    ])
    return prices, quantities


def get_sarp_violation_data() -> tuple[np.ndarray, np.ndarray]:
    """Data that passes GARP but violates SARP (indifference cycle).

    Different bundles that cost the same at each other's prices creates
    mutual weak revealed preference (indifference), which violates SARP.

    NOTE: PyRevealed and R have different WARP definitions:
    - PyRevealed: R[i,j] AND P[j,i] (strict-weak asymmetry)
    - R: R[i,j] AND R[j,i] (any mutual preference)
    This data fails R's WARP but passes PyRevealed's WARP.
    """
    prices = np.array([
        [1.0, 1.0],
        [1.0, 1.0],
    ])
    quantities = np.array([
        [3.0, 1.0],  # Chose more A
        [1.0, 3.0],  # Chose more B (same cost at these prices)
    ])
    return prices, quantities


def get_near_violation_data() -> tuple[np.ndarray, np.ndarray]:
    """Data that barely violates GARP - useful for Afriat efficiency testing.

    Returns data that fails at e=1.0 but passes at lower efficiency.
    """
    # Mild violation that can be "fixed" with small efficiency adjustment
    prices = np.array([
        [1.0, 1.0],
        [1.0, 1.0],
    ])
    quantities = np.array([
        [3.0, 1.0],  # Chose more A, spent 4
        [1.0, 3.0],  # Chose more B, spent 4
    ])
    return prices, quantities


def generate_random_data(n_obs: int, n_goods: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate random data for performance testing."""
    np.random.seed(seed)
    prices = np.random.uniform(0.5, 2.0, (n_obs, n_goods))
    quantities = np.random.uniform(0.1, 10.0, (n_obs, n_goods))
    return prices, quantities


def generate_rational_data(n_obs: int, n_goods: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate GARP-consistent (rational) data for performance testing.

    Uses Cobb-Douglas utility maximization which always satisfies GARP.
    """
    np.random.seed(seed)

    # Random Cobb-Douglas exponents (preferences)
    alphas = np.random.dirichlet(np.ones(n_goods))

    # Random budgets and prices
    budgets = np.random.uniform(10, 100, n_obs)
    prices = np.random.uniform(0.5, 2.0, (n_obs, n_goods))

    # Optimal quantities for Cobb-Douglas: x_i = alpha_i * budget / p_i
    quantities = np.zeros((n_obs, n_goods))
    for t in range(n_obs):
        for i in range(n_goods):
            quantities[t, i] = alphas[i] * budgets[t] / prices[t, i]

    return prices, quantities


# =============================================================================
# R INTERFACE
# =============================================================================


def check_r_available() -> bool:
    """Check if R and revealedPrefs are available."""
    try:
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr

        importr("revealedPrefs")
        return True
    except Exception:
        return False


def r_check_garp(prices: np.ndarray, quantities: np.ndarray) -> bool:
    """Call R's checkGarp and return whether data passes (no violation)."""
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr

    revealedPrefs = importr("revealedPrefs")

    # Convert to R matrices (revealedPrefs uses x=quantities, p=prices)
    r_x = ro.r.matrix(
        ro.FloatVector(quantities.flatten().tolist()),
        nrow=quantities.shape[0],
        byrow=True,
    )
    r_p = ro.r.matrix(
        ro.FloatVector(prices.flatten().tolist()),
        nrow=prices.shape[0],
        byrow=True,
    )

    result = revealedPrefs.checkGarp(r_x, r_p)

    # Extract violation field
    violation = result.rx2("violation")[0]

    return not violation  # Return True if PASSES (no violation)


def r_check_warp(prices: np.ndarray, quantities: np.ndarray) -> bool:
    """Call R's checkWarp and return whether data passes (no violation)."""
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr

    revealedPrefs = importr("revealedPrefs")

    r_x = ro.r.matrix(
        ro.FloatVector(quantities.flatten().tolist()),
        nrow=quantities.shape[0],
        byrow=True,
    )
    r_p = ro.r.matrix(
        ro.FloatVector(prices.flatten().tolist()),
        nrow=prices.shape[0],
        byrow=True,
    )

    result = revealedPrefs.checkWarp(r_x, r_p)
    violation = result.rx2("violation")[0]

    return not violation


def r_check_sarp(prices: np.ndarray, quantities: np.ndarray) -> bool:
    """Call R's checkSarp and return whether data passes (no violation)."""
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr

    revealedPrefs = importr("revealedPrefs")

    r_x = ro.r.matrix(
        ro.FloatVector(quantities.flatten().tolist()),
        nrow=quantities.shape[0],
        byrow=True,
    )
    r_p = ro.r.matrix(
        ro.FloatVector(prices.flatten().tolist()),
        nrow=prices.shape[0],
        byrow=True,
    )

    result = revealedPrefs.checkSarp(r_x, r_p)
    violation = result.rx2("violation")[0]

    return not violation


def r_check_garp_with_efficiency(
    prices: np.ndarray, quantities: np.ndarray, afriat_par: float
) -> bool:
    """Call R's checkGarp with afriat.par parameter."""
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr

    revealedPrefs = importr("revealedPrefs")

    r_x = ro.r.matrix(
        ro.FloatVector(quantities.flatten().tolist()),
        nrow=quantities.shape[0],
        byrow=True,
    )
    r_p = ro.r.matrix(
        ro.FloatVector(prices.flatten().tolist()),
        nrow=prices.shape[0],
        byrow=True,
    )

    # Call with afriat.par parameter
    result = revealedPrefs.checkGarp(r_x, r_p, **{"afriat.par": afriat_par})
    violation = result.rx2("violation")[0]

    return not violation


def r_direct_prefs(prices: np.ndarray, quantities: np.ndarray) -> np.ndarray:
    """Call R's directPrefs and return the preference matrix."""
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr

    revealedPrefs = importr("revealedPrefs")

    r_x = ro.r.matrix(
        ro.FloatVector(quantities.flatten().tolist()),
        nrow=quantities.shape[0],
        byrow=True,
    )
    r_p = ro.r.matrix(
        ro.FloatVector(prices.flatten().tolist()),
        nrow=prices.shape[0],
        byrow=True,
    )

    result = revealedPrefs.directPrefs(r_x, r_p)

    # Convert R matrix to numpy
    T = quantities.shape[0]
    matrix = np.array(result).reshape(T, T)

    return matrix


# =============================================================================
# VALIDATION
# =============================================================================


@dataclass
class AxiomTestResult:
    """Result of a single axiom test."""
    axiom: str
    test_name: str
    pyrevealed_pass: bool
    r_pass: bool
    match: bool


@dataclass
class EfficiencyTestResult:
    """Result of an Afriat efficiency test."""
    efficiency: float
    pyrevealed_pass: bool
    r_pass: bool
    match: bool


@dataclass
class MatrixComparisonResult:
    """Result of a matrix comparison."""
    matrix_name: str
    match: bool
    max_diff: float


@dataclass
class PerformanceResult:
    """Result of a performance comparison."""
    n_observations: int
    pyrevealed_ms: float
    r_ms: float
    speedup: float


def run_axiom_tests() -> list[AxiomTestResult]:
    """Run axiom validation tests for GARP, WARP, and SARP.

    Note: WARP has different definitions in PyRevealed vs R:
    - PyRevealed: R[i,j] AND P[j,i] (strict-weak asymmetry)
    - R revealedPrefs: R[i,j] AND R[j,i] (any mutual preference)

    We only compare WARP on test cases where both definitions agree.
    """
    results = []

    # Test cases - only use ones where WARP definitions agree
    # Skip WARP comparison for sarp_violation (known definitional difference)
    test_cases = [
        ("consistent_3obs", get_consistent_data),
        ("warp_violation", get_warp_violation_data),
        ("garp_3cycle", get_garp_3cycle_data),
    ]

    for name, data_func in test_cases:
        prices, quantities = data_func()
        session = ConsumerSession(prices=prices, quantities=quantities)

        # GARP test
        py_garp = check_garp(session).is_consistent
        r_garp = r_check_garp(prices, quantities)
        results.append(AxiomTestResult(
            axiom="GARP",
            test_name=name,
            pyrevealed_pass=py_garp,
            r_pass=r_garp,
            match=(py_garp == r_garp),
        ))

        # WARP test
        py_warp, _ = check_warp(session)
        r_warp = r_check_warp(prices, quantities)
        results.append(AxiomTestResult(
            axiom="WARP",
            test_name=name,
            pyrevealed_pass=py_warp,
            r_pass=r_warp,
            match=(py_warp == r_warp),
        ))

        # SARP test
        py_sarp, _ = check_sarp(session)
        r_sarp = r_check_sarp(prices, quantities)
        results.append(AxiomTestResult(
            axiom="SARP",
            test_name=name,
            pyrevealed_pass=py_sarp,
            r_pass=r_sarp,
            match=(py_sarp == r_sarp),
        ))

    # Add sarp_violation test (only GARP and SARP - WARP differs by design)
    prices, quantities = get_sarp_violation_data()
    session = ConsumerSession(prices=prices, quantities=quantities)

    # GARP
    py_garp = check_garp(session).is_consistent
    r_garp = r_check_garp(prices, quantities)
    results.append(AxiomTestResult(
        axiom="GARP",
        test_name="sarp_violation",
        pyrevealed_pass=py_garp,
        r_pass=r_garp,
        match=(py_garp == r_garp),
    ))

    # SARP
    py_sarp, _ = check_sarp(session)
    r_sarp = r_check_sarp(prices, quantities)
    results.append(AxiomTestResult(
        axiom="SARP",
        test_name="sarp_violation",
        pyrevealed_pass=py_sarp,
        r_pass=r_sarp,
        match=(py_sarp == r_sarp),
    ))

    return results


def run_efficiency_tests() -> list[EfficiencyTestResult]:
    """Run Afriat efficiency parameter tests."""
    results = []

    # Use the warp_violation data which fails at e=1.0
    prices, quantities = get_warp_violation_data()
    session = ConsumerSession(prices=prices, quantities=quantities)

    # Test at different efficiency levels
    for e in [1.0, 0.9, 0.8, 0.5]:
        # PyRevealed: use AEI to check if data passes at efficiency e
        aei_result = compute_aei(session, tolerance=1e-6)
        py_pass = aei_result.efficiency_index >= e

        # R: use afriat.par
        r_pass = r_check_garp_with_efficiency(prices, quantities, e)

        results.append(EfficiencyTestResult(
            efficiency=e,
            pyrevealed_pass=py_pass,
            r_pass=r_pass,
            match=(py_pass == r_pass),
        ))

    return results


def run_matrix_comparison() -> list[MatrixComparisonResult]:
    """Compare preference matrices between PyRevealed and R."""
    results = []

    prices, quantities = get_consistent_data()
    session = ConsumerSession(prices=prices, quantities=quantities)

    # Get PyRevealed's direct revealed preference matrix
    garp_result = check_garp(session)
    py_matrix = garp_result.direct_revealed_preference.astype(float)

    # Get R's directPrefs matrix
    r_matrix = r_direct_prefs(prices, quantities)

    # Compare - R uses 0/1 integers, PyRevealed uses True/False
    # directPrefs in R returns 1 if i R j (revealed preferred), 0 otherwise
    max_diff = np.max(np.abs(py_matrix - r_matrix))
    match = max_diff < 0.5  # Allow for boolean vs int differences

    results.append(MatrixComparisonResult(
        matrix_name="Direct prefs (R)",
        match=match,
        max_diff=max_diff,
    ))

    return results


# Legacy function for backwards compatibility
def run_correctness_tests() -> list[AxiomTestResult]:
    """Run correctness validation tests (GARP only for backward compat)."""
    all_results = run_axiom_tests()
    # Filter to GARP only for legacy output
    return [r for r in all_results if r.axiom == "GARP"]


def run_performance_tests(scale_levels: list[int], n_goods: int = 10) -> list[PerformanceResult]:
    """Run performance comparison at different scales."""
    results = []

    # JIT warmup with small data first
    warmup_p, warmup_q = generate_random_data(50, n_goods, seed=99)
    warmup_session = ConsumerSession(prices=warmup_p, quantities=warmup_q)
    for _ in range(3):
        check_garp(warmup_session)

    for n_obs in scale_levels:
        # Use rational data (GARP-consistent) for fair performance comparison
        # Random data has many violations which triggers slow cycle reconstruction
        prices, quantities = generate_rational_data(n_obs, n_goods)
        session = ConsumerSession(prices=prices, quantities=quantities)

        # Warmup for this size
        check_garp(session)
        r_check_garp(prices, quantities)

        # PyRevealed timing (average of 3 runs)
        py_times = []
        for _ in range(3):
            start = time.perf_counter()
            check_garp(session)
            py_times.append((time.perf_counter() - start) * 1000)
        py_ms = np.mean(py_times)

        # R timing (average of 3 runs)
        r_times = []
        for _ in range(3):
            start = time.perf_counter()
            r_check_garp(prices, quantities)
            r_times.append((time.perf_counter() - start) * 1000)
        r_ms = np.mean(r_times)

        speedup = r_ms / py_ms if py_ms > 0 else float("inf")

        results.append(PerformanceResult(
            n_observations=n_obs,
            pyrevealed_ms=py_ms,
            r_ms=r_ms,
            speedup=speedup,
        ))

    return results


# =============================================================================
# OUTPUT
# =============================================================================


def print_banner(text: str, char: str = "=", width: int = 70) -> None:
    """Print a banner line."""
    print()
    print(char * width)
    print(f" {text}")
    print(char * width)


def print_axiom_results(results: list[AxiomTestResult]) -> bool:
    """Print axiom test results table. Returns True if all match."""
    print_banner("AXIOM TESTS")

    print(f"{'Axiom':<8} {'Test Case':<20} {'PyRevealed':<12} {'revealedPrefs':<14} {'Match':<10}")
    print("-" * 70)

    all_match = True
    for r in results:
        py_str = "PASS" if r.pyrevealed_pass else "FAIL"
        r_str = "PASS" if r.r_pass else "FAIL"
        match_str = "OK" if r.match else "MISMATCH"
        print(f"{r.axiom:<8} {r.test_name:<20} {py_str:<12} {r_str:<14} {match_str:<10}")
        if not r.match:
            all_match = False

    return all_match


def print_efficiency_results(results: list[EfficiencyTestResult]) -> bool:
    """Print Afriat efficiency test results. Returns True if all match."""
    print_banner("AFRIAT EFFICIENCY TESTS")

    print(f"{'Efficiency':<12} {'PyRevealed':<12} {'revealedPrefs':<14} {'Match':<10}")
    print("-" * 50)

    all_match = True
    for r in results:
        py_str = "PASS" if r.pyrevealed_pass else "FAIL"
        r_str = "PASS" if r.r_pass else "FAIL"
        match_str = "OK" if r.match else "MISMATCH"
        print(f"{r.efficiency:<12.1f} {py_str:<12} {r_str:<14} {match_str:<10}")
        if not r.match:
            all_match = False

    return all_match


def print_matrix_results(results: list[MatrixComparisonResult]) -> bool:
    """Print matrix comparison results. Returns True if all match."""
    print_banner("MATRIX COMPARISON")

    print(f"{'Matrix':<20} {'Match':<10} {'Max Diff':<10}")
    print("-" * 40)

    all_match = True
    for r in results:
        match_str = "OK" if r.match else "MISMATCH"
        print(f"{r.matrix_name:<20} {match_str:<10} {r.max_diff:<10.4f}")
        if not r.match:
            all_match = False

    return all_match


# Legacy function
def print_correctness_results(results: list[AxiomTestResult]) -> bool:
    """Print correctness results table (legacy). Returns True if all match."""
    print_banner("CORRECTNESS VALIDATION")

    print(f"{'Test Case':<25} {'PyRevealed':<15} {'revealedPrefs':<15} {'Match':<10}")
    print("-" * 65)

    all_match = True
    for r in results:
        py_str = "PASS" if r.pyrevealed_pass else "FAIL"
        r_str = "PASS" if r.r_pass else "FAIL"
        match_str = "OK" if r.match else "MISMATCH"
        print(f"{r.test_name:<25} {py_str:<15} {r_str:<15} {match_str:<10}")
        if not r.match:
            all_match = False

    return all_match


def print_performance_results(results: list[PerformanceResult]) -> None:
    """Print performance comparison table."""
    print_banner("PERFORMANCE COMPARISON")

    print(f"{'T':<10} {'PyRevealed':<15} {'revealedPrefs':<15} {'Speedup':<10}")
    print("-" * 50)

    for r in results:
        py_str = format_time(r.pyrevealed_ms)
        r_str = format_time(r.r_ms)
        speedup_str = f"{r.speedup:.0f}x"
        print(f"{r.n_observations:<10} {py_str:<15} {r_str:<15} {speedup_str:<10}")


def format_time(ms: float) -> str:
    """Format time in ms or seconds."""
    if ms >= 1000:
        return f"{ms/1000:.1f}s"
    return f"{ms:.1f}ms"


# =============================================================================
# MAIN
# =============================================================================


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Cross-validation against R revealedPrefs package"
    )
    parser.add_argument("--quick", action="store_true", help="Quick mode (smaller scale)")
    parser.add_argument(
        "--correctness-only", action="store_true", help="Skip performance tests"
    )
    parser.add_argument(
        "--basic", action="store_true", help="Basic mode (GARP only, legacy output)"
    )
    args = parser.parse_args()

    print_banner("PYREVEALED vs R revealedPrefs VALIDATION")

    # Check R availability
    if not check_r_available():
        print("\nERROR: R with revealedPrefs package not available.")
        print("\nTo install:")
        print("  1. Install R (https://cran.r-project.org/)")
        print("  2. In R: install.packages('revealedPrefs')")
        print("  3. pip install rpy2")
        return 1

    print("\nR and revealedPrefs package detected.")

    all_pass = True

    if args.basic:
        # Legacy mode: GARP only
        correctness_results = run_correctness_tests()
        all_pass = print_correctness_results(correctness_results)
    else:
        # Full validation: GARP, WARP, SARP, efficiency, matrices

        # Axiom tests (GARP, WARP, SARP)
        axiom_results = run_axiom_tests()
        axiom_pass = print_axiom_results(axiom_results)
        all_pass = all_pass and axiom_pass

        # Afriat efficiency tests
        efficiency_results = run_efficiency_tests()
        efficiency_pass = print_efficiency_results(efficiency_results)
        all_pass = all_pass and efficiency_pass

        # Matrix comparison tests
        matrix_results = run_matrix_comparison()
        matrix_pass = print_matrix_results(matrix_results)
        all_pass = all_pass and matrix_pass

    if not all_pass:
        print("\nWARNING: Some validation tests FAILED - results do not match!")
        return 1

    # Count tests
    if args.basic:
        n_tests = len(correctness_results)
    else:
        n_tests = len(axiom_results) + len(efficiency_results) + len(matrix_results)

    print(f"\nAll {n_tests} validation tests PASSED.")

    # Performance tests
    if not args.correctness_only:
        scale_levels = [100, 500] if args.quick else [100, 500, 1000, 2000]
        performance_results = run_performance_tests(scale_levels)
        print_performance_results(performance_results)

        avg_speedup = np.mean([r.speedup for r in performance_results])
        print(f"\nAverage speedup: {avg_speedup:.0f}x faster than R")

    print_banner("VALIDATION COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
