from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pyrevealed.core.types import Cycle
from pyrevealed.core.mixins import ResultSummaryMixin
from pyrevealed.core.display import ResultDisplayMixin, ResultPlotMixin

__all__ = [
    "GARPResult",
    "AEIResult",
    "MPIResult",
    "UtilityRecoveryResult",
    "ConsistencyResult",
    "IntegrityResult",
    "ConfusionResult",
    "LatentValueResult",
]


@dataclass(frozen=True)
class GARPResult(ResultDisplayMixin, ResultPlotMixin):
    """
    Result of GARP (Generalized Axiom of Revealed Preference) consistency test.

    GARP is satisfied when there are no cycles in the revealed preference
    relation that include at least one strict preference. A violation indicates
    the consumer made inconsistent choices.

    Attributes:
        is_consistent: True if data satisfies GARP (no violations found)
        violations: List of violation cycles (tuples of observation indices)
        direct_revealed_preference: T x T boolean matrix R where R[i,j] = True
            iff bundle i is directly revealed preferred to bundle j
            (i.e., p_i @ x_i >= p_i @ x_j)
        transitive_closure: T x T boolean matrix R* (transitive closure of R)
        strict_revealed_preference: T x T boolean matrix P where P[i,j] = True
            iff bundle i is strictly revealed preferred to bundle j
            (i.e., p_i @ x_i > p_i @ x_j)
        computation_time_ms: Time taken to compute result in milliseconds
    """

    is_consistent: bool
    violations: list[Cycle]
    direct_revealed_preference: NDArray[np.bool_]
    transitive_closure: NDArray[np.bool_]
    strict_revealed_preference: NDArray[np.bool_]
    computation_time_ms: float

    @property
    def num_violations(self) -> int:
        """Number of violation cycles found."""
        return len(self.violations)

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns 1.0 if consistent (no violations), 0.0 if violations exist.
        """
        return 1.0 if self.is_consistent else 0.0

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("GARP CONSISTENCY REPORT")]

        # Status
        status = m._format_status(self.is_consistent, "CONSISTENT", "VIOLATIONS FOUND")
        lines.append(f"\nStatus: {status}")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("Consistent", self.is_consistent))
        lines.append(m._format_metric("Violations", self.num_violations))
        n_obs = self.direct_revealed_preference.shape[0]
        lines.append(m._format_metric("Observations", n_obs))

        # Show first violation if any
        if self.violations:
            lines.append(m._format_section("First Violation Cycle"))
            lines.append(f"  {self.violations[0]}")
            if len(self.violations) > 1:
                lines.append(f"  ... and {len(self.violations) - 1} more cycles")

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        if self.is_consistent:
            lines.append("  Behavior is consistent with utility maximization.")
            lines.append("  No revealed preference cycles detected.")
        else:
            lines.append("  Behavior violates GARP - inconsistent with utility maximization.")
            lines.append(f"  Found {self.num_violations} preference cycle(s).")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "is_consistent": self.is_consistent,
            "num_violations": self.num_violations,
            "violations": [list(c) for c in self.violations],
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation with [+]/[-] indicator."""
        indicator = "[+]" if self.is_consistent else "[-]"
        status = "CONSISTENT" if self.is_consistent else f"{self.num_violations} violations"
        return f"GARPResult: {indicator} {status} ({self.computation_time_ms:.2f}ms)"

    def short_summary(self) -> str:
        """Return one-liner with [+]/[-] indicator."""
        indicator = "[+]" if self.is_consistent else "[-]"
        status = "CONSISTENT" if self.is_consistent else f"{self.num_violations} violations"
        return f"GARP: {indicator} {status}"


@dataclass(frozen=True)
class AEIResult(ResultDisplayMixin, ResultPlotMixin):
    """
    Result of Afriat Efficiency Index computation.

    The AEI measures how close consumer behavior is to perfect rationality.
    It is defined as: sup{e in [0,1] : <R_e, P_e> is acyclic}
    where R_e is the revealed preference relation with budgets deflated by e.

    An AEI of 1.0 means perfectly consistent behavior.
    An AEI of 0.5 means the consumer wastes ~50% of their budget on
    inconsistent choices.

    Attributes:
        efficiency_index: The computed AEI score in [0, 1]
        is_perfectly_consistent: True if AEI = 1.0 (data satisfies GARP)
        garp_result_at_threshold: GARP result at the efficiency threshold
        binary_search_iterations: Number of iterations in binary search
        tolerance: Convergence tolerance used
        computation_time_ms: Time taken in milliseconds
    """

    efficiency_index: float
    is_perfectly_consistent: bool
    garp_result_at_threshold: GARPResult
    binary_search_iterations: int
    tolerance: float
    computation_time_ms: float

    @property
    def waste_fraction(self) -> float:
        """Fraction of budget wasted on inconsistent choices (1 - AEI)."""
        return 1.0 - self.efficiency_index

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns the efficiency index directly (already in [0, 1]).
        """
        return self.efficiency_index

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("AFRIAT EFFICIENCY INDEX REPORT")]

        # Status
        if self.is_perfectly_consistent:
            status = "PERFECT (AEI = 1.0)"
        elif self.efficiency_index >= 0.95:
            status = "EXCELLENT"
        elif self.efficiency_index >= 0.9:
            status = "GOOD"
        elif self.efficiency_index >= 0.7:
            status = "MODERATE"
        else:
            status = "LOW"
        lines.append(f"\nStatus: {status}")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("Efficiency Index (AEI)", self.efficiency_index))
        lines.append(m._format_metric("Waste Fraction", self.waste_fraction))
        lines.append(m._format_metric("Perfectly Consistent", self.is_perfectly_consistent))
        lines.append(m._format_metric("Binary Search Iterations", self.binary_search_iterations))
        lines.append(m._format_metric("Tolerance", self.tolerance))

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        lines.append(f"  {m._format_interpretation(self.efficiency_index, 'efficiency')}")
        if self.waste_fraction > 0:
            pct = self.waste_fraction * 100
            lines.append(f"  Approximately {pct:.1f}% of budget on inconsistent choices.")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "efficiency_index": self.efficiency_index,
            "is_perfectly_consistent": self.is_perfectly_consistent,
            "waste_fraction": self.waste_fraction,
            "binary_search_iterations": self.binary_search_iterations,
            "tolerance": self.tolerance,
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation with [+]/[-] indicator."""
        indicator = "[+]" if self.efficiency_index >= 0.95 else "[-]"
        return f"AEIResult: {indicator} AEI={self.efficiency_index:.4f} ({self.computation_time_ms:.2f}ms)"

    def short_summary(self) -> str:
        """Return one-liner with [+]/[-] indicator."""
        indicator = "[+]" if self.efficiency_index >= 0.95 else "[-]"
        return f"AEI: {indicator} {self.efficiency_index:.4f}"


@dataclass(frozen=True)
class MPIResult(ResultDisplayMixin, ResultPlotMixin):
    """
    Result of Money Pump Index computation.

    The MPI measures the percentage of total expenditure that could be
    "pumped" from a consumer exhibiting cyclic preferences by an arbitrager.

    For a cycle k1 -> k2 -> ... -> kn -> k1:
    MPI = sum(p_ki @ (x_ki - x_{ki+1})) / sum(p_ki @ x_ki)

    Attributes:
        mpi_value: Maximum MPI across all violation cycles (0 if consistent)
        worst_cycle: The cycle with highest MPI (None if consistent)
        cycle_costs: List of (cycle, mpi) pairs for all violation cycles
        total_expenditure: Sum of all expenditures in the session
        computation_time_ms: Time taken in milliseconds
    """

    mpi_value: float
    worst_cycle: Cycle | None
    cycle_costs: list[tuple[Cycle, float]]
    total_expenditure: float
    computation_time_ms: float

    @property
    def is_consistent(self) -> bool:
        """True if no money pump exists (MPI = 0)."""
        return self.mpi_value == 0.0

    @property
    def num_cycles(self) -> int:
        """Number of violation cycles found."""
        return len(self.cycle_costs)

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns 1 - mpi_value (inverted so higher is better).
        MPI is capped at 1.0 for score calculation.
        """
        return max(0.0, 1.0 - min(1.0, self.mpi_value))

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("MONEY PUMP INDEX REPORT")]

        # Status
        if self.is_consistent:
            status = "NO EXPLOITABILITY"
        elif self.mpi_value < 0.05:
            status = "VERY LOW EXPLOITABILITY"
        elif self.mpi_value < 0.15:
            status = "LOW EXPLOITABILITY"
        else:
            status = "HIGH EXPLOITABILITY"
        lines.append(f"\nStatus: {status}")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("Money Pump Index (MPI)", self.mpi_value))
        lines.append(m._format_metric("Exploitability %", self.mpi_value * 100))
        lines.append(m._format_metric("Number of Cycles", self.num_cycles))
        lines.append(m._format_metric("Total Expenditure", self.total_expenditure))

        # Show worst cycle if any
        if self.worst_cycle:
            lines.append(m._format_section("Worst Violation Cycle"))
            lines.append(f"  Cycle: {self.worst_cycle}")
            # Find MPI for worst cycle
            for cycle, mpi in self.cycle_costs:
                if cycle == self.worst_cycle:
                    lines.append(f"  Cycle MPI: {mpi:.4f}")
                    break

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        lines.append(f"  {m._format_interpretation(self.score(), 'mpi')}")
        if self.mpi_value > 0:
            pct = self.mpi_value * 100
            lines.append(f"  An arbitrager could extract ~{pct:.1f}% of expenditure.")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        return {
            "mpi_value": self.mpi_value,
            "is_consistent": self.is_consistent,
            "num_cycles": self.num_cycles,
            "worst_cycle": list(self.worst_cycle) if self.worst_cycle else None,
            "total_expenditure": self.total_expenditure,
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        """Compact string representation with [+]/[-] indicator."""
        indicator = "[+]" if self.is_consistent else "[-]"
        status = "CONSISTENT" if self.is_consistent else f"MPI={self.mpi_value:.4f}"
        return f"MPIResult: {indicator} {status} ({self.computation_time_ms:.2f}ms)"

    def short_summary(self) -> str:
        """Return one-liner with [+]/[-] indicator."""
        indicator = "[+]" if self.is_consistent else "[-]"
        status = "0.0000" if self.is_consistent else f"{self.mpi_value:.4f}"
        return f"MPI: {indicator} {status}"


@dataclass(frozen=True)
class UtilityRecoveryResult(ResultDisplayMixin, ResultPlotMixin):
    """
    Result of utility recovery via linear programming (Afriat's inequalities).

    If the data satisfies GARP, we can recover utility values U_k and
    Lagrange multipliers (marginal utility of money) lambda_k such that:
    U_k <= U_l + lambda_l * p_l @ (x_k - x_l) for all k, l

    The recovered utility function is piecewise linear and concave.

    Attributes:
        success: True if LP found a feasible solution
        utility_values: Array of U_k values (utility at each observation)
        lagrange_multipliers: Array of lambda_k values (marginal utility of money)
        lp_status: Status message from the LP solver
        residuals: Matrix of Afriat inequality residuals (for verification)
        computation_time_ms: Time taken in milliseconds
    """

    success: bool
    utility_values: NDArray[np.float64] | None
    lagrange_multipliers: NDArray[np.float64] | None
    lp_status: str
    residuals: NDArray[np.float64] | None
    computation_time_ms: float

    @property
    def mean_marginal_utility(self) -> float | None:
        """Average marginal utility of money across observations."""
        if self.lagrange_multipliers is None:
            return None
        return float(np.mean(self.lagrange_multipliers))

    def score(self) -> float:
        """Return scikit-learn style score in [0, 1]. Higher is better.

        Returns 1.0 if utility recovery succeeded, 0.0 if failed.
        """
        return 1.0 if self.success else 0.0

    def summary(self) -> str:
        """Return human-readable summary report."""
        m = ResultSummaryMixin
        lines = [m._format_header("UTILITY RECOVERY REPORT")]

        # Status
        status = m._format_status(self.success, "SUCCESS", "FAILED")
        lines.append(f"\nStatus: {status}")
        lines.append(f"LP Status: {self.lp_status}")

        # Metrics
        lines.append(m._format_section("Metrics"))
        lines.append(m._format_metric("Recovery Successful", self.success))

        if self.success and self.utility_values is not None:
            lines.append(m._format_metric("Utility Values Range",
                                          f"[{np.min(self.utility_values):.4f}, {np.max(self.utility_values):.4f}]"))
            lines.append(m._format_metric("Mean Utility", float(np.mean(self.utility_values))))
            lines.append(m._format_metric("Num Observations", len(self.utility_values)))

        if self.lagrange_multipliers is not None:
            lines.append(m._format_metric("Mean Marginal Utility", self.mean_marginal_utility))
            lines.append(m._format_metric("Lagrange Range",
                                          f"[{np.min(self.lagrange_multipliers):.4f}, {np.max(self.lagrange_multipliers):.4f}]"))

        # Interpretation
        lines.append(m._format_section("Interpretation"))
        if self.success:
            lines.append("  Utility function successfully recovered.")
            lines.append("  The recovered utility is piecewise linear and concave.")
        else:
            lines.append("  Utility recovery failed - data may violate GARP.")
            lines.append("  Run consistency check first to identify violations.")

        lines.append(m._format_footer(self.computation_time_ms))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for serialization."""
        result = {
            "success": self.success,
            "lp_status": self.lp_status,
            "computation_time_ms": self.computation_time_ms,
            "score": self.score(),
        }
        if self.utility_values is not None:
            result["utility_values"] = self.utility_values.tolist()
        if self.lagrange_multipliers is not None:
            result["lagrange_multipliers"] = self.lagrange_multipliers.tolist()
            result["mean_marginal_utility"] = self.mean_marginal_utility
        return result

    def __repr__(self) -> str:
        """Compact string representation with [+]/[-] indicator."""
        indicator = "[+]" if self.success else "[-]"
        if self.success:
            n = len(self.utility_values) if self.utility_values is not None else 0
            return f"UtilityRecoveryResult: {indicator} SUCCESS (n={n}, {self.computation_time_ms:.2f}ms)"
        return f"UtilityRecoveryResult: {indicator} FAILED ({self.computation_time_ms:.2f}ms)"

    def short_summary(self) -> str:
        """Return one-liner with [+]/[-] indicator."""
        indicator = "[+]" if self.success else "[-]"
        status = "SUCCESS" if self.success else "FAILED"
        return f"Utility Recovery: {indicator} {status}"


# =============================================================================
# TECH-FRIENDLY ALIASES (Primary names)
# =============================================================================

# ConsistencyResult: Result of behavioral consistency validation
ConsistencyResult = GARPResult
"""
Tech-friendly alias for GARPResult.

Use this to check if user behavior is internally consistent.
Consistent behavior = not a bot, single user account.
"""

# IntegrityResult: Result of integrity/noise score computation
IntegrityResult = AEIResult
"""
Tech-friendly alias for AEIResult.

The integrity score (0-1) indicates data quality:
- 1.0 = Perfect signal, fully consistent user
- 0.5 = Noisy signal, possible bot or confused user
- <0.5 = Very noisy, likely bot or shared account
"""

# ConfusionResult: Result of confusion/exploitability metric
ConfusionResult = MPIResult
"""
Tech-friendly alias for MPIResult.

The confusion score indicates how exploitable the user's decisions are.
High confusion = bad UX causing irrational choices.
"""

# LatentValueResult: Result of latent preference extraction
LatentValueResult = UtilityRecoveryResult
"""
Tech-friendly alias for UtilityRecoveryResult.

Contains extracted latent preference values that can be used as
features for ML models or for counterfactual simulations.
"""
