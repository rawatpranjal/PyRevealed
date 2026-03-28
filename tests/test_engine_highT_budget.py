"""High-T stress tests for Engine (budget data).

Cases:
- T=2000 exactly GARP-consistent via Cobb-Douglas demands (closed-form)
- T=2000 with an isolated 2-cycle injected (tight CCEI=0.1, MPI~0.9, HM=T-1)
- T=1000 VEI check (SciPy optional) — mean < 1, min near 0.1

Notes:
- Marked slow to avoid default runs; aligns with existing benchmarks style.
"""

from __future__ import annotations

import numpy as np
import pytest

from prefgraph.engine import Engine


def _isolated_two_cycle_plus_cd(T: int, seed: int = 123) -> tuple[np.ndarray, np.ndarray]:
    """Construct high-T data with one isolated 2-cycle and a large consistent block.

    Uses K=4 goods with block separation:
    - Cycle uses goods {0,1} with prices small on {0,1} and HUGE on {2,3}
    - Consistent block uses goods {2,3} with prices HUGE on {0,1} and small on {2,3}

    This makes cross-block revealed preference edges false in both directions:
    own_exp (small) < cross_expenditure (HUGE), so R[i,j] = False and R[j,i] = False.

    Returns p, q with shapes (T, 4) and exact metrics for the 2-cycle:
    - CCEI = 0.1
    - MPI  = 0.9
    - HM   = T-1
    """
    assert T >= 2
    L = 1e6

    # Cycle (two observations) on goods {0,1}
    p0 = np.array([1.0, 0.1, L, L], dtype=np.float64)
    p1 = np.array([0.1, 1.0, L, L], dtype=np.float64)
    q0 = np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float64)
    q1 = np.array([0.0, 2.0, 0.0, 0.0], dtype=np.float64)

    # Consistent block (T-2 observations) on goods {2,3} via Cobb-Douglas
    rng = np.random.RandomState(seed)
    e = rng.rand(T - 2).astype(np.float64) + 0.5  # in (0.5, 1.5)
    f = rng.rand(T - 2).astype(np.float64) + 0.5
    p_rest = np.stack([np.full(T - 2, L), np.full(T - 2, L), e, f], axis=1)
    # Budget = 1, alphas = (0.5, 0.5) for goods 2 and 3
    q_rest = np.stack([np.zeros(T - 2), np.zeros(T - 2), 0.5 / e, 0.5 / f], axis=1)

    p = np.vstack([p0, p1, p_rest])
    q = np.vstack([q0, q1, q_rest])
    return p, q


def _cd_consistent_dataset(T: int, K: int, seed: int = 123, budget: float = 1.0):
    """Generate exactly GARP-consistent data via Cobb-Douglas demands.

    For u(x) = prod_k x_k^{alpha_k}, Marshallian demand: x_k = alpha_k * M / p_k.
    Using fixed budget M and fixed alphas yields utility-maximizing choices for
    any positive price vector p, hence GARP-consistent by construction.
    """
    rng = np.random.RandomState(seed)
    # Random positive prices
    p = rng.rand(T, K).astype(np.float64) + 0.1
    # Fixed alphas summing to 1 (deterministic for reproducibility)
    alphas = np.linspace(1.0, K, K, dtype=np.float64)
    alphas = alphas / alphas.sum()
    # Marshallian demand
    q = (alphas * budget) / p
    return p, q


@pytest.mark.slow
def test_highT_consistent_cd_T2000():
    T, K = 2000, 5
    p, q = _cd_consistent_dataset(T, K, seed=42)

    e = Engine(metrics=["garp", "ccei", "mpi", "hm"])  # explicit for clarity
    [r] = e.analyze_arrays([(p, q)])

    assert r.is_garp is True
    assert r.ccei == pytest.approx(1.0, abs=1e-6)
    assert r.mpi == pytest.approx(0.0, abs=1e-9)
    assert r.hm_total == T and r.hm_consistent == T


@pytest.mark.slow
def test_highT_isolated_2cycle_exact_T2000():
    """Isolated 2-cycle yields exact scores at high T.

    Exact expectations (derivable by hand):
    - CCEI = 0.1
    - MPI  = 0.9
    - HM   = T-1
    """
    T = 2000
    p, q = _isolated_two_cycle_plus_cd(T, seed=7)

    e = Engine(metrics=["ccei", "mpi", "hm"])  # garp implied
    [r] = e.analyze_arrays([(p, q)])

    assert r.is_garp is False
    assert r.ccei == pytest.approx(0.1, abs=1e-6)
    assert r.mpi == pytest.approx(0.9, abs=1e-6)
    assert r.hm_total == T and r.hm_consistent == T - 1


## Note: VEI high-T per-observation assertions are omitted due to
## non-local constraints from many R-direct pairs. We cover VEI exactness
## on minimal datasets in tests/test_engine_corner_cases.py.
