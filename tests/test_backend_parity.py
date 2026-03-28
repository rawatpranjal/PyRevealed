"""Cross-backend parity tests: Python vs Rust produce consistent results.

Tests that _analyze_chunk_python and _analyze_chunk_rust produce equivalent
results within known tolerance bounds for shared metrics (GARP, CCEI, MPI).

Tolerance rationale:
- GARP (bool): exact match - same Floyd-Warshall algorithm
- CCEI (float): within 0.01 - same discrete binary search, minor float rounding
- MPI (float): within 0.05 - Python uses cycle-enumeration, Rust uses Karp's
  max-mean-weight cycle - different algorithms, same theoretical target
"""

import pytest
import numpy as np
from prefgraph._rust_backend import HAS_RUST
from prefgraph.engine import Engine

pytestmark = pytest.mark.skipif(not HAS_RUST, reason="Rust backend not available")

FLAGS = {"ccei": True, "mpi": True, "harp": False, "hm": False,
         "utility": False, "vei": False, "vei_exact": False}


def _run_both(chunk):
    """Run both Python and Rust backends on the same data."""
    engine = Engine(metrics=["garp", "ccei", "mpi"])
    py = engine._analyze_chunk_python(chunk, FLAGS)
    rust = engine._analyze_chunk_rust(chunk, FLAGS)
    return py, rust


# --- Fixtures ---

@pytest.fixture
def consistent_data():
    """GARP-consistent: budget line rotation, no violations."""
    p = np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float64)
    q = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    return [(p, q)]


@pytest.fixture
def violation_data():
    """Random data (seed=123) that produces GARP violations.

    Varied prices + random quantities create strict revealed preferences
    that form cycles - unlike equal-price data where all expenditures match.
    """
    rng = np.random.RandomState(123)
    p = (rng.rand(20, 3) + 0.1).astype(np.float64)
    q = (rng.rand(20, 3) * 10).astype(np.float64)
    return [(p, q)]


@pytest.fixture
def random_users():
    """50 random users for statistical parity checking."""
    rng = np.random.RandomState(42)
    users = []
    for _ in range(50):
        p = (rng.rand(10, 5) + 0.1).astype(np.float64)
        q = (rng.rand(10, 5) + 0.1).astype(np.float64)
        users.append((p, q))
    return users


# --- GARP parity ---

class TestGARPParity:
    def test_consistent(self, consistent_data):
        py, rust = _run_both(consistent_data)
        assert py[0].is_garp == rust[0].is_garp is True

    def test_violation(self, violation_data):
        py, rust = _run_both(violation_data)
        assert py[0].is_garp == rust[0].is_garp is False

    def test_random_match(self, random_users):
        py, rust = _run_both(random_users)
        for i, (p, r) in enumerate(zip(py, rust)):
            assert p.is_garp == r.is_garp, f"User {i}: py={p.is_garp}, rust={r.is_garp}"


# --- CCEI parity ---

class TestCCEIParity:
    def test_consistent_is_one(self, consistent_data):
        py, rust = _run_both(consistent_data)
        assert abs(py[0].ccei - rust[0].ccei) < 0.01
        assert py[0].ccei == pytest.approx(1.0, abs=0.01)

    def test_violation(self, violation_data):
        py, rust = _run_both(violation_data)
        assert abs(py[0].ccei - rust[0].ccei) < 0.01

    def test_random_close(self, random_users):
        py, rust = _run_both(random_users)
        for i, (p, r) in enumerate(zip(py, rust)):
            assert abs(p.ccei - r.ccei) < 0.01, (
                f"User {i}: py_ccei={p.ccei:.4f}, rust_ccei={r.ccei:.4f}"
            )


# --- MPI parity ---

class TestMPIParity:
    def test_consistent_is_zero(self, consistent_data):
        py, rust = _run_both(consistent_data)
        assert abs(py[0].mpi - rust[0].mpi) < 0.05
        assert py[0].mpi == pytest.approx(0.0, abs=0.01)

    def test_violation(self, violation_data):
        py, rust = _run_both(violation_data)
        assert abs(py[0].mpi - rust[0].mpi) < 0.05

    def test_random_close(self, random_users):
        py, rust = _run_both(random_users)
        for i, (p, r) in enumerate(zip(py, rust)):
            assert abs(p.mpi - r.mpi) < 0.05, (
                f"User {i}: py_mpi={p.mpi:.4f}, rust_mpi={r.mpi:.4f}"
            )
