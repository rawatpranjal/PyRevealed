"""Tests for result summary(), score(), to_dict(), and __repr__() methods."""

import numpy as np
import pytest

from pyrevealed.core.result import (
    GARPResult,
    AEIResult,
    MPIResult,
    UtilityRecoveryResult,
    WARPResult,
    SARPResult,
    HoutmanMaksResult,
    AbstractSARPResult,
    CongruenceResult,
    HoutmanMaksAbstractResult,
    OrdinalUtilityResult,
)


class TestGARPResultMethods:
    """Tests for GARPResult summary methods."""

    @pytest.fixture
    def consistent_result(self):
        """Create a consistent GARP result."""
        n = 3
        return GARPResult(
            is_consistent=True,
            violations=[],
            direct_revealed_preference=np.eye(n, dtype=bool),
            transitive_closure=np.eye(n, dtype=bool),
            strict_revealed_preference=np.zeros((n, n), dtype=bool),
            computation_time_ms=1.5,
        )

    @pytest.fixture
    def violation_result(self):
        """Create a GARP result with violations."""
        n = 3
        return GARPResult(
            is_consistent=False,
            violations=[(0, 1, 0), (1, 2, 1)],
            direct_revealed_preference=np.eye(n, dtype=bool),
            transitive_closure=np.eye(n, dtype=bool),
            strict_revealed_preference=np.zeros((n, n), dtype=bool),
            computation_time_ms=2.5,
        )

    def test_score_consistent(self, consistent_result):
        """Test score returns 1.0 for consistent data."""
        assert consistent_result.score() == 1.0

    def test_score_violations(self, violation_result):
        """Test score returns 0.0 for data with violations."""
        assert violation_result.score() == 0.0

    def test_summary_contains_status(self, consistent_result):
        """Test summary contains status information."""
        summary = consistent_result.summary()
        assert "CONSISTENT" in summary
        assert "GARP" in summary

    def test_summary_shows_violations(self, violation_result):
        """Test summary shows violation information."""
        summary = violation_result.summary()
        assert "VIOLATIONS FOUND" in summary
        assert "2" in summary  # Number of violations

    def test_to_dict_keys(self, consistent_result):
        """Test to_dict returns expected keys."""
        d = consistent_result.to_dict()
        assert "is_consistent" in d
        assert "num_violations" in d
        assert "score" in d
        assert "computation_time_ms" in d

    def test_to_dict_values(self, consistent_result):
        """Test to_dict returns correct values."""
        d = consistent_result.to_dict()
        assert d["is_consistent"] is True
        assert d["num_violations"] == 0
        assert d["score"] == 1.0

    def test_repr_consistent(self, consistent_result):
        """Test repr for consistent result."""
        r = repr(consistent_result)
        assert "GARPResult" in r
        assert "consistent" in r.lower()

    def test_repr_violations(self, violation_result):
        """Test repr for result with violations."""
        r = repr(violation_result)
        assert "GARPResult" in r
        assert "2 violations" in r


class TestAEIResultMethods:
    """Tests for AEIResult summary methods."""

    @pytest.fixture
    def perfect_result(self):
        """Create a perfect AEI result."""
        garp = GARPResult(
            is_consistent=True,
            violations=[],
            direct_revealed_preference=np.eye(3, dtype=bool),
            transitive_closure=np.eye(3, dtype=bool),
            strict_revealed_preference=np.zeros((3, 3), dtype=bool),
            computation_time_ms=1.0,
        )
        return AEIResult(
            efficiency_index=1.0,
            is_perfectly_consistent=True,
            garp_result_at_threshold=garp,
            binary_search_iterations=0,
            tolerance=1e-6,
            computation_time_ms=5.0,
        )

    @pytest.fixture
    def imperfect_result(self):
        """Create an imperfect AEI result."""
        garp = GARPResult(
            is_consistent=True,
            violations=[],
            direct_revealed_preference=np.eye(3, dtype=bool),
            transitive_closure=np.eye(3, dtype=bool),
            strict_revealed_preference=np.zeros((3, 3), dtype=bool),
            computation_time_ms=1.0,
        )
        return AEIResult(
            efficiency_index=0.85,
            is_perfectly_consistent=False,
            garp_result_at_threshold=garp,
            binary_search_iterations=10,
            tolerance=1e-6,
            computation_time_ms=15.0,
        )

    def test_score_perfect(self, perfect_result):
        """Test score returns efficiency_index for perfect result."""
        assert perfect_result.score() == 1.0

    def test_score_imperfect(self, imperfect_result):
        """Test score returns efficiency_index for imperfect result."""
        assert imperfect_result.score() == 0.85

    def test_summary_contains_aei(self, perfect_result):
        """Test summary contains AEI information."""
        summary = perfect_result.summary()
        assert "AFRIAT" in summary or "AEI" in summary
        assert "1.0" in summary or "PERFECT" in summary

    def test_to_dict_includes_waste(self, imperfect_result):
        """Test to_dict includes waste fraction."""
        d = imperfect_result.to_dict()
        assert "waste_fraction" in d
        assert d["waste_fraction"] == pytest.approx(0.15)

    def test_repr(self, imperfect_result):
        """Test repr format."""
        r = repr(imperfect_result)
        assert "AEIResult" in r
        assert "0.85" in r


class TestMPIResultMethods:
    """Tests for MPIResult summary methods."""

    @pytest.fixture
    def no_pump_result(self):
        """Create an MPI result with no money pump."""
        return MPIResult(
            mpi_value=0.0,
            worst_cycle=None,
            cycle_costs=[],
            total_expenditure=100.0,
            computation_time_ms=3.0,
        )

    @pytest.fixture
    def pump_result(self):
        """Create an MPI result with money pump."""
        return MPIResult(
            mpi_value=0.12,
            worst_cycle=(0, 1, 2, 0),
            cycle_costs=[((0, 1, 2, 0), 0.12), ((1, 2, 1), 0.05)],
            total_expenditure=100.0,
            computation_time_ms=4.0,
        )

    def test_score_no_pump(self, no_pump_result):
        """Test score returns 1.0 for no money pump."""
        assert no_pump_result.score() == 1.0

    def test_score_with_pump(self, pump_result):
        """Test score returns 1 - mpi_value."""
        assert pump_result.score() == pytest.approx(0.88)

    def test_summary_shows_exploitability(self, pump_result):
        """Test summary shows exploitability information."""
        summary = pump_result.summary()
        assert "EXPLOITABILITY" in summary or "MPI" in summary

    def test_to_dict_includes_cycles(self, pump_result):
        """Test to_dict includes cycle information."""
        d = pump_result.to_dict()
        assert "num_cycles" in d
        assert d["num_cycles"] == 2

    def test_repr_consistent(self, no_pump_result):
        """Test repr for consistent result."""
        r = repr(no_pump_result)
        assert "consistent" in r.lower()


class TestUtilityRecoveryResultMethods:
    """Tests for UtilityRecoveryResult summary methods."""

    @pytest.fixture
    def success_result(self):
        """Create a successful utility recovery result."""
        return UtilityRecoveryResult(
            success=True,
            utility_values=np.array([1.0, 2.0, 1.5]),
            lagrange_multipliers=np.array([0.5, 0.6, 0.55]),
            lp_status="Optimal",
            residuals=np.zeros((3, 3)),
            computation_time_ms=10.0,
        )

    @pytest.fixture
    def failure_result(self):
        """Create a failed utility recovery result."""
        return UtilityRecoveryResult(
            success=False,
            utility_values=None,
            lagrange_multipliers=None,
            lp_status="Infeasible",
            residuals=None,
            computation_time_ms=5.0,
        )

    def test_score_success(self, success_result):
        """Test score returns 1.0 for successful recovery."""
        assert success_result.score() == 1.0

    def test_score_failure(self, failure_result):
        """Test score returns 0.0 for failed recovery."""
        assert failure_result.score() == 0.0

    def test_summary_shows_status(self, success_result):
        """Test summary shows success status."""
        summary = success_result.summary()
        assert "SUCCESS" in summary

    def test_to_dict_includes_utilities(self, success_result):
        """Test to_dict includes utility values."""
        d = success_result.to_dict()
        assert "utility_values" in d
        assert len(d["utility_values"]) == 3


class TestHoutmanMaksResultMethods:
    """Tests for HoutmanMaksResult summary methods."""

    @pytest.fixture
    def consistent_result(self):
        """Create a fully consistent result."""
        return HoutmanMaksResult(
            fraction=0.0,
            removed_observations=[],
            computation_time_ms=2.0,
        )

    @pytest.fixture
    def removal_result(self):
        """Create a result requiring removals."""
        return HoutmanMaksResult(
            fraction=0.2,
            removed_observations=[0, 3],
            computation_time_ms=3.0,
        )

    def test_score_consistent(self, consistent_result):
        """Test score returns 1.0 for consistent data."""
        assert consistent_result.score() == 1.0

    def test_score_removal(self, removal_result):
        """Test score returns 1 - fraction."""
        assert removal_result.score() == 0.8

    def test_summary_shows_removal(self, removal_result):
        """Test summary shows removed observations."""
        summary = removal_result.summary()
        assert "20" in summary or "0.2" in summary  # percentage or fraction


class TestAbstractChoiceResultMethods:
    """Tests for abstract choice result methods."""

    @pytest.fixture
    def sarp_consistent(self):
        """Create a consistent AbstractSARPResult."""
        n = 4
        return AbstractSARPResult(
            is_consistent=True,
            violations=[],
            revealed_preference_matrix=np.eye(n, dtype=bool),
            transitive_closure=np.eye(n, dtype=bool),
            computation_time_ms=1.0,
        )

    @pytest.fixture
    def congruence_result(self, sarp_consistent):
        """Create a CongruenceResult."""
        return CongruenceResult(
            is_congruent=True,
            satisfies_sarp=True,
            maximality_violations=[],
            sarp_result=sarp_consistent,
            computation_time_ms=2.0,
        )

    def test_sarp_score(self, sarp_consistent):
        """Test AbstractSARPResult score."""
        assert sarp_consistent.score() == 1.0

    def test_congruence_score(self, congruence_result):
        """Test CongruenceResult score."""
        assert congruence_result.score() == 1.0

    def test_sarp_summary(self, sarp_consistent):
        """Test AbstractSARPResult summary."""
        summary = sarp_consistent.summary()
        assert "SARP" in summary
        assert "CONSISTENT" in summary

    def test_congruence_to_dict(self, congruence_result):
        """Test CongruenceResult to_dict."""
        d = congruence_result.to_dict()
        assert d["is_rationalizable"] is True
        assert d["satisfies_sarp"] is True


class TestOrdinalUtilityResultMethods:
    """Tests for OrdinalUtilityResult methods."""

    @pytest.fixture
    def success_result(self):
        """Create a successful ordinal utility result."""
        return OrdinalUtilityResult(
            success=True,
            utility_ranking={0: 0, 1: 1, 2: 2, 3: 3},
            utility_values=np.array([3.0, 2.0, 1.0, 0.0]),
            preference_order=[0, 1, 2, 3],
            num_items=4,
            is_complete=True,
            computation_time_ms=1.5,
        )

    def test_score(self, success_result):
        """Test score returns 1.0 for success."""
        assert success_result.score() == 1.0

    def test_summary(self, success_result):
        """Test summary contains preference order."""
        summary = success_result.summary()
        assert "ORDINAL" in summary
        assert "SUCCESS" in summary

    def test_to_dict(self, success_result):
        """Test to_dict includes preference order."""
        d = success_result.to_dict()
        assert d["preference_order"] == [0, 1, 2, 3]
        assert d["most_preferred"] == 0


class TestScoreRange:
    """Tests to verify all scores are in [0, 1] range."""

    def test_garp_score_range(self):
        """Test GARPResult score is in valid range."""
        for is_consistent in [True, False]:
            result = GARPResult(
                is_consistent=is_consistent,
                violations=[] if is_consistent else [(0, 1, 0)],
                direct_revealed_preference=np.eye(2, dtype=bool),
                transitive_closure=np.eye(2, dtype=bool),
                strict_revealed_preference=np.zeros((2, 2), dtype=bool),
                computation_time_ms=1.0,
            )
            assert 0.0 <= result.score() <= 1.0

    def test_aei_score_range(self):
        """Test AEIResult score is in valid range."""
        garp = GARPResult(
            is_consistent=True,
            violations=[],
            direct_revealed_preference=np.eye(2, dtype=bool),
            transitive_closure=np.eye(2, dtype=bool),
            strict_revealed_preference=np.zeros((2, 2), dtype=bool),
            computation_time_ms=1.0,
        )
        for aei in [0.0, 0.5, 0.85, 1.0]:
            result = AEIResult(
                efficiency_index=aei,
                is_perfectly_consistent=(aei == 1.0),
                garp_result_at_threshold=garp,
                binary_search_iterations=5,
                tolerance=1e-6,
                computation_time_ms=5.0,
            )
            assert 0.0 <= result.score() <= 1.0

    def test_mpi_score_range(self):
        """Test MPIResult score is in valid range."""
        for mpi in [0.0, 0.1, 0.5, 1.0, 1.5]:  # Include > 1.0 edge case
            result = MPIResult(
                mpi_value=mpi,
                worst_cycle=None if mpi == 0 else (0, 1, 0),
                cycle_costs=[],
                total_expenditure=100.0,
                computation_time_ms=1.0,
            )
            assert 0.0 <= result.score() <= 1.0
