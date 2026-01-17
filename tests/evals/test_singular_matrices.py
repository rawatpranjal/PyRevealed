"""
EVAL: Singular and ill-conditioned matrix tests.

These tests expose linear algebra failures in regression and optimization.
"""

import numpy as np
import pytest
from pyrevealed.core.session import BehaviorLog


class TestMulticollinearity:
    """EVAL: Multicollinear prices cause singular regression matrices."""

    def test_slutsky_multicollinear_prices(self, multicollinear_prices):
        """EVAL: Slutsky regression with perfectly correlated prices.

        In integrability.py around line 239:
            XtX_inv = np.linalg.pinv(XtX)  # Use pseudo-inverse

        Multicollinearity makes XtX singular, requiring pseudo-inverse.
        """
        from pyrevealed.algorithms.integrability import compute_slutsky_matrix_regression

        S = compute_slutsky_matrix_regression(multicollinear_prices)

        # Should not crash, but results may be unreliable
        assert S.shape == (2, 2)

        # Check for inf/nan
        if not np.all(np.isfinite(S)):
            pytest.xfail(f"Multicollinear data produces non-finite Slutsky: {S}")

    def test_slutsky_constant_prices(self, constant_prices):
        """EVAL: Slutsky with constant prices - zero variance."""
        from pyrevealed.algorithms.integrability import compute_slutsky_matrix_regression

        S = compute_slutsky_matrix_regression(constant_prices)

        # Zero variance in prices means no price effect can be estimated
        assert S.shape == (2, 2)


class TestProportionalBundles:
    """EVAL: Proportional bundles create degenerate preference relations."""

    def test_garp_proportional_bundles(self, proportional_bundles):
        """EVAL: GARP with all bundles on same ray."""
        from pyrevealed.algorithms.garp import check_garp

        result = check_garp(proportional_bundles)

        # Proportional bundles can still have GARP violations
        # depending on prices
        assert hasattr(result, 'is_consistent')

    def test_utility_recovery_proportional(self, proportional_bundles):
        """EVAL: Utility recovery with collinear bundles."""
        from pyrevealed.algorithms.utility import recover_utility

        result = recover_utility(proportional_bundles)

        # Utility should still be recoverable (rank 1 data)
        assert hasattr(result, 'success')


class TestRankDeficiency:
    """EVAL: Rank deficient matrices in LP constraints."""

    def test_afriat_constraints_rank_deficient(self):
        """EVAL: Afriat LP constraints with redundant observations.

        When bundles are identical, constraint matrix has redundant rows.
        """
        from pyrevealed.algorithms.utility import recover_utility

        # Identical bundles at different prices
        log = BehaviorLog(
            cost_vectors=np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
            action_vectors=np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
        )

        result = recover_utility(log)

        # Should handle redundant constraints
        assert hasattr(result, 'success')

    def test_welfare_lp_rank_deficient(self):
        """EVAL: Welfare LP with rank deficient constraint matrix."""
        from pyrevealed.algorithms.welfare import compute_cv_exact

        # Very similar observations
        baseline = BehaviorLog(
            cost_vectors=np.array([[1.0, 1.0], [1.0, 1.0]]),
            action_vectors=np.array([[1.0, 1.0], [1.0 + 1e-10, 1.0 + 1e-10]]),
        )

        try:
            cv, success = compute_cv_exact(baseline, baseline)
            # Should either succeed or fail gracefully
            assert np.isfinite(cv) or not success
        except Exception as e:
            pytest.xfail(f"CV computation failed on near-identical data: {e}")


class TestConditionNumber:
    """EVAL: High condition number matrices."""

    def test_expenditure_matrix_condition(self, high_condition_number_log):
        """EVAL: Expenditure matrix with condition number > 1e15."""
        E = high_condition_number_log.spend_matrix

        cond = np.linalg.cond(E)

        # Document the actual condition number
        print(f"Expenditure matrix condition number: {cond:.2e}")

        # Algorithms using this matrix may fail
        if cond > 1e15:
            pass  # This is expected - document the limitation

    def test_garp_high_condition(self, high_condition_number_log):
        """EVAL: GARP check with ill-conditioned expenditure matrix."""
        from pyrevealed.algorithms.garp import check_garp

        result = check_garp(high_condition_number_log)

        # Should not crash
        assert hasattr(result, 'is_consistent')

    def test_aei_high_condition(self, high_condition_number_log):
        """EVAL: AEI binary search with ill-conditioned data."""
        from pyrevealed.algorithms.aei import compute_aei

        result = compute_aei(high_condition_number_log)

        assert np.isfinite(result.efficiency_index), (
            f"AEI should be finite, got {result.efficiency_index}"
        )


class TestSingularEigenvalue:
    """EVAL: Eigenvalue problems with singular matrices."""

    def test_slutsky_nsd_singular(self):
        """EVAL: NSD test on singular Slutsky matrix."""
        from pyrevealed.algorithms.integrability import check_slutsky_nsd

        # Singular matrix (rank 1)
        S = np.array([[1.0, 2.0], [2.0, 4.0]])

        is_nsd, eigenvalues, max_eig, pvalue = check_slutsky_nsd(S)

        # Should handle singular matrix
        assert np.all(np.isfinite(eigenvalues))

    def test_slutsky_nsd_zero_matrix(self):
        """EVAL: NSD test on zero matrix."""
        from pyrevealed.algorithms.integrability import check_slutsky_nsd

        S = np.zeros((3, 3))

        is_nsd, eigenvalues, max_eig, pvalue = check_slutsky_nsd(S)

        # Zero matrix is trivially NSD
        assert is_nsd, "Zero matrix should be NSD"


class TestPseudoInverse:
    """EVAL: Pseudo-inverse behavior with singular matrices."""

    def test_pinv_nearly_singular(self):
        """EVAL: np.linalg.pinv with nearly singular matrix."""
        # Create a nearly singular matrix
        A = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-15]])

        pinv_A = np.linalg.pinv(A)

        # Pseudo-inverse may have large values
        assert np.all(np.isfinite(pinv_A)), f"Non-finite pseudo-inverse: {pinv_A}"

    def test_regression_with_pinv(self):
        """EVAL: Slutsky regression using pinv on singular XtX."""
        from pyrevealed.algorithms.integrability import compute_slutsky_matrix_regression

        # Data that makes XtX nearly singular
        log = BehaviorLog(
            cost_vectors=np.array([
                [1.0, 1.0],
                [1.0, 1.0 + 1e-12],
                [1.0, 1.0 + 2e-12],
            ]),
            action_vectors=np.array([
                [2.0, 3.0],
                [2.0, 3.0],
                [2.0, 3.0],
            ]),
        )

        S = compute_slutsky_matrix_regression(log)

        # May fall back to finite differences
        assert S.shape == (2, 2)


class TestLinearDependence:
    """EVAL: Linearly dependent constraints in LP problems."""

    def test_afriat_linearly_dependent_constraints(self):
        """EVAL: Afriat LP with linearly dependent inequality constraints."""
        from pyrevealed.algorithms.utility import recover_utility

        # Create data with duplicate observations
        log = BehaviorLog(
            cost_vectors=np.array([[1.0, 2.0], [1.0, 2.0], [2.0, 1.0]]),
            action_vectors=np.array([[3.0, 1.0], [3.0, 1.0], [1.0, 3.0]]),
        )

        result = recover_utility(log)

        # LP solver should handle redundant constraints
        assert hasattr(result, 'success')

    def test_rum_exact_dependent_constraints(self):
        """EVAL: RUM LP with dependent probability constraints."""
        from pyrevealed.core.session import StochasticChoiceLog
        from pyrevealed.algorithms.stochastic import test_rum_consistency

        # Same menu repeated
        log = StochasticChoiceLog(
            menus=[frozenset({0, 1}), frozenset({0, 1})],
            choice_frequencies=[
                {0: 60, 1: 40},
                {0: 60, 1: 40},  # Same frequencies
            ],
        )

        result = test_rum_consistency(log)

        # Should handle redundant constraints
        assert hasattr(result, 'is_rum_consistent')
