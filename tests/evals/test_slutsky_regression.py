"""
EVAL: Slutsky matrix regression failures.

Tests for integrability check edge cases and regression failures.
"""

import numpy as np
import pytest
from pyrevealed.core.session import BehaviorLog


class TestSlutskyRegressionFailures:
    """EVAL: Slutsky matrix estimation regression failures."""

    def test_slutsky_regression_multicollinear(self, multicollinear_prices):
        """EVAL: Slutsky regression with multicollinear prices.

        In integrability.py around line 239:
            XtX_inv = np.linalg.pinv(XtX)  # Uses pseudo-inverse

        Multicollinearity makes XtX singular.
        """
        from pyrevealed.algorithms.integrability import compute_slutsky_matrix_regression

        S = compute_slutsky_matrix_regression(multicollinear_prices)

        # Should not crash (uses pinv)
        assert S.shape == (2, 2)

        # But results may be unreliable
        if np.any(np.abs(S) > 1e10):
            pytest.xfail("Large Slutsky values from multicollinear data")

    def test_slutsky_regression_constant_prices(self, constant_prices):
        """EVAL: Slutsky regression with constant prices (zero variance)."""
        from pyrevealed.algorithms.integrability import compute_slutsky_matrix_regression

        S = compute_slutsky_matrix_regression(constant_prices)

        # Zero price variance means no price effect can be estimated
        assert S.shape == (2, 2)

    def test_slutsky_regression_underdetermined(self, two_observations_log):
        """EVAL: Slutsky regression with T < N+1 (underdetermined)."""
        from pyrevealed.algorithms.integrability import compute_slutsky_matrix_regression

        S = compute_slutsky_matrix_regression(two_observations_log)

        # T=2, N=2 means we have fewer observations than parameters
        assert S.shape == (2, 2)


class TestSlutskySymmetry:
    """EVAL: Slutsky symmetry test edge cases."""

    def test_symmetry_perfectly_symmetric(self):
        """EVAL: Symmetry test for perfectly symmetric matrix."""
        from pyrevealed.algorithms.integrability import check_slutsky_symmetry

        S = np.array([[1.0, 2.0], [2.0, 3.0]])

        is_sym, violations, max_asym = check_slutsky_symmetry(S)

        assert is_sym, "Symmetric matrix should pass symmetry test"
        assert max_asym == 0.0

    def test_symmetry_asymmetric(self):
        """EVAL: Symmetry test for asymmetric matrix."""
        from pyrevealed.algorithms.integrability import check_slutsky_symmetry

        S = np.array([[1.0, 2.0], [3.0, 4.0]])

        is_sym, violations, max_asym = check_slutsky_symmetry(S)

        assert not is_sym, "Asymmetric matrix should fail symmetry test"
        assert max_asym > 0.0

    def test_symmetry_near_symmetric(self):
        """EVAL: Symmetry test for nearly symmetric matrix."""
        from pyrevealed.algorithms.integrability import check_slutsky_symmetry

        S = np.array([[1.0, 2.0], [2.0 + 1e-8, 3.0]])

        is_sym_strict, _, _ = check_slutsky_symmetry(S, tolerance=1e-10)
        is_sym_loose, _, _ = check_slutsky_symmetry(S, tolerance=1e-6)

        # Should fail strict but pass loose
        assert not is_sym_strict
        assert is_sym_loose


class TestSlutskyNSD:
    """EVAL: Slutsky negative semi-definiteness test."""

    def test_nsd_negative_definite(self):
        """EVAL: NSD test for negative definite matrix."""
        from pyrevealed.algorithms.integrability import check_slutsky_nsd

        S = np.array([[-2.0, 0.5], [0.5, -2.0]])

        is_nsd, eigenvalues, max_eig, pvalue = check_slutsky_nsd(S)

        assert is_nsd, "Negative definite matrix should be NSD"
        assert max_eig < 0

    def test_nsd_positive_eigenvalue(self):
        """EVAL: NSD test for matrix with positive eigenvalue."""
        from pyrevealed.algorithms.integrability import check_slutsky_nsd

        S = np.array([[1.0, 0.0], [0.0, -1.0]])

        is_nsd, eigenvalues, max_eig, pvalue = check_slutsky_nsd(S)

        assert not is_nsd, "Matrix with positive eigenvalue is not NSD"
        assert max_eig > 0

    def test_nsd_singular_matrix(self):
        """EVAL: NSD test for singular matrix (has zero eigenvalue)."""
        from pyrevealed.algorithms.integrability import check_slutsky_nsd

        S = np.array([[1.0, 2.0], [2.0, 4.0]])  # Rank 1

        is_nsd, eigenvalues, max_eig, pvalue = check_slutsky_nsd(S)

        # Singular matrix with non-negative eigenvalues is not NSD
        # (unless all eigenvalues are zero or negative)
        assert hasattr(eigenvalues, '__len__')


class TestIntegrabilityTest:
    """EVAL: Full integrability test edge cases."""

    def test_integrability_simple(self):
        """EVAL: Integrability test for simple data."""
        from pyrevealed.algorithms.integrability import test_integrability

        log = BehaviorLog(
            cost_vectors=np.array([
                [1.0, 2.0],
                [2.0, 1.0],
                [1.5, 1.5],
            ]),
            action_vectors=np.array([
                [4.0, 1.0],
                [1.0, 4.0],
                [2.5, 2.5],
            ]),
        )

        result = test_integrability(log)

        assert hasattr(result, 'is_integrable')
        assert hasattr(result, 'slutsky_matrix')

    def test_integrability_single_observation(self, single_observation_log):
        """EVAL: Integrability test with T=1."""
        from pyrevealed.algorithms.integrability import test_integrability

        result = test_integrability(single_observation_log)

        # Should handle gracefully
        assert hasattr(result, 'is_integrable')


class TestSlutskyMethods:
    """EVAL: Different Slutsky estimation methods."""

    def test_slutsky_finite_difference(self):
        """EVAL: Slutsky via finite difference method."""
        from pyrevealed.algorithms.integrability import compute_slutsky_matrix

        log = BehaviorLog(
            cost_vectors=np.array([
                [1.0, 2.0],
                [1.0 + 0.01, 2.0],  # Small price change
                [1.0, 2.0 + 0.01],
            ]),
            action_vectors=np.array([
                [4.0, 1.0],
                [3.9, 1.05],
                [4.1, 0.95],
            ]),
        )

        S = compute_slutsky_matrix(log)

        assert S.shape == (2, 2)
        assert np.all(np.isfinite(S))

    def test_slutsky_stone_geary(self):
        """EVAL: Slutsky via Stone-Geary estimation."""
        from pyrevealed.algorithms.integrability import compute_slutsky_matrix_stone_geary

        log = BehaviorLog(
            cost_vectors=np.array([
                [1.0, 2.0],
                [2.0, 1.0],
                [1.5, 1.5],
                [1.0, 1.0],
            ]),
            action_vectors=np.array([
                [4.0, 1.0],
                [1.0, 4.0],
                [2.5, 2.5],
                [3.0, 3.0],
            ]),
        )

        S = compute_slutsky_matrix_stone_geary(log)

        assert S.shape == (2, 2)


class TestSlutskyDecomposition:
    """EVAL: Slutsky decomposition edge cases."""

    def test_slutsky_decomposition_simple(self):
        """EVAL: Slutsky decomposition into substitution and income effects."""
        from pyrevealed.algorithms.integrability import compute_slutsky_matrix

        log = BehaviorLog(
            cost_vectors=np.array([
                [1.0, 2.0],
                [2.0, 1.0],
                [1.5, 1.5],
            ]),
            action_vectors=np.array([
                [4.0, 1.0],
                [1.0, 4.0],
                [2.5, 2.5],
            ]),
        )

        S = compute_slutsky_matrix(log)

        # Slutsky = Marshallian + Income effect * quantity
        # For normal goods, own-price Slutsky should be negative
        print(f"Slutsky diagonal: {np.diag(S)}")


class TestLargeNGoods:
    """EVAL: Slutsky estimation with many goods."""

    def test_slutsky_n100(self, large_n_100):
        """EVAL: Slutsky matrix for N=100 goods."""
        from pyrevealed.algorithms.integrability import compute_slutsky_matrix

        S = compute_slutsky_matrix(large_n_100)

        assert S.shape == (100, 100)

        # Check for numerical issues
        assert np.all(np.isfinite(S)), "Non-finite values in large Slutsky matrix"

        # Check symmetry (should be approximately symmetric)
        asymmetry = np.max(np.abs(S - S.T))
        print(f"Max asymmetry for N=100: {asymmetry:.2e}")
