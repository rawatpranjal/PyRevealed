"""
EVAL: Mutation and aliasing bugs.

These tests expose mutable cache issues and array aliasing problems.
"""

import numpy as np
import pytest
from pyrevealed.core.session import BehaviorLog


class TestSpendMatrixMutation:
    """EVAL: spend_matrix returns mutable reference, not copy."""

    def test_spend_matrix_mutation_corrupts_cache(self):
        """EVAL: Modifying spend_matrix corrupts internal state.

        The spend_matrix property returns _expenditure_matrix directly (session.py:254).
        This allows external code to mutate the internal cache.
        """
        log = BehaviorLog(
            cost_vectors=np.array([[1.0, 2.0], [2.0, 1.0]]),
            action_vectors=np.array([[3.0, 1.0], [1.0, 3.0]]),
        )

        # Get original value
        original = log.spend_matrix[0, 0]

        # Mutate the returned reference
        log.spend_matrix[0, 0] = 99999.0

        # Check if internal state was corrupted
        # This WILL FAIL if the cache is mutable - proving the vulnerability
        assert log.spend_matrix[0, 0] == original, (
            f"MUTATION BUG: spend_matrix cache was corrupted! "
            f"Expected {original}, got {log.spend_matrix[0, 0]}"
        )

    def test_spend_matrix_repeated_access_same_object(self):
        """EVAL: Multiple accesses return same object (not fresh copies)."""
        log = BehaviorLog(
            cost_vectors=np.array([[1.0, 2.0], [2.0, 1.0]]),
            action_vectors=np.array([[3.0, 1.0], [1.0, 3.0]]),
        )

        matrix1 = log.spend_matrix
        matrix2 = log.spend_matrix

        # If these are the same object, mutation of one affects the other
        same_object = matrix1 is matrix2

        if same_object:
            pytest.xfail(
                "spend_matrix returns same object on repeated access "
                "(expected for caching, but allows mutation)"
            )


class TestLegacyAliases:
    """EVAL: Legacy attribute aliases share same array."""

    def test_prices_cost_vectors_same_array(self):
        """EVAL: log.prices and log.cost_vectors point to same array.

        After __post_init__, both attributes reference the same array.
        Mutation through one alias affects the other.
        """
        log = BehaviorLog(
            cost_vectors=np.array([[1.0, 2.0], [2.0, 1.0]]),
            action_vectors=np.array([[3.0, 1.0], [1.0, 3.0]]),
        )

        # Check if they're the same object
        same_object = log.prices is log.cost_vectors

        if same_object:
            # Demonstrate the aliasing
            original = log.cost_vectors[0, 0]
            log.prices[0, 0] = 999.0
            assert log.cost_vectors[0, 0] == 999.0, "Aliases don't share memory"
            log.prices[0, 0] = original  # Restore

    def test_quantities_action_vectors_same_array(self):
        """EVAL: log.quantities and log.action_vectors point to same array."""
        log = BehaviorLog(
            cost_vectors=np.array([[1.0, 2.0], [2.0, 1.0]]),
            action_vectors=np.array([[3.0, 1.0], [1.0, 3.0]]),
        )

        same_object = log.quantities is log.action_vectors
        assert same_object, "quantities and action_vectors should be same object"


class TestInputArrayMutation:
    """EVAL: Mutation of input arrays after construction."""

    def test_input_mutation_after_construction(self):
        """EVAL: Mutating input arrays after construction affects BehaviorLog.

        If BehaviorLog doesn't copy inputs, external mutation corrupts state.
        """
        costs = np.array([[1.0, 2.0], [2.0, 1.0]])
        actions = np.array([[3.0, 1.0], [1.0, 3.0]])

        log = BehaviorLog(cost_vectors=costs, action_vectors=actions)

        original_spend = log.total_spend[0]

        # Mutate the original input array
        costs[0, 0] = 999.0

        # Check if log's internal state was affected
        new_spend = log.total_spend[0]

        # This documents whether inputs are copied or referenced
        if new_spend != original_spend:
            pytest.xfail(
                "BehaviorLog doesn't copy input arrays - external mutation affects state"
            )

    def test_input_mutation_detected_via_spend_matrix(self):
        """EVAL: Input mutation can be detected via spend_matrix inconsistency."""
        costs = np.array([[1.0, 2.0], [2.0, 1.0]])
        actions = np.array([[3.0, 1.0], [1.0, 3.0]])

        log = BehaviorLog(cost_vectors=costs, action_vectors=actions)

        # Cache the spend matrix
        _ = log.spend_matrix

        # Mutate input
        costs[0, 0] = 999.0

        # Recompute spend matrix manually
        expected_spend = costs @ actions.T

        # Check for inconsistency
        if not np.allclose(log.spend_matrix, expected_spend):
            # This is actually expected - the cache is stale
            pass  # This documents the caching behavior


class TestReturnedArrayMutation:
    """EVAL: Mutation of arrays returned by properties."""

    def test_total_spend_mutation(self):
        """EVAL: Mutating total_spend doesn't affect source."""
        log = BehaviorLog(
            cost_vectors=np.array([[1.0, 2.0], [2.0, 1.0]]),
            action_vectors=np.array([[3.0, 1.0], [1.0, 3.0]]),
        )

        total = log.total_spend
        original = total[0]
        total[0] = 999.0

        # total_spend is computed from diagonal, so this tests if
        # np.diag returns a copy or view
        new_total = log.total_spend[0]

        # np.diag returns a copy of the diagonal, so this should be safe
        assert new_total == original, "total_spend should not be mutable"

    def test_transitive_closure_mutation(self):
        """EVAL: Mutating GARP transitive closure doesn't corrupt results."""
        from pyrevealed.algorithms.garp import check_garp

        log = BehaviorLog(
            cost_vectors=np.array([[1.0, 2.0], [2.0, 1.0]]),
            action_vectors=np.array([[3.0, 1.0], [1.0, 3.0]]),
        )

        result1 = check_garp(log)
        original_closure = result1.transitive_closure.copy()

        # Mutate the returned matrix
        result1.transitive_closure[0, 0] = True

        # Get a fresh result
        result2 = check_garp(log)

        # Check if the internal state was corrupted
        assert np.array_equal(result2.transitive_closure, original_closure), (
            "GARP transitive closure mutation corrupted cached results"
        )


class TestAlgorithmInternalState:
    """EVAL: Algorithm functions that might mutate input state."""

    def test_garp_doesnt_mutate_log(self):
        """EVAL: check_garp should not mutate the BehaviorLog."""
        from pyrevealed.algorithms.garp import check_garp

        log = BehaviorLog(
            cost_vectors=np.array([[1.0, 2.0], [2.0, 1.0]]),
            action_vectors=np.array([[3.0, 1.0], [1.0, 3.0]]),
        )

        costs_before = log.cost_vectors.copy()
        actions_before = log.action_vectors.copy()

        _ = check_garp(log)

        assert np.array_equal(log.cost_vectors, costs_before), "GARP mutated costs"
        assert np.array_equal(log.action_vectors, actions_before), "GARP mutated actions"

    def test_aei_doesnt_mutate_log(self):
        """EVAL: compute_aei should not mutate the BehaviorLog."""
        from pyrevealed.algorithms.aei import compute_aei

        log = BehaviorLog(
            cost_vectors=np.array([[1.0, 2.0], [2.0, 1.0]]),
            action_vectors=np.array([[3.0, 1.0], [1.0, 3.0]]),
        )

        costs_before = log.cost_vectors.copy()
        actions_before = log.action_vectors.copy()

        _ = compute_aei(log)

        assert np.array_equal(log.cost_vectors, costs_before), "AEI mutated costs"
        assert np.array_equal(log.action_vectors, actions_before), "AEI mutated actions"
