"""Tests for the new tech-friendly API (v0.3.0).

This file tests:
1. BehavioralAuditor class
2. PreferenceEncoder class
3. New function names (validate_consistency, compute_integrity_score, etc.)
4. New data container names (BehaviorLog, RiskChoiceLog, EmbeddingChoiceLog)
5. Backward compatibility with old names
"""

import numpy as np
import pytest

# Test new imports work
from pyrevealed import (
    # High-level classes
    BehavioralAuditor,
    AuditReport,
    PreferenceEncoder,
    # Data containers - new names
    BehaviorLog,
    RiskChoiceLog,
    EmbeddingChoiceLog,
    # Data containers - old names (backward compatibility)
    ConsumerSession,
    RiskSession,
    SpatialSession,
    # Functions - new names
    validate_consistency,
    validate_consistency_weak,
    compute_integrity_score,
    compute_confusion_metric,
    fit_latent_values,
    build_value_function,
    predict_choice,
    find_preference_anchor,
    validate_embedding_consistency,
    discover_independent_groups,
    compute_cross_impact,
    # Functions - old names (backward compatibility)
    check_garp,
    compute_aei,
    compute_mpi,
    recover_utility,
    find_ideal_point,
    check_separability,
    # Result types - new names
    ConsistencyResult,
    IntegrityResult,
    ConfusionResult,
    LatentValueResult,
    PreferenceAnchorResult,
    FeatureIndependenceResult,
    # Result types - old names
    GARPResult,
    AEIResult,
    MPIResult,
)
# Import with alias to avoid pytest collecting it as a test
from pyrevealed import test_feature_independence as check_feature_independence


# =============================================================================
# FIXTURES WITH NEW NAMES
# =============================================================================

@pytest.fixture
def consistent_log() -> BehaviorLog:
    """BehaviorLog with consistent behavior using new parameter names."""
    return BehaviorLog(
        cost_vectors=np.array([
            [1.0, 2.0],
            [2.0, 1.0],
            [1.5, 1.5],
        ]),
        action_vectors=np.array([
            [4.0, 1.0],
            [1.0, 4.0],
            [2.0, 2.0],
        ]),
        user_id="consistent_user"
    )


@pytest.fixture
def inconsistent_log() -> BehaviorLog:
    """BehaviorLog with GARP violation using new parameter names."""
    return BehaviorLog(
        cost_vectors=np.array([
            [1.0, 0.1],
            [0.1, 1.0],
        ]),
        action_vectors=np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ]),
        user_id="inconsistent_user"
    )


@pytest.fixture
def embedding_log() -> EmbeddingChoiceLog:
    """EmbeddingChoiceLog for preference anchor tests."""
    return EmbeddingChoiceLog(
        item_features=np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]),
        choice_sets=[[0, 1], [0, 2], [0, 3]],
        choices=[0, 0, 0],
    )


# =============================================================================
# BEHAVIORLOG TESTS
# =============================================================================

class TestBehaviorLog:
    """Tests for the new BehaviorLog data container."""

    def test_new_parameter_names(self):
        """BehaviorLog accepts new parameter names."""
        log = BehaviorLog(
            cost_vectors=np.array([[1.0, 2.0]]),
            action_vectors=np.array([[3.0, 1.0]]),
            user_id="test_user"
        )
        assert log.num_observations == 1
        assert log.num_goods == 2
        assert log.user_id == "test_user"

    def test_old_parameter_names(self):
        """BehaviorLog accepts old parameter names for backward compatibility."""
        log = BehaviorLog(
            prices=np.array([[1.0, 2.0]]),
            quantities=np.array([[3.0, 1.0]]),
            session_id="test_session"
        )
        assert log.num_observations == 1
        assert log.num_goods == 2
        # Old names should work
        assert log.session_id == "test_session"

    def test_mixed_parameters(self):
        """BehaviorLog accepts mix of old and new names."""
        log = BehaviorLog(
            cost_vectors=np.array([[1.0, 2.0]]),
            quantities=np.array([[3.0, 1.0]]),  # Old name
            user_id="test_user"
        )
        assert log.num_observations == 1

    def test_alias_consistency(self):
        """ConsumerSession is an alias for BehaviorLog."""
        # Both should create equivalent objects
        log = BehaviorLog(
            cost_vectors=np.array([[1.0, 2.0]]),
            action_vectors=np.array([[3.0, 1.0]]),
        )
        session = ConsumerSession(
            prices=np.array([[1.0, 2.0]]),
            quantities=np.array([[3.0, 1.0]]),
        )
        assert type(log).__name__ == type(session).__name__


# =============================================================================
# BEHAVIORAL AUDITOR TESTS
# =============================================================================

class TestBehavioralAuditor:
    """Tests for the BehavioralAuditor class."""

    def test_validate_history_consistent(self, consistent_log):
        """validate_history returns True for consistent behavior."""
        auditor = BehavioralAuditor()
        assert auditor.validate_history(consistent_log) is True

    def test_validate_history_inconsistent(self, inconsistent_log):
        """validate_history returns False for inconsistent behavior."""
        auditor = BehavioralAuditor()
        assert auditor.validate_history(inconsistent_log) is False

    def test_get_integrity_score_consistent(self, consistent_log):
        """Consistent behavior has high integrity score."""
        auditor = BehavioralAuditor()
        score = auditor.get_integrity_score(consistent_log)
        assert score == 1.0

    def test_get_integrity_score_inconsistent(self, inconsistent_log):
        """Inconsistent behavior has lower integrity score."""
        auditor = BehavioralAuditor()
        score = auditor.get_integrity_score(inconsistent_log)
        assert score < 1.0

    def test_get_confusion_score_consistent(self, consistent_log):
        """Consistent behavior has low confusion score."""
        auditor = BehavioralAuditor()
        score = auditor.get_confusion_score(consistent_log)
        assert score == 0.0

    def test_get_confusion_score_inconsistent(self, inconsistent_log):
        """Inconsistent behavior has higher confusion score."""
        auditor = BehavioralAuditor()
        score = auditor.get_confusion_score(inconsistent_log)
        assert score > 0.0

    def test_full_audit_returns_report(self, consistent_log):
        """full_audit returns an AuditReport."""
        auditor = BehavioralAuditor()
        report = auditor.full_audit(consistent_log)
        assert isinstance(report, AuditReport)
        assert hasattr(report, 'is_consistent')
        assert hasattr(report, 'integrity_score')
        assert hasattr(report, 'confusion_score')
        assert hasattr(report, 'bot_risk')
        assert hasattr(report, 'shared_account_risk')
        assert hasattr(report, 'ux_confusion_risk')

    def test_full_audit_consistent_low_risk(self, consistent_log):
        """Consistent behavior has low risk scores."""
        auditor = BehavioralAuditor()
        report = auditor.full_audit(consistent_log)
        assert report.is_consistent is True
        assert report.integrity_score == 1.0
        assert report.confusion_score == 0.0
        assert report.bot_risk <= 0.3

    def test_full_audit_inconsistent_higher_risk(self, inconsistent_log):
        """Inconsistent behavior has higher risk scores."""
        auditor = BehavioralAuditor()
        report = auditor.full_audit(inconsistent_log)
        assert report.is_consistent is False
        assert report.bot_risk > 0.3

    def test_precision_parameter(self, consistent_log):
        """Auditor accepts precision parameter."""
        auditor = BehavioralAuditor(precision=1e-10)
        assert auditor.precision == 1e-10
        # Should still work
        assert auditor.validate_history(consistent_log) is True

    def test_get_details_methods(self, consistent_log):
        """Detail methods return proper result types."""
        auditor = BehavioralAuditor()

        consistency_details = auditor.get_consistency_details(consistent_log)
        assert hasattr(consistency_details, 'is_consistent')

        integrity_details = auditor.get_integrity_details(consistent_log)
        assert hasattr(integrity_details, 'efficiency_index')

        confusion_details = auditor.get_confusion_details(consistent_log)
        assert hasattr(confusion_details, 'mpi_value')


# =============================================================================
# PREFERENCE ENCODER TESTS
# =============================================================================

class TestPreferenceEncoder:
    """Tests for the PreferenceEncoder class."""

    def test_fit_returns_self(self, consistent_log):
        """fit() returns self for method chaining."""
        encoder = PreferenceEncoder()
        result = encoder.fit(consistent_log)
        assert result is encoder

    def test_is_fitted_property(self, consistent_log):
        """is_fitted property works correctly."""
        encoder = PreferenceEncoder()
        assert encoder.is_fitted is False
        encoder.fit(consistent_log)
        assert encoder.is_fitted is True

    def test_extract_latent_values(self, consistent_log):
        """extract_latent_values returns array of correct shape."""
        encoder = PreferenceEncoder()
        encoder.fit(consistent_log)
        values = encoder.extract_latent_values()
        assert isinstance(values, np.ndarray)
        assert len(values) == consistent_log.num_observations

    def test_extract_marginal_weights(self, consistent_log):
        """extract_marginal_weights returns array of correct shape."""
        encoder = PreferenceEncoder()
        encoder.fit(consistent_log)
        weights = encoder.extract_marginal_weights()
        assert isinstance(weights, np.ndarray)
        assert len(weights) == consistent_log.num_observations

    def test_get_value_function(self, consistent_log):
        """get_value_function returns callable."""
        encoder = PreferenceEncoder()
        encoder.fit(consistent_log)
        value_fn = encoder.get_value_function()
        assert callable(value_fn)
        # Should be able to evaluate
        value = value_fn(np.array([2.0, 2.0]))
        assert isinstance(value, (int, float))

    def test_predict_choice(self, consistent_log):
        """predict_choice returns prediction or None."""
        encoder = PreferenceEncoder()
        encoder.fit(consistent_log)
        prediction = encoder.predict_choice(
            cost_vector=np.array([1.0, 1.0]),
            resource_limit=10.0
        )
        # Should return array or None
        assert prediction is None or isinstance(prediction, np.ndarray)

    def test_get_fit_details(self, consistent_log):
        """get_fit_details returns LatentValueResult."""
        encoder = PreferenceEncoder()
        encoder.fit(consistent_log)
        details = encoder.get_fit_details()
        assert hasattr(details, 'utility_values')
        assert hasattr(details, 'lagrange_multipliers')

    def test_not_fitted_raises_error(self):
        """Methods raise error when not fitted."""
        encoder = PreferenceEncoder()
        with pytest.raises(ValueError, match="not fitted"):
            encoder.extract_latent_values()

    def test_precision_parameter(self, consistent_log):
        """Encoder accepts precision parameter."""
        encoder = PreferenceEncoder(precision=1e-10)
        assert encoder.precision == 1e-10
        encoder.fit(consistent_log)
        assert encoder.is_fitted


# =============================================================================
# FUNCTION ALIAS TESTS
# =============================================================================

class TestFunctionAliases:
    """Tests that new function names work correctly."""

    def test_validate_consistency(self, consistent_log):
        """validate_consistency is alias for check_garp."""
        result_new = validate_consistency(consistent_log)
        result_old = check_garp(consistent_log)
        assert result_new.is_consistent == result_old.is_consistent

    def test_compute_integrity_score(self, consistent_log):
        """compute_integrity_score is alias for compute_aei."""
        result_new = compute_integrity_score(consistent_log)
        result_old = compute_aei(consistent_log)
        assert result_new.efficiency_index == result_old.efficiency_index

    def test_compute_confusion_metric(self, consistent_log):
        """compute_confusion_metric is alias for compute_mpi."""
        result_new = compute_confusion_metric(consistent_log)
        result_old = compute_mpi(consistent_log)
        assert result_new.mpi_value == result_old.mpi_value

    def test_fit_latent_values(self, consistent_log):
        """fit_latent_values is alias for recover_utility."""
        result_new = fit_latent_values(consistent_log)
        result_old = recover_utility(consistent_log)
        np.testing.assert_array_almost_equal(
            result_new.utility_values,
            result_old.utility_values
        )

    def test_find_preference_anchor(self, embedding_log):
        """find_preference_anchor is alias for find_ideal_point."""
        result_new = find_preference_anchor(embedding_log)
        result_old = find_ideal_point(embedding_log)
        np.testing.assert_array_almost_equal(
            result_new.ideal_point,
            result_old.ideal_point
        )

    def test_check_feature_independence(self, consistent_log):
        """test_feature_independence is alias for check_separability."""
        result_new = check_feature_independence(consistent_log, [0], [1])
        result_old = check_separability(consistent_log, [0], [1])
        assert result_new.is_separable == result_old.is_separable


# =============================================================================
# RESULT TYPE ALIAS TESTS
# =============================================================================

class TestResultTypeAliases:
    """Tests that result type aliases work correctly."""

    def test_consistency_result_alias(self, consistent_log):
        """ConsistencyResult is alias for GARPResult."""
        result = validate_consistency(consistent_log)
        assert isinstance(result, ConsistencyResult)
        assert isinstance(result, GARPResult)

    def test_integrity_result_alias(self, consistent_log):
        """IntegrityResult is alias for AEIResult."""
        result = compute_integrity_score(consistent_log)
        assert isinstance(result, IntegrityResult)
        assert isinstance(result, AEIResult)

    def test_confusion_result_alias(self, consistent_log):
        """ConfusionResult is alias for MPIResult."""
        result = compute_confusion_metric(consistent_log)
        assert isinstance(result, ConfusionResult)
        assert isinstance(result, MPIResult)

    def test_latent_value_result_alias(self, consistent_log):
        """LatentValueResult is alias for UtilityRecoveryResult."""
        result = fit_latent_values(consistent_log)
        assert isinstance(result, LatentValueResult)

    def test_preference_anchor_result_alias(self, embedding_log):
        """PreferenceAnchorResult is alias for IdealPointResult."""
        result = find_preference_anchor(embedding_log)
        assert isinstance(result, PreferenceAnchorResult)

    def test_feature_independence_result_alias(self, consistent_log):
        """FeatureIndependenceResult is alias for SeparabilityResult."""
        result = check_feature_independence(consistent_log, [0], [1])
        assert isinstance(result, FeatureIndependenceResult)


# =============================================================================
# BACKWARD COMPATIBILITY TESTS
# =============================================================================

class TestBackwardCompatibility:
    """Tests that old API still works."""

    def test_consumer_session_still_works(self):
        """ConsumerSession works as before."""
        session = ConsumerSession(
            prices=np.array([[1.0, 2.0]]),
            quantities=np.array([[3.0, 1.0]]),
            session_id="test"
        )
        result = check_garp(session)
        assert result.is_consistent is True

    def test_old_functions_work(self, simple_consistent_session):
        """Old function names still work."""
        # These should not raise
        check_garp(simple_consistent_session)
        compute_aei(simple_consistent_session)
        compute_mpi(simple_consistent_session)
        recover_utility(simple_consistent_session)

    def test_old_imports_work(self):
        """Old names are importable."""
        from pyrevealed import (
            ConsumerSession,
            RiskSession,
            SpatialSession,
            check_garp,
            check_warp,
            compute_aei,
            compute_mpi,
            recover_utility,
            find_ideal_point,
            check_separability,
        )
        # Just test they're callable/classes
        assert callable(check_garp)
        assert callable(compute_aei)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for the new API workflow."""

    def test_full_workflow_auditor(self, consistent_log):
        """Test complete auditor workflow."""
        auditor = BehavioralAuditor()

        # Step 1: Quick validation
        is_consistent = auditor.validate_history(consistent_log)
        assert is_consistent is True

        # Step 2: Get scores
        integrity = auditor.get_integrity_score(consistent_log)
        confusion = auditor.get_confusion_score(consistent_log)
        assert integrity == 1.0
        assert confusion == 0.0

        # Step 3: Full audit
        report = auditor.full_audit(consistent_log)
        assert report.bot_risk < 0.5

    def test_full_workflow_encoder(self, consistent_log):
        """Test complete encoder workflow."""
        encoder = PreferenceEncoder()

        # Step 1: Fit
        encoder.fit(consistent_log)
        assert encoder.is_fitted

        # Step 2: Extract features
        values = encoder.extract_latent_values()
        weights = encoder.extract_marginal_weights()
        assert len(values) == 3
        assert len(weights) == 3

        # Step 3: Get value function
        value_fn = encoder.get_value_function()
        v = value_fn(np.array([2.0, 2.0]))
        # Value function returns non-negative value (can be 0 for certain inputs)
        assert v >= 0

    def test_mixed_old_new_api(self, consistent_log):
        """Old and new APIs can be used together."""
        # Use new auditor class
        auditor = BehavioralAuditor()
        report = auditor.full_audit(consistent_log)

        # Use old function
        garp_result = check_garp(consistent_log)

        # Results should agree
        assert report.is_consistent == garp_result.is_consistent
