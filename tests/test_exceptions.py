"""Tests for custom exceptions and error handling in PyRevealed."""

import warnings

import numpy as np
import pytest

from pyrevealed import (
    # Data containers
    BehaviorLog,
    RiskChoiceLog,
    EmbeddingChoiceLog,
    LancasterLog,
    PreferenceEncoder,
    # Exceptions
    PyRevealedError,
    DataValidationError,
    DimensionError,
    ValueRangeError,
    NaNInfError,
    OptimizationError,
    NotFittedError,
    InsufficientDataError,
    # Warnings
    DataQualityWarning,
    NumericalInstabilityWarning,
    # Functions
    test_feature_independence,
    test_cross_price_effect,
)


class TestExceptionHierarchy:
    """Test that exception hierarchy is correct."""

    def test_pyrevealed_error_is_value_error(self):
        """PyRevealedError should inherit from ValueError for backward compat."""
        assert issubclass(PyRevealedError, ValueError)

    def test_data_validation_error_hierarchy(self):
        """DataValidationError and subclasses should inherit from PyRevealedError."""
        assert issubclass(DataValidationError, PyRevealedError)
        assert issubclass(DimensionError, DataValidationError)
        assert issubclass(ValueRangeError, DataValidationError)
        assert issubclass(NaNInfError, DataValidationError)

    def test_computation_exceptions_hierarchy(self):
        """Computation exceptions should inherit from PyRevealedError."""
        assert issubclass(OptimizationError, PyRevealedError)
        assert issubclass(NotFittedError, PyRevealedError)
        assert issubclass(InsufficientDataError, PyRevealedError)

    def test_warnings_hierarchy(self):
        """Warning classes should inherit from UserWarning."""
        assert issubclass(DataQualityWarning, UserWarning)
        assert issubclass(NumericalInstabilityWarning, UserWarning)

    def test_catch_all_pyrevealed_errors(self):
        """All library errors should be catchable with PyRevealedError."""
        with pytest.raises(PyRevealedError):
            BehaviorLog(
                cost_vectors=np.array([[1, 2]]),
                action_vectors=np.array([[1, 2, 3]]),  # Wrong shape
            )

    def test_catch_all_value_errors(self):
        """All library errors should be catchable with ValueError (backward compat)."""
        with pytest.raises(ValueError):
            BehaviorLog(
                cost_vectors=np.array([[1, 2]]),
                action_vectors=np.array([[1, 2, 3]]),  # Wrong shape
            )


class TestBehaviorLogNaNHandling:
    """Test NaN/Inf handling in BehaviorLog."""

    def test_nan_raises_by_default(self):
        """NaN values should raise NaNInfError by default."""
        prices = np.array([[1.0, 2.0], [np.nan, 1.0]])
        quantities = np.array([[1.0, 2.0], [1.0, 2.0]])

        with pytest.raises(NaNInfError) as exc_info:
            BehaviorLog(cost_vectors=prices, action_vectors=quantities)

        assert "NaN/Inf" in str(exc_info.value)
        assert "nan_policy" in str(exc_info.value)

    def test_inf_raises_by_default(self):
        """Inf values should raise NaNInfError by default."""
        prices = np.array([[1.0, np.inf], [2.0, 1.0]])
        quantities = np.array([[1.0, 2.0], [1.0, 2.0]])

        with pytest.raises(NaNInfError):
            BehaviorLog(cost_vectors=prices, action_vectors=quantities)

    def test_nan_policy_drop(self):
        """nan_policy='drop' should silently drop affected rows."""
        prices = np.array([[1.0, 2.0], [np.nan, 1.0], [2.0, 1.0]])
        quantities = np.array([[1.0, 2.0], [1.0, 2.0], [2.0, 1.0]])

        log = BehaviorLog(
            cost_vectors=prices, action_vectors=quantities, nan_policy="drop"
        )

        assert log.num_records == 2
        assert np.allclose(log.cost_vectors[0], [1.0, 2.0])
        assert np.allclose(log.cost_vectors[1], [2.0, 1.0])

    def test_nan_policy_warn(self):
        """nan_policy='warn' should warn and drop affected rows."""
        prices = np.array([[1.0, 2.0], [np.nan, 1.0], [2.0, 1.0]])
        quantities = np.array([[1.0, 2.0], [1.0, 2.0], [2.0, 1.0]])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            log = BehaviorLog(
                cost_vectors=prices, action_vectors=quantities, nan_policy="warn"
            )

            assert len(w) == 1
            assert issubclass(w[0].category, DataQualityWarning)
            assert "Dropping" in str(w[0].message)

        assert log.num_records == 2

    def test_nan_in_quantities(self):
        """NaN in action_vectors should also be detected."""
        prices = np.array([[1.0, 2.0], [2.0, 1.0]])
        quantities = np.array([[1.0, np.nan], [1.0, 2.0]])

        with pytest.raises(NaNInfError):
            BehaviorLog(cost_vectors=prices, action_vectors=quantities)


class TestBehaviorLogValidation:
    """Test validation error messages in BehaviorLog."""

    def test_dimension_mismatch_error(self):
        """Mismatched shapes should raise DimensionError with helpful message."""
        prices = np.array([[1.0, 2.0, 3.0]])
        quantities = np.array([[1.0, 2.0]])

        with pytest.raises(DimensionError) as exc_info:
            BehaviorLog(cost_vectors=prices, action_vectors=quantities)

        error_msg = str(exc_info.value)
        assert "does not match" in error_msg
        assert "(1, 3)" in error_msg
        assert "(1, 2)" in error_msg
        assert "Hint" in error_msg

    def test_non_2d_error(self):
        """Non-2D arrays should raise DimensionError."""
        prices = np.array([1.0, 2.0])  # 1D array
        quantities = np.array([1.0, 2.0])

        with pytest.raises(DimensionError) as exc_info:
            BehaviorLog(cost_vectors=prices, action_vectors=quantities)

        assert "2D" in str(exc_info.value)
        assert "Hint" in str(exc_info.value)

    def test_negative_cost_error(self):
        """Non-positive costs should raise ValueRangeError with positions."""
        prices = np.array([[1.0, -2.0], [2.0, 0.0]])
        quantities = np.array([[1.0, 2.0], [1.0, 2.0]])

        with pytest.raises(ValueRangeError) as exc_info:
            BehaviorLog(cost_vectors=prices, action_vectors=quantities)

        error_msg = str(exc_info.value)
        assert "non-positive costs" in error_msg
        assert "positions" in error_msg
        assert "Hint" in error_msg

    def test_negative_action_error(self):
        """Negative actions should raise ValueRangeError."""
        prices = np.array([[1.0, 2.0]])
        quantities = np.array([[1.0, -2.0]])

        with pytest.raises(ValueRangeError) as exc_info:
            BehaviorLog(cost_vectors=prices, action_vectors=quantities)

        assert "negative actions" in str(exc_info.value)

    def test_empty_data_error(self):
        """Empty arrays should raise InsufficientDataError."""
        prices = np.array([]).reshape(0, 2)
        quantities = np.array([]).reshape(0, 2)

        with pytest.raises(InsufficientDataError):
            BehaviorLog(cost_vectors=prices, action_vectors=quantities)


class TestRiskChoiceLogValidation:
    """Test validation in RiskChoiceLog."""

    def test_nan_detection(self):
        """NaN in risk data should raise NaNInfError."""
        safe = np.array([50.0, np.nan])
        outcomes = np.array([[100.0, 0.0], [200.0, 0.0]])
        probs = np.array([[0.5, 0.5], [0.5, 0.5]])
        choices = np.array([True, False])

        with pytest.raises(NaNInfError):
            RiskChoiceLog(safe, outcomes, probs, choices)

    def test_probability_sum_error(self):
        """Probabilities not summing to 1 should raise ValueRangeError."""
        safe = np.array([50.0, 100.0])
        outcomes = np.array([[100.0, 0.0], [200.0, 0.0]])
        probs = np.array([[0.5, 0.3], [0.5, 0.5]])  # First row sums to 0.8
        choices = np.array([True, False])

        with pytest.raises(ValueRangeError) as exc_info:
            RiskChoiceLog(safe, outcomes, probs, choices)

        assert "sum to 1" in str(exc_info.value)


class TestEmbeddingChoiceLogValidation:
    """Test validation in EmbeddingChoiceLog (SpatialSession)."""

    def test_nan_detection(self):
        """NaN in item features should raise NaNInfError."""
        features = np.array([[0.0, 0.0], [1.0, np.nan]])
        choice_sets = [[0, 1]]
        choices = [0]

        with pytest.raises(NaNInfError):
            EmbeddingChoiceLog(features, choice_sets, choices)

    def test_insufficient_choice_set(self):
        """Choice sets with < 2 items should raise InsufficientDataError."""
        features = np.array([[0.0, 0.0], [1.0, 1.0]])
        choice_sets = [[0]]  # Only 1 item
        choices = [0]

        with pytest.raises(InsufficientDataError):
            EmbeddingChoiceLog(features, choice_sets, choices)

    def test_invalid_choice_index(self):
        """Choice not in choice set should raise ValueRangeError."""
        features = np.array([[0.0, 0.0], [1.0, 1.0]])
        choice_sets = [[0, 1]]
        choices = [2]  # Invalid index

        with pytest.raises(ValueRangeError):
            EmbeddingChoiceLog(features, choice_sets, choices)


class TestLancasterLogValidation:
    """Test validation in LancasterLog."""

    def test_nan_detection(self):
        """NaN in any array should raise NaNInfError."""
        A = np.array([[95.0, 4.4], [105.0, np.nan]])
        prices = np.array([[1.0, 0.5], [0.8, 0.6]])
        quantities = np.array([[2, 3], [4, 1]])

        with pytest.raises(NaNInfError):
            LancasterLog(cost_vectors=prices, action_vectors=quantities, attribute_matrix=A)

    def test_rank_deficiency_warning(self):
        """Rank-deficient attribute matrix should emit DataQualityWarning."""
        A = np.array([[1.0, 2.0], [2.0, 4.0]])  # Rank 1 (columns are multiples)
        prices = np.array([[1.0, 0.5], [0.8, 0.6]])
        quantities = np.array([[2, 3], [4, 1]])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            LancasterLog(cost_vectors=prices, action_vectors=quantities, attribute_matrix=A)

            # Should have rank-deficiency warning
            quality_warnings = [x for x in w if issubclass(x.category, DataQualityWarning)]
            assert len(quality_warnings) >= 1
            assert "rank-deficient" in str(quality_warnings[0].message)


class TestPreferenceEncoderNotFitted:
    """Test NotFittedError in PreferenceEncoder."""

    def test_extract_before_fit(self):
        """Extracting features before fit should raise NotFittedError."""
        encoder = PreferenceEncoder()

        with pytest.raises(NotFittedError) as exc_info:
            encoder.extract_latent_values()

        assert "not fitted" in str(exc_info.value).lower()
        assert "Hint" in str(exc_info.value)

    def test_value_function_before_fit(self):
        """Getting value function before fit should raise NotFittedError."""
        encoder = PreferenceEncoder()

        with pytest.raises(NotFittedError):
            encoder.get_value_function()


class TestAlgorithmExceptions:
    """Test exceptions in algorithm modules."""

    def test_separability_overlap_error(self):
        """Overlapping groups should raise DataValidationError."""
        prices = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 4.0, 3.0]])
        quantities = np.array([[4.0, 1.0, 2.0, 1.0], [1.0, 4.0, 1.0, 2.0]])
        log = BehaviorLog(cost_vectors=prices, action_vectors=quantities)

        with pytest.raises(DataValidationError) as exc_info:
            test_feature_independence(log, [0, 1], [1, 2])  # Index 1 overlaps

        assert "overlap" in str(exc_info.value).lower()

    def test_separability_index_out_of_range(self):
        """Invalid good indices should raise ValueRangeError."""
        prices = np.array([[1.0, 2.0], [2.0, 1.0]])
        quantities = np.array([[4.0, 1.0], [1.0, 4.0]])
        log = BehaviorLog(cost_vectors=prices, action_vectors=quantities)

        with pytest.raises(ValueRangeError):
            test_feature_independence(log, [0], [5])  # Index 5 out of range

    def test_gross_substitutes_same_good(self):
        """Same good_g and good_h should raise DataValidationError."""
        prices = np.array([[1.0, 2.0], [2.0, 1.0]])
        quantities = np.array([[4.0, 1.0], [1.0, 4.0]])
        log = BehaviorLog(cost_vectors=prices, action_vectors=quantities)

        with pytest.raises(DataValidationError):
            test_cross_price_effect(log, 0, 0)  # Same good

    def test_gross_substitutes_index_out_of_range(self):
        """Invalid good indices should raise ValueRangeError."""
        prices = np.array([[1.0, 2.0], [2.0, 1.0]])
        quantities = np.array([[4.0, 1.0], [1.0, 4.0]])
        log = BehaviorLog(cost_vectors=prices, action_vectors=quantities)

        with pytest.raises(ValueRangeError):
            test_cross_price_effect(log, 0, 5)  # Index 5 out of range


class TestWarningControl:
    """Test that warnings can be controlled."""

    def test_suppress_data_quality_warning(self):
        """DataQualityWarning should be suppressible."""
        prices = np.array([[1.0, 2.0], [np.nan, 1.0], [2.0, 1.0]])
        quantities = np.array([[1.0, 2.0], [1.0, 2.0], [2.0, 1.0]])

        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("ignore", category=DataQualityWarning)
            BehaviorLog(cost_vectors=prices, action_vectors=quantities, nan_policy="warn")

            # Should have no DataQualityWarning
            quality_warnings = [x for x in w if issubclass(x.category, DataQualityWarning)]
            assert len(quality_warnings) == 0

    def test_promote_warning_to_error(self):
        """DataQualityWarning should be promotable to error."""
        prices = np.array([[1.0, 2.0], [np.nan, 1.0], [2.0, 1.0]])
        quantities = np.array([[1.0, 2.0], [1.0, 2.0], [2.0, 1.0]])

        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=DataQualityWarning)

            with pytest.raises(DataQualityWarning):
                BehaviorLog(
                    cost_vectors=prices, action_vectors=quantities, nan_policy="warn"
                )
