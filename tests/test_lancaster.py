"""Tests for Lancaster Characteristics Model."""

import numpy as np
import pytest

from pyrevealed import (
    LancasterLog,
    CharacteristicsLog,
    BehaviorLog,
    transform_to_characteristics,
    validate_consistency,
    compute_integrity_score,
    LancasterResult,
)


class TestLancasterLogCreation:
    """Test LancasterLog instantiation and validation."""

    def test_basic_creation(self, simple_lancaster_data):
        """Test basic LancasterLog creation."""
        prices, quantities, A = simple_lancaster_data
        log = LancasterLog(
            cost_vectors=prices,
            action_vectors=quantities,
            attribute_matrix=A,
        )
        assert log.num_observations == 3
        assert log.num_products == 2
        assert log.num_characteristics == 2

    def test_with_user_id(self, simple_lancaster_data):
        """Test LancasterLog with user_id."""
        prices, quantities, A = simple_lancaster_data
        log = LancasterLog(
            cost_vectors=prices,
            action_vectors=quantities,
            attribute_matrix=A,
            user_id="test_user",
        )
        assert log.user_id == "test_user"

    def test_with_metadata(self, simple_lancaster_data):
        """Test LancasterLog with metadata."""
        prices, quantities, A = simple_lancaster_data
        log = LancasterLog(
            cost_vectors=prices,
            action_vectors=quantities,
            attribute_matrix=A,
            metadata={"characteristic_names": ["calories", "fiber"]},
        )
        assert log.metadata["characteristic_names"] == ["calories", "fiber"]

    def test_legacy_parameter_names(self, simple_lancaster_data):
        """Test that prices/quantities aliases work."""
        prices, quantities, A = simple_lancaster_data
        log = LancasterLog(
            prices=prices,
            quantities=quantities,
            attribute_matrix=A,
        )
        assert log.num_observations == 3
        # Legacy aliases should be in sync
        np.testing.assert_array_equal(log.prices, log.cost_vectors)
        np.testing.assert_array_equal(log.quantities, log.action_vectors)

    def test_characteristics_log_alias(self, simple_lancaster_data):
        """Test that CharacteristicsLog is an alias for LancasterLog."""
        prices, quantities, A = simple_lancaster_data
        log = CharacteristicsLog(
            cost_vectors=prices,
            action_vectors=quantities,
            attribute_matrix=A,
        )
        assert isinstance(log, LancasterLog)

    def test_dimension_mismatch_error(self):
        """Test error when A rows don't match N products."""
        prices = np.array([[1.0, 2.0]])
        quantities = np.array([[1.0, 1.0]])
        A = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ])  # 3 rows, but only 2 products

        with pytest.raises(ValueError, match="attribute_matrix rows"):
            LancasterLog(
                cost_vectors=prices,
                action_vectors=quantities,
                attribute_matrix=A,
            )

    def test_quantities_prices_shape_mismatch(self):
        """Test error when prices and quantities shapes don't match."""
        prices = np.array([[1.0, 2.0]])
        quantities = np.array([[1.0, 1.0, 1.0]])  # Wrong shape
        A = np.array([[1.0, 2.0], [3.0, 4.0]])

        with pytest.raises(ValueError, match="action_vectors shape"):
            LancasterLog(
                cost_vectors=prices,
                action_vectors=quantities,
                attribute_matrix=A,
            )

    def test_negative_cost_error(self):
        """Test error when costs are negative."""
        prices = np.array([[-1.0, 2.0]])  # Negative price
        quantities = np.array([[1.0, 1.0]])
        A = np.array([[1.0, 2.0], [3.0, 4.0]])

        with pytest.raises(ValueError, match="strictly positive"):
            LancasterLog(
                cost_vectors=prices,
                action_vectors=quantities,
                attribute_matrix=A,
            )

    def test_zero_cost_error(self):
        """Test error when costs are zero."""
        prices = np.array([[0.0, 2.0]])  # Zero price
        quantities = np.array([[1.0, 1.0]])
        A = np.array([[1.0, 2.0], [3.0, 4.0]])

        with pytest.raises(ValueError, match="strictly positive"):
            LancasterLog(
                cost_vectors=prices,
                action_vectors=quantities,
                attribute_matrix=A,
            )

    def test_negative_quantity_error(self):
        """Test error when quantities are negative."""
        prices = np.array([[1.0, 2.0]])
        quantities = np.array([[-1.0, 1.0]])  # Negative quantity
        A = np.array([[1.0, 2.0], [3.0, 4.0]])

        with pytest.raises(ValueError, match="non-negative"):
            LancasterLog(
                cost_vectors=prices,
                action_vectors=quantities,
                attribute_matrix=A,
            )

    def test_negative_attribute_error(self):
        """Test error when attribute matrix has negative values."""
        prices = np.array([[1.0, 2.0]])
        quantities = np.array([[1.0, 1.0]])
        A = np.array([[1.0, -2.0], [3.0, 4.0]])  # Negative attribute

        with pytest.raises(ValueError, match="non-negative"):
            LancasterLog(
                cost_vectors=prices,
                action_vectors=quantities,
                attribute_matrix=A,
            )

    def test_zero_product_row_error(self):
        """Test error when a product has no characteristics."""
        prices = np.array([[1.0, 2.0]])
        quantities = np.array([[1.0, 1.0]])
        A = np.array([
            [1.0, 2.0],
            [0.0, 0.0],
        ])  # Product 2 has no characteristics

        with pytest.raises(ValueError, match="no characteristics"):
            LancasterLog(
                cost_vectors=prices,
                action_vectors=quantities,
                attribute_matrix=A,
            )

    def test_rank_deficient_warning(self, rank_deficient_attribute_matrix):
        """Test warning for rank-deficient A."""
        prices = np.array([[1.0, 2.0]])
        quantities = np.array([[1.0, 1.0]])

        with pytest.warns(UserWarning, match="rank-deficient"):
            LancasterLog(
                cost_vectors=prices,
                action_vectors=quantities,
                attribute_matrix=rank_deficient_attribute_matrix,
            )

    def test_unused_characteristic_warning(self):
        """Test warning when a characteristic is not in any product."""
        prices = np.array([[1.0, 2.0]])
        quantities = np.array([[1.0, 1.0]])
        A = np.array([
            [1.0, 0.0],  # Only has characteristic 1
            [2.0, 0.0],  # Only has characteristic 1
        ])  # Characteristic 2 is never used

        with pytest.warns(UserWarning, match="not present in any product"):
            LancasterLog(
                cost_vectors=prices,
                action_vectors=quantities,
                attribute_matrix=A,
            )


class TestCharacteristicsTransformation:
    """Test the Z = X @ A transformation."""

    def test_characteristics_quantities_shape(self, simple_lancaster_data):
        """Test output shape is T x K."""
        prices, quantities, A = simple_lancaster_data
        log = LancasterLog(
            cost_vectors=prices,
            action_vectors=quantities,
            attribute_matrix=A,
        )

        assert log.characteristics_quantities.shape == (3, 2)  # T=3, K=2

    def test_characteristics_quantities_values(self, identity_attribute_matrix):
        """Test Z = X @ A calculation with identity-like A."""
        A = identity_attribute_matrix
        prices = np.array([[1.0, 1.0]])
        quantities = np.array([[3.0, 5.0]])

        log = LancasterLog(
            cost_vectors=prices,
            action_vectors=quantities,
            attribute_matrix=A,
        )

        expected_Z = np.array([[3.0, 5.0]])  # Same as quantities for identity A
        np.testing.assert_array_almost_equal(
            log.characteristics_quantities, expected_Z
        )

    def test_characteristics_aggregation(self):
        """Test that characteristics aggregate correctly across products."""
        # Two products, each providing some of characteristic 1
        A = np.array([
            [2.0, 0.0],
            [3.0, 0.0],
        ])  # Only char 1, different amounts
        prices = np.array([[1.0, 1.0]])
        quantities = np.array([[1.0, 1.0]])  # 1 of each product

        log = LancasterLog(
            cost_vectors=prices,
            action_vectors=quantities,
            attribute_matrix=A,
        )

        # Total characteristic 1 = 1*2 + 1*3 = 5
        assert log.characteristics_quantities[0, 0] == 5.0
        assert log.characteristics_quantities[0, 1] == 0.0

    def test_zero_quantities_produce_zero_characteristics(self):
        """Test that zero quantities produce zero characteristics."""
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        prices = np.array([[1.0, 1.0]])
        quantities = np.array([[0.0, 0.0]])  # No purchases

        log = LancasterLog(
            cost_vectors=prices,
            action_vectors=quantities,
            attribute_matrix=A,
        )

        np.testing.assert_array_equal(
            log.characteristics_quantities, np.array([[0.0, 0.0]])
        )


class TestShadowPrices:
    """Test NNLS shadow price computation."""

    def test_shadow_prices_shape(self, simple_lancaster_data):
        """Test shadow prices have shape T x K."""
        prices, quantities, A = simple_lancaster_data
        log = LancasterLog(
            cost_vectors=prices,
            action_vectors=quantities,
            attribute_matrix=A,
        )

        assert log.shadow_prices.shape == (3, 2)  # T=3, K=2

    def test_shadow_prices_non_negative(self, simple_lancaster_data):
        """Test all shadow prices are non-negative (NNLS constraint)."""
        prices, quantities, A = simple_lancaster_data
        log = LancasterLog(
            cost_vectors=prices,
            action_vectors=quantities,
            attribute_matrix=A,
        )

        assert np.all(log.shadow_prices >= 0)

    def test_shadow_prices_exact_solution(self, identity_attribute_matrix):
        """Test shadow prices are exact when A is identity."""
        A = identity_attribute_matrix
        prices = np.array([[2.0, 3.0], [1.5, 4.0]])
        quantities = np.array([[1.0, 1.0], [2.0, 2.0]])

        log = LancasterLog(
            cost_vectors=prices,
            action_vectors=quantities,
            attribute_matrix=A,
        )

        # Shadow prices should match product prices for identity A
        np.testing.assert_array_almost_equal(log.shadow_prices, prices)

    def test_nnls_residuals_exist(self, simple_lancaster_data):
        """Test that NNLS residuals are computed."""
        prices, quantities, A = simple_lancaster_data
        log = LancasterLog(
            cost_vectors=prices,
            action_vectors=quantities,
            attribute_matrix=A,
        )

        assert log.nnls_residuals.shape == (3,)  # T observations
        assert np.all(log.nnls_residuals >= 0)

    def test_nnls_residuals_zero_for_exact_fit(self, identity_attribute_matrix):
        """Test residuals are zero when fit is exact."""
        A = identity_attribute_matrix
        prices = np.array([[2.0, 3.0]])
        quantities = np.array([[1.0, 1.0]])

        log = LancasterLog(
            cost_vectors=prices,
            action_vectors=quantities,
            attribute_matrix=A,
        )

        np.testing.assert_almost_equal(log.nnls_residuals[0], 0.0)


class TestBehaviorLogGeneration:
    """Test that behavior_log produces valid BehaviorLog."""

    def test_behavior_log_type(self, simple_lancaster_data):
        """Test that behavior_log returns BehaviorLog instance."""
        prices, quantities, A = simple_lancaster_data
        log = LancasterLog(
            cost_vectors=prices,
            action_vectors=quantities,
            attribute_matrix=A,
        )

        assert isinstance(log.behavior_log, BehaviorLog)

    def test_behavior_log_dimensions(self, simple_lancaster_data):
        """Test behavior_log has T records and K features."""
        prices, quantities, A = simple_lancaster_data
        log = LancasterLog(
            cost_vectors=prices,
            action_vectors=quantities,
            attribute_matrix=A,
        )

        char_log = log.behavior_log
        assert char_log.num_records == 3  # T observations
        assert char_log.num_features == 2  # K characteristics

    def test_behavior_log_user_id_suffix(self, simple_lancaster_data):
        """Test behavior_log user_id gets _characteristics suffix."""
        prices, quantities, A = simple_lancaster_data
        log = LancasterLog(
            cost_vectors=prices,
            action_vectors=quantities,
            attribute_matrix=A,
            user_id="test_user",
        )

        assert log.behavior_log.user_id == "test_user_characteristics"

    def test_behavior_log_metadata(self, simple_lancaster_data):
        """Test behavior_log preserves and extends metadata."""
        prices, quantities, A = simple_lancaster_data
        log = LancasterLog(
            cost_vectors=prices,
            action_vectors=quantities,
            attribute_matrix=A,
            metadata={"custom_field": "value"},
        )

        char_log = log.behavior_log
        assert char_log.metadata["custom_field"] == "value"
        assert char_log.metadata["lancaster_source"] is True
        assert char_log.metadata["num_products"] == 2
        assert char_log.metadata["num_characteristics"] == 2

    def test_behavior_log_with_algorithms(self, well_specified_lancaster_data):
        """Test that behavior_log works with standard algorithms."""
        prices, quantities, A = well_specified_lancaster_data
        log = LancasterLog(
            cost_vectors=prices,
            action_vectors=quantities,
            attribute_matrix=A,
        )

        char_log = log.behavior_log

        # Should not raise errors
        consistency = validate_consistency(char_log)
        integrity = compute_integrity_score(char_log)

        assert hasattr(consistency, "is_consistent")
        assert 0 <= integrity.efficiency_index <= 1

    def test_behavior_log_caching(self, simple_lancaster_data):
        """Test that behavior_log is cached."""
        prices, quantities, A = simple_lancaster_data
        log = LancasterLog(
            cost_vectors=prices,
            action_vectors=quantities,
            attribute_matrix=A,
        )

        # Access twice, should return same object
        log1 = log.behavior_log
        log2 = log.behavior_log
        assert log1 is log2


class TestValuationReport:
    """Test valuation_report method."""

    def test_report_structure(self, simple_lancaster_data):
        """Test report returns correct dataclass."""
        prices, quantities, A = simple_lancaster_data
        log = LancasterLog(
            cost_vectors=prices,
            action_vectors=quantities,
            attribute_matrix=A,
        )

        report = log.valuation_report()

        assert isinstance(report, LancasterResult)
        assert len(report.mean_shadow_prices) == 2
        assert len(report.shadow_price_std) == 2
        assert len(report.shadow_price_cv) == 2
        assert len(report.spend_shares) == 2

    def test_spend_shares_sum_to_one(self, well_specified_lancaster_data):
        """Test spend shares sum to approximately 1."""
        prices, quantities, A = well_specified_lancaster_data
        log = LancasterLog(
            cost_vectors=prices,
            action_vectors=quantities,
            attribute_matrix=A,
        )

        report = log.valuation_report()

        np.testing.assert_almost_equal(report.spend_shares.sum(), 1.0)

    def test_well_specified_flag(self, well_specified_lancaster_data):
        """Test is_well_specified is True for full rank A."""
        prices, quantities, A = well_specified_lancaster_data
        log = LancasterLog(
            cost_vectors=prices,
            action_vectors=quantities,
            attribute_matrix=A,
        )

        report = log.valuation_report()

        assert report.is_well_specified is True
        assert report.attribute_matrix_rank == 2

    def test_characteristic_names_from_metadata(self, simple_lancaster_data):
        """Test characteristic names come from metadata."""
        prices, quantities, A = simple_lancaster_data
        log = LancasterLog(
            cost_vectors=prices,
            action_vectors=quantities,
            attribute_matrix=A,
            metadata={"characteristic_names": ["calories", "fiber"]},
        )

        report = log.valuation_report()
        assert report.characteristic_names == ["calories", "fiber"]

    def test_most_valued_characteristic(self, identity_attribute_matrix):
        """Test most_valued_characteristic property."""
        A = identity_attribute_matrix
        prices = np.array([[1.0, 5.0]])  # Char 2 is more expensive
        quantities = np.array([[1.0, 1.0]])

        log = LancasterLog(
            cost_vectors=prices,
            action_vectors=quantities,
            attribute_matrix=A,
        )

        report = log.valuation_report()
        assert report.most_valued_characteristic == 1  # Index of char 2

    def test_computation_time_tracked(self, simple_lancaster_data):
        """Test computation time is recorded."""
        prices, quantities, A = simple_lancaster_data
        log = LancasterLog(
            cost_vectors=prices,
            action_vectors=quantities,
            attribute_matrix=A,
        )

        report = log.valuation_report()
        assert report.computation_time_ms >= 0

    def test_residual_threshold(self, simple_lancaster_data):
        """Test residual_threshold parameter affects problematic_observations."""
        prices, quantities, A = simple_lancaster_data
        log = LancasterLog(
            cost_vectors=prices,
            action_vectors=quantities,
            attribute_matrix=A,
        )

        # Very strict threshold should flag more observations
        report_strict = log.valuation_report(residual_threshold=0.001)
        # Very lenient threshold should flag fewer
        report_lenient = log.valuation_report(residual_threshold=1.0)

        assert len(report_lenient.problematic_observations) <= len(
            report_strict.problematic_observations
        )


class TestTransformToCharacteristics:
    """Test convenience function."""

    def test_from_behavior_log(self, simple_lancaster_data):
        """Test creating LancasterLog from BehaviorLog."""
        prices, quantities, A = simple_lancaster_data

        original_log = BehaviorLog(cost_vectors=prices, action_vectors=quantities)
        lancaster_log = transform_to_characteristics(original_log, A)

        assert isinstance(lancaster_log, LancasterLog)
        assert lancaster_log.num_observations == 3

    def test_preserves_user_id(self, simple_lancaster_data):
        """Test that user_id is preserved."""
        prices, quantities, A = simple_lancaster_data

        original_log = BehaviorLog(
            cost_vectors=prices,
            action_vectors=quantities,
            user_id="original_user",
        )
        lancaster_log = transform_to_characteristics(original_log, A)

        assert lancaster_log.user_id == "original_user"

    def test_preserves_metadata(self, simple_lancaster_data):
        """Test that original metadata is preserved."""
        prices, quantities, A = simple_lancaster_data

        original_log = BehaviorLog(
            cost_vectors=prices,
            action_vectors=quantities,
            metadata={"original_key": "original_value"},
        )
        lancaster_log = transform_to_characteristics(original_log, A)

        assert lancaster_log.metadata["original_key"] == "original_value"

    def test_characteristic_names_added(self, simple_lancaster_data):
        """Test that characteristic names are passed through."""
        prices, quantities, A = simple_lancaster_data

        original_log = BehaviorLog(cost_vectors=prices, action_vectors=quantities)
        lancaster_log = transform_to_characteristics(
            original_log, A, characteristic_names=["calories", "fiber"]
        )

        report = lancaster_log.valuation_report()
        assert report.characteristic_names == ["calories", "fiber"]


class TestPropertiesAndAliases:
    """Test property aliases and convenience methods."""

    def test_num_records_alias(self, simple_lancaster_data):
        """Test num_records is alias for num_observations."""
        prices, quantities, A = simple_lancaster_data
        log = LancasterLog(
            cost_vectors=prices,
            action_vectors=quantities,
            attribute_matrix=A,
        )

        assert log.num_records == log.num_observations

    def test_num_features_alias(self, simple_lancaster_data):
        """Test num_features is alias for num_characteristics."""
        prices, quantities, A = simple_lancaster_data
        log = LancasterLog(
            cost_vectors=prices,
            action_vectors=quantities,
            attribute_matrix=A,
        )

        assert log.num_features == log.num_characteristics


class TestIntegrationWithAlgorithms:
    """Integration tests with other PyRevealed algorithms."""

    def test_consistency_check_works(self, well_specified_lancaster_data):
        """Test running validate_consistency on characteristics log."""
        prices, quantities, A = well_specified_lancaster_data
        log = LancasterLog(
            cost_vectors=prices,
            action_vectors=quantities,
            attribute_matrix=A,
        )

        result = validate_consistency(log.behavior_log)
        assert hasattr(result, "is_consistent")

    def test_integrity_score_works(self, well_specified_lancaster_data):
        """Test running compute_integrity_score on characteristics log."""
        prices, quantities, A = well_specified_lancaster_data
        log = LancasterLog(
            cost_vectors=prices,
            action_vectors=quantities,
            attribute_matrix=A,
        )

        result = compute_integrity_score(log.behavior_log)
        assert 0 <= result.efficiency_index <= 1

    def test_product_vs_characteristics_consistency(self, simple_lancaster_data):
        """Test comparing consistency in product vs characteristics space."""
        prices, quantities, A = simple_lancaster_data

        # Product space analysis
        product_log = BehaviorLog(cost_vectors=prices, action_vectors=quantities)
        product_result = validate_consistency(product_log)

        # Characteristics space analysis
        lancaster_log = LancasterLog(
            cost_vectors=prices, action_vectors=quantities, attribute_matrix=A
        )
        char_result = validate_consistency(lancaster_log.behavior_log)

        # Both should be valid results
        assert hasattr(product_result, "is_consistent")
        assert hasattr(char_result, "is_consistent")
