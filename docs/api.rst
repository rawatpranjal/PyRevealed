API
===

High-Level Classes
------------------

BehavioralAuditor
^^^^^^^^^^^^^^^^^

.. autoclass:: pyrevealed.BehavioralAuditor
   :members:
   :undoc-members:
   :show-inheritance:

AuditReport
^^^^^^^^^^^

.. autoclass:: pyrevealed.AuditReport
   :members:
   :undoc-members:

PreferenceEncoder
^^^^^^^^^^^^^^^^^

.. autoclass:: pyrevealed.PreferenceEncoder
   :members:
   :undoc-members:
   :show-inheritance:

Summary Classes
---------------

BehavioralSummary
^^^^^^^^^^^^^^^^^

.. autoclass:: pyrevealed.BehavioralSummary
   :members:
   :undoc-members:

PanelSummary
^^^^^^^^^^^^

.. autoclass:: pyrevealed.PanelSummary
   :members:
   :undoc-members:

Data Containers
---------------

BehaviorLog
^^^^^^^^^^^

.. autoclass:: pyrevealed.BehaviorLog
   :members:
   :undoc-members:

BehaviorPanel
^^^^^^^^^^^^^

.. autoclass:: pyrevealed.BehaviorPanel
   :members:
   :undoc-members:

MenuChoicePanel
^^^^^^^^^^^^^^^

.. autoclass:: pyrevealed.MenuChoicePanel
   :members:
   :undoc-members:

RiskChoiceLog
^^^^^^^^^^^^^

.. autoclass:: pyrevealed.RiskChoiceLog
   :members:
   :undoc-members:

EmbeddingChoiceLog
^^^^^^^^^^^^^^^^^^

.. autoclass:: pyrevealed.EmbeddingChoiceLog
   :members:
   :undoc-members:

Consistency Functions
---------------------

.. autofunction:: pyrevealed.validate_consistency

.. autofunction:: pyrevealed.validate_consistency_weak

.. autofunction:: pyrevealed.validate_sarp

.. autofunction:: pyrevealed.validate_smooth_preferences

.. autofunction:: pyrevealed.validate_strict_consistency

.. autofunction:: pyrevealed.validate_price_preferences

Efficiency Functions
--------------------

.. autofunction:: pyrevealed.compute_integrity_score

.. autofunction:: pyrevealed.compute_confusion_metric

.. autofunction:: pyrevealed.compute_minimal_outlier_fraction

.. autofunction:: pyrevealed.compute_granular_integrity

.. autofunction:: pyrevealed.compute_test_power

Preference Structure Functions
------------------------------

.. autofunction:: pyrevealed.validate_proportional_scaling

.. autofunction:: pyrevealed.test_income_invariance

.. autofunction:: pyrevealed.test_feature_independence

.. autofunction:: pyrevealed.test_cross_price_effect

.. autofunction:: pyrevealed.compute_cross_price_matrix

Utility Recovery
----------------

.. autofunction:: pyrevealed.fit_latent_values

.. autofunction:: pyrevealed.build_value_function

.. autofunction:: pyrevealed.predict_choice

Embedding Analysis
------------------

.. autofunction:: pyrevealed.find_preference_anchor

.. autofunction:: pyrevealed.validate_embedding_consistency

.. autofunction:: pyrevealed.compute_signal_strength

Risk Analysis
-------------

.. autofunction:: pyrevealed.compute_risk_profile

.. autofunction:: pyrevealed.check_expected_utility_axioms

.. autofunction:: pyrevealed.classify_risk_type

Menu Choice Functions
---------------------

MenuChoiceLog
^^^^^^^^^^^^^

.. autoclass:: pyrevealed.MenuChoiceLog
   :members:
   :undoc-members:

MenuPreferenceEncoder
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyrevealed.MenuPreferenceEncoder
   :members:
   :undoc-members:
   :show-inheritance:

MenuAuditReport
^^^^^^^^^^^^^^^

.. autoclass:: pyrevealed.MenuAuditReport
   :members:
   :undoc-members:

Menu Consistency Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyrevealed.validate_menu_warp

.. autofunction:: pyrevealed.validate_menu_sarp

.. autofunction:: pyrevealed.validate_menu_consistency

.. autofunction:: pyrevealed.compute_menu_efficiency

.. autofunction:: pyrevealed.fit_menu_preferences

Integrability (Slutsky Conditions)
----------------------------------

Test whether observed demand data is consistent with integrability conditions.
Based on Chambers & Echenique (2016) Chapter 6.4-6.5.

.. autofunction:: pyrevealed.test_integrability

.. autofunction:: pyrevealed.compute_slutsky_matrix

.. autofunction:: pyrevealed.check_slutsky_symmetry

.. autofunction:: pyrevealed.check_slutsky_nsd

Welfare Analysis
----------------

Analyze welfare changes from price variations using compensating and equivalent variation.
Based on Chambers & Echenique (2016) Chapter 7.3-7.4.

.. autofunction:: pyrevealed.analyze_welfare_change

.. autofunction:: pyrevealed.compute_compensating_variation

.. autofunction:: pyrevealed.compute_equivalent_variation

.. autofunction:: pyrevealed.recover_cost_function

.. autofunction:: pyrevealed.compute_consumer_surplus

.. autofunction:: pyrevealed.compute_deadweight_loss

Additive Separability
---------------------

Test whether preferences are additively separable across goods.
Based on Chambers & Echenique (2016) Chapter 9.3.

.. autofunction:: pyrevealed.test_additive_separability

.. autofunction:: pyrevealed.identify_additive_groups

.. autofunction:: pyrevealed.check_no_cross_effects

Compensated Demand
------------------

Analyze substitution and income effects via Slutsky decomposition.
Based on Chambers & Echenique (2016) Chapter 10.3.

.. autofunction:: pyrevealed.decompose_price_effects

.. autofunction:: pyrevealed.compute_hicksian_demand

.. autofunction:: pyrevealed.check_compensated_law_of_demand

.. autofunction:: pyrevealed.compute_slutsky_decomposition

.. autofunction:: pyrevealed.estimate_compensated_demand

General Metric Preferences
--------------------------

Analyze preferences with general distance metrics beyond Euclidean.
Based on Chambers & Echenique (2016) Chapter 11.3-11.4.

EmbeddingChoiceLog
^^^^^^^^^^^^^^^^^^

.. autoclass:: pyrevealed.EmbeddingChoiceLog
   :members:
   :undoc-members:

.. autofunction:: pyrevealed.find_ideal_point_general

.. autofunction:: pyrevealed.determine_best_metric

.. autofunction:: pyrevealed.test_metric_rationality

Stochastic Choice
-----------------

Analyze probabilistic choice data using random utility models.
Based on Chambers & Echenique (2016) Chapter 13.

StochasticChoiceLog
^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyrevealed.StochasticChoiceLog
   :members:
   :undoc-members:

.. autofunction:: pyrevealed.fit_random_utility_model

.. autofunction:: pyrevealed.test_mcfadden_axioms

.. autofunction:: pyrevealed.estimate_choice_probabilities

.. autofunction:: pyrevealed.check_independence_irrelevant_alternatives

.. autofunction:: pyrevealed.fit_luce_model

Limited Attention
-----------------

Test rationality under limited attention and estimate consideration sets.
Based on Chambers & Echenique (2016) Chapter 14.

.. autofunction:: pyrevealed.test_attention_rationality

.. autofunction:: pyrevealed.estimate_consideration_sets

.. autofunction:: pyrevealed.compute_salience_weights

.. autofunction:: pyrevealed.test_attention_filter

Production Theory
-----------------

Analyze firm behavior using revealed preference methods for production.
Based on Chambers & Echenique (2016) Chapter 15.

ProductionLog
^^^^^^^^^^^^^

.. autoclass:: pyrevealed.ProductionLog
   :members:
   :undoc-members:

.. autofunction:: pyrevealed.test_profit_maximization

.. autofunction:: pyrevealed.check_cost_minimization

.. autofunction:: pyrevealed.estimate_returns_to_scale

.. autofunction:: pyrevealed.compute_technical_efficiency

Dataset Loaders
---------------

.. autofunction:: pyrevealed.datasets.load_dunnhumby

.. autofunction:: pyrevealed.datasets.load_open_ecommerce

.. autofunction:: pyrevealed.datasets.load_uci_retail

.. autofunction:: pyrevealed.datasets.list_datasets

Exceptions and Warnings
-----------------------

PyRevealed provides custom exceptions that inherit from ``ValueError`` for
backward compatibility.

Base Exception
^^^^^^^^^^^^^^

.. autoexception:: pyrevealed.PyRevealedError
   :show-inheritance:

Data Validation Exceptions
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoexception:: pyrevealed.DataValidationError
   :show-inheritance:

.. autoexception:: pyrevealed.DimensionError
   :show-inheritance:

.. autoexception:: pyrevealed.ValueRangeError
   :show-inheritance:

.. autoexception:: pyrevealed.NaNInfError
   :show-inheritance:

Computation Exceptions
^^^^^^^^^^^^^^^^^^^^^^

.. autoexception:: pyrevealed.OptimizationError
   :show-inheritance:

.. autoexception:: pyrevealed.NotFittedError
   :show-inheritance:

.. autoexception:: pyrevealed.InsufficientDataError
   :show-inheritance:

Warnings
^^^^^^^^

.. autoclass:: pyrevealed.DataQualityWarning
   :show-inheritance:

.. autoclass:: pyrevealed.NumericalInstabilityWarning
   :show-inheritance:

Troubleshooting
---------------

**Common Errors**

- ``ValueRangeError: Found non-positive costs`` --- All prices must be > 0. Check for zeros or missing data encoded as 0.
- ``DimensionError: cost_vectors shape does not match`` --- Prices and quantities must have the same shape (T x N).
- ``NaNInfError: Found NaN/Inf values`` --- Use ``nan_policy="drop"`` to automatically remove bad rows: ``BehaviorLog(..., nan_policy="drop")``.
- ``InsufficientDataError: Must have at least 2 observations`` --- Need T >= 2 for meaningful analysis.
- ``ImportError: pandas is required`` --- Install with ``pip install pyrevealed[datasets]`` for dataset loaders.

**Tips**

- For large panels, the first call may be slow due to Numba JIT compilation. Subsequent calls are fast.
- If ``compute_integrity_score`` is slow for T > 500, the SCC-optimized path activates automatically.
- Memory usage scales as O(T^2) per user due to the T x T revealed preference matrices.
