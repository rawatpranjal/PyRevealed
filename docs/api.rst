API
===

One-Liner API
-------------

.. autofunction:: prefgraph.analyze

Engine (Batch Scoring)
----------------------

Engine
^^^^^^

.. autoclass:: prefgraph.Engine
   :members:
   :undoc-members:

EngineResult
^^^^^^^^^^^^

.. autoclass:: prefgraph.EngineResult
   :members:
   :undoc-members:

High-Level Classes
------------------

BehavioralAuditor
^^^^^^^^^^^^^^^^^

.. autoclass:: prefgraph.BehavioralAuditor
   :members:
   :undoc-members:
   :show-inheritance:

AuditReport
^^^^^^^^^^^

.. autoclass:: prefgraph.AuditReport
   :members:
   :undoc-members:

PreferenceEncoder
^^^^^^^^^^^^^^^^^

.. autoclass:: prefgraph.PreferenceEncoder
   :members:
   :undoc-members:
   :show-inheritance:

Summary Classes
---------------

BehavioralSummary
^^^^^^^^^^^^^^^^^

.. autoclass:: prefgraph.BehavioralSummary
   :members:
   :undoc-members:

PanelSummary
^^^^^^^^^^^^

.. autoclass:: prefgraph.PanelSummary
   :members:
   :undoc-members:

Data Containers
---------------

BehaviorLog
^^^^^^^^^^^

.. autoclass:: prefgraph.BehaviorLog
   :members:
   :undoc-members:

BehaviorPanel
^^^^^^^^^^^^^

.. autoclass:: prefgraph.BehaviorPanel
   :members:
   :undoc-members:

MenuChoicePanel
^^^^^^^^^^^^^^^

.. autoclass:: prefgraph.MenuChoicePanel
   :members:
   :undoc-members:

RiskChoiceLog
^^^^^^^^^^^^^

.. autoclass:: prefgraph.RiskChoiceLog
   :members:
   :undoc-members:

EmbeddingChoiceLog
^^^^^^^^^^^^^^^^^^

.. autoclass:: prefgraph.EmbeddingChoiceLog
   :members:
   :undoc-members:

Consistency Functions
---------------------

.. autofunction:: prefgraph.validate_consistency

.. autofunction:: prefgraph.validate_consistency_weak

.. autofunction:: prefgraph.validate_sarp

.. autofunction:: prefgraph.validate_smooth_preferences

.. autofunction:: prefgraph.validate_strict_consistency

.. autofunction:: prefgraph.validate_price_preferences

Efficiency Functions
--------------------

.. autofunction:: prefgraph.compute_integrity_score

.. autofunction:: prefgraph.compute_confusion_metric

.. autofunction:: prefgraph.compute_minimal_outlier_fraction

.. autofunction:: prefgraph.compute_granular_integrity

.. autofunction:: prefgraph.compute_test_power

Preference Structure Functions
------------------------------

.. autofunction:: prefgraph.validate_proportional_scaling

.. autofunction:: prefgraph.test_income_invariance

.. autofunction:: prefgraph.test_feature_independence

.. autofunction:: prefgraph.test_cross_price_effect

.. autofunction:: prefgraph.compute_cross_price_matrix

Utility Recovery
----------------

.. autofunction:: prefgraph.fit_latent_values

.. autofunction:: prefgraph.build_value_function

.. autofunction:: prefgraph.predict_choice

Embedding Analysis
------------------

.. autofunction:: prefgraph.find_preference_anchor

.. autofunction:: prefgraph.validate_embedding_consistency

.. autofunction:: prefgraph.compute_signal_strength

Risk Analysis
-------------

.. autofunction:: prefgraph.compute_risk_profile

.. autofunction:: prefgraph.check_expected_utility_axioms

.. autofunction:: prefgraph.classify_risk_type

Menu Choice Functions
---------------------

MenuChoiceLog
^^^^^^^^^^^^^

.. autoclass:: prefgraph.MenuChoiceLog
   :members:
   :undoc-members:

MenuPreferenceEncoder
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: prefgraph.MenuPreferenceEncoder
   :members:
   :undoc-members:
   :show-inheritance:

MenuAuditReport
^^^^^^^^^^^^^^^

.. autoclass:: prefgraph.MenuAuditReport
   :members:
   :undoc-members:

Menu Consistency Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: prefgraph.validate_menu_warp

.. autofunction:: prefgraph.validate_menu_sarp

.. autofunction:: prefgraph.validate_menu_consistency

.. autofunction:: prefgraph.compute_menu_efficiency

.. autofunction:: prefgraph.fit_menu_preferences

Integrability (Slutsky Conditions)
----------------------------------

Test whether observed demand data is consistent with integrability conditions.
Based on Chambers & Echenique (2016) Chapter 6.4-6.5.

.. autofunction:: prefgraph.test_integrability

.. autofunction:: prefgraph.compute_slutsky_matrix

.. autofunction:: prefgraph.check_slutsky_symmetry

.. autofunction:: prefgraph.check_slutsky_nsd

Welfare Analysis
----------------

Analyze welfare changes from price variations using compensating and equivalent variation.
Based on Chambers & Echenique (2016) Chapter 7.3-7.4.

.. autofunction:: prefgraph.analyze_welfare_change

.. autofunction:: prefgraph.compute_compensating_variation

.. autofunction:: prefgraph.compute_equivalent_variation

.. autofunction:: prefgraph.recover_cost_function

.. autofunction:: prefgraph.compute_consumer_surplus

.. autofunction:: prefgraph.compute_deadweight_loss

Additive Separability
---------------------

Test whether preferences are additively separable across goods.
Based on Chambers & Echenique (2016) Chapter 9.3.

.. autofunction:: prefgraph.test_additive_separability

.. autofunction:: prefgraph.identify_additive_groups

.. autofunction:: prefgraph.check_no_cross_effects

Compensated Demand
------------------

Analyze substitution and income effects via Slutsky decomposition.
Based on Chambers & Echenique (2016) Chapter 10.3.

.. autofunction:: prefgraph.decompose_price_effects

.. autofunction:: prefgraph.compute_hicksian_demand

.. autofunction:: prefgraph.check_compensated_law_of_demand

.. autofunction:: prefgraph.compute_slutsky_decomposition

.. autofunction:: prefgraph.estimate_compensated_demand

General Metric Preferences
--------------------------

Analyze preferences with general distance metrics beyond Euclidean.
Based on Chambers & Echenique (2016) Chapter 11.3-11.4.

EmbeddingChoiceLog
^^^^^^^^^^^^^^^^^^

.. autoclass:: prefgraph.EmbeddingChoiceLog
   :members:
   :undoc-members:

.. autofunction:: prefgraph.find_ideal_point_general

.. autofunction:: prefgraph.determine_best_metric

.. autofunction:: prefgraph.test_metric_rationality

Stochastic Choice
-----------------

Analyze probabilistic choice data using random utility models.
Based on Chambers & Echenique (2016) Chapter 13.

StochasticChoiceLog
^^^^^^^^^^^^^^^^^^^

.. autoclass:: prefgraph.StochasticChoiceLog
   :members:
   :undoc-members:

.. autofunction:: prefgraph.fit_random_utility_model

.. autofunction:: prefgraph.test_mcfadden_axioms

.. autofunction:: prefgraph.estimate_choice_probabilities

.. autofunction:: prefgraph.check_independence_irrelevant_alternatives

.. autofunction:: prefgraph.fit_luce_model

Limited Attention
-----------------

Test rationality under limited attention and estimate consideration sets.
Based on Chambers & Echenique (2016) Chapter 14.

.. autofunction:: prefgraph.test_attention_rationality

.. autofunction:: prefgraph.estimate_consideration_sets

.. autofunction:: prefgraph.compute_salience_weights

.. autofunction:: prefgraph.test_attention_filter

Production Theory
-----------------

Analyze firm behavior using revealed preference methods for production.
Based on Chambers & Echenique (2016) Chapter 15.

ProductionLog
^^^^^^^^^^^^^

.. autoclass:: prefgraph.ProductionLog
   :members:
   :undoc-members:

.. autofunction:: prefgraph.test_profit_maximization

.. autofunction:: prefgraph.check_cost_minimization

.. autofunction:: prefgraph.estimate_returns_to_scale

.. autofunction:: prefgraph.compute_technical_efficiency

Dataset Loaders
---------------

.. autofunction:: prefgraph.datasets.load_demo

.. autofunction:: prefgraph.datasets.load_dunnhumby

.. autofunction:: prefgraph.datasets.load_open_ecommerce

.. autofunction:: prefgraph.datasets.load_uci_retail

.. autofunction:: prefgraph.datasets.load_retailrocket

.. autofunction:: prefgraph.datasets.load_instacart

.. autofunction:: prefgraph.datasets.load_instacart_menu_v2

.. autofunction:: prefgraph.datasets.load_yoochoose

.. autofunction:: prefgraph.datasets.load_olist

.. autofunction:: prefgraph.datasets.load_m5

.. autofunction:: prefgraph.datasets.load_rees46

.. autofunction:: prefgraph.datasets.load_online_retail_ii

.. autofunction:: prefgraph.datasets.load_hm

.. autofunction:: prefgraph.datasets.load_pakistan

.. autofunction:: prefgraph.datasets.load_favorita

.. autofunction:: prefgraph.datasets.load_taobao

.. autofunction:: prefgraph.datasets.list_datasets

Exceptions and Warnings
-----------------------

PrefGraph provides custom exceptions that inherit from ``ValueError`` for
backward compatibility.

Base Exception
^^^^^^^^^^^^^^

.. autoexception:: prefgraph.PrefGraphError
   :show-inheritance:

Data Validation Exceptions
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoexception:: prefgraph.DataValidationError
   :show-inheritance:

.. autoexception:: prefgraph.DimensionError
   :show-inheritance:

.. autoexception:: prefgraph.ValueRangeError
   :show-inheritance:

.. autoexception:: prefgraph.NaNInfError
   :show-inheritance:

Computation Exceptions
^^^^^^^^^^^^^^^^^^^^^^

.. autoexception:: prefgraph.OptimizationError
   :show-inheritance:

.. autoexception:: prefgraph.NotFittedError
   :show-inheritance:

.. autoexception:: prefgraph.InsufficientDataError
   :show-inheritance:

Warnings
^^^^^^^^

.. autoclass:: prefgraph.DataQualityWarning
   :show-inheritance:

.. autoclass:: prefgraph.NumericalInstabilityWarning
   :show-inheritance:

Troubleshooting
---------------

**Common Errors**

- ``ValueRangeError: Found non-positive costs`` --- All prices must be > 0. Check for zeros or missing data encoded as 0.
- ``DimensionError: cost_vectors shape does not match`` --- Prices and quantities must have the same shape (T x N).
- ``NaNInfError: Found NaN/Inf values`` --- Use ``nan_policy="drop"`` to automatically remove bad rows: ``rp.analyze(df, ..., nan_policy="drop")`` or ``BehaviorLog(..., nan_policy="drop")``.
- ``InsufficientDataError: Must have at least 2 observations`` --- Need T >= 2 for meaningful analysis.
- ``ImportError: pandas is required`` --- Install with ``pip install prefgraph[datasets]`` for dataset loaders.

**Tips**

- For large panels, the first call may be slow due to Numba JIT compilation. Subsequent calls are fast.
- If ``compute_integrity_score`` is slow for T > 500, the SCC-optimized path activates automatically.
- Memory usage scales as O(T^2) per user due to the T x T revealed preference matrices.
