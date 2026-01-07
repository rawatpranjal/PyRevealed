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

Data Containers
---------------

BehaviorLog
^^^^^^^^^^^

.. autoclass:: pyrevealed.BehaviorLog
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

Exceptions and Warnings
-----------------------

PyRevealed provides custom exceptions that inherit from ``ValueError`` for
backward compatibility. See :doc:`troubleshooting` for detailed usage examples.

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
