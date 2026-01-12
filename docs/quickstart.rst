Quickstart
==========

Basic Usage
-----------

Create a behavior log and run consistency checks:

.. code-block:: python

   from pyrevealed import BehaviorLog, validate_consistency, compute_integrity_score, compute_confusion_metric
   import numpy as np

   # Create a behavior log from observed choices
   log = BehaviorLog(
       cost_vectors=np.array([      # Prices at each observation (T x N)
           [1.0, 2.0],              # Observation 0: price of A=1, B=2
           [2.0, 1.0],              # Observation 1: price of A=2, B=1
       ]),
       action_vectors=np.array([    # Quantities chosen (T x N)
           [3.0, 1.0],              # Observation 0: bought 3 of A, 1 of B
           [1.0, 3.0],              # Observation 1: bought 1 of A, 3 of B
       ])
   )

   # Test consistency
   is_consistent = validate_consistency(log)
   print(f"Consistent: {is_consistent}")

   # Compute integrity score (0-1, higher = more consistent)
   integrity = compute_integrity_score(log)
   print(f"Integrity Score: {integrity.efficiency_index:.3f}")

   # Compute confusion metric (0-1, higher = more preference cycles)
   confusion = compute_confusion_metric(log)
   print(f"Confusion Metric: {confusion.mpi_value:.3f}")

Using BehavioralAuditor
-----------------------

For a higher-level API:

.. code-block:: python

   from pyrevealed import BehavioralAuditor, BehaviorLog
   import numpy as np

   log = BehaviorLog(
       cost_vectors=np.array([[1.0, 2.0], [2.0, 1.0]]),
       action_vectors=np.array([[3.0, 1.0], [1.0, 3.0]])
   )

   auditor = BehavioralAuditor()

   # Quick consistency check
   if auditor.validate_history(log):
       print("User behavior is consistent")

   # Full audit
   report = auditor.full_audit(log)
   print(f"Consistent: {report.is_consistent}")
   print(f"Integrity: {report.integrity_score:.2f}")
   print(f"Confusion: {report.confusion_score:.2f}")

Available Tests
---------------

Consistency Tests (Boolean)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Function
     - What it tests
     - True means
   * - ``validate_consistency(log)``
     - Transitive preference cycles
     - Utility-maximizing behavior
   * - ``validate_consistency_weak(log)``
     - Direct preference contradictions
     - No direct reversals
   * - ``validate_sarp(log)``
     - Indifference cycles
     - No mutual preferences
   * - ``validate_smooth_preferences(log)``
     - Differentiable utility
     - Can compute elasticities
   * - ``validate_strict_consistency(log)``
     - Strict cycles only (lenient)
     - Approximately consistent
   * - ``validate_price_preferences(log)``
     - Price preference consistency
     - Seeks lower prices

Efficiency Scores (0-1)
^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Function
     - What it measures
   * - ``compute_integrity_score(log)``
     - How consistent? (higher = more consistent)
   * - ``compute_confusion_metric(log)``
     - Welfare loss (lower = fewer cycles)
   * - ``compute_minimal_outlier_fraction(log)``
     - Fraction of observations to remove for consistency
   * - ``compute_granular_integrity(log)``
     - Per-observation efficiency scores
   * - ``compute_test_power(log)``
     - Statistical power of consistency test

Preference Structure (Boolean)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Function
     - What it tests
     - True means
   * - ``validate_proportional_scaling(log)``
     - Preferences scale with budget
     - Homothetic (Cobb-Douglas)
   * - ``test_income_invariance(log)``
     - Constant marginal utility of money
     - No income effects
   * - ``test_feature_independence(log, groups)``
     - Group A independent of group B
     - Separable preferences
   * - ``test_cross_price_effect(log, g, h)``
     - Substitute/complement relationship
     - Returns relationship type
