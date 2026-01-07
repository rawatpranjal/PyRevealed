Quick Start
===========

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

   # Compute confusion metric (0-1, higher = more exploitable)
   confusion = compute_confusion_metric(log)
   print(f"Confusion Metric: {confusion.mpi_value:.3f}")

Using BehavioralAuditor
-----------------------

For a higher-level API with risk assessments:

.. code-block:: python

   from pyrevealed import BehavioralAuditor, BehaviorLog
   import numpy as np

   log = BehaviorLog(
       cost_vectors=np.array([[1.0, 2.0], [2.0, 1.0]]),
       action_vectors=np.array([[3.0, 1.0], [1.0, 3.0]])
   )

   auditor = BehavioralAuditor()

   # Quick checks
   if auditor.validate_history(log):
       print("User behavior is consistent")

   # Full audit with risk scores
   report = auditor.full_audit(log)
   print(f"Bot risk: {report.bot_risk:.2f}")
   print(f"Shared account risk: {report.shared_account_risk:.2f}")
   print(f"UX confusion risk: {report.ux_confusion_risk:.2f}")

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
     - Approximately rational
   * - ``validate_price_preferences(log)``
     - Price preference consistency
     - Seeks lower prices

Efficiency Scores (0-1)
^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Function
     - What it measures
     - Interpretation
   * - ``compute_integrity_score(log)``
     - Fraction consistent with utility max
     - 1.0 = perfect, <0.7 = bot risk
   * - ``compute_confusion_metric(log)``
     - Exploitability via preference cycles
     - 0.0 = safe, >0.3 = confused
   * - ``compute_minimal_outlier_fraction(log)``
     - Observations to remove for consistency
     - <0.1 = almost rational
   * - ``compute_granular_integrity(log)``
     - Per-observation efficiency
     - Identifies problem observations
   * - ``compute_test_power(log)``
     - Statistical significance of tests
     - >0.5 = meaningful result

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
     - Separate mental budgets
   * - ``test_cross_price_effect(log, g, h)``
     - Substitute/complement relationship
     - Returns relationship type
