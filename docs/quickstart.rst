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

Output:

.. code-block:: text

   Consistent: True
   Integrity Score: 1.000
   Confusion Metric: 0.000

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

   # Get scikit-learn style score (0-1, higher = better)
   print(f"Score: {report.score():.2f}")

   # Print detailed summary report
   print(report.summary())

Output:

.. code-block:: text

   User behavior is consistent
   Consistent: True
   Integrity: 1.00
   Confusion: 0.00
   Score: 1.00
   ================================================================================
                               BEHAVIORAL AUDIT REPORT
   ================================================================================

   Overall Status: PASS

   Core Metrics:
   ------------
     Consistent (GARP) .................. Yes
     Integrity Score (AEI) ........... 1.0000
     Confusion Score (MPI) ........... 0.0000

   Interpretation:
   --------------
     Integrity: Perfect consistency - behavior fully rationalized
     Confusion: No exploitability - choices are fully consistent

   Recommendation:
   --------------
     Behavior is highly consistent with utility maximization.
     User signal is clean and reliable.

Summary and Score Methods
-------------------------

All result objects have ``summary()`` and ``score()`` methods for easy interpretation:

.. code-block:: python

   from pyrevealed import BehaviorLog, compute_integrity_score, compute_confusion_metric
   import numpy as np

   log = BehaviorLog(
       cost_vectors=np.array([[1.0, 2.0], [2.0, 1.0]]),
       action_vectors=np.array([[3.0, 1.0], [1.0, 3.0]])
   )

   # Get integrity result
   result = compute_integrity_score(log)

   # Scikit-learn compatible score (0-1)
   print(f"Score: {result.score():.2f}")  # 1.00

   # Human-readable summary
   print(result.summary())

Output:

.. code-block:: text

   Score: 1.00
   ================================================================================
                            AFRIAT EFFICIENCY INDEX REPORT
   ================================================================================

   Status: PERFECT (AEI = 1.0)

   Metrics:
   -------
     Efficiency Index (AEI) ........... 1.0000
     Waste Fraction ................... 0.0000
     Perfectly Consistent ................ Yes

   Interpretation:
   --------------
     Perfect consistency - behavior fully rationalized by utility maximization

Utility Recovery
----------------

Recover the latent utility values that rationalize observed choices:

.. code-block:: python

   from pyrevealed import BehaviorLog, fit_latent_values
   import numpy as np

   log = BehaviorLog(
       cost_vectors=np.array([[1.0, 2.0], [2.0, 1.0]]),
       action_vectors=np.array([[3.0, 1.0], [1.0, 3.0]])
   )

   # Recover utility function via Afriat's inequalities
   result = fit_latent_values(log)

   if result.success:
       print(f"Recovery successful!")
       print(f"Utility values: {result.utility_values}")
       print(f"Lagrange multipliers: {result.lagrange_multipliers}")
   else:
       print(f"Recovery failed: {result.lp_status}")

Output:

.. code-block:: text

   Recovery successful!
   Utility values: [0.e+00 2.e-06]
   Lagrange multipliers: [1.e-06 1.e-06]

The recovered utility values and Lagrange multipliers (marginal utility of money)
satisfy Afriat's inequalities, meaning they rationalize the observed choices.

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

Menu-Based Choice Analysis
--------------------------

For discrete choices without prices (surveys, recommendations, votes),
use ``MenuChoiceLog`` and the menu choice functions:

.. code-block:: python

   from pyrevealed import MenuChoiceLog, validate_menu_sarp, fit_menu_preferences
   from pyrevealed import compute_menu_efficiency

   # Menu choices: which item was chosen from each available set
   log = MenuChoiceLog(
       menus=[
           frozenset({0, 1, 2}),  # Menu 1: Pizza, Burger, Salad available
           frozenset({1, 2}),     # Menu 2: Burger, Salad available
           frozenset({0, 2}),     # Menu 3: Pizza, Salad available
       ],
       choices=[0, 1, 0],  # Chose Pizza, Burger, Pizza
       item_labels=["Pizza", "Burger", "Salad"]
   )

   # Check SARP consistency (no preference cycles)
   result = validate_menu_sarp(log)
   print(f"Consistent: {result.is_consistent}")

   # Compute efficiency (fraction of consistent choices)
   eff = compute_menu_efficiency(log)
   print(f"Efficiency: {eff.efficiency_index:.2f}")

   # Recover preference ranking
   prefs = fit_menu_preferences(log)
   if prefs.success:
       print(f"Preference order: {prefs.preference_order}")
       # e.g., [0, 1, 2] means Pizza > Burger > Salad

Output:

.. code-block:: text

   Consistent: True
   Efficiency: 1.00
   Preference order: [0, 1, 2]

Using BehavioralAuditor for Menu Choices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``BehavioralAuditor`` class also supports menu-based analysis:

.. code-block:: python

   from pyrevealed import BehavioralAuditor, MenuChoiceLog

   log = MenuChoiceLog(
       menus=[frozenset({0, 1, 2}), frozenset({1, 2})],
       choices=[0, 1]
   )

   auditor = BehavioralAuditor()

   # Quick consistency check
   if auditor.validate_menu_history(log):
       print("Menu choices are consistent")

   # Full menu audit
   report = auditor.full_menu_audit(log)
   print(f"Rationalizable: {report.is_rationalizable}")
   print(f"Efficiency: {report.efficiency_score:.2f}")

Menu Choice Functions
^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Function
     - What it does
   * - ``validate_menu_warp(log)``
     - Check WARP (no direct preference reversals)
   * - ``validate_menu_sarp(log)``
     - Check SARP (no preference cycles of any length)
   * - ``validate_menu_consistency(log)``
     - Check full rationalizability (Congruence axiom)
   * - ``compute_menu_efficiency(log)``
     - Houtman-Maks efficiency (fraction consistent)
   * - ``fit_menu_preferences(log)``
     - Recover ordinal preference ranking

Advanced Analysis
-----------------

Stochastic Choice (Random Utility Models)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For probabilistic choice data where each menu has observed choice frequencies:

.. code-block:: python

   from pyrevealed import StochasticChoiceLog, fit_random_utility_model, test_mcfadden_axioms
   import numpy as np

   # Create stochastic choice data
   log = StochasticChoiceLog(
       menus=[
           frozenset({0, 1, 2}),  # Full menu
           frozenset({0, 1}),     # Subset
           frozenset({1, 2}),     # Another subset
       ],
       choice_frequencies=[
           {0: 50, 1: 30, 2: 20},  # Frequencies from menu 1
           {0: 70, 1: 30},         # Frequencies from menu 2
           {1: 40, 2: 60},         # Frequencies from menu 3
       ],
       item_labels=["Pizza", "Burger", "Salad"]
   )

   # Test if data satisfies RUM axioms
   axioms = test_mcfadden_axioms(log)
   print(f"RUM-consistent: {axioms.is_consistent}")
   print(f"IIA holds: {axioms.iia_holds}")

   # Fit random utility model
   result = fit_random_utility_model(log)
   if result.success:
       print(f"Utility estimates: {result.utility_values}")

Output:

.. code-block:: text

   RUM-consistent: True
   IIA holds: True
   Utility estimates: [0.5, 0.3, 0.2]

Production Theory (Firm Behavior)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Analyze firm behavior with input-output data:

.. code-block:: python

   from pyrevealed import ProductionLog, test_profit_maximization, estimate_returns_to_scale
   import numpy as np

   # Create production data
   log = ProductionLog(
       input_prices=np.array([
           [1.0, 2.0],  # Wages for labor, capital at t=0
           [1.5, 1.5],  # Wages at t=1
           [2.0, 1.0],  # Wages at t=2
       ]),
       input_quantities=np.array([
           [10, 5],   # Labor, capital used at t=0
           [8, 8],    # Inputs at t=1
           [5, 10],   # Inputs at t=2
       ]),
       output_prices=np.array([
           [10.0],  # Output price at t=0
           [12.0],  # Output price at t=1
           [11.0],  # Output price at t=2
       ]),
       output_quantities=np.array([
           [20],  # Output at t=0
           [22],  # Output at t=1
           [18],  # Output at t=2
       ])
   )

   # Test profit maximization
   result = test_profit_maximization(log)
   print(f"Profit-maximizing: {result.is_consistent}")

   # Estimate returns to scale
   rts = estimate_returns_to_scale(log)
   print(f"Returns to scale: {rts.estimate:.2f}")

Output:

.. code-block:: text

   Profit-maximizing: True
   Returns to scale: 1.05

Welfare Analysis
^^^^^^^^^^^^^^^^

Analyze welfare changes from price variations:

.. code-block:: python

   from pyrevealed import BehaviorLog, analyze_welfare_change
   from pyrevealed import compute_compensating_variation, compute_equivalent_variation
   import numpy as np

   # Before and after price change
   log = BehaviorLog(
       cost_vectors=np.array([
           [1.0, 2.0],  # Old prices
           [1.5, 2.0],  # New prices (good 1 became more expensive)
       ]),
       action_vectors=np.array([
           [4.0, 1.0],  # Old consumption
           [3.0, 1.5],  # New consumption
       ])
   )

   # Full welfare analysis
   result = analyze_welfare_change(log, base_period=0, comparison_period=1)
   print(f"Welfare decreased: {result.welfare_decreased}")
   print(f"Compensating Variation: ${result.cv:.2f}")
   print(f"Equivalent Variation: ${result.ev:.2f}")

Limited Attention
^^^^^^^^^^^^^^^^^

Test if choices can be rationalized with limited consideration:

.. code-block:: python

   from pyrevealed import MenuChoiceLog, test_attention_rationality, estimate_consideration_sets

   # Choices that might violate SARP due to limited attention
   log = MenuChoiceLog(
       menus=[
           frozenset({0, 1, 2}),
           frozenset({0, 1}),
           frozenset({1, 2}),
       ],
       choices=[0, 1, 2],  # Might form a cycle if all items considered
       item_labels=["A", "B", "C"]
   )

   # Test if rationalizable with attention filter
   result = test_attention_rationality(log)
   print(f"Attention-rationalizable: {result.is_rationalizable}")

   # Estimate consideration sets
   if result.is_rationalizable:
       sets = estimate_consideration_sets(log)
       print(f"Estimated consideration sets: {sets.consideration_sets}")

Integrability Testing
^^^^^^^^^^^^^^^^^^^^^

Test Slutsky conditions for demand integrability:

.. code-block:: python

   from pyrevealed import BehaviorLog, test_integrability, compute_slutsky_matrix
   import numpy as np

   # Demand data with price variations
   log = BehaviorLog(
       cost_vectors=np.array([
           [1.0, 2.0, 3.0],
           [1.1, 2.0, 3.0],  # Small price change in good 1
           [1.0, 2.1, 3.0],  # Small price change in good 2
           [1.0, 2.0, 3.1],  # Small price change in good 3
       ]),
       action_vectors=np.array([
           [3.0, 2.0, 1.0],
           [2.8, 2.1, 1.0],
           [3.1, 1.9, 1.0],
           [3.0, 2.0, 0.9],
       ])
   )

   # Test integrability
   result = test_integrability(log)
   print(f"Integrable: {result.is_integrable}")
   print(f"Slutsky symmetric: {result.slutsky_symmetric}")
   print(f"Slutsky NSD: {result.slutsky_nsd}")

   # Compute Slutsky matrix
   slutsky = compute_slutsky_matrix(log)
   print(f"Slutsky matrix:\n{slutsky.matrix}")

New Analysis Functions
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Function
     - What it does
   * - ``test_integrability(log)``
     - Test Slutsky symmetry and negative semi-definiteness
   * - ``analyze_welfare_change(log)``
     - Compute CV/EV for price changes
   * - ``test_additive_separability(log)``
     - Test if preferences are additively separable
   * - ``decompose_price_effects(log)``
     - Slutsky decomposition into substitution/income effects
   * - ``find_ideal_point_general(log)``
     - Find ideal point with general metrics
   * - ``fit_random_utility_model(log)``
     - Fit RUM to stochastic choice data
   * - ``test_attention_rationality(log)``
     - Test rationality with limited consideration
   * - ``test_profit_maximization(log)``
     - Test firm profit maximization
