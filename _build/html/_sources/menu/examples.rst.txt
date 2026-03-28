Examples
========

SARP / WARP Consistency
-----------------------

Test whether a user's menu choices form a consistent ranking:

.. code-block:: python

   from prefgraph import MenuChoiceLog, validate_menu_warp, validate_menu_sarp

   log = MenuChoiceLog(
       menus=[
           frozenset({0, 1, 2}),  # chose 0 from {Pizza, Burger, Salad}
           frozenset({1, 2, 3}),  # chose 1 from {Burger, Salad, Pasta}
           frozenset({0, 3}),     # chose 0 from {Pizza, Pasta}
           frozenset({0, 1, 3}),  # chose 0 from {Pizza, Burger, Pasta}
       ],
       choices=[0, 1, 0, 0],
       item_labels=["Pizza", "Burger", "Salad", "Pasta"],
   )

   warp = validate_menu_warp(log)
   sarp = validate_menu_sarp(log)
   print(f"WARP: {warp.is_consistent}  SARP: {sarp.is_consistent}")

.. code-block:: text

   WARP: True  SARP: True

Detecting Violations
--------------------

A WARP violation: choosing differently from the same pair.

.. code-block:: python

   from prefgraph import MenuChoiceLog, validate_menu_warp

   log = MenuChoiceLog(
       menus=[frozenset({0, 1}), frozenset({0, 1})],
       choices=[0, 1],
   )
   result = validate_menu_warp(log)
   print(f"WARP: {result.is_consistent}  Violations: {result.violations}")

.. code-block:: text

   WARP: False  Violations: [(0, 1)]

Houtman-Maks Efficiency
-----------------------

How many observations to remove to restore consistency:

.. code-block:: python

   from prefgraph import MenuChoiceLog, compute_menu_efficiency

   log = MenuChoiceLog(
       menus=[frozenset({0, 1}), frozenset({0, 1}), frozenset({1, 2}), frozenset({0, 2})],
       choices=[0, 1, 1, 0],
   )
   result = compute_menu_efficiency(log)
   print(f"Efficiency: {result.efficiency_index:.2f}")
   print(f"Removed: {result.removed_observations}")

.. code-block:: text

   Efficiency: 0.75
   Removed: [1]

Ordinal Utility Recovery
------------------------

Recover a preference ranking from consistent choices:

.. code-block:: python

   from prefgraph import MenuChoiceLog, fit_menu_preferences

   log = MenuChoiceLog(
       menus=[frozenset({0, 1, 2}), frozenset({1, 2}), frozenset({0, 2})],
       choices=[0, 1, 0],
   )
   result = fit_menu_preferences(log)
   print(f"Preference order: {result.preference_order}")
   print(f"Utility values: {result.utility_values}")

.. code-block:: text

   Preference order: [0, 1, 2]
   Utility values: [3. 2. 1.]

Limited Attention (WARP-LA)
---------------------------

Test whether violations can be explained by inattention rather than irrationality:

.. code-block:: python

   from prefgraph import MenuChoiceLog, test_warp_la

   log = MenuChoiceLog(
       menus=[frozenset({0, 1, 2}), frozenset({0, 1}), frozenset({1, 2}), frozenset({0, 2})],
       choices=[0, 0, 1, 2],
   )
   result = test_warp_la(log)
   print(f"WARP(LA): {result.satisfies_warp_la}")

.. code-block:: text

   WARP(LA): True

Stochastic Choice (RUM)
------------------------

Fit a random utility model to choice frequency data:

.. code-block:: python

   from prefgraph import StochasticChoiceLog, fit_random_utility_model

   log = StochasticChoiceLog(
       menus=[frozenset({0, 1, 2}), frozenset({0, 1}), frozenset({1, 2})],
       choice_frequencies=[
           {0: 60, 1: 30, 2: 10},
           {0: 70, 1: 30},
           {1: 55, 2: 45},
       ],
       item_labels=["Apple", "Banana", "Cherry"],
   )
   result = fit_random_utility_model(log, model_type="logit")
   print(f"Log-likelihood: {result.log_likelihood:.2f}")
   print(f"Satisfies IIA: {result.satisfies_iia}")

.. code-block:: text

   Log-likelihood: -89.34
   Satisfies IIA: True

Risk Preferences
----------------

Classify risk attitudes from lottery choices:

.. code-block:: python

   import numpy as np
   from prefgraph import RiskChoiceLog, compute_risk_profile

   log = RiskChoiceLog(
       safe_values=np.array([50.0, 40.0, 30.0, 20.0, 10.0]),
       risky_outcomes=np.array([[100.0, 0.0]] * 5),
       risky_probabilities=np.array([[0.5, 0.5]] * 5),
       choices=np.array([False, False, False, True, True]),
   )
   result = compute_risk_profile(log)
   print(f"Risk category: {result.risk_category}")
   print(f"Consistency: {result.consistency_score:.0%}")

.. code-block:: text

   Risk category: risk_averse
   Consistency: 100%

Context Effects (Decoy Detection)
---------------------------------

Detect whether adding items shifts choice probabilities:

.. code-block:: python

   from prefgraph import StochasticChoiceLog, detect_decoy_effect

   log = StochasticChoiceLog(
       menus=[
           frozenset({0, 1, 2}),
           frozenset({0, 1, 2, 3}),  # decoy added
           frozenset({0, 1}),
           frozenset({1, 2}),
       ],
       choice_frequencies=[
           {0: 30, 1: 45, 2: 25},
           {0: 22, 1: 38, 2: 35, 3: 5},
           {0: 40, 1: 60},
           {1: 55, 2: 45},
       ],
       total_observations_per_menu=[100, 100, 100, 100],
       item_labels=["Basic", "Standard", "Premium", "Decoy"],
   )
   result = detect_decoy_effect(log, threshold=0.05)
   print(f"Decoy effect: {result.has_decoy_effect}")
   print(f"Magnitude: {result.magnitude:.1%}")

.. code-block:: text

   Decoy effect: True
   Magnitude: 10.0%

Ranking and Pairwise Comparison
-------------------------------

Fit a Bradley-Terry model from pairwise data:

.. code-block:: python

   from prefgraph import fit_bradley_terry

   # (winner, loser, count)
   comparisons = [
       (0, 1, 15), (1, 0, 5),
       (0, 2, 18), (2, 0, 2),
       (1, 2, 12), (2, 1, 8),
   ]
   result = fit_bradley_terry(comparisons, method="mle")
   print(f"Ranking: {result.ranking}")
   for item, score in sorted(result.scores.items(), key=lambda x: -x[1]):
       print(f"  Item {item}: {score:.3f}")

.. code-block:: text

   Ranking: [0, 1, 2]
     Item 0: 1.523
     Item 1: 0.412
     Item 2: 0.000

Batch Menu Scoring (Engine)
---------------------------

Score thousands of users via the Rust engine:

.. code-block:: python

   from prefgraph import MenuChoicePanel
   from prefgraph.engine import Engine

   panel = MenuChoicePanel.from_dataframe(
       df, user_col="user_id", menu_col="shown_items", choice_col="clicked"
   )
   engine = Engine()
   results = engine.analyze_menus(panel.to_engine_tuples())

   for r in results[:3]:
       print(r)
