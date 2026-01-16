Tutorial 2: Menu-Based Choice Analysis
=======================================

This tutorial covers discrete choice analysis from menus without prices.
Useful for surveys, recommendations, voting, and any domain where items
are chosen from finite sets.

Topics covered:

- MenuChoiceLog construction
- WARP and SARP consistency testing
- Full rationalizability (Congruence)
- Houtman-Maks efficiency index
- Ordinal preference recovery
- Limited attention models

Prerequisites
-------------

- Python 3.10+
- Basic familiarity with revealed preference concepts
- Completed Tutorial 1 (recommended)

.. note::

   Menu-based choice differs from budget-based analysis: there are no prices
   or budgets, only menus of available options and observed choices.


Part 1: The Data (MenuChoiceLog)
--------------------------------

A ``MenuChoiceLog`` stores a sequence of menu-choice pairs:

- **Menus**: Sets of available items at each observation
- **Choices**: The item chosen from each menu

This data structure is used for abstract choice theory (Chapters 1-2 of
Chambers & Echenique 2016).

.. code-block:: python

   from pyrevealed import MenuChoiceLog

   # A user's choices from restaurant menus
   log = MenuChoiceLog(
       menus=[
           frozenset({0, 1, 2}),  # Menu 1: Pizza, Burger, Salad
           frozenset({1, 2, 3}),  # Menu 2: Burger, Salad, Pasta
           frozenset({0, 3}),     # Menu 3: Pizza, Pasta
           frozenset({0, 1, 3}),  # Menu 4: Pizza, Burger, Pasta
       ],
       choices=[0, 1, 0, 0],  # Chose Pizza, Burger, Pizza, Pizza
       item_labels=["Pizza", "Burger", "Salad", "Pasta"],
   )

   print(f"Observations: {log.num_observations}")  # 4
   print(f"Unique items: {log.num_items}")         # 4

Output:

.. code-block:: text

   Observations: 4
   Unique items: 4

Creating from Recommendation Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For recommendation systems, use the convenience method:

.. code-block:: python

   from pyrevealed import MenuChoiceLog

   # User saw 3 recommendation slates and clicked one item each time
   shown_items = [[0, 1, 2, 3], [1, 2, 4, 5], [0, 3, 4]]
   clicked_items = [1, 4, 0]

   log = MenuChoiceLog.from_recommendations(
       shown_items=shown_items,
       clicked_items=clicked_items,
       item_labels=["News", "Sports", "Tech", "Entertainment", "Science", "Business"],
       user_id="user_123",
   )


Part 2: Testing WARP
--------------------

The **Weak Axiom of Revealed Preference (WARP)** prohibits direct preference
reversals. If x is chosen over y, then y cannot be chosen over x.

Formally: If x is chosen when y was available, then y cannot be chosen
from any menu containing x.

.. code-block:: python

   from pyrevealed import MenuChoiceLog, validate_menu_warp

   # WARP violation: choose 0 over 1, then 1 over 0
   violation_log = MenuChoiceLog(
       menus=[frozenset({0, 1}), frozenset({0, 1})],
       choices=[0, 1],  # Contradictory choices
   )

   result = validate_menu_warp(violation_log)

   print(f"Satisfies WARP: {result.is_consistent}")
   print(f"Violations: {result.violations}")

Output:

.. code-block:: text

   Satisfies WARP: False
   Violations: [(0, 1)]

Consistent Example
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # No WARP violation: always choose 0 when available
   consistent_log = MenuChoiceLog(
       menus=[frozenset({0, 1}), frozenset({1, 2}), frozenset({0, 2})],
       choices=[0, 1, 0],  # 0 > 1 > 2
   )

   result = validate_menu_warp(consistent_log)
   print(f"Satisfies WARP: {result.is_consistent}")  # True


Part 3: Testing SARP
--------------------

The **Strong Axiom of Revealed Preference (SARP)** extends WARP to prohibit
preference cycles of any length. The transitive closure of revealed
preferences must be acyclic.

.. code-block:: python

   from pyrevealed import validate_menu_sarp

   # SARP violation via 3-cycle: 0 > 1 > 2 > 0
   cycle_log = MenuChoiceLog(
       menus=[
           frozenset({0, 1}),  # Chose 0 over 1
           frozenset({1, 2}),  # Chose 1 over 2
           frozenset({0, 2}),  # Chose 2 over 0 (closes cycle)
       ],
       choices=[0, 1, 2],
   )

   result = validate_menu_sarp(cycle_log)

   print(f"Satisfies SARP: {result.is_consistent}")
   print(f"Cycles found: {result.violations}")

Output:

.. code-block:: text

   Satisfies SARP: False
   Cycles found: [(0, 1, 2)]

WARP vs SARP
~~~~~~~~~~~~

.. list-table:: Comparison of WARP and SARP
   :header-rows: 1
   :widths: 25 35 40

   * - Axiom
     - Checks For
     - Implication
   * - WARP
     - Direct reversals (2-cycles)
     - Pairwise consistency
   * - SARP
     - All cycles (any length)
     - Transitivity of preferences


Part 4: Full Rationalizability (Congruence)
-------------------------------------------

**Congruence** is the strongest condition. It requires:

1. SARP: No preference cycles
2. Maximality: The chosen item must be maximal in the menu under the
   revealed preference ordering

A dataset satisfies Congruence if and only if it can be rationalized by
a strict preference ordering (Richter's Theorem).

.. code-block:: python

   from pyrevealed import validate_menu_consistency

   # Test for full rationalizability
   log = MenuChoiceLog(
       menus=[
           frozenset({0, 1, 2}),
           frozenset({1, 2}),
           frozenset({0, 2}),
       ],
       choices=[0, 1, 0],  # Reveals 0 > 1 > 2
   )

   result = validate_menu_consistency(log)

   print(f"Rationalizable: {result.is_congruent}")
   print(f"Satisfies SARP: {result.satisfies_sarp}")
   print(f"Maximality violations: {result.maximality_violations}")

Output:

.. code-block:: text

   Rationalizable: True
   Satisfies SARP: True
   Maximality violations: []

.. list-table:: Consistency Hierarchy
   :header-rows: 1
   :widths: 30 35 35

   * - Condition
     - Strength
     - Interpretation
   * - WARP
     - Weakest
     - No direct contradictions
   * - SARP
     - Intermediate
     - No indirect contradictions
   * - Congruence
     - Strongest
     - Fully rationalizable by strict order


Part 5: Efficiency Index (Houtman-Maks)
---------------------------------------

The **Houtman-Maks efficiency index** measures the minimum fraction of
observations that must be removed to achieve SARP consistency.

.. math::

   HM = 1 - \frac{\text{observations removed}}{\text{total observations}}

A score of 1.0 means fully consistent; lower values indicate more violations.

.. code-block:: python

   from pyrevealed import compute_menu_efficiency

   # Data with one inconsistent observation
   log = MenuChoiceLog(
       menus=[
           frozenset({0, 1}),
           frozenset({0, 1}),  # Inconsistent with first
           frozenset({1, 2}),
           frozenset({0, 2}),
       ],
       choices=[0, 1, 1, 0],
   )

   result = compute_menu_efficiency(log)

   print(f"Efficiency: {result.efficiency_index:.2f}")
   print(f"Removed observations: {result.removed_observations}")
   print(f"Remaining: {result.remaining_observations}")

Output:

.. code-block:: text

   Efficiency: 0.75
   Removed observations: [1]
   Remaining: [0, 2, 3]

Interpreting Efficiency
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Houtman-Maks Interpretation
   :header-rows: 1
   :widths: 25 75

   * - Efficiency
     - Interpretation
   * - 1.00
     - Fully consistent with rational choice
   * - 0.90+
     - Minor inconsistencies
   * - 0.75-0.90
     - Moderate inconsistencies
   * - < 0.75
     - Substantial departures from rationality


Part 6: Recovering Preferences
------------------------------

For SARP-consistent data, we can recover the ordinal preference ranking
using topological sort of the revealed preference graph.

.. code-block:: python

   from pyrevealed import fit_menu_preferences

   log = MenuChoiceLog(
       menus=[
           frozenset({0, 1, 2}),
           frozenset({1, 2}),
           frozenset({0, 2}),
       ],
       choices=[0, 1, 0],
   )

   result = fit_menu_preferences(log)

   if result.success:
       print(f"Preference order: {result.preference_order}")
       print(f"Utility ranking: {result.utility_ranking}")
       print(f"Utility values: {result.utility_values}")
   else:
       print("Cannot recover preferences (SARP violated)")

Output:

.. code-block:: text

   Preference order: [0, 1, 2]
   Utility ranking: {0: 0, 1: 1, 2: 2}
   Utility values: [3. 2. 1.]

The preference order ``[0, 1, 2]`` means item 0 is most preferred, then 1, then 2.


Part 7: Limited Attention Models
--------------------------------

Sometimes apparent irrationality stems from limited attention rather than
inconsistent preferences. The **attention model** allows for consideration
sets smaller than the full menu.

A choice is attention-rational if there exists:

1. A preference ordering over items
2. A consideration set function (which items are noticed)

Such that each choice is optimal among considered items.

.. code-block:: python

   from pyrevealed import test_attention_rationality

   # Data that violates SARP but might be attention-rational
   log = MenuChoiceLog(
       menus=[
           frozenset({0, 1, 2}),
           frozenset({0, 1, 2}),
       ],
       choices=[0, 2],  # Different choices from same menu
   )

   result = test_attention_rationality(log)

   print(f"Attention-rational: {result.is_attention_rational}")
   print(f"Attention parameter: {result.attention_parameter:.2f}")
   print(f"Inattention rate: {result.inattention_rate:.2%}")
   print(f"Consideration sets: {result.consideration_sets}")

Output:

.. code-block:: text

   Attention-rational: True
   Attention parameter: 0.67
   Inattention rate: 50.00%
   Consideration sets: [{0}, {2}]

The model rationalizes the data by assuming the user only considered
item 0 in observation 1 and only item 2 in observation 2.

Estimating Consideration Sets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pyrevealed import estimate_consideration_sets, compute_salience_weights

   log = MenuChoiceLog(
       menus=[frozenset({0, 1, 2, 3})] * 10,
       choices=[0, 0, 0, 1, 0, 0, 2, 0, 0, 0],  # Mostly choose 0
   )

   # Estimate what items are typically considered
   consideration_sets = estimate_consideration_sets(log, method="greedy")

   # Compute salience weights (how often each item is noticed)
   salience = compute_salience_weights(log, consideration_sets)

   print(f"Salience weights: {salience}")

Salience weights near 1.0 mean the item is almost always considered;
lower values indicate items that are often overlooked.


Part 8: Application Example
---------------------------

Consider a recommender system where we want to understand user preferences:

.. code-block:: python

   import numpy as np
   from pyrevealed import (
       MenuChoiceLog,
       validate_menu_warp,
       validate_menu_sarp,
       compute_menu_efficiency,
       fit_menu_preferences,
       test_attention_rationality,
   )

   # Simulate user clicks on recommendation slates
   np.random.seed(42)
   n_items = 10
   n_observations = 50

   # True preference: lower index = higher preference (with noise)
   menus = []
   choices = []

   for _ in range(n_observations):
       # Random slate of 5 items
       slate = frozenset(np.random.choice(n_items, size=5, replace=False))
       menus.append(slate)

       # Choose item with probability proportional to (n_items - index)
       items = list(slate)
       probs = np.array([n_items - i for i in items], dtype=float)
       probs /= probs.sum()
       choice = np.random.choice(items, p=probs)
       choices.append(choice)

   log = MenuChoiceLog(
       menus=menus,
       choices=choices,
       item_labels=[f"Item_{i}" for i in range(n_items)],
   )

   # Full analysis
   print("=== Consistency Analysis ===")
   warp = validate_menu_warp(log)
   print(f"WARP satisfied: {warp.is_consistent}")
   print(f"WARP violations: {len(warp.violations)}")

   sarp = validate_menu_sarp(log)
   print(f"SARP satisfied: {sarp.is_consistent}")
   print(f"SARP cycles: {len(sarp.violations)}")

   efficiency = compute_menu_efficiency(log)
   print(f"Houtman-Maks efficiency: {efficiency.efficiency_index:.2%}")

   # Try to recover preferences
   prefs = fit_menu_preferences(log)
   if prefs.success:
       print(f"\nRecovered preference order: {prefs.preference_order[:5]}...")
   else:
       print("\nPreferences not fully recoverable (SARP violated)")

   # Check attention rationality
   attention = test_attention_rationality(log)
   print(f"Attention-rational: {attention.is_attention_rational}")
   print(f"Average attention: {attention.attention_parameter:.2%}")

Example output:

.. code-block:: text

   === Consistency Analysis ===
   WARP satisfied: False
   WARP violations: 12
   SARP satisfied: False
   SARP cycles: 8
   Houtman-Maks efficiency: 78.00%

   Preferences not fully recoverable (SARP violated)
   Attention-rational: True
   Average attention: 85.00%


Part 9: Best Practices
----------------------

When to Use Menu-Based Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Use Menu Analysis When
     - Use Budget Analysis When
   * - No meaningful prices exist
     - Prices affect choices
   * - Discrete choice from finite set
     - Continuous quantity choices
   * - Surveys, voting, recommendations
     - Consumer purchases
   * - Comparing items directly
     - Budget constraints matter

Tips for Analysis
~~~~~~~~~~~~~~~~~

1. **Start with WARP** — it's the weakest test. If WARP fails, SARP will too.

2. **Use efficiency index** — don't just report pass/fail. The efficiency
   score quantifies how close behavior is to rational.

3. **Consider attention models** — apparent inconsistency may reflect
   limited attention rather than irrational preferences.

4. **Check sample size** — more observations provide stronger tests but
   also more opportunities for violations.

5. **Report all metrics** — different metrics capture different aspects
   of consistency:

   - WARP/SARP: binary consistency
   - Houtman-Maks: proportion of consistent observations
   - Attention parameter: degree of limited attention


Function Reference
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Purpose
     - Function
   * - WARP test
     - ``validate_menu_warp()``
   * - SARP test
     - ``validate_menu_sarp()``
   * - Full rationalizability
     - ``validate_menu_consistency()``
   * - Houtman-Maks efficiency
     - ``compute_menu_efficiency()``
   * - Preference recovery
     - ``fit_menu_preferences()``
   * - Attention rationality
     - ``test_attention_rationality()``
   * - Consideration sets
     - ``estimate_consideration_sets()``
   * - Salience weights
     - ``compute_salience_weights()``


See Also
--------

- :doc:`tutorial` — Budget-based revealed preference (GARP, CCEI)
- :doc:`tutorial_advanced` — Stochastic choice models
- :doc:`api` — Full API documentation
- :doc:`theory` — Mathematical foundations (Chapters 1-2, 14)
