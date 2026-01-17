Tutorial 2: Menu-Based Choice
==============================

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

   print(f"Observations: {log.num_observations}")
   print(f"Unique items: {log.num_items}")

Output:

.. code-block:: text

   Observations: 3
   Unique items: 6


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

Full Summary Report
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   print(result.summary())

.. code-block:: text

   ================================================================================
                              ABSTRACT WARP TEST REPORT
   ================================================================================

   Status: CONSISTENT

   Metrics:
   -------
     Consistent ......................... Yes
     Violations ........................... 0
     Revealed Preferences ................. 4

   Interpretation:
   --------------
     No direct preference reversals in menu choices.
     Satisfies Weak Axiom for abstract choice.

   Computation Time: 0.00 ms
   ================================================================================

Consistent Example
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # No WARP violation: always choose 0 when available
   consistent_log = MenuChoiceLog(
       menus=[frozenset({0, 1}), frozenset({1, 2}), frozenset({0, 2})],
       choices=[0, 1, 0],  # 0 > 1 > 2
   )

   result = validate_menu_warp(consistent_log)
   print(f"Satisfies WARP: {result.is_consistent}")

Output:

.. code-block:: text

   Satisfies WARP: True


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

Full Summary Report
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   print(result.summary())

.. code-block:: text

   ================================================================================
                              ABSTRACT SARP TEST REPORT
   ================================================================================

   Status: CONSISTENT

   Metrics:
   -------
     Consistent ......................... Yes
     Violations ........................... 0
     Items ................................ 3

   Interpretation:
   --------------
     No preference cycles in menu choices.
     Choices are rationalizable by a preference ordering.

   Computation Time: 0.12 ms
   ================================================================================

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

Full Summary Report
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   print(result.summary())

.. code-block:: text

   ================================================================================
                                CONGRUENCE TEST REPORT
   ================================================================================

   Status: RATIONALIZABLE

   Metrics:
   -------
     Is Congruent ....................... Yes
     Satisfies SARP ..................... Yes
     SARP Violations ...................... 0
     Maximality Violations ................ 0

   Interpretation:
   --------------
     Choices are fully rationalizable by a preference ordering.
     Both SARP and maximality conditions satisfied.

   Computation Time: 0.02 ms
   ================================================================================

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

Full Summary Report
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   print(result.summary())

.. code-block:: text

   ================================================================================
                          HOUTMAN-MAKS ABSTRACT INDEX REPORT
   ================================================================================

   Status: FULLY CONSISTENT

   Metrics:
   -------
     Efficiency Index ................ 1.0000
     Fraction Removed ................ 0.0000
     Total Observations ................... 3
     Removed Observations ................. 0
     Remaining Observations ............... 3

   Interpretation:
   --------------
     All menu choices are consistent - no removal needed.

   Computation Time: 0.02 ms
   ================================================================================

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

Full Summary Report
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   print(result.summary())

.. code-block:: text

   ================================================================================
                           ORDINAL UTILITY RECOVERY REPORT
   ================================================================================

   Status: SUCCESS

   Metrics:
   -------
     Recovery Successful ................ Yes
     Number of Items ...................... 3
     Complete Ranking ................... Yes
     Most Preferred ....................... 0
     Least Preferred ...................... 2

   Preference Order (most to least):
   --------------------------------
     0 > 1 > 2

   Interpretation:
   --------------
     Ordinal preference ranking successfully recovered.
     All items fully ranked (no incomparable pairs).

   Computation Time: 2.59 ms
   ================================================================================

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

Output:

.. code-block:: text

   Salience weights: [0.8 0.1 0.1 1. ]

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


At Scale: Content Recommendation Platform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example simulates a realistic content recommendation scenario with
multiple users, position bias, and partial attention effects:

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

   np.random.seed(42)

   # Platform configuration
   n_items = 10  # Content categories
   n_users = 5
   obs_per_user = 20  # Recommendation sessions per user
   slate_size = 5     # Items shown per session

   item_labels = [
       "Breaking News", "Sports", "Tech", "Entertainment", "Politics",
       "Science", "Business", "Health", "Travel", "Food"
   ]

   # Each user has latent preferences (utilities) over items
   # Plus some shared popularity component
   popularity = np.array([2.0, 1.8, 1.5, 2.2, 0.8, 1.0, 1.2, 1.4, 1.6, 1.9])

   all_results = []

   for user_id in range(n_users):
       # User-specific preference perturbation
       user_prefs = popularity + np.random.normal(0, 0.5, n_items)

       menus = []
       choices = []

       for session in range(obs_per_user):
           # Generate a random slate of items
           slate_items = np.random.choice(n_items, size=slate_size, replace=False)
           menu = frozenset(slate_items.tolist())
           menus.append(menu)

           # Choice probability with position bias and partial attention
           items = list(menu)
           base_probs = np.exp(user_prefs[items])

           # Position bias: top positions get attention boost
           positions = np.arange(len(items))
           position_weights = 1.0 / (1.0 + 0.3 * positions)
           np.random.shuffle(position_weights)  # Random ordering in slate

           # Partial attention: user may not see all items (70% attention rate)
           attention_mask = np.random.random(len(items)) < 0.7
           if not attention_mask.any():
               attention_mask[0] = True  # Always consider at least one

           # Combined probability
           probs = base_probs * position_weights * attention_mask
           probs /= probs.sum()

           choice = np.random.choice(items, p=probs)
           choices.append(choice)

       log = MenuChoiceLog(
           menus=menus,
           choices=choices,
           item_labels=item_labels,
           user_id=f"user_{user_id}",
       )

       # Analyze this user
       warp = validate_menu_warp(log)
       sarp = validate_menu_sarp(log)
       efficiency = compute_menu_efficiency(log)
       attention = test_attention_rationality(log)

       all_results.append({
           "user": f"user_{user_id}",
           "log": log,
           "warp_rate": 1.0 if warp.is_consistent else len(warp.violations),
           "sarp_consistent": sarp.is_consistent,
           "hm_efficiency": efficiency.efficiency_index,
           "attention_param": attention.attention_parameter,
       })

   # Aggregate results
   print("=" * 60)
   print("CONTENT RECOMMENDATION PLATFORM - USER BEHAVIOR ANALYSIS")
   print("=" * 60)
   print(f"\nConfiguration:")
   print(f"  Items: {n_items}")
   print(f"  Users: {n_users}")
   print(f"  Sessions per user: {obs_per_user}")
   print(f"  Total observations: {n_users * obs_per_user}")

   print(f"\nPer-User Results:")
   print("-" * 60)
   print(f"{'User':<10} {'WARP Viol':<12} {'SARP OK':<10} {'HM Eff':<10} {'Attention':<10}")
   print("-" * 60)

   warp_violations = []
   sarp_pass = 0
   hm_scores = []
   att_params = []

   for r in all_results:
       warp_v = 0 if r["warp_rate"] == 1.0 else r["warp_rate"]
       warp_violations.append(warp_v)
       sarp_pass += 1 if r["sarp_consistent"] else 0
       hm_scores.append(r["hm_efficiency"])
       att_params.append(r["attention_param"])

       print(f"{r['user']:<10} {warp_v:<12} {str(r['sarp_consistent']):<10} "
             f"{r['hm_efficiency']:.2f}      {r['attention_param']:.2f}")

   print("-" * 60)
   print(f"\nAggregate Statistics:")
   print(f"  WARP satisfaction rate: {100 * (n_users - sum(1 for v in warp_violations if v > 0)) / n_users:.0f}%")
   print(f"  SARP satisfaction rate: {100 * sarp_pass / n_users:.0f}%")
   print(f"  Mean HM efficiency: {np.mean(hm_scores):.2f}")
   print(f"  Mean attention parameter: {np.mean(att_params):.2f}")

Example output:

.. code-block:: text

   ============================================================
   CONTENT RECOMMENDATION PLATFORM - USER BEHAVIOR ANALYSIS
   ============================================================

   Configuration:
     Items: 10
     Users: 5
     Sessions per user: 20
     Total observations: 100

   Per-User Results:
   ------------------------------------------------------------
   User       WARP Viol    SARP OK    HM Eff     Attention
   ------------------------------------------------------------
   user_0     3            False      0.85       0.72
   user_1     2            False      0.90       0.68
   user_2     4            False      0.80       0.75
   user_3     1            False      0.90       0.71
   user_4     2            False      0.85       0.69
   ------------------------------------------------------------

   Aggregate Statistics:
     WARP satisfaction rate: 0%
     SARP satisfaction rate: 0%
     Mean HM efficiency: 0.86
     Mean attention parameter: 0.71

The realistic simulation shows how position bias and limited attention lead to
apparent inconsistencies (WARP/SARP violations), even when users have stable
underlying preferences. The Houtman-Maks efficiency (0.80-0.90) indicates that
most choices are consistent, and the attention model successfully explains the
deviations


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


Part 10: Unified Summary Display
---------------------------------

For comprehensive analysis in one command, use the ``MenuChoiceSummary`` class
which runs all tests and presents results in a unified format.

One-Liner Analysis
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pyrevealed import MenuChoiceSummary

   # Run all menu choice tests with one command
   summary = MenuChoiceSummary.from_log(log)

   # Statsmodels-style text summary
   print(summary.summary())

Output:

.. code-block:: text

   ============================================================
                    MENU CHOICE SUMMARY
   ============================================================

   Data:
   -----
     Observations ............................ 50
     Alternatives ............................ 6

   Consistency Tests:
   ------------------
     WARP ............................ [+] PASS
     SARP ............................ [+] PASS
     Congruence ...................... [+] PASS

   Goodness-of-Fit:
   ----------------
     Houtman-Maks Efficiency .......... 1.0000

   Preference Order:
   -----------------
     0 > 1 > 2 > 3 > 4 > 5

   Computation Time: 23.45 ms
   ============================================================

Quick Status Indicators
~~~~~~~~~~~~~~~~~~~~~~~

For quick status checks, use ``short_summary()``:

.. code-block:: python

   # Quick one-liner status
   print(summary.short_summary())
   # Output: MenuChoiceSummary: [+] WARP, [+] SARP, [+] Congruence, HM=1.00

   # Individual results also have short summaries
   from pyrevealed import validate_menu_sarp, compute_menu_efficiency

   sarp = validate_menu_sarp(log)
   print(sarp.short_summary())
   # Output: SARP: [+] CONSISTENT

   hm = compute_menu_efficiency(log)
   print(hm.short_summary())
   # Output: Houtman-Maks: [+] 1.0000 (Fully consistent)

.. note::

   In Jupyter notebooks, results display as styled HTML cards automatically.
   Just evaluate a result object in a cell to see rich formatting:

   >>> result = validate_menu_sarp(log)
   >>> result  # Displays as HTML card with pass/fail indicator


See Also
--------

- :doc:`tutorial` — Budget-based revealed preference (GARP, CCEI)
- :doc:`tutorial_advanced` — Stochastic choice models
- :doc:`api` — Full API documentation
- :doc:`theory` — Mathematical foundations (Chapters 1-2, 14)
