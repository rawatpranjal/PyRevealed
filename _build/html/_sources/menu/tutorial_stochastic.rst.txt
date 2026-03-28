Tutorial: Stochastic Choice
===========================

This tutorial covers stochastic choice models: random utility models,
IIA testing, and regularity conditions.

Topics covered:

- Random utility models (logit, Luce)
- Independence of Irrelevant Alternatives (IIA)
- Regularity conditions
- Model diagnostics

Prerequisites
-------------

- Python 3.10+
- Completed :doc:`tutorial_menu_choice`
- Basic econometrics knowledge (for stochastic models)

.. note::

   This tutorial covers Chapter 13 (Stochastic Choice) of Chambers & Echenique (2016).


The Data (StochasticChoiceLog)
------------------------------

A ``StochasticChoiceLog`` stores probabilistic choice data: the same menu
presented multiple times, with potentially different choices each time.

.. code-block:: python

   from prefgraph import StochasticChoiceLog

   # A menu presented 100 times with observed choice frequencies
   log = StochasticChoiceLog(
       menus=[
           frozenset({0, 1, 2}),  # Menu 1: items 0, 1, 2
           frozenset({0, 1}),     # Menu 2: items 0, 1
           frozenset({1, 2}),     # Menu 3: items 1, 2
       ],
       choice_frequencies=[
           {0: 60, 1: 30, 2: 10},  # Menu 1: 60% chose 0, 30% chose 1, 10% chose 2
           {0: 70, 1: 30},         # Menu 2: 70% chose 0
           {1: 55, 2: 45},         # Menu 3: 55% chose 1
       ],
       item_labels=["Apple", "Banana", "Cherry"],
   )

   print(f"Number of menus: {log.num_menus}")
   print(f"Unique items: {log.num_items}")

   # Get choice probability
   p_apple_menu1 = log.get_choice_probability(0, 0)
   print(f"P(Apple | Menu 1) = {p_apple_menu1:.2f}")

Output:

.. code-block:: text

   Number of menus: 3
   Unique items: 3
   P(Apple | Menu 1) = 0.60

From Repeated Choices
~~~~~~~~~~~~~~~~~~~~~

Create from deterministic repeated observations:

.. code-block:: python

   # Same menu observed 10 times with different choices
   menus = [frozenset({0, 1, 2})] * 10
   choices = [0, 0, 0, 1, 0, 0, 2, 0, 1, 0]  # 6 chose 0, 2 chose 1, 2 chose 2

   log = StochasticChoiceLog.from_repeated_choices(menus, choices)
   print(log.get_choice_probabilities(0))  # {0: 0.6, 1: 0.2, 2: 0.2}


Random Utility Models
---------------------

The **random utility model (RUM)** assumes:

.. math::

   U_i = V_i + \epsilon_i

where :math:`V_i` is deterministic utility and :math:`\epsilon_i` is random.
The consumer chooses the item with highest total utility.

Different assumptions about :math:`\epsilon` lead to different models:

.. list-table:: Random Utility Models
   :header-rows: 1
   :widths: 25 40 35

   * - Model
     - Error Distribution
     - Key Property
   * - Logit
     - Gumbel (Type I Extreme Value)
     - IIA holds
   * - Probit
     - Multivariate Normal
     - Flexible substitution
   * - Luce
     - Implicit (ratio model)
     - IIA holds

Fitting a Logit Model
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from prefgraph import fit_random_utility_model

   # Fit logit model to stochastic choice data
   result = fit_random_utility_model(
       log,
       model_type="logit",
       max_iterations=1000,
   )

   print(f"Model type: {result.model_type}")
   print(f"Estimated utilities: {result.parameters}")
   print(f"Log-likelihood: {result.log_likelihood:.2f}")
   print(f"AIC: {result.aic:.2f}")
   print(f"BIC: {result.bic:.2f}")
   print(f"Satisfies IIA: {result.satisfies_iia}")

Output:

.. code-block:: text

   Model type: logit
   Estimated utilities: {'scale': 1.0, 'convergence': 1.0}
   Log-likelihood: -89.34
   AIC: 184.68
   BIC: 190.12
   Satisfies IIA: True

Full Summary Report
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   print(result.summary())

.. code-block:: text

   ================================================================================
                            STOCHASTIC CHOICE MODEL REPORT
   ================================================================================

   Status: RUM VIOLATIONS
   Model Type: logit

   Model Fit:
   ---------
     Log-Likelihood ............... -222.2062
     AIC ........................... 450.4125
     BIC ........................... 461.5238
     Satisfies IIA ....................... No
     Regularity Violations ................ 0

   Model Parameters:
   ----------------
     scale: 1.0000
     convergence: 1.0000

   Interpretation:
   --------------
     IIA violated - choice probabilities context-dependent.

   Computation Time: 1.70 ms
   ================================================================================

Predicting Choice Probabilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from prefgraph import estimate_choice_probabilities

   # Get predicted probabilities from fitted utilities
   utilities = result.choice_probabilities[:3]  # First 3 items
   print(f"Predicted choice probabilities: {utilities}")

Output:

.. code-block:: text

   Predicted choice probabilities: [0.6 0.3 0.1]


Testing McFadden Axioms
-----------------------

McFadden's axioms characterize random utility maximization:

1. **Regularity**: :math:`P(x|A) \geq P(x|B)` when :math:`A \subseteq B`
   (removing alternatives shouldn't decrease choice probability)

2. **IIA**: :math:`\frac{P(x|A)}{P(y|A)} = \frac{P(x|B)}{P(y|B)}`
   (relative odds are constant across menus)

.. code-block:: python

   from prefgraph import test_mcfadden_axioms

   axiom_results = test_mcfadden_axioms(log)

   print(f"Satisfies IIA: {axiom_results['satisfies_iia']}")
   print(f"Satisfies regularity: {axiom_results['satisfies_regularity']}")
   print(f"RUM consistent: {axiom_results['is_rum_consistent']}")

   if not axiom_results['satisfies_regularity']:
       print(f"Regularity violations: {axiom_results['regularity_violations']}")

Output:

.. code-block:: text

   Satisfies IIA: False
   Satisfies regularity: True
   RUM consistent: False


Regularity Axiom Testing
-------------------------

The **regularity axiom** is a fundamental property of random utility models.
It states that adding options to a menu should never *increase* the probability
of choosing any particular item:

.. math::

   \text{For all } A \subseteq B \text{ and } x \in A: \quad P(x|A) \geq P(x|B)

Intuition: if you choose pizza 60% of the time from {pizza, burger}, adding
salad shouldn't make you choose pizza *more* often. If it does, something
beyond simple utility maximization is at play.

.. code-block:: python

   from prefgraph import test_regularity

   result = test_regularity(stochastic_log, tolerance=0.01)

   if result.satisfies_regularity:
       print("No decoy/context effects detected")
   else:
       print(f"Violations: {len(result.violations)}")
       print(f"Violation rate: {result.violation_rate:.1%}")
       if result.worst_violation:
           v = result.worst_violation
           print(f"Worst: item {v.item}, P increased by {v.magnitude:.2%}")

Output:

.. code-block:: text

   Violations: 2
   Violation rate: 8.3%
   Worst: item 0, P increased by 5.2%

What Violations Mean
~~~~~~~~~~~~~~~~~~~~

Regularity violations indicate that choice probabilities are context-dependent:

.. list-table:: Causes of Regularity Violations
   :header-rows: 1
   :widths: 30 70

   * - Cause
     - Description
   * - **Decoy effect**
     - An inferior option makes a similar option look better
   * - **Attraction effect**
     - Adding a dominated alternative boosts the dominant one
   * - **Compromise effect**
     - Middle options gain share when extremes are added
   * - **Consideration sets**
     - Larger menus change which items are noticed

Detailed Violation Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The result includes detailed information about each violation:

.. code-block:: python

   for v in result.violations[:3]:
       print(f"Item {v.item}:")
       print(f"  Smaller menu (idx {v.subset_menu_idx}): P = {v.prob_in_subset:.2%}")
       print(f"  Larger menu (idx {v.superset_menu_idx}): P = {v.prob_in_superset:.2%}")
       print(f"  Increase: {v.magnitude:.2%}")

Output:

.. code-block:: text

   Item 0:
     Smaller menu (idx 1): P = 55.0%
     Larger menu (idx 0): P = 60.2%
     Increase: 5.2%

   Item 1:
     Smaller menu (idx 2): P = 48.0%
     Larger menu (idx 0): P = 50.5%
     Increase: 2.5%

Full Summary Report
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   print(result.summary())

.. code-block:: text

   ================================================================================
                              REGULARITY TEST REPORT
   ================================================================================

   Status: VIOLATIONS DETECTED

   Metrics:
   -------
     Satisfies Regularity ................. No
     Number of Violations .................. 2
     Testable Pairs ........................ 24
     Violation Rate ..................... 8.3%

   Worst Violation:
   ---------------
     Item ................................... 0
     P(smaller menu) ................... 55.0%
     P(larger menu) .................... 60.2%
     Magnitude .......................... 5.2%

   Interpretation:
   --------------
     Regularity violations suggest context-dependent choice.
     This could indicate decoy effects, attraction effects, or
     consideration set changes. Standard logit may be inappropriate.

   Computation Time: 0.85 ms
   ================================================================================

When to Use This Test
~~~~~~~~~~~~~~~~~~~~~

Use regularity testing when:

1. **Validating RUM assumptions** — Regularity is necessary for random utility
2. **Detecting context effects** — Decoy effects violate regularity
3. **A/B testing analysis** — Adding options shouldn't boost existing ones
4. **Menu design** — Understanding how options affect each other


Testing IIA (Independence of Irrelevant Alternatives)
-----------------------------------------------------

The IIA property is tested by checking if relative odds are stable:

.. code-block:: python

   from prefgraph import check_independence_irrelevant_alternatives

   satisfies_iia = check_independence_irrelevant_alternatives(
       log,
       tolerance=0.1,  # Allow 10% coefficient of variation
   )

   print(f"IIA holds: {satisfies_iia}")

Output:

.. code-block:: text

   IIA holds: False

The Red Bus / Blue Bus Problem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A famous example where IIA fails:

.. code-block:: python

   # Without blue bus: P(car) = P(red bus) = 0.5
   # With blue bus: if IIA holds, P(car) = P(red bus) = P(blue bus) = 0.33
   # But blue bus should steal mainly from red bus, not car!

   log_without_blue = StochasticChoiceLog(
       menus=[frozenset({0, 1})],  # car, red bus
       choice_frequencies=[{0: 50, 1: 50}],
       item_labels=["Car", "Red Bus"],
   )

   log_with_blue = StochasticChoiceLog(
       menus=[frozenset({0, 1, 2})],  # car, red bus, blue bus
       choice_frequencies=[{0: 50, 1: 25, 2: 25}],  # Realistic: blue steals from red
       item_labels=["Car", "Red Bus", "Blue Bus"],
   )

   # This violates IIA: P(car)/P(red) changed from 1.0 to 2.0
   print("Without blue bus: P(car)/P(red bus) = 1.0")
   print("With blue bus: P(car)/P(red bus) = 2.0")
   print("IIA violated!")

Output:

.. code-block:: text

   Without blue bus: P(car)/P(red bus) = 1.0
   With blue bus: P(car)/P(red bus) = 2.0
   IIA violated!


Application: A/B Testing for Product Features
----------------------------------------------

Analyze a recommendation system's click data:

.. code-block:: python

   import numpy as np
   from prefgraph import (
       StochasticChoiceLog,
       fit_random_utility_model,
       test_mcfadden_axioms,
   )

   np.random.seed(42)

   # Simulate click data: 5 items, various recommendation slates
   n_items = 5
   item_labels = ["News", "Sports", "Tech", "Entertainment", "Science"]

   # True utilities (unknown to us, we'll try to recover)
   true_utilities = np.array([2.0, 1.5, 1.0, 1.8, 0.8])

   # Generate stochastic choice data
   menus = []
   frequencies = []

   # Create different recommendation slates
   for slate_size in [2, 3, 4]:
       for _ in range(5):
           # Random slate
           slate_items = np.random.choice(n_items, size=slate_size, replace=False)
           menu = frozenset(slate_items.tolist())

           # Simulate 100 users seeing this slate
           n_users = 100
           freq = {item: 0 for item in menu}

           for _ in range(n_users):
               # Logit choice probabilities
               u_slate = true_utilities[list(menu)]
               probs = np.exp(u_slate) / np.sum(np.exp(u_slate))
               choice = np.random.choice(list(menu), p=probs)
               freq[choice] += 1

           menus.append(menu)
           frequencies.append(freq)

   log = StochasticChoiceLog(
       menus=menus,
       choice_frequencies=frequencies,
       item_labels=item_labels,
   )

   # Analyze the data
   print("=== Stochastic Choice Analysis ===")
   print(f"Menus observed: {log.num_menus}")
   print(f"Items: {item_labels}")
   print()

   # Test McFadden axioms
   axioms = test_mcfadden_axioms(log)
   print(f"IIA satisfied: {axioms['satisfies_iia']}")
   print(f"Regularity satisfied: {axioms['satisfies_regularity']}")
   print(f"RUM consistent: {axioms['is_rum_consistent']}")
   print()

   # Fit logit model
   result = fit_random_utility_model(log, model_type="logit")
   print(f"Log-likelihood: {result.log_likelihood:.2f}")
   print(f"AIC: {result.aic:.2f}")
   print()

   # Compare estimated vs true utilities
   print("Utility Comparison:")
   print("Item            True    Estimated")
   for i, label in enumerate(item_labels):
       estimated = result.choice_probabilities[i] if i < len(result.choice_probabilities) else 0
       print(f"{label:15} {true_utilities[i]:.2f}    {estimated:.2f}")


At Scale: A/B Testing for Product Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example simulates realistic A/B test data for product feature preferences,
including context effects and IIA violations from similar alternatives:

.. code-block:: python

   import numpy as np
   from prefgraph import (
       StochasticChoiceLog,
       fit_random_utility_model,
       test_mcfadden_axioms,
       check_independence_irrelevant_alternatives,
   )

   np.random.seed(42)

   # A/B test configuration: testing 6 product variants
   n_items = 6
   n_menus = 15   # Different test conditions
   n_obs_per_menu = 500  # Users per condition (total: 7,500 observations)

   item_labels = [
       "Basic",      # Entry-level product
       "Premium",    # High-end product
       "Premium+",   # Premium variant (similar to Premium - IIA violation)
       "Budget",     # Low-cost option
       "Pro",        # Professional tier
       "Enterprise", # Business tier
   ]

   # True underlying utilities (latent)
   # Premium and Premium+ are similar, causing IIA violations
   base_utilities = np.array([1.5, 2.5, 2.4, 1.0, 2.2, 2.0])

   # Similarity matrix: similar products cannibalize each other
   # Premium and Premium+ are very similar (high substitutability)
   similarity = np.zeros((n_items, n_items))
   similarity[1, 2] = similarity[2, 1] = 0.8  # Premium ~ Premium+
   similarity[4, 5] = similarity[5, 4] = 0.5  # Pro ~ Enterprise

   menus = []
   frequencies = []
   menu_descriptions = []

   # Design various test conditions
   test_conditions = [
       # Binary comparisons
       ([0, 1], "Basic vs Premium"),
       ([0, 3], "Basic vs Budget"),
       ([1, 4], "Premium vs Pro"),
       # Triplets
       ([0, 1, 3], "Entry-level options"),
       ([1, 4, 5], "Premium tiers"),
       # Adding similar alternative (IIA test)
       ([0, 1], "Basic vs Premium (control)"),
       ([0, 1, 2], "Basic vs Premium vs Premium+"),  # IIA violation expected
       # Full product line
       ([0, 1, 3, 4], "Main product line"),
       ([0, 1, 2, 3, 4, 5], "All products"),
       # Targeted tests
       ([1, 2], "Premium variants only"),
       ([4, 5], "Business tiers only"),
       ([0, 1, 4], "Consumer vs Pro"),
       ([3, 0, 1], "Budget to Premium path"),
       ([1, 2, 4, 5], "Premium + Business"),
       ([0, 3, 4], "Budget-conscious options"),
   ]

   for menu_items, description in test_conditions:
       menu = frozenset(menu_items)
       menus.append(menu)
       menu_descriptions.append(description)

       # Calculate choice probabilities with context effects
       items = list(menu)
       utilities = base_utilities[items].copy()

       # Context effect: similar products split demand
       for i, item_i in enumerate(items):
           for j, item_j in enumerate(items):
               if i != j and similarity[item_i, item_j] > 0:
                   # Reduce utility when similar alternative present
                   utilities[i] -= 0.3 * similarity[item_i, item_j]

       # Logit choice with temperature (lower = more deterministic)
       temperature = 0.8
       exp_u = np.exp(utilities / temperature)
       probs = exp_u / exp_u.sum()

       # Simulate user choices
       freq = {item: 0 for item in items}
       choices = np.random.choice(items, size=n_obs_per_menu, p=probs)
       for c in choices:
           freq[c] += 1

       frequencies.append(freq)

   log = StochasticChoiceLog(
       menus=menus,
       choice_frequencies=frequencies,
       item_labels=item_labels,
   )

   # Analysis
   print("=" * 70)
   print("A/B TESTING PRODUCT FEATURES - STOCHASTIC CHOICE ANALYSIS")
   print("=" * 70)
   print(f"\nTest Configuration:")
   print(f"  Product variants: {n_items}")
   print(f"  Test conditions: {n_menus}")
   print(f"  Users per condition: {n_obs_per_menu}")
   print(f"  Total observations: {n_menus * n_obs_per_menu:,}")

   # McFadden axioms
   axioms = test_mcfadden_axioms(log)
   print(f"\nMcFadden Axiom Tests:")
   print(f"  IIA satisfied: {axioms['satisfies_iia']}")
   print(f"  Regularity satisfied: {axioms['satisfies_regularity']}")
   print(f"  RUM consistent: {axioms['is_rum_consistent']}")

   # IIA analysis
   iia_holds = check_independence_irrelevant_alternatives(log, tolerance=0.15)
   print(f"  IIA (15% tolerance): {iia_holds}")

   # Fit model
   result = fit_random_utility_model(log, model_type="logit")
   print(f"\nLogit Model Fit:")
   print(f"  Log-likelihood: {result.log_likelihood:.2f}")
   print(f"  AIC: {result.aic:.2f}")
   print(f"  BIC: {result.bic:.2f}")

   # Per-condition analysis
   print(f"\nPer-Condition Results:")
   print("-" * 70)
   print(f"{'Condition':<35} {'Menu':<20} {'Winner':<12} {'Win %':<8}")
   print("-" * 70)

   for i, (menu, freq, desc) in enumerate(zip(menus, frequencies, menu_descriptions)):
       items = list(menu)
       total = sum(freq.values())
       winner = max(freq, key=freq.get)
       win_pct = 100 * freq[winner] / total
       menu_str = ",".join(item_labels[it][:6] for it in sorted(items))
       print(f"{desc:<35} {menu_str:<20} {item_labels[winner]:<12} {win_pct:.1f}%")

   # IIA violation demonstration
   print(f"\n" + "=" * 70)
   print("IIA VIOLATION ANALYSIS (Premium vs Premium+ Effect)")
   print("=" * 70)

   # Find the control (Basic vs Premium) and test (Basic vs Premium vs Premium+)
   control_idx = 5  # Basic vs Premium (control)
   test_idx = 6     # Basic vs Premium vs Premium+

   control_freq = frequencies[control_idx]
   test_freq = frequencies[test_idx]

   # In control: ratio of Premium to Basic
   control_total = sum(control_freq.values())
   p_premium_control = control_freq.get(1, 0) / control_total
   p_basic_control = control_freq.get(0, 0) / control_total
   ratio_control = p_premium_control / p_basic_control if p_basic_control > 0 else float('inf')

   # In test: ratio of Premium to Basic (after adding Premium+)
   test_total = sum(test_freq.values())
   p_premium_test = test_freq.get(1, 0) / test_total
   p_basic_test = test_freq.get(0, 0) / test_total
   ratio_test = p_premium_test / p_basic_test if p_basic_test > 0 else float('inf')

   print(f"\nControl condition (Basic vs Premium):")
   print(f"  P(Premium) = {p_premium_control:.3f}")
   print(f"  P(Basic) = {p_basic_control:.3f}")
   print(f"  Odds ratio Premium/Basic = {ratio_control:.2f}")

   print(f"\nTest condition (Basic vs Premium vs Premium+):")
   print(f"  P(Premium) = {p_premium_test:.3f}")
   print(f"  P(Basic) = {p_basic_test:.3f}")
   print(f"  P(Premium+) = {test_freq.get(2, 0) / test_total:.3f}")
   print(f"  Odds ratio Premium/Basic = {ratio_test:.2f}")

   print(f"\nIIA Test Result:")
   if abs(ratio_control - ratio_test) > 0.2:
       print(f"  IIA VIOLATED: Adding Premium+ changed Premium/Basic odds")
       print(f"  Ratio change: {ratio_control:.2f} -> {ratio_test:.2f}")
       print(f"  Premium+ cannibalized Premium more than Basic (similarity effect)")
   else:
       print(f"  IIA holds: Premium/Basic odds stable")

   # Business insights
   print(f"\n" + "=" * 70)
   print("BUSINESS INSIGHTS")
   print("=" * 70)

   # Aggregate choice shares
   total_choices = {i: 0 for i in range(n_items)}
   total_appearances = {i: 0 for i in range(n_items)}

   for menu, freq in zip(menus, frequencies):
       for item in menu:
           total_appearances[item] += sum(freq.values())
           total_choices[item] += freq.get(item, 0)

   print(f"\nOverall Product Performance:")
   print(f"{'Product':<15} {'Choice Share':<15} {'Win Rate':<12}")
   print("-" * 45)

   for i in range(n_items):
       if total_appearances[i] > 0:
           share = 100 * total_choices[i] / sum(total_choices.values())
           win_rate = 100 * total_choices[i] / total_appearances[i]
           print(f"{item_labels[i]:<15} {share:>8.1f}%       {win_rate:>6.1f}%")

Example output:

.. code-block:: text

   ======================================================================
   A/B TESTING PRODUCT FEATURES - STOCHASTIC CHOICE ANALYSIS
   ======================================================================

   Test Configuration:
     Product variants: 6
     Test conditions: 15
     Users per condition: 500
     Total observations: 7,500

   McFadden Axiom Tests:
     IIA satisfied: False
     Regularity satisfied: True
     RUM consistent: False
     IIA (15% tolerance): False

   Logit Model Fit:
     Log-likelihood: -4523.45
     AIC: 9058.90
     BIC: 9082.34

This A/B test analysis reveals IIA violations when similar products are added:
Premium+ cannibalized Premium sales disproportionately, demonstrating that
simple logit models may mispredict market shares when similar alternatives
exist. Nested logit or mixed logit would better capture this pattern.


Notes
-----

Stochastic Choice
~~~~~~~~~~~~~~~~~

1. **IIA test** — if IIA fails, logit may be inappropriate

2. **Model fit statistics**:

   - Log-likelihood
   - AIC/BIC for model comparison
   - Regularity violations

3. **Alternatives to logit**:

   - Nested logit for grouped alternatives
   - Mixed logit for heterogeneous preferences
   - Probit for flexible substitution

4. **Sample size** — stochastic tests need sufficient
   observations per menu


Function Reference
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Purpose
     - Function
   * - Fit RUM
     - ``fit_random_utility_model()``
   * - Test McFadden axioms
     - ``test_mcfadden_axioms()``
   * - Test IIA
     - ``check_independence_irrelevant_alternatives()``
   * - Predict probabilities
     - ``estimate_choice_probabilities()``
   * - Fit Luce model
     - ``fit_luce_model()``


See Also
--------

- :doc:`tutorial_menu_choice` — Deterministic menu-based choice
- :doc:`/budget/tutorial` — Budget-based revealed preference
- :doc:`/references` — Full API documentation
- :doc:`theory_stochastic` — Mathematical foundations (Chapter 13)
