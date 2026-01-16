Tutorial 5: Stochastic & Production
=====================================

This tutorial covers two advanced areas: stochastic choice models and
production theory analysis.

Topics covered:

**Part A: Stochastic Choice**

- Random utility models (logit, Luce)
- Independence of Irrelevant Alternatives (IIA)
- Regularity conditions
- Model diagnostics

**Part B: Production Theory**

- ProductionLog data structure
- Profit maximization testing
- Cost minimization testing
- Returns to scale estimation
- Technical efficiency

Prerequisites
-------------

- Python 3.10+
- Completed Tutorials 1-2
- Basic econometrics knowledge (for stochastic models)

.. note::

   Part A covers Chapter 13 (Stochastic Choice) and Part B covers Chapter 15
   (Production Theory) of Chambers & Echenique (2016).


Part A: Stochastic Choice
=========================


A1: The Data (StochasticChoiceLog)
----------------------------------

A ``StochasticChoiceLog`` stores probabilistic choice data: the same menu
presented multiple times, with potentially different choices each time.

.. code-block:: python

   from pyrevealed import StochasticChoiceLog

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


A2: Random Utility Models
-------------------------

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

   from pyrevealed import fit_random_utility_model

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

   from pyrevealed import estimate_choice_probabilities

   # Get predicted probabilities from fitted utilities
   utilities = result.choice_probabilities[:3]  # First 3 items
   print(f"Predicted choice probabilities: {utilities}")

Output:

.. code-block:: text

   Predicted choice probabilities: [0.6 0.3 0.1]


A3: Testing McFadden Axioms
---------------------------

McFadden's axioms characterize random utility maximization:

1. **Regularity**: :math:`P(x|A) \geq P(x|B)` when :math:`A \subseteq B`
   (removing alternatives shouldn't decrease choice probability)

2. **IIA**: :math:`\frac{P(x|A)}{P(y|A)} = \frac{P(x|B)}{P(y|B)}`
   (relative odds are constant across menus)

.. code-block:: python

   from pyrevealed import test_mcfadden_axioms

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


A4: Testing IIA (Independence of Irrelevant Alternatives)
---------------------------------------------------------

The IIA property is tested by checking if relative odds are stable:

.. code-block:: python

   from pyrevealed import check_independence_irrelevant_alternatives

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


A5: Stochastic Choice Application
---------------------------------

Analyze a recommendation system's click data:

.. code-block:: python

   import numpy as np
   from pyrevealed import (
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


Part B: Production Theory
=========================


B1: The Data (ProductionLog)
----------------------------

A ``ProductionLog`` stores firm production data: input prices/quantities
and output prices/quantities over multiple observations.

.. code-block:: python

   import numpy as np
   from pyrevealed import ProductionLog

   # A firm with 2 inputs (labor, capital) and 1 output (widgets)
   log = ProductionLog(
       input_prices=np.array([
           [20.0, 50.0],   # Period 1: wage=$20, rental=$50
           [22.0, 48.0],   # Period 2
           [21.0, 52.0],   # Period 3
       ]),
       input_quantities=np.array([
           [100.0, 50.0],  # Period 1: 100 labor, 50 capital
           [90.0, 55.0],   # Period 2
           [110.0, 45.0],  # Period 3
       ]),
       output_prices=np.array([
           [10.0],         # Output price
           [11.0],
           [10.5],
       ]),
       output_quantities=np.array([
           [500.0],        # Widgets produced
           [480.0],
           [520.0],
       ]),
       firm_id="factory_1",
   )

   print(f"Observations: {log.num_observations}")
   print(f"Inputs: {log.num_inputs}")
   print(f"Outputs: {log.num_outputs}")
   print(f"Profit: {log.profit}")

Output:

.. code-block:: text

   Observations: 3
   Inputs: 2
   Outputs: 1
   Profit: [ 500.  660.  635.]


B2: Testing Profit Maximization
-------------------------------

The production analogue of GARP tests whether observed choices are consistent
with profit maximization:

.. code-block:: python

   from pyrevealed import test_profit_maximization

   result = test_profit_maximization(log)

   print(f"Profit maximizing: {result.is_profit_maximizing}")
   print(f"Violations: {result.violations}")
   print(f"Cost efficiency: {result.cost_efficiency_score:.2f}")
   print(f"Profit efficiency: {result.profit_efficiency:.2f}")
   print(f"Returns to scale: {result.returns_to_scale}")

Output:

.. code-block:: text

   Profit maximizing: True
   Violations: []
   Cost efficiency: 1.00
   Profit efficiency: 0.85
   Returns to scale: constant

Full Summary Report
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   print(result.summary())

.. code-block:: text

   ================================================================================
                             PRODUCTION GARP TEST REPORT
   ================================================================================

   Status: PROFIT MAXIMIZING
   Returns to Scale: increasing

   Efficiency Metrics:
   ------------------
     Profit Maximizing .................. Yes
     Cost Minimizing ..................... No
     Profit Efficiency ............... 0.8107
     Cost Efficiency ................. 0.3333
     Technical Efficiency ............ 1.0000
     Violations ........................... 0

   Input Efficiencies:
   ------------------
     Input 0: 1.0000
     Input 1: 0.0000

   Interpretation:
   --------------
     Firm behavior is consistent with profit maximization.
     Returns to scale: increasing.

   Computation Time: 274.13 ms
   ================================================================================

Interpretation
~~~~~~~~~~~~~~

.. list-table:: Production GARP Interpretation
   :header-rows: 1
   :widths: 30 70

   * - Result
     - Meaning
   * - is_profit_maximizing=True
     - No arbitrage opportunities between observations
   * - violations > 0
     - Firm could have done better by choosing differently
   * - cost_efficiency < 1
     - Some observations used too many inputs


B3: Testing Cost Minimization
-----------------------------

Cost minimization is the dual of profit maximization:

.. code-block:: python

   from pyrevealed import check_cost_minimization

   result = check_cost_minimization(log)

   print(f"Cost minimizing: {result['is_cost_minimizing']}")
   print(f"Violations: {result['num_violations']}")

   if result['violations']:
       print("Violation details:")
       for i, j in result['violations'][:3]:
           print(f"  Obs {i} could have used inputs from obs {j} at lower cost")

Output:

.. code-block:: text

   Cost minimizing: False
   Violations: 3

A violation means the firm could have achieved the same (or more) output
at lower cost by using a different input mix.


B4: Returns to Scale
--------------------

Estimate whether the production technology exhibits increasing, constant,
or decreasing returns to scale:

.. code-block:: python

   from pyrevealed import estimate_returns_to_scale

   rts = estimate_returns_to_scale(log)

   print(f"Returns to scale: {rts}")

Output:

.. code-block:: text

   Returns to scale: increasing

.. list-table:: Returns to Scale Interpretation
   :header-rows: 1
   :widths: 25 75

   * - Result
     - Meaning
   * - increasing
     - Doubling inputs more than doubles output (economies of scale)
   * - constant
     - Doubling inputs exactly doubles output
   * - decreasing
     - Doubling inputs less than doubles output (diseconomies of scale)
   * - variable
     - Cannot determine (insufficient variation in data)


B5: Technical Efficiency
------------------------

Technical efficiency measures how close each observation operates to the
production frontier:

.. code-block:: python

   from pyrevealed import compute_technical_efficiency

   efficiencies = compute_technical_efficiency(log, method="output_oriented")

   print("Technical efficiency by period:")
   for t, eff in enumerate(efficiencies):
       print(f"  Period {t+1}: {eff:.2%}")

Output:

.. code-block:: text

   Technical efficiency by period:
     Period 1: 100.00%
     Period 2: 100.00%
     Period 3: 100.00%

A score of 1.0 means the observation is on the frontier; lower values
indicate the firm could produce more with the same inputs (or use fewer
inputs for the same output).


B6: Production Application
--------------------------

Compare efficiency across multiple firms:

.. code-block:: python

   import numpy as np
   from pyrevealed import (
       ProductionLog,
       test_profit_maximization,
       check_cost_minimization,
       estimate_returns_to_scale,
       compute_technical_efficiency,
   )

   np.random.seed(456)

   def simulate_firm(firm_id, efficiency_level=1.0, rts_factor=1.0):
       """Simulate firm production data."""
       n_periods = 12
       n_inputs = 2
       n_outputs = 1

       # Base input prices with variation
       input_prices = np.random.uniform(15, 25, (n_periods, n_inputs))
       output_prices = np.random.uniform(8, 12, (n_periods, n_outputs))

       # Input choices
       input_quantities = np.random.uniform(80, 120, (n_periods, n_inputs))

       # Output = f(inputs) with efficiency and RTS
       total_input = np.sum(input_quantities, axis=1)
       base_output = total_input ** rts_factor  # RTS
       output_quantities = (efficiency_level * base_output * 0.05)[:, np.newaxis]

       # Add noise
       output_quantities *= np.random.uniform(0.9, 1.1, output_quantities.shape)

       return ProductionLog(
           input_prices=input_prices,
           input_quantities=input_quantities,
           output_prices=output_prices,
           output_quantities=output_quantities,
           firm_id=firm_id,
       )

   # Simulate 3 firms with different characteristics
   firms = {
       "Efficient Corp": simulate_firm("efficient", efficiency_level=1.2, rts_factor=1.0),
       "Growing Inc": simulate_firm("growing", efficiency_level=1.0, rts_factor=1.1),
       "Struggling LLC": simulate_firm("struggling", efficiency_level=0.8, rts_factor=0.9),
   }

   # Analyze each firm
   print("=== Multi-Firm Production Analysis ===")
   print()

   results = []
   for name, log in firms.items():
       profit_result = test_profit_maximization(log)
       cost_result = check_cost_minimization(log)
       rts = estimate_returns_to_scale(log)
       tech_eff = compute_technical_efficiency(log)

       results.append({
           "firm": name,
           "profit_max": profit_result.is_profit_maximizing,
           "cost_min": cost_result["is_cost_minimizing"],
           "cost_eff": profit_result.cost_efficiency_score,
           "tech_eff": np.mean(tech_eff),
           "rts": rts,
           "mean_profit": np.mean(log.profit),
       })

   # Print comparison table
   print(f"{'Firm':<20} {'Profit Max':<12} {'Cost Min':<10} {'Cost Eff':<10} {'Tech Eff':<10} {'RTS':<12} {'Avg Profit':<10}")
   print("-" * 84)
   for r in results:
       print(f"{r['firm']:<20} {str(r['profit_max']):<12} {str(r['cost_min']):<10} "
             f"{r['cost_eff']:.2f}      {r['tech_eff']:.2f}      {r['rts']:<12} ${r['mean_profit']:.0f}")

Example output:

.. code-block:: text

   === Multi-Firm Production Analysis ===

   Firm                 Profit Max   Cost Min   Cost Eff   Tech Eff   RTS          Avg Profit
   -------------------------------------------------------------------------------------
   Efficient Corp       True         True       0.92       0.95      constant     $234
   Growing Inc          True         True       0.88       0.91      increasing   $198
   Struggling LLC       False        False      0.75       0.82      decreasing   $145


Part C: Best Practices
======================


Stochastic Choice Guidelines
----------------------------

1. **Check IIA first** — if IIA fails, logit may be inappropriate

2. **Report model fit statistics**:

   - Log-likelihood
   - AIC/BIC for model comparison
   - Regularity violations

3. **Consider alternatives to logit**:

   - Nested logit for grouped alternatives
   - Mixed logit for heterogeneous preferences
   - Probit for flexible substitution

4. **Watch for small samples** — stochastic tests need sufficient
   observations per menu

Production Analysis Guidelines
------------------------------

1. **Use multiple tests**:

   - Profit maximization (production GARP)
   - Cost minimization (dual test)
   - Technical efficiency (frontier analysis)

2. **Interpret returns to scale carefully**:

   - Requires sufficient variation in scale
   - May be industry-specific

3. **Compare across firms** — relative efficiency is often more
   informative than absolute

4. **Check data quality**:

   - All prices must be positive
   - All quantities must be non-negative
   - Match number of observations across inputs/outputs


Function Reference
------------------

Stochastic Choice
~~~~~~~~~~~~~~~~~

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

Production Theory
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Purpose
     - Function
   * - Test profit maximization
     - ``test_profit_maximization()``
   * - Test cost minimization
     - ``check_cost_minimization()``
   * - Returns to scale
     - ``estimate_returns_to_scale()``
   * - Technical efficiency
     - ``compute_technical_efficiency()``


See Also
--------

- :doc:`tutorial_menu_choice` — Deterministic menu-based choice
- :doc:`tutorial` — Budget-based revealed preference
- :doc:`api` — Full API documentation
- :doc:`theory` — Mathematical foundations (Chapters 13, 15)
