Tutorial 1
==========

What can 2 years of grocery data tell us about human decision-making? In this
tutorial, you'll analyze real shopping behavior from 2,222 households and
learn to properly interpret revealed preference tests.

By the end of this tutorial, you'll be able to:

- Load and prepare behavioral data for analysis
- Test whether choices are internally consistent (GARP)
- **Assess whether the test is meaningful** (power analysis)
- Measure efficiency and exploitability (CCEI, MPI)
- Test preference structure (separability)
- Transform to characteristics space (Lancaster model)

Prerequisites
-------------

- Python 3.8+
- Basic familiarity with NumPy and pandas
- Understanding of basic statistics (means, correlations)

.. note::

   The full code for this tutorial is available in the ``dunnhumby/`` directory
   of the PyRevealed repository.


Part 1: The Data
----------------

The **Dunnhumby "The Complete Journey"** dataset contains 2 years of grocery
transactions from approximately 2,500 households. We focus on 10 product
categories.

.. list-table:: Dataset Overview
   :header-rows: 1
   :widths: 40 60

   * - Metric
     - Value
   * - Households analyzed
     - 2,222
   * - Product categories
     - 10 (Soda, Milk, Bread, Cheese, Chips, Soup, Yogurt, Beef, Pizza, Lunchmeat)
   * - Time period
     - 2 years
   * - Aggregation
     - **Monthly** (24 observations per household)

Aggregation Choice
~~~~~~~~~~~~~~~~~~

We aggregate transactions to **monthly** observations. Why not weekly?

- **Weekly is too sparse**: Many households have zero purchases in most
  categories each week
- **Monthly aligns with budgeting**: Households plan grocery spending monthly
- **Reduces noise**: Smooths out random shopping timing

.. code-block:: bash

   pip install pyrevealed[viz]

   # Download the Dunnhumby dataset (requires Kaggle API)
   cd dunnhumby && ./download_data.sh

Assumption Checklist
~~~~~~~~~~~~~~~~~~~~

Before running RP tests, verify these assumptions are reasonable for your data:

.. list-table:: Assumptions Required for RP Testing
   :header-rows: 1
   :widths: 30 35 35

   * - Assumption
     - Grocery Data
     - Concern Level
   * - Stable preferences
     - 2 years is long
     - Medium
   * - Single decision-maker
     - Household, not individual
     - Medium
   * - Budget exhaustion
     - Monthly grocery budget
     - Medium
   * - Category homogeneity
     - "Milk" = all milk products
     - Medium
   * - No stockpiling
     - Monthly smooths this
     - Low

**Key point**: Real data violates all of these to some degree. This doesn't
invalidate the analysis—it affects interpretation.


Part 2: Building BehaviorLogs
-----------------------------

A ``BehaviorLog`` captures a series of choices. Each observation consists of:

- **Prices**: What each option cost at the time of choice
- **Quantities**: How much of each option the user chose

.. code-block:: python

   import numpy as np
   from pyrevealed import BehaviorLog

   # For a single household: 24 months, 10 products
   prices = np.array([
       [2.50, 3.20, 2.10, 4.50, 3.00, 1.80, 2.90, 8.50, 5.00, 6.20],  # Month 1
       [2.45, 3.30, 2.15, 4.40, 2.90, 1.85, 3.00, 8.20, 5.10, 6.00],  # Month 2
       # ... more months
   ])

   quantities = np.array([
       [2.0, 1.5, 3.0, 0.5, 1.0, 2.0, 1.0, 0.5, 0.0, 0.5],  # Month 1
       [1.5, 2.0, 2.5, 0.5, 1.5, 1.5, 1.0, 0.5, 1.0, 0.0],  # Month 2
       # ... more months
   ])

   log = BehaviorLog(
       cost_vectors=prices,
       action_vectors=quantities,
       user_id="household_123"
   )

   print(f"Observations: {log.num_records}")  # 24
   print(f"Products: {log.num_goods}")        # 10

Price Imputation
~~~~~~~~~~~~~~~~

For categories with zero purchases, we need prices. Use market medians:

.. code-block:: python

   def build_behavior_log(transactions_df, household_id, categories, price_oracle):
       """Transform transactions to BehaviorLog format."""
       hh_df = transactions_df[transactions_df['household_id'] == household_id]
       hh_df['period'] = hh_df['date'].dt.to_period('M')

       # Build quantity matrix
       quantities = hh_df.pivot_table(
           index='period',
           columns='category',
           values='quantity',
           aggfunc='sum',
           fill_value=0
       )

       # Build price matrix: user's price if purchased, oracle otherwise
       prices = pd.DataFrame(index=quantities.index, columns=categories)
       for period in quantities.index:
           for cat in categories:
               if quantities.loc[period, cat] > 0:
                   mask = (hh_df['period'] == period) & (hh_df['category'] == cat)
                   prices.loc[period, cat] = hh_df[mask]['price'].median()
               else:
                   prices.loc[period, cat] = price_oracle.loc[period, cat]

       return BehaviorLog(
           cost_vectors=prices.values,
           action_vectors=quantities.values,
           user_id=household_id
       )


Part 3: Testing Consistency (GARP)
----------------------------------

The Generalized Axiom of Revealed Preference (GARP) tests whether choices can
be explained by **any** utility function.

**The idea**: If you chose bundle A when B was affordable, you reveal A ≿ B.
GARP checks whether these revealed preferences are transitive (no cycles).

.. code-block:: python

   from pyrevealed import validate_consistency

   # Test a single household
   result = validate_consistency(log)

   if result.is_consistent:
       print("GARP satisfied — a utility function exists")
   else:
       print(f"GARP violated — {result.num_violations} contradictions found")

Testing All Households
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   consistent_count = 0
   for household_id, session_data in sessions.items():
       result = validate_consistency(session_data.behavior_log)
       if result.is_consistent:
           consistent_count += 1

   total = len(sessions)
   print(f"GARP pass rate: {consistent_count}/{total} ({100*consistent_count/total:.1f}%)")

**Expected result**: ~5-15% pass rate. This is typical for real consumption data.

Why Do Most Households Fail?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A GARP failure doesn't mean "irrational." It means choices can't be explained
by a **single, stable utility function**. Possible causes:

1. **Preference evolution**: Tastes change over 2 years
2. **Multiple decision-makers**: Different family members shop
3. **Context dependence**: Holiday shopping ≠ regular shopping
4. **Measurement error**: Price imputation is imperfect
5. **Stockpiling**: Buy extra during sales (even monthly data may miss this)

.. note::

   The consistency test tells us whether a single utility function explains the
   data—not whether people are "rational" in some deeper sense.


Part 4: Assessing Test Power
----------------------------

**Critical question**: Is passing GARP meaningful, or would random behavior
also pass?

The Bronars (1987) test answers this by simulating random behavior on the same
budgets and checking how often it violates GARP.

.. code-block:: python

   from pyrevealed import compute_test_power

   # Test power for a sample of households
   power_scores = []
   for household_id in sample_households:
       log = sessions[household_id].behavior_log
       result = compute_test_power(log, n_simulations=500)
       power_scores.append(result.power)

   print(f"Mean Bronars power: {np.mean(power_scores):.3f}")

Interpreting Power
~~~~~~~~~~~~~~~~~~

.. list-table:: Power Interpretation Guide
   :header-rows: 1
   :widths: 25 75

   * - Power
     - Interpretation
   * - > 0.90
     - Excellent — random behavior almost always fails GARP
   * - 0.70 - 0.90
     - Good — passing GARP is meaningful
   * - 0.50 - 0.70
     - Moderate — interpret with caution
   * - < 0.50
     - Weak — GARP test is uninformative

**Expected for this data**: With 24 monthly observations and 10 goods, power
should be > 0.90. Passing GARP is highly informative.

**Why power matters**: If power is low (sparse data, few observations), both
rational and random consumers pass GARP. The test tells us nothing.


Part 5: Measuring Efficiency (CCEI)
-----------------------------------

For households that fail GARP, how badly do they fail?

The **Afriat Efficiency Index (AEI)** or **Critical Cost Efficiency Index
(CCEI)** measures what fraction of behavior is consistent with utility
maximization. A score of 1.0 means perfectly consistent.

.. code-block:: python

   from pyrevealed import compute_integrity_score

   ccei_scores = []
   for household_id, session_data in sessions.items():
       result = compute_integrity_score(session_data.behavior_log)
       ccei_scores.append(result.efficiency_index)

   print(f"Mean CCEI: {np.mean(ccei_scores):.3f}")
   print(f"Median CCEI: {np.median(ccei_scores):.3f}")
   print(f"CCEI ≥ 0.95: {np.mean(np.array(ccei_scores) >= 0.95)*100:.1f}%")
   print(f"CCEI < 0.70: {np.mean(np.array(ccei_scores) < 0.70)*100:.1f}%")

Benchmark Comparison
~~~~~~~~~~~~~~~~~~~~

How do these results compare to controlled experiments?

.. list-table:: Dunnhumby vs. CKMS (2014) Lab Experiments
   :header-rows: 1
   :widths: 40 30 30

   * - Metric
     - Dunnhumby (Grocery)
     - CKMS (Lab)
   * - GARP pass rate
     - ~5-15%
     - 22.8%
   * - Mean CCEI
     - ~0.80-0.85
     - 0.881
   * - Median CCEI
     - ~0.85-0.90
     - 0.95
   * - CCEI ≥ 0.95
     - ~20-30%
     - 45.2%

**Why lower consistency?** Real grocery data has more noise, longer time
horizons, and multiple decision-makers. Lab experiments are cleaner.

CCEI Interpretation Guide
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - CCEI
     - Interpretation
   * - 1.00
     - Perfectly consistent — GARP satisfied
   * - 0.95+
     - Near-consistent — minor deviations
   * - 0.85-0.95
     - Moderate — typical for real data
   * - 0.70-0.85
     - Substantial deviations
   * - < 0.70
     - Severe — check data quality


Part 6: Exploitability (MPI)
----------------------------

The **Money Pump Index (MPI)** measures how exploitable preference cycles are.
If someone prefers A to B to C to A, a seller can "pump" money from them.

.. code-block:: python

   from pyrevealed import compute_confusion_metric

   mpi_scores = []
   for household_id, session_data in sessions.items():
       result = compute_confusion_metric(session_data.behavior_log)
       mpi_scores.append(result.mpi_value)

   print(f"Mean MPI: {np.mean(mpi_scores):.3f}")

**Result**: Mean MPI around 0.2-0.25. Strong negative correlation with CCEI
(r ≈ -0.85) — more consistent users are less exploitable.

.. list-table:: MPI Interpretation
   :header-rows: 1
   :widths: 20 80

   * - MPI
     - Interpretation
   * - 0
     - Unexploitable (consistent)
   * - 0.1-0.2
     - Low exploitability
   * - 0.2-0.3
     - Moderate
   * - > 0.3
     - High exploitability


Part 7: Preference Structure (Separability)
-------------------------------------------

**What separability tests**: Whether preferences over group A are independent
of consumption in group B. Formally, weak separability asks if:

.. math::

   U(x_A, x_B) = V(u_A(x_A), u_B(x_B))

This means the marginal rate of substitution *within* group A doesn't depend
on how much of group B you consume.

.. warning::

   **This is NOT mental accounting** in the Thaler sense. Separability is about
   utility function structure, not psychological budgeting. A person can have
   non-separable preferences but still use mental accounts, and vice versa.

.. code-block:: python

   from pyrevealed import test_feature_independence

   # Define product groups
   DAIRY = [1, 3, 6]      # Milk, Cheese, Yogurt (indices)
   PROTEIN = [7, 9]       # Beef, Lunchmeat

   separability_results = []
   for household_id, session_data in sessions.items():
       result = test_feature_independence(
           session_data.behavior_log,
           group_a=DAIRY,
           group_b=PROTEIN
       )
       separability_results.append(result.is_separable)

   print(f"Separable: {100*np.mean(separability_results):.1f}%")

**Interpretation**: Low separability rates don't mean "people don't
compartmentalize." They mean preferences don't decompose in the specific
mathematical way the test checks.


Part 8: Cross-Price Effects
---------------------------

Test whether goods are **gross substitutes** or **gross complements**:

- **Substitutes**: :math:`\partial x_j / \partial p_i > 0` (price of i up →
  demand for j up)
- **Complements**: :math:`\partial x_j / \partial p_i < 0` (price of i up →
  demand for j down)

.. code-block:: python

   from pyrevealed import test_cross_price_effect

   for household_id, log in sample_logs.items():
       result = test_cross_price_effect(log, good_g=1, good_h=2)  # Milk vs Bread
       # result.relationship: 'substitute', 'complement', or 'independent'

.. warning::

   **This is NOT co-purchase frequency.** People often confuse:

   - **Complements** (economic): Cross-price derivative < 0
   - **Co-purchased**: Bought together frequently

   Milk and bread might be co-purchased (basket composition) but still be
   economic substitutes (if bread price rises, buy more milk for cereal
   instead of toast).


Part 9: The Lancaster Model
---------------------------

The Lancaster model posits that consumers derive utility from **characteristics**
(nutrition, taste), not products directly:

.. math::

   U(x) = u(Zx)

where :math:`Z` maps products to characteristics.

When Does Lancaster Help?
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - CCEI Increases
     - CCEI Decreases
   * - Consumer optimizes over characteristics
     - Consumer has product-specific preferences
   * - Products are imperfect substitutes for characteristics
     - Brand loyalty matters
   * - Characteristics matrix is well-specified
     - Characteristics matrix is wrong

.. code-block:: python

   from pyrevealed import transform_to_characteristics

   # Nutritional characteristics: [Protein, Carbs, Fat, Sodium]
   Z = np.array([
       [0, 39, 0, 15],      # Soda
       [8, 12, 8, 120],     # Milk
       [9, 49, 3, 490],     # Bread
       [25, 1, 33, 620],    # Cheese
       # ... etc
   ])

   lancaster_log = transform_to_characteristics(log, Z)
   result = validate_consistency(lancaster_log)

Results Comparison
~~~~~~~~~~~~~~~~~~

.. list-table:: Product Space vs Characteristics Space
   :header-rows: 1
   :widths: 35 30 35

   * - Metric
     - Product Space
     - Characteristics Space
   * - Mean CCEI
     - ~0.84
     - ~0.89 (+5%)
   * - GARP pass rate
     - ~5%
     - ~8% (+60%)

**Interpretation**: Some households are better explained by a characteristics
model. This doesn't mean they're "rescued" from irrationality—they're just
better fit by a different model. Households whose CCEI *decreases* in
characteristics space have product-specific preferences (brand loyalty).


Part 10: Summary and Best Practices
-----------------------------------

Key Findings
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Category
     - Finding
   * - **GARP pass rate**
     - ~5-15% (lower than lab experiments)
   * - **Mean CCEI**
     - ~0.80-0.85 (moderate consistency)
   * - **Bronars power**
     - >0.90 (test is meaningful)
   * - **MPI**
     - ~0.2-0.25 (correlated with CCEI)
   * - **Separability**
     - Generally low (preferences don't decompose cleanly)
   * - **Lancaster**
     - +5% CCEI for some households

Best Practices
~~~~~~~~~~~~~~

1. **Always compute power** before interpreting GARP results
2. **Report CCEI distribution**, not just pass rates
3. **Be explicit about aggregation** (monthly vs weekly matters)
4. **Acknowledge assumptions** — real data violates all of them
5. **Compare to benchmarks** — CKMS (2014) provides reference
6. **Don't over-interpret** — GARP failure ≠ "irrationality"

When to Use Each Test
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Question
     - Test
   * - Can behavior be rationalized?
     - ``validate_consistency()``
   * - How close to rational?
     - ``compute_integrity_score()``
   * - Is the test meaningful?
     - ``compute_test_power()``
   * - How exploitable?
     - ``compute_confusion_metric()``
   * - Do preferences separate by group?
     - ``test_feature_independence()``
   * - Substitute or complement?
     - ``test_cross_price_effect()``


Next Steps
----------

- :doc:`tutorial_ecommerce` — Apply these methods to Amazon purchase data
- :doc:`api` — Full function documentation
- :doc:`theory` — Mathematical foundations
- ``examples/`` directory — More advanced use cases
