Tutorial 1
==========

This tutorial analyzes 2 years of grocery data from 2,222 households using
revealed preference methods.

Topics covered:

- Data preparation and BehaviorLog construction
- GARP consistency testing
- Power analysis (Bronars test)
- Efficiency metrics (CCEI, MPI, Houtman-Maks)
- Preference structure (separability, cross-price effects)
- Lancaster characteristics model

Prerequisites
-------------

- Python 3.10+
- Basic familiarity with NumPy and pandas
- Understanding of basic statistics (means, correlations)

.. note::

   The full code for this tutorial is available in the ``dunnhumby/`` directory
   of the PyRevealed repository.


.. _important-assumptions:

Important Assumptions
---------------------

Revealed preference tests rely on several assumptions. Real data typically
violates these to some degree, which affects interpretation.

.. list-table:: Assumptions Required for RP Testing
   :header-rows: 1
   :widths: 30 35 35

   * - Assumption
     - Grocery Data
     - Concern Level
   * - Stable preferences
     - Extended period
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
     - Monthly aggregation mitigates
     - Low

What GARP Failure Means
~~~~~~~~~~~~~~~~~~~~~~~

GARP failure means no single, stable utility function rationalizes the observed
choices. Common causes:

- Preferences changed over the observation period
- Multiple household members with different preferences
- Context-dependent choices (holidays, special occasions)
- Measurement error in prices or quantities
- Stockpiling behavior


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

Transactions are aggregated to **monthly** observations:

- Weekly data is too sparse (many zero-purchase periods)
- Monthly aligns with household budgeting cycles
- Reduces noise from random shopping timing

.. code-block:: bash

   pip install pyrevealed[viz]

   # Download the Dunnhumby dataset (requires Kaggle API)
   cd dunnhumby && ./download_data.sh


Part 2: Building BehaviorLogs
-----------------------------

A ``BehaviorLog`` stores a sequence of choice observations:

- **Prices**: Price vector at time of choice
- **Quantities**: Quantity vector chosen

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

For zero-purchase categories, impute prices using market medians:

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
be explained by a utility function. Choosing bundle A when B was affordable
reveals A ≿ B. GARP checks whether these revealed preferences form a consistent
(acyclic) ordering.

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

Typical GARP pass rates for field data range from 5-15%, reflecting assumption
violations over long time horizons (see :ref:`important-assumptions`).


Part 4: Assessing Test Power
----------------------------

The Bronars (1987) test assesses whether GARP has discriminative power by
simulating random behavior on the same budgets and measuring violation rates.

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

.. list-table:: Power Interpretation
   :header-rows: 1
   :widths: 25 75

   * - Power
     - Interpretation
   * - > 0.90
     - Random behavior almost always violates GARP
   * - 0.70 - 0.90
     - GARP results are informative
   * - 0.50 - 0.70
     - Limited discriminative power
   * - < 0.50
     - GARP cannot distinguish consistent from random behavior

With 24 observations and 10 goods, power typically exceeds 0.90.


Part 5: Measuring Efficiency (CCEI)
-----------------------------------

The **Critical Cost Efficiency Index (CCEI)**, also called the Afriat Efficiency
Index, quantifies the degree of GARP violation. A CCEI of 1.0 indicates perfect
consistency; lower values indicate larger budget adjustments needed to
rationalize behavior.

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

Lower consistency in field data reflects measurement noise, longer time horizons,
and multiple decision-makers per household.

CCEI Interpretation
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - CCEI
     - Interpretation
   * - 1.00
     - GARP satisfied
   * - 0.95+
     - Minor deviations
   * - 0.85-0.95
     - Typical for field data
   * - 0.70-0.85
     - Substantial deviations
   * - < 0.70
     - Large deviations; verify data quality


Part 6: Welfare Loss (MPI)
--------------------------

The **Money Pump Index (MPI)** measures potential welfare loss from preference
cycles (e.g., A ≻ B ≻ C ≻ A).

.. code-block:: python

   from pyrevealed import compute_confusion_metric

   mpi_scores = []
   for household_id, session_data in sessions.items():
       result = compute_confusion_metric(session_data.behavior_log)
       mpi_scores.append(result.mpi_value)

   print(f"Mean MPI: {np.mean(mpi_scores):.3f}")

Mean MPI in this data is 0.2-0.25, with strong negative correlation to CCEI
(r ≈ -0.85).

.. list-table:: MPI Interpretation
   :header-rows: 1
   :widths: 20 80

   * - MPI
     - Interpretation
   * - 0
     - No preference cycles
   * - 0.1-0.2
     - Minor cycles
   * - 0.2-0.3
     - Moderate cycles
   * - > 0.3
     - Substantial cycles

Houtman-Maks Index
~~~~~~~~~~~~~~~~~~

The Houtman-Maks Index measures the minimum fraction of observations to remove
for GARP consistency.

.. code-block:: python

   from pyrevealed import compute_minimal_outlier_fraction

   result = compute_minimal_outlier_fraction(log)
   print(f"Observations to remove: {result.fraction:.1%}")

CCEI, MPI, and Houtman-Maks capture different aspects of inconsistency.


Part 7: Preference Structure (Separability)
-------------------------------------------

Weak separability tests whether preferences over group A are independent of
consumption in group B:

.. math::

   U(x_A, x_B) = V(u_A(x_A), u_B(x_B))

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


Part 8: Cross-Price Effects
---------------------------

Test whether goods are gross substitutes (:math:`\partial x_j / \partial p_i > 0`)
or complements (:math:`\partial x_j / \partial p_i < 0`).

.. code-block:: python

   from pyrevealed import test_cross_price_effect

   for household_id, log in sample_logs.items():
       result = test_cross_price_effect(log, good_g=1, good_h=2)  # Milk vs Bread
       # result.relationship: 'substitute', 'complement', or 'independent'


Part 9: The Lancaster Model
---------------------------

The Lancaster model assumes utility derives from characteristics (e.g., nutrition)
rather than products directly: :math:`U(x) = u(Zx)` where :math:`Z` maps products
to characteristics.

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

Households with decreased CCEI in characteristics space have product-specific
preferences.


Part 10: Summary and Best Practices
-----------------------------------

Key Findings
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Metric
     - Value
   * - GARP pass rate
     - 5-15%
   * - Mean CCEI
     - 0.80-0.85
   * - Bronars power
     - >0.90
   * - Mean MPI
     - 0.2-0.25
   * - Lancaster improvement
     - +5% CCEI

Best Practices
~~~~~~~~~~~~~~

1. Compute power before interpreting GARP results
2. Report CCEI distribution, not just pass rates
3. Document aggregation choices
4. Consider assumption applicability
5. Compare to published benchmarks (e.g., CKMS 2014)

Function Reference
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Purpose
     - Function
   * - GARP consistency
     - ``validate_consistency()``
   * - CCEI / efficiency index
     - ``compute_integrity_score()``
   * - Bronars power
     - ``compute_test_power()``
   * - Money Pump Index
     - ``compute_confusion_metric()``
   * - Per-observation CCEI
     - ``compute_granular_integrity()``
   * - Houtman-Maks Index
     - ``compute_minimal_outlier_fraction()``
   * - Weak separability
     - ``test_feature_independence()``
   * - Homotheticity (HARP)
     - ``validate_proportional_scaling()``
   * - Cross-price effects
     - ``test_cross_price_effect()``
   * - Utility recovery
     - ``fit_latent_values()``


See Also
--------

- :doc:`tutorial_ecommerce` — E-commerce application
- :doc:`api` — API documentation
- :doc:`theory` — Mathematical foundations
