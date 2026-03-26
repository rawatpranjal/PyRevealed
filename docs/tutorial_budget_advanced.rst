Tutorial 1a: Advanced Budget Analysis
=====================================

This tutorial covers advanced topics in budget-constrained revealed preference
analysis, including homothetic preferences, the Lancaster characteristics model,
and latent utility recovery.

Topics covered:

- Homothetic preferences (HARP)
- The Lancaster characteristics model
- Piecewise-linear utility recovery (Afriat's theorem)
- Marginal utility of money

Prerequisites
-------------

- Python 3.10+
- Completed :doc:`tutorial` (Basics)
- Basic understanding of linear programming

.. note::

   These methods are grounded in the Lancaster (1966) model and Afriat (1967)
   construction theorem.


Part 1: Homothetic Preferences (HARP)
--------------------------------------

**HARP (Homothetic Axiom of Revealed Preference)** tests whether demand scales
proportionally with income. Homothetic preferences mean the consumer buys the
same *proportions* of goods regardless of budget level—only the scale changes.

.. code-block:: python

   from pyrevealed import BehaviorLog, validate_proportional_scaling

   # log from Tutorial 1
   result = validate_proportional_scaling(log)

   if result.is_consistent:
       print("Homothetic preferences: demand scales proportionally")
   else:
       print(f"Non-homothetic: {len(result.violations)} scaling violations")
       print(f"Max expenditure ratio product: {result.max_cycle_product:.3f}")

Output:

.. code-block:: text

   Non-homothetic: 3 scaling violations
   Max expenditure ratio product: 1.234

When HARP Matters
~~~~~~~~~~~~~~~~~

HARP is a **stronger** requirement than GARP. Use it when:

- Aggregating demand across different income levels
- Extrapolating demand to unobserved budget levels
- Testing constant-returns-to-scale demand models
- Validating Cobb-Douglas or CES utility assumptions

.. list-table:: HARP vs GARP
   :header-rows: 1
   :widths: 30 35 35

   * - Condition
     - GARP
     - HARP
   * - Consistent ordinal preferences
     - Required
     - Required
   * - Proportional budget shares
     - Not required
     - Required
   * - Typical pass rate (field data)
     - 5-15%
     - 1-5%


Part 2: The Lancaster Model
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

   import numpy as np
   from pyrevealed import transform_to_characteristics, validate_consistency

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
   print(f"Lancaster consistent: {result.is_consistent}")

Output:

.. code-block:: text

   Lancaster consistent: True

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


Part 3: Utility Recovery
-------------------------

For GARP-consistent households, we can recover the utility function that
rationalizes their choices using Afriat's theorem.

.. code-block:: python

   from pyrevealed import fit_latent_values

   # For a GARP-consistent household
   result = fit_latent_values(log)

   if result.success:
       print(f"Recovery successful!")
       print(f"Utility values: {result.utility_values[:5]}...")  # First 5
       print(f"Marginal utility of money: {result.lagrange_multipliers[:5]}...")
   else:
       print(f"Recovery failed: {result.lp_status}")

Output:

.. code-block:: text

   Recovery successful!
   Utility values: [0.000e+00 1.234e-05 2.468e-05 3.702e-05 4.936e-05]...
   Marginal utility of money: [1.000e-06 1.000e-06 1.000e-06 1.000e-06 1.000e-06]...

Interpreting Results
~~~~~~~~~~~~~~~~~~~~

The recovered values satisfy Afriat's inequalities:

.. math::

   u_s - u_t \leq \lambda_t p_t \cdot (x_s - x_t) \quad \forall s, t

Where:

- :math:`u_t` = utility at observation :math:`t`
- :math:`\lambda_t` = marginal utility of money at :math:`t`

.. list-table:: Utility Recovery Interpretation
   :header-rows: 1
   :widths: 30 70

   * - Value
     - Meaning
   * - ``utility_values``
     - Ordinal utility indices (relative ranking matters)
   * - ``lagrange_multipliers``
     - Marginal utility of money (shadow price of budget)
   * - ``success=True``
     - A rationalizing utility function exists
   * - ``success=False``
     - GARP violated; no consistent utility exists


See Also
--------

- :doc:`tutorial` — Basic budget analysis
- :doc:`tutorial_demand_analysis` — Slutsky and integrability
- :doc:`tutorial_welfare` — CV/EV and deadweight loss
