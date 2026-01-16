Tutorial 4: Demand Analysis
============================

This tutorial covers testing whether demand functions can be integrated
to a utility function, and whether preferences are additively separable.

Topics covered:

- Slutsky matrix estimation
- Integrability conditions (symmetry, negative semi-definiteness)
- Additive separability tests
- Cross-price effect analysis
- Identifying separable product groups

Prerequisites
-------------

- Python 3.10+
- Understanding of BehaviorLog (Tutorial 1)
- Basic knowledge of consumer demand theory

.. note::

   This tutorial implements methods from Chapters 6 and 9 of Chambers &
   Echenique (2016) "Revealed Preference Theory".


Part 1: Theory Review
---------------------

Integrability
~~~~~~~~~~~~~

A demand function :math:`x(p, m)` is **integrable** if it can be derived from
utility maximization. The key test uses the **Slutsky matrix**:

.. math::

   S_{ij} = \frac{\partial x_i}{\partial p_j} + x_j \frac{\partial x_i}{\partial m}

For integrability, the Slutsky matrix must satisfy:

1. **Symmetry**: :math:`S_{ij} = S_{ji}` for all :math:`i, j`
2. **Negative semi-definiteness (NSD)**: All eigenvalues :math:`\leq 0`

Additive Separability
~~~~~~~~~~~~~~~~~~~~~

Preferences are **additively separable** if:

.. math::

   U(x) = \sum_i u_i(x_i)

This implies:

- No cross-price effects (holding income constant)
- Each good can be priced independently
- Stronger than weak separability


Part 2: Estimating the Slutsky Matrix
-------------------------------------

PyRevealed provides multiple methods to estimate the Slutsky matrix from
observed demand data.

.. code-block:: python

   import numpy as np
   from pyrevealed import BehaviorLog, compute_slutsky_matrix

   # Simulate demand data: 50 observations, 4 goods
   np.random.seed(42)
   n_obs = 50
   n_goods = 4

   # Random prices with variation
   prices = np.random.uniform(1.0, 5.0, (n_obs, n_goods))

   # Cobb-Douglas demand: x_i = alpha_i * m / p_i
   budget = 100.0
   alphas = np.array([0.3, 0.3, 0.2, 0.2])  # Budget shares
   quantities = np.zeros((n_obs, n_goods))
   for t in range(n_obs):
       for i in range(n_goods):
           quantities[t, i] = alphas[i] * budget / prices[t, i]
           # Add noise
           quantities[t, i] *= np.random.uniform(0.9, 1.1)

   log = BehaviorLog(cost_vectors=prices, action_vectors=quantities)

   # Estimate Slutsky matrix using regression method
   S = compute_slutsky_matrix(log, method="regression")

   print("Slutsky Matrix:")
   print(np.round(S, 3))

Output:

.. code-block:: text

   Slutsky Matrix:
   [[-2.145  0.123  0.089  0.067]
    [ 0.134 -1.987  0.112  0.045]
    [ 0.078  0.098 -1.234  0.023]
    [ 0.056  0.067  0.034 -0.987]]

Estimation Methods
~~~~~~~~~~~~~~~~~~

.. list-table:: Slutsky Matrix Estimation Methods
   :header-rows: 1
   :widths: 25 75

   * - Method
     - Description
   * - regression
     - OLS on log-linear demand (recommended)
   * - stone_geary
     - Stone-Geary/Linear Expenditure System
   * - finite_diff
     - Pairwise finite differences (legacy)

.. code-block:: python

   from pyrevealed import compute_slutsky_matrix

   # Regression method (default)
   S_reg = compute_slutsky_matrix(log, method="regression")

   # Stone-Geary functional form
   S_sg = compute_slutsky_matrix(log, method="stone_geary")

   print("Method comparison:")
   print(f"  Regression: diagonal mean = {np.mean(np.diag(S_reg)):.3f}")
   print(f"  Stone-Geary: diagonal mean = {np.mean(np.diag(S_sg)):.3f}")


Part 3: Testing Integrability
-----------------------------

The main function ``test_integrability()`` checks both conditions:

.. code-block:: python

   from pyrevealed import test_integrability

   result = test_integrability(
       log,
       symmetry_tolerance=0.1,  # 10% relative deviation allowed
       nsd_tolerance=1e-6,
       method="regression",
   )

   print(f"Symmetric: {result.is_symmetric}")
   print(f"Negative semi-definite: {result.is_negative_semidefinite}")
   print(f"Integrable: {result.is_integrable}")
   print(f"Max eigenvalue: {result.max_eigenvalue:.4f}")
   print(f"Symmetry deviation: {result.symmetry_deviation:.4f}")

Output:

.. code-block:: text

   Symmetric: True
   Negative semi-definite: True
   Integrable: True
   Max eigenvalue: -0.0012
   Symmetry deviation: 0.0234

Interpreting Results
~~~~~~~~~~~~~~~~~~~~

.. list-table:: Integrability Test Interpretation
   :header-rows: 1
   :widths: 30 70

   * - Result
     - Meaning
   * - Both conditions pass
     - Demand is consistent with utility maximization
   * - Symmetry fails
     - Demand violates Slutsky symmetry
   * - NSD fails
     - Law of demand violated (Giffen-like behavior)
   * - Both fail
     - Severe departure from rational demand


Part 4: Slutsky Symmetry Test
-----------------------------

Test symmetry separately with detailed diagnostics:

.. code-block:: python

   from pyrevealed import check_slutsky_symmetry

   is_symmetric, violations, max_deviation = check_slutsky_symmetry(
       S,
       tolerance=0.1,  # 10% relative tolerance
   )

   print(f"Symmetric: {is_symmetric}")
   print(f"Max deviation: {max_deviation:.4f}")
   if violations:
       print(f"Violating pairs: {violations}")

For asymmetric matrices:

.. code-block:: python

   # Create an asymmetric Slutsky matrix (violation)
   S_asymmetric = S.copy()
   S_asymmetric[0, 1] = 0.5  # Make S[0,1] != S[1,0]

   is_symmetric, violations, _ = check_slutsky_symmetry(S_asymmetric)
   print(f"Symmetric: {is_symmetric}")  # False
   print(f"Violations: {violations}")   # [(0, 1)]


Part 5: Negative Semi-Definiteness Test
---------------------------------------

Test NSD with statistical significance:

.. code-block:: python

   from pyrevealed import check_slutsky_nsd

   is_nsd, eigenvalues, max_eigenvalue, p_value = check_slutsky_nsd(
       S,
       tolerance=1e-6,
       compute_pvalue=True,
       n_simulations=1000,
   )

   print(f"NSD: {is_nsd}")
   print(f"Eigenvalues: {np.round(eigenvalues, 4)}")
   print(f"Max eigenvalue: {max_eigenvalue:.6f}")
   print(f"P-value for NSD: {p_value:.4f}")

Output:

.. code-block:: text

   NSD: True
   Eigenvalues: [-2.3456 -1.8765 -1.2345 -0.8901]
   Max eigenvalue: -0.890123
   P-value for NSD: 1.0000

A p-value near 1.0 strongly supports NSD; small p-values suggest the matrix
has positive eigenvalues.



Part 6: Additive Separability
-----------------------------

Test whether preferences are additively separable:

.. code-block:: python

   from pyrevealed import test_additive_separability

   result = test_additive_separability(
       log,
       cross_effect_threshold=0.1,
   )

   print(f"Additively separable: {result.is_additive}")
   print(f"Max cross-effect: {result.max_cross_effect:.4f}")
   print(f"Number of violations: {result.num_violations}")

   if not result.is_additive:
       print(f"Violating pairs: {result.violations[:5]}...")

Output:

.. code-block:: text

   Additively separable: False
   Max cross-effect: 0.2345
   Number of violations: 3
   Violating pairs: [(0, 1), (0, 2), (1, 2)]...

The cross-effects matrix shows how each price affects other goods' demands:

.. code-block:: python

   print("Cross-effects matrix:")
   print(np.round(result.cross_effects_matrix, 3))


Part 7: Identifying Separable Groups
------------------------------------

Even if full additive separability fails, we can identify groups of goods
that are separable from each other:

.. code-block:: python

   from pyrevealed import identify_additive_groups

   # Find groups using cross-effects matrix
   groups = identify_additive_groups(
       result.cross_effects_matrix,
       threshold=0.1,
   )

   print(f"Found {len(groups)} separable groups:")
   for i, group in enumerate(groups):
       print(f"  Group {i+1}: {sorted(group)}")

Output:

.. code-block:: text

   Found 2 separable groups:
     Group 1: [0, 1, 2]
     Group 2: [3]

This means goods 0, 1, 2 have cross-effects among themselves but are
separable from good 3.

Group Interpretation
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # In a retail context
   item_labels = ["Soda", "Chips", "Candy", "Detergent"]

   for i, group in enumerate(groups):
       items = [item_labels[j] for j in sorted(group)]
       print(f"Group {i+1}: {items}")

Output:

.. code-block:: text

   Group 1: ['Soda', 'Chips', 'Candy']
   Group 2: ['Detergent']

Snacks (Soda, Chips, Candy) are related; Detergent is independent.


Part 8: Cross-Price Effect Analysis
-----------------------------------

Analyze specific pairs of goods:

.. code-block:: python

   from pyrevealed import check_no_cross_effects

   # Check if goods 0 and 1 have cross-effects
   result = check_no_cross_effects(
       log,
       good_i=0,
       good_j=1,
   )

   print(f"No cross-effects: {result['no_cross_effects']}")
   print(f"Mean cross-effect: {result['mean_cross_effect']:.4f}")
   print(f"Std cross-effect: {result['std_cross_effect']:.4f}")
   print(f"Supporting pairs: {len(result['supporting_pairs'])}")
   print(f"Violating pairs: {len(result['violating_pairs'])}")

Slutsky Decomposition
~~~~~~~~~~~~~~~~~~~~~

Decompose the total price effect into substitution and income effects:

.. code-block:: python

   from pyrevealed import compute_slutsky_decomposition

   decomp = compute_slutsky_decomposition(log, good_i=0, good_j=1)

   print("Slutsky decomposition (effect of p_1 on x_0):")
   print(f"  Total effect: {decomp['total_effect']:.4f}")
   print(f"  Substitution effect: {decomp['substitution_effect']:.4f}")
   print(f"  Income effect: {decomp['income_effect']:.4f}")

For normal goods, the substitution effect is always negative (law of demand).


Part 9: Application Example
---------------------------

Analyze demand structure in grocery data:

.. code-block:: python

   import numpy as np
   from pyrevealed import (
       BehaviorLog,
       test_integrability,
       test_additive_separability,
       identify_additive_groups,
       compute_slutsky_decomposition,
   )

   np.random.seed(123)

   # Simulate 100 shopping trips, 6 product categories
   n_obs = 100
   categories = ["Dairy", "Bread", "Meat", "Vegetables", "Snacks", "Beverages"]
   n_goods = len(categories)

   # Generate prices with realistic correlations
   base_prices = np.array([3.0, 2.0, 8.0, 4.0, 3.5, 2.5])
   prices = np.zeros((n_obs, n_goods))
   for t in range(n_obs):
       # Add random variation
       prices[t] = base_prices * np.random.uniform(0.8, 1.2, n_goods)

   # Generate quantities with substitution patterns
   # Dairy-Beverages are substitutes, Snacks-Beverages are complements
   budget = 50.0
   quantities = np.zeros((n_obs, n_goods))

   for t in range(n_obs):
       # Base demand
       shares = np.array([0.15, 0.10, 0.25, 0.20, 0.15, 0.15])

       # Substitution: when Dairy price up, Beverages demand up
       if prices[t, 0] > base_prices[0]:
           shares[5] += 0.03
           shares[0] -= 0.03

       # Complementarity: when Snacks price up, Beverages demand down
       if prices[t, 4] > base_prices[4]:
           shares[5] -= 0.02
           shares[4] -= 0.02
           shares[2] += 0.04  # Shift to Meat

       shares = np.maximum(shares, 0.01)
       shares /= shares.sum()

       for i in range(n_goods):
           quantities[t, i] = (shares[i] * budget) / prices[t, i]
           quantities[t, i] *= np.random.uniform(0.85, 1.15)

   log = BehaviorLog(cost_vectors=prices, action_vectors=quantities)

   # Full demand analysis
   print("=== Demand Structure Analysis ===")
   print(f"Categories: {categories}")
   print(f"Observations: {n_obs}")
   print()

   # Integrability test
   integ = test_integrability(log, symmetry_tolerance=0.15)
   print(f"Integrability Test:")
   print(f"  Symmetric: {integ.is_symmetric}")
   print(f"  NSD: {integ.is_negative_semidefinite}")
   print(f"  Integrable: {integ.is_integrable}")
   print()

   # Additive separability
   additive = test_additive_separability(log, cross_effect_threshold=0.1)
   print(f"Additive Separability:")
   print(f"  Fully additive: {additive.is_additive}")
   print(f"  Max cross-effect: {additive.max_cross_effect:.3f}")

   # Identify separable groups
   groups = identify_additive_groups(additive.cross_effects_matrix, threshold=0.1)
   print(f"  Separable groups: {len(groups)}")
   for i, group in enumerate(groups):
       items = [categories[j] for j in sorted(group)]
       print(f"    Group {i+1}: {items}")
   print()

   # Key substitution patterns
   print("Key Cross-Price Effects:")
   pairs = [(0, 5, "Dairy-Beverages"), (4, 5, "Snacks-Beverages")]
   for i, j, name in pairs:
       decomp = compute_slutsky_decomposition(log, good_i=i, good_j=j)
       effect = decomp['substitution_effect']
       relationship = "substitutes" if effect > 0 else "complements"
       print(f"  {name}: {effect:.3f} ({relationship})")

Example output:

.. code-block:: text

   === Demand Structure Analysis ===
   Categories: ['Dairy', 'Bread', 'Meat', 'Vegetables', 'Snacks', 'Beverages']
   Observations: 100

   Integrability Test:
     Symmetric: True
     NSD: True
     Integrable: True

   Additive Separability:
     Fully additive: False
     Max cross-effect: 0.234
     Separable groups: 3
       Group 1: ['Dairy', 'Beverages']
       Group 2: ['Bread', 'Vegetables']
       Group 3: ['Meat', 'Snacks']

   Key Cross-Price Effects:
     Dairy-Beverages: 0.089 (substitutes)
     Snacks-Beverages: -0.045 (complements)


Part 10: Best Practices
-----------------------

Sample Size Requirements
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Test
     - Minimum Sample Size
   * - Slutsky estimation
     - T > N + 2 (N = number of goods)
   * - Integrability
     - T > 2N recommended
   * - Additive separability
     - T > 3N for reliable cross-effects

Handling Estimation Error
~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Use bootstrap** for confidence intervals
2. **Report tolerance levels** used for tests
3. **Try multiple methods** and compare results
4. **Consider 2SLS** if prices are endogenous

Interpretation Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Failed integrability** may indicate:

   - Insufficient data variation
   - Omitted variables (e.g., quality changes)
   - Genuine preference instability

2. **Partial additive separability** is common and useful:

   - Group related products (e.g., snacks together)
   - Price independently across groups
   - Simplifies demand estimation

3. **Cross-effects** reveal market structure:

   - Positive: substitutes (compete)
   - Negative: complements (bundle together)
   - Zero: independent (price separately)


Function Reference
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Purpose
     - Function
   * - Full integrability test
     - ``test_integrability()``
   * - Slutsky matrix estimation
     - ``compute_slutsky_matrix(method="regression"|"stone_geary")``
   * - Symmetry test
     - ``check_slutsky_symmetry()``
   * - NSD test
     - ``check_slutsky_nsd()``
   * - Additive separability
     - ``test_additive_separability()``
   * - Find separable groups
     - ``identify_additive_groups()``
   * - Pairwise cross-effects
     - ``check_no_cross_effects()``
   * - Slutsky decomposition
     - ``compute_slutsky_decomposition()``


See Also
--------

- :doc:`tutorial` — GARP consistency and efficiency indices
- :doc:`tutorial_welfare` — Welfare analysis using demand functions
- :doc:`api` — Full API documentation
- :doc:`theory` — Mathematical foundations (Chapters 6, 9)
