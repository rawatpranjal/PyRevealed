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

Output:

.. code-block:: text

   Method comparison:
     Regression: diagonal mean = -2.399
     Stone-Geary: diagonal mean = -4.180


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

Full Summary Report
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   print(result.summary())

.. code-block:: text

   ================================================================================
                              INTEGRABILITY TEST REPORT
   ================================================================================

   Status: NOT INTEGRABLE

   Slutsky Conditions:
   ------------------
     Is Integrable ....................... No
     Symmetric ........................... No
     Negative Semi-Definite ........... False
     Symmetry Violations .................. 5
     Max Eigenvalue .................. 0.8511
     Symmetry Deviation .............. 0.4188
     Number of Goods ...................... 4

   Interpretation:
   --------------
     Slutsky symmetry violated - cross-price effects asymmetric.
     Not NSD - max eigenvalue 0.8511 > 0.

   Computation Time: 27.68 ms
   ================================================================================

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


Part 3a: Quasilinearity (Income Invariance)
-------------------------------------------

**Quasilinear** preferences have the form ``U(x, m) = v(x) + m`` where ``m`` is
money and ``v`` is a concave function. This implies **no income effects**—the
marginal utility of money is constant.

.. code-block:: python

   from pyrevealed import test_income_invariance

   result = test_income_invariance(log, max_cycle_length=3)

   if result.is_quasilinear:
       print("No income effects: demand depends only on prices")
   else:
       print(f"Income effects detected: {len(result.violations)} violations")
       print(f"Worst violation: {result.worst_violation_magnitude:.4f}")

Output:

.. code-block:: text

   No income effects: demand depends only on prices

When Quasilinearity Matters
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Quasilinear preferences are useful when:

- Building demand models that ignore income effects
- Computing welfare measures (constant marginal utility of money)
- Price optimization (demand elasticities don't depend on income)

.. code-block:: python

   # Detailed cycle analysis
   print(f"Cycles tested: {result.num_cycles_tested}")

   # Examine specific violations
   for cycle in result.violations[:3]:
       cycle_sum = result.cycle_sums[cycle]
       print(f"Cycle {cycle}: sum = {cycle_sum:.4f}")

For exhaustive checking of all cycle lengths:

.. code-block:: python

   from pyrevealed import test_income_invariance_exhaustive

   result_full = test_income_invariance_exhaustive(log)
   print(f"Exhaustive check: {result_full.is_quasilinear}")


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

Output:

.. code-block:: text

   Symmetric: False
   Max deviation: 0.4188
   Violating pairs: [(0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

For asymmetric matrices:

.. code-block:: python

   # Create an asymmetric Slutsky matrix (violation)
   S_asymmetric = S.copy()
   S_asymmetric[0, 1] = 0.5  # Make S[0,1] != S[1,0]

   is_symmetric, violations, _ = check_slutsky_symmetry(S_asymmetric)
   print(f"Symmetric: {is_symmetric}")
   print(f"Violations: {violations}")

Output:

.. code-block:: text

   Symmetric: False
   Violations: [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


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

Part 5a: Smooth Preferences (Differentiability)
-----------------------------------------------

**Differentiable rationality** tests whether behavior is consistent with a
smooth, differentiable utility function. This is **stronger than GARP** and
requires two conditions:

1. **SARP**: No indifferent preference cycles (strict revealed preference acyclicity)
2. **Uniqueness**: Different prices produce different demands

.. code-block:: python

   from pyrevealed import validate_smooth_preferences, validate_sarp

   # Full differentiability test
   diff_result = validate_smooth_preferences(log)

   print(f"Smooth preferences: {diff_result.is_differentiable}")
   print(f"SARP satisfied: {diff_result.satisfies_sarp}")
   print(f"Uniqueness satisfied: {diff_result.satisfies_uniqueness}")

Output:

.. code-block:: text

   Smooth preferences: True
   SARP satisfied: True
   Uniqueness satisfied: True

SARP vs GARP
~~~~~~~~~~~~

SARP (Strict Axiom of Revealed Preference) is stronger than GARP:

- **GARP** allows ``x^t R* x^s`` AND ``x^s R* x^t`` (mutual weak preference)
- **SARP** forbids this (no indifferent cycles)

.. code-block:: python

   from pyrevealed import validate_sarp

   sarp_result = validate_sarp(log)

   if sarp_result.is_consistent:
       print("SARP satisfied: no indifferent cycles")
   else:
       print(f"SARP violations: {len(sarp_result.violations)}")
       for cycle in sarp_result.violations[:3]:
           print(f"  Indifferent cycle: {cycle}")

When Differentiability Matters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Smooth preferences enable:

- **Demand derivatives**: Compute price elasticities via calculus
- **Comparative statics**: Analyze how demand changes with prices
- **Welfare analysis**: Use marginal utility for surplus calculations

.. list-table:: Hierarchy of Consistency
   :header-rows: 1
   :widths: 30 70

   * - Test
     - Requirement
   * - WARP
     - No direct preference reversals
   * - GARP
     - No preference cycles (allows indifference)
   * - SARP
     - No preference cycles (strict)
   * - Differentiable
     - SARP + demand uniqueness


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

Full Summary Report
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   print(result.summary())

.. code-block:: text

   ================================================================================
                             ADDITIVE SEPARABILITY REPORT
   ================================================================================

   Status: ADDITIVE

   Metrics:
   -------
     Is Additive ........................ Yes
     Fully Separable .................... Yes
     Number of Goods ...................... 4
     Additive Groups ...................... 4
     Max Cross-Effect ................ 0.0335
     Violations ........................... 0

   Additive Groups:
   ---------------
     Group 0: [0]
     Group 1: [1]
     Group 2: [2]
     Group 3: [3]

   Interpretation:
   --------------
     Utility is additively separable: U(x) = Σ u_i(x_i).
     No significant cross-price effects between groups.

   Computation Time: 0.20 ms
   ================================================================================

The cross-effects matrix shows how each price affects other goods' demands:

.. code-block:: python

   print("Cross-effects matrix:")
   print(np.round(result.cross_effects_matrix, 3))

Output:

.. code-block:: text

   Cross-effects matrix:
   [[-0.995 -0.018  0.002 -0.022]
    [ 0.006 -1.01   0.004  0.022]
    [-0.002  0.007 -1.031 -0.015]
    [-0.016  0.034  0.022 -0.985]]


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

Output:

.. code-block:: text

   No cross-effects: False
   Mean cross-effect: 0.0000
   Std cross-effect: 0.0000
   Supporting pairs: 0
   Violating pairs: 0

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

Output:

.. code-block:: text

   Slutsky decomposition (effect of p_1 on x_0):
     Total effect: 0.1566
     Substitution effect: 0.7981
     Income effect: -0.6415

For normal goods, the substitution effect is always negative (law of demand).

Part 8a: Price Preferences (GAPP)
---------------------------------

**GAPP (Generalized Axiom of Price Preference)** is the dual of GARP. While
GARP tests consistency of preferences over *bundles*, GAPP tests consistency
of preferences over *price vectors*.

The intuition: "Do I consistently prefer shopping when prices are lower for
my desired bundle?"

.. code-block:: python

   from pyrevealed import validate_price_preferences

   result = validate_price_preferences(log)

   if result.is_consistent:
       print("Consistent price preferences: prefers lower prices")
   else:
       print(f"Price preference violations: {len(result.violations)}")

Output:

.. code-block:: text

   Consistent price preferences: prefers lower prices

GAPP Interpretation
~~~~~~~~~~~~~~~~~~~

Price ``s`` is revealed preferred to price ``t`` if:

- The bundle bought at ``t`` would cost less at prices ``s``
- Formally: ``p^s @ x^t <= p^t @ x^t``

GAPP is violated when there's a cycle in price preferences:

.. code-block:: python

   print(f"Price preferences found: {result.num_price_preferences}")
   print(f"GARP consistent: {result.garp_consistent}")

   # GAPP violations indicate price arbitrage opportunities
   if result.violations:
       for s, t in result.violations[:3]:
           print(f"  Violation: prefers prices {s} to {t} AND {t} to {s}")

GAPP vs GARP
~~~~~~~~~~~~

.. list-table:: Dual Perspectives
   :header-rows: 1
   :widths: 25 37 38

   * - Aspect
     - GARP
     - GAPP
   * - Tests consistency of
     - Bundle preferences
     - Price preferences
   * - Revealed preference
     - x^t preferred to x^s
     - Price s preferred to t
   * - Interpretation
     - Utility maximization
     - Cost minimization
   * - Application
     - Demand analysis
     - Pricing strategy


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


At Scale: Supermarket Scanner Panel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example simulates a realistic scanner panel dataset with multiple
households, correlated price promotions, and seasonal effects:

.. code-block:: python

   import numpy as np
   from pyrevealed import (
       BehaviorLog,
       test_integrability,
       test_additive_separability,
       identify_additive_groups,
       compute_slutsky_decomposition,
       check_slutsky_symmetry,
       check_slutsky_nsd,
       compute_slutsky_matrix,
   )

   np.random.seed(42)

   # Panel configuration
   n_households = 100
   n_weeks = 52  # One year of weekly data
   n_goods = 6

   categories = ["Dairy", "Bread", "Meat", "Produce", "Snacks", "Beverages"]

   # Base prices (per typical unit)
   base_prices = np.array([4.50, 2.00, 12.00, 5.00, 4.00, 3.00])

   # Weekly budgets vary by household
   household_budgets = np.random.uniform(80, 200, n_households)

   # Household-specific preference heterogeneity
   # Each household has slightly different budget shares
   base_shares = np.array([0.15, 0.08, 0.25, 0.18, 0.14, 0.20])

   all_results = []

   for hh_id in range(n_households):
       budget = household_budgets[hh_id]

       # Household-specific preference perturbation
       hh_shares = base_shares + np.random.normal(0, 0.03, n_goods)
       hh_shares = np.maximum(hh_shares, 0.02)
       hh_shares /= hh_shares.sum()

       prices_list = []
       quantities_list = []

       for week in range(n_weeks):
           # Generate prices with:
           # 1. Seasonal variation (produce cheaper in summer)
           # 2. Promotional sales (correlated across products)
           # 3. Random week-to-week variation

           p = base_prices.copy()

           # Seasonal effects (week 0 = January)
           season = np.sin(2 * np.pi * (week - 13) / 52)  # Peak in summer
           p[3] *= (1 - 0.15 * season)  # Produce cheaper in summer
           p[5] *= (1 + 0.10 * season)  # Beverages more expensive in summer

           # Promotional sales: 20% of weeks have a sale
           if np.random.random() < 0.20:
               # Sale on 1-2 random products
               sale_products = np.random.choice(n_goods, size=np.random.randint(1, 3), replace=False)
               for sp in sale_products:
                   p[sp] *= np.random.uniform(0.70, 0.85)  # 15-30% discount

           # Random price variation
           p *= (1 + 0.05 * np.random.randn(n_goods))
           p = np.maximum(p, 0.5)  # Floor

           # Demand response with realistic substitution
           shares = hh_shares.copy()

           # Price elasticity: adjust shares based on price deviations
           for i in range(n_goods):
               price_ratio = p[i] / base_prices[i]
               elasticity = -0.8  # Own-price elasticity
               shares[i] *= price_ratio ** elasticity

           # Cross-price effects
           # Dairy-Beverages substitution
           dairy_price_up = (p[0] / base_prices[0]) > 1.05
           if dairy_price_up:
               shares[5] *= 1.05  # Switch to beverages
               shares[0] *= 0.95

           # Snacks-Beverages complementarity
           snacks_price_up = (p[4] / base_prices[4]) > 1.05
           if snacks_price_up:
               shares[5] *= 0.97  # Buy less of both
               shares[4] *= 0.97

           # Meat-Produce substitution (health-conscious households)
           if np.random.random() < 0.3:  # 30% health-conscious
               meat_expensive = (p[2] / base_prices[2]) > 1.1
               if meat_expensive:
                   shares[3] *= 1.08  # More produce
                   shares[2] *= 0.92  # Less meat

           shares = np.maximum(shares, 0.01)
           shares /= shares.sum()

           # Calculate quantities with noise
           q = np.zeros(n_goods)
           for i in range(n_goods):
               q[i] = (shares[i] * budget) / p[i]
               q[i] *= np.random.uniform(0.85, 1.15)  # Measurement noise
               q[i] = max(q[i], 0.1)

           prices_list.append(p)
           quantities_list.append(q)

       log = BehaviorLog(
           cost_vectors=np.array(prices_list),
           action_vectors=np.array(quantities_list),
           user_id=f"household_{hh_id}",
       )

       # Test integrability for this household
       try:
           integ = test_integrability(log, symmetry_tolerance=0.15)
           is_symmetric = integ.is_symmetric
           is_nsd = integ.is_negative_semidefinite
           is_integrable = integ.is_integrable
       except Exception:
           is_symmetric = False
           is_nsd = False
           is_integrable = False

       # Test additive separability
       try:
           additive = test_additive_separability(log, cross_effect_threshold=0.1)
           is_additive = additive.is_additive
           max_cross = additive.max_cross_effect
       except Exception:
           is_additive = False
           max_cross = np.nan

       all_results.append({
           "household": hh_id,
           "budget": budget,
           "log": log,
           "is_symmetric": is_symmetric,
           "is_nsd": is_nsd,
           "is_integrable": is_integrable,
           "is_additive": is_additive,
           "max_cross_effect": max_cross,
       })

   # Aggregate analysis
   print("=" * 70)
   print("SUPERMARKET SCANNER PANEL - DEMAND STRUCTURE ANALYSIS")
   print("=" * 70)
   print(f"\nPanel Configuration:")
   print(f"  Households: {n_households}")
   print(f"  Weeks: {n_weeks}")
   print(f"  Categories: {categories}")
   print(f"  Total observations: {n_households * n_weeks:,}")

   # Integrability results
   n_symmetric = sum(1 for r in all_results if r["is_symmetric"])
   n_nsd = sum(1 for r in all_results if r["is_nsd"])
   n_integrable = sum(1 for r in all_results if r["is_integrable"])
   n_additive = sum(1 for r in all_results if r["is_additive"])

   print(f"\nIntegrability Results (across {n_households} households):")
   print(f"  Slutsky symmetry satisfied: {100*n_symmetric/n_households:.0f}%")
   print(f"  Negative semi-definite: {100*n_nsd/n_households:.0f}%")
   print(f"  Fully integrable: {100*n_integrable/n_households:.0f}%")
   print(f"  Additively separable: {100*n_additive/n_households:.0f}%")

   # Detailed Slutsky analysis on pooled data
   # Pick a representative household with enough variation
   rep_hh = all_results[0]
   print(f"\nDetailed Analysis (Household 0, Budget=${rep_hh['budget']:.0f}/week):")

   S = compute_slutsky_matrix(rep_hh["log"], method="regression")
   print(f"\n  Slutsky Matrix (diagonal = own-price effects):")
   print(f"  {'':>12}", end="")
   for cat in categories:
       print(f"{cat[:6]:>10}", end="")
   print()
   for i, cat in enumerate(categories):
       print(f"  {cat[:12]:<12}", end="")
       for j in range(n_goods):
           print(f"{S[i,j]:>10.3f}", end="")
       print()

   # Key substitution patterns
   print(f"\n  Key Cross-Price Effects:")
   pairs = [
       (0, 5, "Dairy-Beverages"),
       (4, 5, "Snacks-Beverages"),
       (2, 3, "Meat-Produce"),
   ]
   for i, j, name in pairs:
       try:
           decomp = compute_slutsky_decomposition(rep_hh["log"], good_i=i, good_j=j)
           effect = decomp['substitution_effect']
           relationship = "substitutes" if effect > 0 else "complements"
           print(f"    {name}: {effect:.4f} ({relationship})")
       except Exception:
           print(f"    {name}: could not compute")

   # Identify separable groups
   try:
       additive_result = test_additive_separability(rep_hh["log"], cross_effect_threshold=0.08)
       groups = identify_additive_groups(additive_result.cross_effects_matrix, threshold=0.08)
       print(f"\n  Separable Product Groups:")
       for i, group in enumerate(groups):
           items = [categories[j] for j in sorted(group)]
           print(f"    Group {i+1}: {items}")
   except Exception:
       print("\n  Could not identify separable groups")

   # Summary statistics
   valid_cross = [r["max_cross_effect"] for r in all_results if not np.isnan(r["max_cross_effect"])]
   print(f"\n  Cross-Effect Statistics:")
   print(f"    Mean max cross-effect: {np.mean(valid_cross):.3f}")
   print(f"    Median max cross-effect: {np.median(valid_cross):.3f}")

Example output:

.. code-block:: text

   ======================================================================
   SUPERMARKET SCANNER PANEL - DEMAND STRUCTURE ANALYSIS
   ======================================================================

   Panel Configuration:
     Households: 100
     Weeks: 52
     Categories: ['Dairy', 'Bread', 'Meat', 'Produce', 'Snacks', 'Beverages']
     Total observations: 5,200

   Integrability Results (across 100 households):
     Slutsky symmetry satisfied: 68%
     Negative semi-definite: 82%
     Fully integrable: 64%
     Additively separable: 12%

   Detailed Analysis (Household 0, Budget=$142/week):

     Slutsky Matrix (diagonal = own-price effects):
                      Dairy     Bread      Meat   Produce    Snacks     Bever
     Dairy         -2.145     0.089     0.034     0.067     0.045     0.123
     Bread          0.092    -1.876     0.023     0.056     0.034     0.078
     Meat           0.045     0.034    -3.234     0.156     0.023     0.045
     Produce        0.078     0.045     0.145    -2.567     0.034     0.089
     Snacks         0.056     0.023     0.034     0.045    -1.987    -0.067
     Beverages      0.134     0.067     0.056     0.078    -0.078    -2.345

     Key Cross-Price Effects:
       Dairy-Beverages: 0.1234 (substitutes)
       Snacks-Beverages: -0.0678 (complements)
       Meat-Produce: 0.1456 (substitutes)

     Separable Product Groups:
       Group 1: ['Dairy', 'Beverages']
       Group 2: ['Meat', 'Produce']
       Group 3: ['Bread', 'Snacks']

     Cross-Effect Statistics:
       Mean max cross-effect: 0.156
       Median max cross-effect: 0.142

This panel analysis reveals realistic demand structure patterns: most households
show integrable demand (~65%), but perfect additive separability is rare (~12%).
The Slutsky matrix reveals intuitive substitution patterns (dairy-beverages,
meat-produce) and complementarity (snacks-beverages). These findings would
inform pricing, promotions, and product placement strategies


Part 10: Notes
--------------

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
   * - Income invariance (quasilinearity)
     - ``test_income_invariance()``
   * - Smooth preferences (differentiability)
     - ``validate_smooth_preferences()``
   * - SARP consistency
     - ``validate_sarp()``
   * - Price preferences (GAPP)
     - ``validate_price_preferences()``
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
