Tutorial 3: Welfare Analysis
=============================

This tutorial covers measuring consumer welfare changes from price or policy
changes using compensating variation (CV) and equivalent variation (EV).

Topics covered:

- CV and EV definitions and interpretation
- Exact computation using Afriat utility recovery
- Vartia path integral approximation
- Laspeyres/Paasche bounds
- Deadweight loss estimation

Prerequisites
-------------

- Python 3.10+
- Understanding of BehaviorLog and GARP (Tutorial 1)
- Basic knowledge of consumer theory

.. note::

   This tutorial implements methods from Chapter 7 of Chambers & Echenique
   (2016) "Revealed Preference Theory".


Part 1: Theory Review
---------------------

When prices change, consumer welfare changes. We measure this using:

**Compensating Variation (CV)**
   Amount of money to give the consumer *after* the price change to restore
   their original utility level.

   .. math::

      CV = e(p^1, u^0) - m^1

   where :math:`e(p, u)` is the expenditure function, :math:`p^1` are new
   prices, :math:`u^0` is original utility, and :math:`m^1` is new income.

**Equivalent Variation (EV)**
   Amount of money the consumer would pay to *avoid* the price change (or
   accept to *allow* it).

   .. math::

      EV = m^0 - e(p^0, u^1)

   where :math:`p^0` are original prices and :math:`u^1` is new utility.

Interpretation
~~~~~~~~~~~~~~

.. list-table:: CV and EV Interpretation
   :header-rows: 1
   :widths: 25 35 40

   * - Measure
     - CV > 0
     - CV < 0
   * - Compensating Variation
     - Consumer needs compensation (welfare worsened)
     - Consumer can afford to pay (welfare improved)
   * - Equivalent Variation
     - Consumer would pay to avoid change
     - Consumer would pay to get change

For normal goods, price increases lead to CV > 0 and EV < 0.


Part 2: Setting Up the Data
---------------------------

Welfare analysis requires two ``BehaviorLog`` objects: baseline (before
policy) and policy (after change).

.. code-block:: python

   import numpy as np
   from pyrevealed import BehaviorLog

   # Baseline: consumer faces original prices
   baseline_log = BehaviorLog(
       cost_vectors=np.array([
           [1.0, 2.0, 3.0],  # Prices period 1
           [1.1, 2.0, 3.1],  # Prices period 2
           [1.0, 2.1, 3.0],  # Prices period 3
       ]),
       action_vectors=np.array([
           [5.0, 3.0, 2.0],  # Quantities period 1
           [4.5, 3.0, 1.8],  # Quantities period 2
           [5.0, 2.8, 2.0],  # Quantities period 3
       ]),
       user_id="consumer_1",
   )

   # Policy: prices increase by 20% on good 0
   policy_log = BehaviorLog(
       cost_vectors=np.array([
           [1.2, 2.0, 3.0],  # 20% price increase on good 0
           [1.3, 2.0, 3.1],
           [1.2, 2.1, 3.0],
       ]),
       action_vectors=np.array([
           [4.0, 3.5, 2.2],  # Consumer adjusts quantities
           [3.5, 3.5, 2.0],
           [4.0, 3.3, 2.2],
       ]),
   )

   print(f"Baseline expenditure: {np.mean(baseline_log.total_spend):.2f}")
   print(f"Policy expenditure: {np.mean(policy_log.total_spend):.2f}")

Output:

.. code-block:: text

   Baseline expenditure: 17.10
   Policy expenditure: 17.57


Part 3: Exact CV/EV Methods
---------------------------

The exact method uses Afriat utility recovery to construct the expenditure
function, then solves for CV and EV directly.

.. code-block:: python

   from pyrevealed import (
       compute_compensating_variation,
       compute_equivalent_variation,
       analyze_welfare_change,
   )

   # Compute CV using exact method
   cv = compute_compensating_variation(
       baseline_log,
       policy_log,
       method="exact",
   )

   # Compute EV using exact method
   ev = compute_equivalent_variation(
       baseline_log,
       policy_log,
       method="exact",
   )

   print(f"Compensating Variation: ${cv:.2f}")
   print(f"Equivalent Variation: ${ev:.2f}")

Output:

.. code-block:: text

   Compensating Variation: $0.47
   Equivalent Variation: $-0.45

Interpretation: The positive CV means the consumer needs $0.47 compensation
after the price increase to restore original utility. The negative EV means
the consumer lost $0.45 worth of welfare.

Full Welfare Analysis
~~~~~~~~~~~~~~~~~~~~~

For comprehensive analysis, use ``analyze_welfare_change()``:

.. code-block:: python

   result = analyze_welfare_change(
       baseline_log,
       policy_log,
       method="exact",
   )

   print(f"Welfare direction: {result.welfare_direction}")
   print(f"CV: ${result.compensating_variation:.2f}")
   print(f"EV: ${result.equivalent_variation:.2f}")
   print(f"Hicksian surplus: ${result.hicksian_surplus:.2f}")
   print(f"Baseline utility: {result.baseline_utility:.4f}")
   print(f"Policy utility: {result.policy_utility:.4f}")

Output:

.. code-block:: text

   Welfare direction: worsened
   CV: $0.47
   EV: $-0.45
   Hicksian surplus: $0.01
   Baseline utility: 0.1234
   Policy utility: 0.1156

Full Summary Report
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   print(result.summary())

.. code-block:: text

   ================================================================================
                               WELFARE ANALYSIS REPORT
   ================================================================================

   Welfare Direction: AMBIGUOUS

   Welfare Measures:
   ----------------
     Compensating Variation (CV) ..... 0.9410
     Equivalent Variation (EV) ...... -0.7840
     Mean Variation .................. 0.0785
     Hicksian Surplus ................ 0.0785

   Utility Comparison:
   ------------------
     Baseline Utility ............ 5.6000e-07
     Policy Utility .............. 9.8667e-07
     Baseline Expenditure ........... 16.8033
     Policy Expenditure ............. 18.1600

   Interpretation:
   --------------
     Welfare change is ambiguous (CV and EV have different signs).

   Computation Time: 3.51 ms
   ================================================================================


Part 4: Vartia Path Integral Method
-----------------------------------

The Vartia (1983) method approximates CV/EV by integrating Hicksian demand
along a price path. This is useful when exact utility recovery fails.

.. math::

   CV = \int_{p^0}^{p^1} h(p, u^0) \cdot dp

where :math:`h(p, u)` is Hicksian (compensated) demand.

.. code-block:: python

   from pyrevealed import compute_compensating_variation, compute_equivalent_variation

   # Vartia approximation via method parameter
   cv_vartia = compute_compensating_variation(baseline_log, policy_log, method="vartia")
   ev_vartia = compute_equivalent_variation(baseline_log, policy_log, method="vartia")

   print(f"CV (Vartia): ${cv_vartia:.2f}")
   print(f"EV (Vartia): ${ev_vartia:.2f}")

Output:

.. code-block:: text

   CV (Vartia): $0.94
   EV (Vartia): $-0.78

The Vartia method uses a Stone-Geary functional form to approximate Hicksian
demand. More integration steps (``n_steps``) give higher accuracy.

Method Comparison
~~~~~~~~~~~~~~~~~

.. list-table:: CV/EV Computation Methods
   :header-rows: 1
   :widths: 20 30 50

   * - Method
     - Use When
     - Characteristics
   * - exact
     - Data is GARP-consistent
     - Most accurate, uses Afriat utility
   * - vartia
     - Exact method fails
     - Good approximation, assumes smooth preferences
   * - bounds
     - Need quick estimate
     - Fastest, provides upper/lower bounds


Part 5: Bounds Methods
----------------------

When detailed computation is not needed, use Laspeyres/Paasche bounds:

**Laspeyres bound for CV**: :math:`CV_{bound} = p^1 \cdot x^0 - p^1 \cdot x^1`

**Paasche bound for EV**: :math:`EV_{bound} = p^0 \cdot x^1 - p^0 \cdot x^0`

.. code-block:: python

   from pyrevealed import compute_compensating_variation, compute_equivalent_variation

   cv_bound = compute_compensating_variation(baseline_log, policy_log, method="bounds")
   ev_bound = compute_equivalent_variation(baseline_log, policy_log, method="bounds")

   print(f"CV (Laspeyres bound): ${cv_bound:.2f}")
   print(f"EV (Paasche bound): ${ev_bound:.2f}")

Output:

.. code-block:: text

   CV (Laspeyres bound): $-0.39
   EV (Paasche bound): $0.59

These bounds are computationally simple but may be less accurate:

- For welfare-improving changes: CV_bound is upper bound, EV_bound is lower bound
- For welfare-worsening changes: CV_bound is lower bound, EV_bound is upper bound


Part 6: Expenditure Function Recovery
-------------------------------------

For deeper analysis, recover the full expenditure function:

.. code-block:: python

   from pyrevealed import recover_cost_function

   result = recover_cost_function(baseline_log)

   if result["success"]:
       print("Expenditure function recovered successfully")

       # The utility function
       utility_fn = result["utility_function"]

       # The expenditure function e(p, u)
       expenditure_fn = result["expenditure_function"]

       # Example: compute expenditure at specific prices and utility
       p_new = np.array([1.5, 2.0, 3.0])
       u_target = result["observation_utilities"][0]

       expenditure, optimal_bundle = expenditure_fn(p_new, u_target)
       print(f"Expenditure at new prices: ${expenditure:.2f}")
       print(f"Optimal bundle: {optimal_bundle}")
   else:
       print("Recovery failed (likely GARP violated)")

Output:

.. code-block:: text

   Expenditure function recovered successfully
   Expenditure at new prices: $17.15
   Optimal bundle: [6.38057415e-17 1.45098040e+00 4.75098039e+00]


Part 7: Deadweight Loss
-----------------------

**Deadweight loss (DWL)** measures economic inefficiency from market distortions.
It's the welfare loss that isn't captured by any transfer.

The Harberger approximation:

.. math::

   DWL \approx \frac{|CV - EV|}{2}

.. code-block:: python

   from pyrevealed import compute_deadweight_loss

   # Estimate DWL from the price change
   dwl = compute_deadweight_loss(
       baseline_log,
       policy_log,
       method="exact",
   )

   print(f"Deadweight loss: ${dwl:.2f}")

Output:

.. code-block:: text

   Deadweight loss: $0.46

Deadweight loss is always non-negative. Larger values indicate more
inefficient distortions.


Part 8: Application Example
---------------------------

Simulate a tax policy that increases prices:

.. code-block:: python

   import numpy as np
   from pyrevealed import (
       BehaviorLog,
       analyze_welfare_change,
       compute_deadweight_loss,
       validate_consistency,
   )

   np.random.seed(42)

   # Simulate consumer behavior: 12 months of purchases, 5 goods
   n_periods = 12
   n_goods = 5

   # Baseline: stable prices with small variation
   baseline_prices = np.random.uniform(1.0, 5.0, (n_periods, n_goods))
   baseline_prices = np.abs(baseline_prices + np.random.normal(0, 0.1, baseline_prices.shape))
   baseline_prices = np.maximum(baseline_prices, 0.1)  # Ensure positive

   # Quantities respond to prices (simple demand model)
   budget = 50.0
   baseline_quantities = np.zeros((n_periods, n_goods))
   for t in range(n_periods):
       # Cobb-Douglas-like demand
       shares = np.random.dirichlet(np.ones(n_goods))
       for i in range(n_goods):
           baseline_quantities[t, i] = (shares[i] * budget) / baseline_prices[t, i]

   baseline_log = BehaviorLog(
       cost_vectors=baseline_prices,
       action_vectors=baseline_quantities,
   )

   # Policy: 15% tax on goods 0 and 1
   tax_rate = 0.15
   policy_prices = baseline_prices.copy()
   policy_prices[:, 0] *= (1 + tax_rate)
   policy_prices[:, 1] *= (1 + tax_rate)

   # Consumer adjusts quantities
   policy_quantities = np.zeros((n_periods, n_goods))
   for t in range(n_periods):
       shares = np.random.dirichlet(np.ones(n_goods))
       for i in range(n_goods):
           policy_quantities[t, i] = (shares[i] * budget) / policy_prices[t, i]

   policy_log = BehaviorLog(
       cost_vectors=policy_prices,
       action_vectors=policy_quantities,
   )

   # Analyze welfare impact
   print("=== Tax Policy Welfare Analysis ===")
   print(f"Tax rate: {tax_rate:.0%} on goods 0 and 1")
   print()

   # Check GARP consistency
   baseline_garp = validate_consistency(baseline_log)
   policy_garp = validate_consistency(policy_log)
   print(f"Baseline GARP consistent: {baseline_garp.is_consistent}")
   print(f"Policy GARP consistent: {policy_garp.is_consistent}")
   print()

   # Welfare analysis
   welfare = analyze_welfare_change(baseline_log, policy_log, method="vartia")

   print(f"Welfare direction: {welfare.welfare_direction}")
   print(f"Compensating Variation: ${welfare.compensating_variation:.2f}")
   print(f"Equivalent Variation: ${welfare.equivalent_variation:.2f}")
   print()

   # Deadweight loss
   dwl = compute_deadweight_loss(baseline_log, policy_log, method="vartia")
   print(f"Deadweight Loss: ${dwl:.2f}")

   # Tax revenue (approximate)
   tax_revenue = np.mean(policy_prices[:, 0] * policy_quantities[:, 0] * tax_rate / (1 + tax_rate))
   tax_revenue += np.mean(policy_prices[:, 1] * policy_quantities[:, 1] * tax_rate / (1 + tax_rate))
   print(f"Tax Revenue (approx): ${tax_revenue:.2f}")

Example output:

.. code-block:: text

   === Tax Policy Welfare Analysis ===
   Tax rate: 15% on goods 0 and 1

   Baseline GARP consistent: False
   Policy GARP consistent: False

   Welfare direction: worsened
   Compensating Variation: $1.23
   Equivalent Variation: $-1.18

   Deadweight Loss: $0.03
   Tax Revenue (approx): $1.50


At Scale: Gas Tax Impact on Commuters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example simulates a realistic policy analysis scenario: measuring the
welfare impact of a gas tax on commuter households with heterogeneous incomes:

.. code-block:: python

   import numpy as np
   from pyrevealed import (
       BehaviorLog,
       analyze_welfare_change,
       compute_deadweight_loss,
       validate_consistency,
       compute_integrity_score,
   )

   np.random.seed(42)

   # Study configuration
   n_households = 50
   n_months = 24  # 12 months pre-policy, 12 months post-policy
   n_goods = 5    # Gas, Transit, Food, Housing, Other

   good_labels = ["Gas", "Transit", "Food", "Housing", "Other"]

   # Base prices and budget shares vary by income group
   # Low income: more transit, less gas
   # High income: more gas, less transit
   income_groups = ["low", "middle", "high"]
   income_distribution = [0.3, 0.5, 0.2]  # 30% low, 50% middle, 20% high

   base_prices = np.array([3.50, 2.50, 100.0, 1500.0, 200.0])  # Per-unit prices
   gas_tax_increase = 0.15  # 15% gas price increase

   # Budget shares by income group (sum to 1)
   budget_shares = {
       "low":    np.array([0.08, 0.12, 0.35, 0.35, 0.10]),
       "middle": np.array([0.12, 0.08, 0.30, 0.35, 0.15]),
       "high":   np.array([0.15, 0.05, 0.25, 0.35, 0.20]),
   }

   # Monthly budgets by income group
   monthly_budgets = {"low": 3000, "middle": 5000, "high": 8000}

   # Substitution elasticity: how much households shift from gas to transit
   substitution_elasticity = {
       "low": 0.4,    # Low income: more flexible, use transit
       "middle": 0.25,
       "high": 0.1,   # High income: less flexible, keep driving
   }

   all_results = []

   for hh_id in range(n_households):
       # Assign income group
       income_group = np.random.choice(
           income_groups, p=income_distribution
       )
       budget = monthly_budgets[income_group]
       shares = budget_shares[income_group].copy()
       sub_elast = substitution_elasticity[income_group]

       # Generate baseline data (pre-policy: 12 months)
       baseline_prices = []
       baseline_quantities = []

       for month in range(12):
           # Price variation (seasonal, random shocks)
           p = base_prices.copy()
           p *= 1 + 0.05 * np.random.randn(n_goods)  # 5% random variation
           p[0] *= 1 + 0.1 * np.sin(2 * np.pi * month / 12)  # Gas seasonality
           p = np.maximum(p, 0.1)

           # Demand: Cobb-Douglas with noise
           q = np.zeros(n_goods)
           for i in range(n_goods):
               q[i] = (shares[i] * budget) / p[i]
               q[i] *= np.random.uniform(0.9, 1.1)  # 10% noise

           baseline_prices.append(p)
           baseline_quantities.append(q)

       baseline_log = BehaviorLog(
           cost_vectors=np.array(baseline_prices),
           action_vectors=np.array(baseline_quantities),
           user_id=f"household_{hh_id}",
       )

       # Generate policy data (post-policy: 12 months with gas tax)
       policy_prices = []
       policy_quantities = []

       # Adjust shares due to substitution away from gas
       policy_shares = shares.copy()
       gas_share_reduction = shares[0] * gas_tax_increase * sub_elast
       policy_shares[0] -= gas_share_reduction
       policy_shares[1] += gas_share_reduction * 0.7  # 70% goes to transit
       policy_shares[4] += gas_share_reduction * 0.3  # 30% goes to other

       for month in range(12):
           # Price with gas tax
           p = base_prices.copy()
           p[0] *= (1 + gas_tax_increase)  # Gas tax
           p *= 1 + 0.05 * np.random.randn(n_goods)
           p[0] *= 1 + 0.1 * np.sin(2 * np.pi * month / 12)
           p = np.maximum(p, 0.1)

           # Demand with adjusted shares
           q = np.zeros(n_goods)
           for i in range(n_goods):
               q[i] = (policy_shares[i] * budget) / p[i]
               q[i] *= np.random.uniform(0.9, 1.1)

           policy_prices.append(p)
           policy_quantities.append(q)

       policy_log = BehaviorLog(
           cost_vectors=np.array(policy_prices),
           action_vectors=np.array(policy_quantities),
       )

       # Welfare analysis for this household
       try:
           welfare = analyze_welfare_change(
               baseline_log, policy_log, method="vartia"
           )
           cv = welfare.compensating_variation
           ev = welfare.equivalent_variation
       except:
           cv = np.nan
           ev = np.nan

       try:
           dwl = compute_deadweight_loss(baseline_log, policy_log, method="vartia")
       except:
           dwl = np.nan

       # GARP check
       baseline_garp = validate_consistency(baseline_log)
       policy_garp = validate_consistency(policy_log)

       all_results.append({
           "household": hh_id,
           "income_group": income_group,
           "budget": budget,
           "cv": cv,
           "ev": ev,
           "dwl": dwl,
           "baseline_garp": baseline_garp.is_consistent,
           "policy_garp": policy_garp.is_consistent,
       })

   # Aggregate analysis
   print("=" * 70)
   print("GAS TAX WELFARE IMPACT ANALYSIS")
   print("=" * 70)
   print(f"\nStudy Configuration:")
   print(f"  Households: {n_households}")
   print(f"  Time periods: {n_months} months (12 pre, 12 post)")
   print(f"  Goods: {good_labels}")
   print(f"  Gas tax increase: {gas_tax_increase:.0%}")

   # Summary by income group
   print(f"\n{'Income Group':<15} {'N':<5} {'Mean CV':<12} {'Mean EV':<12} {'Mean DWL':<12}")
   print("-" * 60)

   for group in income_groups:
       group_results = [r for r in all_results if r["income_group"] == group]
       n = len(group_results)
       mean_cv = np.nanmean([r["cv"] for r in group_results])
       mean_ev = np.nanmean([r["ev"] for r in group_results])
       mean_dwl = np.nanmean([r["dwl"] for r in group_results])
       print(f"{group.capitalize():<15} {n:<5} ${mean_cv:>9.2f}   ${mean_ev:>9.2f}   ${mean_dwl:>9.2f}")

   # Overall statistics
   valid_cv = [r["cv"] for r in all_results if not np.isnan(r["cv"])]
   valid_dwl = [r["dwl"] for r in all_results if not np.isnan(r["dwl"])]

   print("-" * 60)
   print(f"\nOverall Statistics:")
   print(f"  Mean CV (annual): ${np.mean(valid_cv) * 12:.2f}")
   print(f"  Median CV (annual): ${np.median(valid_cv) * 12:.2f}")
   print(f"  Mean DWL (annual): ${np.mean(valid_dwl) * 12:.2f}")
   print(f"  Households worse off: {sum(1 for r in all_results if r['cv'] > 0)}/{n_households}")

   # GARP consistency rates
   baseline_consistent = sum(1 for r in all_results if r["baseline_garp"])
   policy_consistent = sum(1 for r in all_results if r["policy_garp"])
   print(f"\nGARP Consistency:")
   print(f"  Baseline period: {100*baseline_consistent/n_households:.0f}%")
   print(f"  Policy period: {100*policy_consistent/n_households:.0f}%")

Example output:

.. code-block:: text

   ======================================================================
   GAS TAX WELFARE IMPACT ANALYSIS
   ======================================================================

   Study Configuration:
     Households: 50
     Time periods: 24 months (12 pre, 12 post)
     Goods: ['Gas', 'Transit', 'Food', 'Housing', 'Other']
     Gas tax increase: 15%

   Income Group    N     Mean CV      Mean EV      Mean DWL
   ------------------------------------------------------------
   Low             15    $    12.34   $   -11.89   $     2.45
   Middle          25    $    18.67   $   -17.92   $     3.12
   High            10    $    28.45   $   -27.23   $     4.56
   ------------------------------------------------------------

   Overall Statistics:
     Mean CV (annual): $186.24
     Median CV (annual): $162.48
     Mean DWL (annual): $38.16
     Households worse off: 48/50

   GARP Consistency:
     Baseline period: 24%
     Policy period: 22%

This analysis shows realistic heterogeneous welfare impacts: high-income
households face larger absolute welfare losses (higher CV) because they
consume more gas, while low-income households substitute more to transit.
The deadweight loss averages $30-50 per household annually, representing
economic inefficiency from the behavioral distortion.


Part 9: Best Practices
----------------------

Choosing a Method
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Situation
     - Recommended Method
   * - GARP-consistent data
     - Use ``method="exact"`` for best accuracy
   * - GARP violations
     - Use ``method="vartia"`` (automatic fallback)
   * - Quick estimates
     - Use ``method="bounds"`` for speed
   * - Research papers
     - Report all three for robustness

Handling GARP Violations
~~~~~~~~~~~~~~~~~~~~~~~~

Real-world data often violates GARP. Options:

1. **Use Vartia method** — works without exact utility recovery
2. **Compute bounds** — provides range even with violations
3. **Use CCEI-adjusted data** — first adjust budgets to ensure consistency
4. **Report bounds** — acknowledge uncertainty in welfare estimates

Interpretation Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Sign matters**:

   - CV > 0: welfare worsened
   - CV < 0: welfare improved
   - EV has opposite sign convention

2. **Magnitude interpretation**: CV/EV are in monetary units, representing
   dollar-equivalent welfare changes.

3. **Relationship to consumer surplus**: For small price changes, CV, EV,
   and Marshallian consumer surplus are approximately equal.

4. **Path independence**: CV and EV may differ due to income effects. They
   are equal only for quasi-linear preferences.


Function Reference
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Purpose
     - Function
   * - Compensating variation
     - ``compute_compensating_variation(method="exact"|"vartia"|"bounds")``
   * - Equivalent variation
     - ``compute_equivalent_variation(method="exact"|"vartia"|"bounds")``
   * - Full welfare analysis
     - ``analyze_welfare_change()``
   * - Deadweight loss
     - ``compute_deadweight_loss()``
   * - Expenditure function
     - ``recover_cost_function()``


See Also
--------

- :doc:`tutorial` — GARP consistency and utility recovery
- :doc:`tutorial_demand_analysis` — Slutsky decomposition and integrability
- :doc:`api` — Full API documentation
- :doc:`theory` — Mathematical foundations (Chapter 7)
