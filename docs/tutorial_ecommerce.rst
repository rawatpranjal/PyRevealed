Tutorial 2
==========

This tutorial analyzes 1.85 million Amazon transactions from 5,027 US consumers
using revealed preference methods.

Topics covered:

- Theory vs. e-commerce data: assumptions and limitations
- Data processing pipeline for transaction logs
- GARP testing and efficiency index computation
- Scaling to thousands of users
- Power analysis and result interpretation

Prerequisites
-------------

- Completed :doc:`tutorial` (Dunnhumby basics)
- Python 3.8+ with NumPy, pandas, matplotlib
- Basic understanding of revealed preference theory

.. note::

   The full code for this tutorial is available in the
   ``datasets/open_ecommerce/`` directory of the PyRevealed repository.


Part 1: The Data Challenge
--------------------------

Revealed preference theory assumes a consumer solves:

.. math::

   \max_{x} U(x) \quad \text{subject to} \quad p \cdot x \leq m

We observe price-quantity pairs :math:`(p_t, x_t)` across choice occasions.
But e-commerce data looks very different:

.. list-table:: Theory vs. Reality
   :header-rows: 1
   :widths: 50 50

   * - Theory Requires
     - E-Commerce Provides
   * - Fixed, finite set of goods
     - Millions of unique products
   * - Discrete choice occasions
     - Continuous shopping
   * - Budget-exhausting choices
     - Partial budget (Amazon ≠ all spending)
   * - Exogenous prices
     - Consumers choose *when* to buy

**The Dataset:** Open E-Commerce 1.0 from Harvard Dataverse contains Amazon
purchase histories from 5,027 US consumers spanning 2018-2023.

.. list-table:: Dataset Overview
   :header-rows: 1
   :widths: 40 60

   * - Metric
     - Value
   * - Raw transactions
     - 1.85 million
   * - Time span
     - 66 months (Jan 2018 - Jul 2023)
   * - Product categories
     - 50 Amazon types
   * - Qualifying users
     - 4,744 (with ≥5 active months)


Part 2: Theory Review
---------------------

Key concepts for revealed preference analysis.

Revealed Preference Relations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For :math:`T` observations with prices :math:`p_t` and quantities :math:`x_t`:

**Direct Revealed Preference (R):**

.. math::

   x_i \, R \, x_j \iff p_i \cdot x_i \geq p_i \cdot x_j

Bundle :math:`x_j` was affordable when :math:`x_i` was chosen.

**Strict Revealed Preference (P):**

.. math::

   x_i \, P \, x_j \iff p_i \cdot x_i > p_i \cdot x_j

**Transitive Closure (R*):** :math:`x_i R^* x_j` if there's a chain of
revealed preferences connecting them.

The Axioms
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 45 40

   * - Axiom
     - Condition
     - Meaning
   * - **WARP**
     - If :math:`x_i R x_j`, then NOT :math:`x_j P x_i`
     - No direct contradictions
   * - **GARP**
     - If :math:`x_i R^* x_j`, then NOT :math:`x_j P x_i`
     - No contradictions via chains
   * - **SARP**
     - If :math:`x_i R^* x_j` (i≠j), then NOT :math:`x_j R^* x_i`
     - No indifference cycles

**Afriat's Theorem:** GARP holds ⟺ a well-behaved utility function exists.


Part 3: Data Processing Pipeline
--------------------------------

Transforming raw transactions into analyzable matrices.

Step 1: Load and Clean
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   import numpy as np

   # Load raw data
   df = pd.read_csv('amazon-purchases.csv')

   # Standardize columns
   df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
   df = df.rename(columns={
       'order_date': 'date',
       'purchase_price_per_unit': 'price',
       'survey_responseid': 'user_id'
   })

   # Parse dates and create periods
   df['date'] = pd.to_datetime(df['date'])
   df['period'] = df['date'].dt.to_period('M')

   # Filter invalid prices
   df = df[(df['price'] >= 0.01) & (df['price'] <= 1000)]

   print(f"Loaded {len(df):,} transactions")
   print(f"Users: {df['user_id'].nunique():,}")
   print(f"Categories: {df['category'].nunique()}")

Step 2: Build Price Oracle
~~~~~~~~~~~~~~~~~~~~~~~~~~

We only observe prices when someone buys. For unpurchased categories, we
impute using market medians:

.. code-block:: python

   def build_price_oracle(df, categories):
       """Build market price grid: median price per category per period."""
       price_oracle = df.pivot_table(
           index='period',
           columns='category',
           values='price',
           aggfunc='median'
       )
       # Fill gaps
       price_oracle = price_oracle.ffill().bfill()
       return price_oracle[categories]

   categories = sorted(df['category'].unique())
   price_oracle = build_price_oracle(df, categories)
   print(f"Price oracle: {price_oracle.shape} (periods × categories)")

Step 3: Build User Matrices
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each user, create price and quantity matrices:

.. code-block:: python

   def build_user_matrices(df, user_id, price_oracle, categories):
       """Build T×N price and quantity matrices for one user."""
       user_df = df[df['user_id'] == user_id]

       # Aggregate quantities by period and category
       qty = user_df.pivot_table(
           index='period',
           columns='category',
           values='quantity',
           aggfunc='sum',
           fill_value=0
       ).reindex(columns=categories, fill_value=0)

       active_periods = qty.index.tolist()

       # Price matrix: user's price if purchased, oracle price otherwise
       prices = pd.DataFrame(index=active_periods, columns=categories)
       for period in active_periods:
           for cat in categories:
               if qty.loc[period, cat] > 0:
                   # User's actual price
                   mask = (user_df['period'] == period) & (user_df['category'] == cat)
                   prices.loc[period, cat] = user_df[mask]['price'].median()
               else:
                   # Market price
                   prices.loc[period, cat] = price_oracle.loc[period, cat]

       return prices.values, qty.values, active_periods


Part 4: Implementing the Algorithms
-----------------------------------

GARP testing and AEI computation from scratch.

Cost Matrix
~~~~~~~~~~~

The cost matrix :math:`E_{ij} = p_i \cdot x_j` (cost of bundle j at prices i):

.. code-block:: python

   def compute_cost_matrix(prices, quantities):
       """E[i,j] = p_i · x_j"""
       T = prices.shape[0]
       E = np.zeros((T, T))
       for i in range(T):
           for j in range(T):
               E[i, j] = np.dot(prices[i], quantities[j])
       return E

Revealed Preference Relations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def compute_rp_relations(prices, quantities):
       """Compute R (weak) and P (strict) revealed preference."""
       E = compute_cost_matrix(prices, quantities)
       expenditures = np.diag(E)  # e_i = p_i · x_i

       R = expenditures[:, None] >= E  # e_i >= p_i · x_j
       P = expenditures[:, None] > E   # e_i > p_i · x_j

       return R, P

GARP Test
~~~~~~~~~

.. code-block:: python

   def transitive_closure(R):
       """Floyd-Warshall algorithm for R*."""
       R_star = R.copy()
       T = R.shape[0]
       for k in range(T):
           for i in range(T):
               for j in range(T):
                   R_star[i,j] = R_star[i,j] or (R_star[i,k] and R_star[k,j])
       return R_star

   def check_garp(prices, quantities):
       """Check GARP: If x_i R* x_j, then NOT x_j P x_i."""
       R, P = compute_rp_relations(prices, quantities)
       R_star = transitive_closure(R)
       T = R.shape[0]

       for i in range(T):
           for j in range(T):
               if i != j and R_star[i,j] and P[j,i]:
                   return False  # Violation!
       return True

AEI via Binary Search
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def compute_aei(prices, quantities, tol=1e-6):
       """Find largest e in [0,1] where scaled GARP holds."""

       def garp_at_e(e):
           E = compute_cost_matrix(prices, quantities)
           expenditures = np.diag(E)
           R_e = (e * expenditures[:, None]) >= E
           R_star = transitive_closure(R_e)
           P = expenditures[:, None] > E
           T = E.shape[0]
           for i in range(T):
               for j in range(T):
                   if i != j and R_star[i,j] and P[j,i]:
                       return False
           return True

       lo, hi = 0.0, 1.0
       while hi - lo > tol:
           mid = (lo + hi) / 2
           if garp_at_e(mid):
               lo = mid
           else:
               hi = mid
       return lo


Part 5: Analyzing One User
--------------------------

Testing the implementation on a single user:

.. code-block:: python

   # Pick a user with enough data
   test_user = df['user_id'].value_counts().index[0]

   prices, quantities, periods = build_user_matrices(
       df, test_user, price_oracle, categories
   )

   print(f"User: {test_user}")
   print(f"Observations: {len(periods)}")
   print(f"Categories: {quantities.shape[1]}")

   # Run tests
   garp_pass = check_garp(prices, quantities)
   aei = compute_aei(prices, quantities)

   print(f"\nResults:")
   print(f"  GARP: {'PASS' if garp_pass else 'FAIL'}")
   print(f"  AEI: {aei:.4f}")

   if aei >= 0.95:
       print("\n  Interpretation: Highly consistent")
   elif aei >= 0.80:
       print("\n  Interpretation: Moderately consistent")
   else:
       print("\n  Interpretation: Significant departures from consistency")


Part 6: Scaling to All Users
----------------------------

Analyzing the full dataset:

.. code-block:: python

   from tqdm import tqdm

   def analyze_user(df, user_id, price_oracle, categories, min_periods=5):
       """Full RP analysis for one user."""
       prices, quantities, periods = build_user_matrices(
           df, user_id, price_oracle, categories
       )

       if len(periods) < min_periods:
           return None

       return {
           'user_id': user_id,
           'n_periods': len(periods),
           'garp_pass': check_garp(prices, quantities),
           'aei': compute_aei(prices, quantities),
           'total_spend': (prices * quantities).sum()
       }

   # Analyze all users
   results = []
   for user_id in tqdm(df['user_id'].unique()):
       result = analyze_user(df, user_id, price_oracle, categories)
       if result:
           results.append(result)

   results_df = pd.DataFrame(results)
   print(f"Analyzed {len(results_df)} users")

Aggregate Results
~~~~~~~~~~~~~~~~~

.. code-block:: python

   print("=" * 50)
   print("AGGREGATE RESULTS")
   print("=" * 50)

   print(f"\nGARP Pass Rate: {results_df['garp_pass'].mean()*100:.1f}%")
   print(f"\nAEI Distribution:")
   print(f"  Mean:   {results_df['aei'].mean():.3f}")
   print(f"  Median: {results_df['aei'].median():.3f}")
   print(f"  AEI ≥ 0.95: {(results_df['aei'] >= 0.95).mean()*100:.1f}%")
   print(f"  AEI < 0.70: {(results_df['aei'] < 0.70).mean()*100:.1f}%")


Part 7: Comparison to Benchmarks
--------------------------------

Comparison to controlled experiments.

.. list-table:: E-Commerce vs. CKMS (2014) Lab Experiments
   :header-rows: 1
   :widths: 40 30 30

   * - Metric
     - Open E-Commerce
     - CKMS Lab
   * - GARP pass rate
     - 8-13%
     - ~45%
   * - Mean AEI
     - 0.85
     - 0.88
   * - Median AEI
     - 0.89
     - 0.95
   * - AEI ≥ 0.95
     - ~25%
     - ~45%

Lower consistency in field data reflects:

1. Category aggregation creates artificial violations
2. Preferences may change over 5 years
3. Multiple household members on one account
4. Gift purchases violate individual utility
5. Stockpiling during sales


Part 8: Heterogeneity Analysis
------------------------------

Examining correlations between consistency and user characteristics:

.. code-block:: python

   import matplotlib.pyplot as plt

   fig, axes = plt.subplots(1, 3, figsize=(15, 4))

   # AEI vs observations
   axes[0].scatter(results_df['n_periods'], results_df['aei'], alpha=0.3, s=10)
   axes[0].set_xlabel('Active Months')
   axes[0].set_ylabel('AEI')
   axes[0].set_title('AEI vs Active Months')

   # AEI vs spending
   axes[1].scatter(np.log10(results_df['total_spend']), results_df['aei'], alpha=0.3, s=10)
   axes[1].set_xlabel('Log10(Total Spend)')
   axes[1].set_ylabel('AEI')
   axes[1].set_title('AEI vs Total Spend')

   # AEI histogram
   axes[2].hist(results_df['aei'], bins=50, edgecolor='black', alpha=0.7)
   axes[2].axvline(results_df['aei'].mean(), color='red', linestyle='--', label='Mean')
   axes[2].set_xlabel('AEI')
   axes[2].set_ylabel('Count')
   axes[2].set_title('AEI Distribution')
   axes[2].legend()

   plt.tight_layout()
   plt.savefig('heterogeneity_analysis.png', dpi=150)

.. list-table:: Heterogeneity Correlations
   :header-rows: 1
   :widths: 40 30 30

   * - Variable
     - Correlation
     - Interpretation
   * - Active months
     - -0.15 to -0.25
     - More data reveals more inconsistency
   * - Log(total spend)
     - +0.05 to +0.10
     - Weak positive correlation
   * - Categories purchased
     - -0.10 to -0.20
     - Broader baskets harder to rationalize


Part 9: Power Analysis
----------------------

The Bronars test assesses whether GARP has discriminative power.

The Bronars Test
~~~~~~~~~~~~~~~~

.. code-block:: python

   def bronars_power(prices, n_simulations=500):
       """Fraction of random behaviors that violate GARP."""
       T, n = prices.shape
       violations = 0

       for _ in range(n_simulations):
           # Random budget allocation via Dirichlet
           random_qty = np.zeros((T, n))
           for t in range(T):
               shares = np.random.dirichlet(np.ones(n))
               budget = np.median(prices[t]) * n
               random_qty[t] = (shares * budget) / prices[t]

           if not check_garp(prices, random_qty):
               violations += 1

       return violations / n_simulations

   # Test on sample user
   power = bronars_power(prices)
   print(f"Bronars power: {power:.2f}")

**Interpretation:**

- Power > 0.90: Excellent—random behavior almost always fails GARP
- Power 0.70-0.90: Good—results are meaningful
- Power < 0.70: Caution—test has low discriminatory power

For the Open E-Commerce dataset, power typically exceeds 0.90.


Part 10: Using PyRevealed
-------------------------

The above shows what happens under the hood. In practice, use PyRevealed:

.. code-block:: python

   from pyrevealed import BehaviorLog, validate_consistency, compute_integrity_score

   # Create BehaviorLog
   log = BehaviorLog(
       cost_vectors=prices,
       action_vectors=quantities,
       user_id=f"user_{user_id}"
   )

   # Run analysis
   is_consistent = validate_consistency(log)
   result = compute_integrity_score(log)

   print(f"GARP: {'PASS' if is_consistent else 'FAIL'}")
   print(f"AEI: {result.efficiency_index:.4f}")


Running the Full Pipeline
-------------------------

Use the provided scripts for the complete analysis:

.. code-block:: bash

   # Download data (299 MB)
   python datasets/open_ecommerce/download.py

   # Full analysis (all users)
   python datasets/open_ecommerce/run_all.py

   # Quick test (500 users)
   python datasets/open_ecommerce/run_all.py --quick

This generates:

- ``output/user_results.csv`` — Full results
- ``output/aei_distribution.png`` — Histogram
- ``output/spend_vs_aei.png`` — Scatter plot


Key Takeaways
-------------

1. **E-commerce data requires assumptions** — category aggregation, temporal
   aggregation, price imputation all affect results

2. **Behavior is approximately consistent** — mean AEI ~0.85 means 85%
   of behavior can be explained by utility maximization

3. **Lower consistency than lab experiments** — reflects data complexity
   and assumption violations

4. **More data = more violations** — consumers with longer histories show
   lower AEI, suggesting measurement issues

5. **Power analysis matters** — verify your test can distinguish consistent
   from random behavior


Exercises
---------

1. **Preference Recovery:** Use ``fit_latent_values()`` to recover utility
   levels for a consistent consumer.

2. **COVID Effect:** Split data into pre-COVID (before March 2020) and
   during-COVID. Compare AEI distributions.

3. **Category Analysis:** Which product categories drive the most violations?

4. **Alternative Aggregation:** Try weekly instead of monthly periods.
   How do results change?


References
----------

**Theory:**

- Afriat, S. (1967). The construction of utility functions from expenditure
  data. *International Economic Review*, 8(1), 67-77.

- Varian, H. (1982). The nonparametric approach to demand analysis.
  *Econometrica*, 50(4), 945-973.

**Empirical Studies:**

- Choi, S., Kariv, S., Müller, W., & Silverman, D. (2014). Who is (more)
  rational? *American Economic Review*, 104(6), 1518-1550.

- Bronars, S. (1987). The power of nonparametric tests of preference
  maximization. *Econometrica*, 55(3), 693-698.

**Data:**

- Goldfarb, A., Tucker, C., & Wang, Y. (2024). Open e-commerce 1.0.
  *Scientific Data*, 11(527).
