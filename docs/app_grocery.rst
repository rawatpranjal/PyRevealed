Application: Grocery Scanner Data
=================================

Score household shopping behavior for economic rationality using
revealed preference theory on loyalty-card scanner data.

.. contents:: On this page
   :local:
   :depth: 2

Introduction
------------

Every week, millions of households make grocery purchases across product
categories at posted prices. A natural question: are these choices consistent
with *any* utility function? If a household buys more beef when it's expensive
and less when it's cheap, that's a revealed preference violation --- no
well-behaved utility function can explain it.

Dean & Martin (2016) applied GARP to 977 households' grocery scanner data
across 38 product categories, finding marked heterogeneity in rationality
scores correlated with demographics. Echenique, Lee & Shum (2011) used the
same type of data to compute the Money Pump Index, quantifying how much
money an arbitrageur could extract from inconsistent shoppers.

**What you'll learn:**

- How to map grocery transactions into revealed preference data
- The formal GARP test and three goodness-of-fit scores (CCEI, MPI, HM)
- Exploratory analysis of real scanner data (Dunnhumby, 2,222 households)
- How to segment customers by rationality and interpret the scores

**Companion script:** ``applications/01_grocery_scanner.py``

Formal Setup
------------

Notation
~~~~~~~~

A household makes :math:`T` shopping trips. On trip :math:`t`, they face
prices :math:`p^t \in \mathbb{R}^n_{++}` across :math:`n` product categories
and purchase quantities :math:`x^t \in \mathbb{R}^n_+`. Their expenditure is
:math:`p^t \cdot x^t`.

The **budget set** at observation :math:`t` is:

.. math::

   B(p^t, m^t) = \{ x \in \mathbb{R}^n_+ : p^t \cdot x \leq p^t \cdot x^t \}

Bundle :math:`x^t` is **directly revealed preferred** to :math:`x^s` (written
:math:`x^t \, R_0 \, x^s`) if :math:`x^s` was affordable when :math:`x^t`
was chosen:

.. math::

   x^t \, R_0 \, x^s \iff p^t \cdot x^t \geq p^t \cdot x^s

The preference is **strict** (:math:`x^t \, P_0 \, x^s`) if the inequality
is strict. Let :math:`R^*` denote the transitive closure of :math:`R_0`.

GARP
~~~~

The **Generalized Axiom of Revealed Preference** (Varian, 1982) states:

.. math::

   \text{GARP: } \quad x^t \, R^* \, x^s \implies \lnot (x^s \, P_0 \, x^t)

In words: if :math:`x^t` is transitively revealed preferred to :math:`x^s`,
then :math:`x^s` cannot be strictly directly revealed preferred to :math:`x^t`.
GARP holds if and only if there exists a non-satiated utility function
rationalizing the data (Afriat's theorem, 1967).

Scores
~~~~~~

When GARP fails, three scores measure *how badly*:

**CCEI** (Afriat Efficiency Index): the largest :math:`e \in [0,1]` such that
GARP holds when budgets are relaxed by factor :math:`e`:

.. math::

   \text{CCEI} = \sup \left\{ e \in [0,1] : e \cdot (p^t \cdot x^t) \geq p^t \cdot x^s \text{ implies no violations} \right\}

Interpretation: :math:`1 - \text{CCEI}` is the fraction of income wasted by
choosing inconsistently. CCEI = 1 means perfectly rational.

**MPI** (Money Pump Index, Echenique et al. 2011): the maximum fraction of
expenditure an arbitrageur could extract by exploiting preference cycles:

.. math::

   \text{MPI} = \max_{\text{cycle } C} \frac{\sum_{(t,s) \in C} \left( p^t \cdot x^t - p^t \cdot x^s \right)}{\sum_{(t,s) \in C} p^t \cdot x^t}

**HM** (Houtman-Maks, 1985): the minimum fraction of observations to remove
to restore GARP consistency:

.. math::

   \text{HM} = \min \left\{ \frac{|S|}{T} : \text{removing observations } S \text{ makes data GARP-consistent} \right\}

Data
----

This application uses the **Dunnhumby "The Complete Journey"** dataset:
2,500 households tracked over 2 years (104 weeks) across 10 staple
product categories.

Loading
~~~~~~~

.. code-block:: python

   import sys
   from pathlib import Path

   # Load Dunnhumby pipeline
   sys.path.insert(0, str(Path("dunnhumby")))
   from data_loader import load_filtered_data
   from price_oracle import get_master_price_grid
   from session_builder import build_all_sessions

   filtered = load_filtered_data()
   price_grid = get_master_price_grid(filtered)
   households = build_all_sessions(filtered, price_grid)

   print(f"Households: {len(households)}")
   print(f"Price grid: {price_grid.shape}")  # (104 weeks, 10 categories)

.. code-block:: text

   Households: 2222
   Price grid: (104, 10)

.. note::

   The Dunnhumby dataset requires a Kaggle download. Run
   ``dunnhumby/download_data.sh`` first. If unavailable, the companion
   script falls back to simulated data.

EDA: Product Categories
~~~~~~~~~~~~~~~~~~~~~~~

The 10 product categories and their average weekly prices:

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 15

   * - Category
     - Mean Price ($)
     - Std ($)
     - Purchase Freq (%)
   * - Soft Drinks
     - 3.42
     - 1.08
     - 78%
   * - Fluid Milk
     - 2.89
     - 0.94
     - 72%
   * - Bread/Rolls
     - 2.51
     - 0.83
     - 68%
   * - Cheese
     - 4.67
     - 1.52
     - 61%
   * - Bag Snacks
     - 3.28
     - 1.15
     - 55%
   * - Soup
     - 1.89
     - 0.72
     - 43%
   * - Yogurt
     - 3.15
     - 1.01
     - 52%
   * - Beef
     - 6.24
     - 2.18
     - 38%
   * - Frozen Pizza
     - 4.53
     - 1.44
     - 35%
   * - Lunchmeat
     - 3.98
     - 1.31
     - 41%

EDA: Household Activity
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np

   obs_counts = [h.num_observations for h in households.values()]
   print(f"Active weeks per household:")
   print(f"  Min: {min(obs_counts)}  Median: {int(np.median(obs_counts))}"
         f"  Max: {max(obs_counts)}")
   print(f"  Mean: {np.mean(obs_counts):.1f}  Std: {np.std(obs_counts):.1f}")

.. code-block:: text

   Active weeks per household:
     Min: 10  Median: 42  Max: 104
     Mean: 43.2  Std: 25.8

Households with more active weeks provide more data for GARP testing but
are also more likely to show violations (more chances for inconsistency).

Algorithm
---------

The GARP test proceeds in three steps:

.. code-block:: text

   GARP-TEST(prices P[T×n], quantities X[T×n]):
   ─────────────────────────────────────────────
   1. BUILD PREFERENCE GRAPH                    O(T²)
      For each pair (t,s):
        R₀[t,s] ← (pₜ·xₜ ≥ pₜ·xₛ)   // weak RP
        P₀[t,s] ← (pₜ·xₜ > pₜ·xₛ)   // strict RP

   2. TRANSITIVE CLOSURE                        O(T³)
      R* ← Floyd-Warshall(R₀)
      // R*[t,s] = 1 iff there exists a chain
      // t R₀ k₁ R₀ k₂ ... R₀ s

   3. CHECK FOR VIOLATIONS                      O(T²)
      For each pair (t,s):
        if R*[t,s] AND P₀[s,t]:
          Record violation (t → ... → s → t)
      Return: is_consistent, violations

   Total: O(T³) dominated by Floyd-Warshall

For CCEI, binary search over :math:`T^2` candidate efficiency ratios
:math:`e = (p^t \cdot x^s) / (p^t \cdot x^t)`, running GARP at each.
This yields exact CCEI in :math:`O(T^3 \log T)` time.

Pipeline Walkthrough
--------------------

Single household
~~~~~~~~~~~~~~~~

.. code-block:: python

   from pyrevealed import BehaviorLog, validate_consistency
   from pyrevealed import compute_integrity_score, compute_confusion_metric
   from pyrevealed.algorithms.mpi import compute_houtman_maks_index

   # Pick one household
   hh = list(households.values())[0]
   log = hh.behavior_log

   print(f"Household: {log.user_id}")
   print(f"Observations: {log.num_records}")
   print(f"Goods: {log.num_features}")

.. code-block:: text

   Household: household_457
   Observations: 63
   Goods: 10

Step 1 --- GARP test:

.. code-block:: python

   garp = validate_consistency(log)
   print(f"GARP consistent: {garp.is_consistent}")
   print(f"Violation cycles: {len(garp.violations)}")

.. code-block:: text

   GARP consistent: False
   Violation cycles: 847

Step 2 --- CCEI (efficiency score):

.. code-block:: python

   ccei = compute_integrity_score(log, tolerance=1e-4)
   print(f"CCEI: {ccei.efficiency_index:.4f}")
   print(f"Budget waste: {(1 - ccei.efficiency_index) * 100:.1f}%")

.. code-block:: text

   CCEI: 0.8325
   Budget waste: 16.8%

Step 3 --- MPI (exploitability):

.. code-block:: python

   mpi = compute_confusion_metric(log)
   print(f"MPI: {mpi.mpi_value:.4f}")

.. code-block:: text

   MPI: 0.2140

Step 4 --- Houtman-Maks (outlier fraction):

.. code-block:: python

   hm = compute_houtman_maks_index(log)
   print(f"Observations to remove: {hm.removed_count}/{log.num_records}")
   print(f"Fraction removed: {hm.fraction:.3f}")

.. code-block:: text

   Observations to remove: 15/63
   Fraction removed: 0.238

.. note::

   This household's CCEI of 0.83 means that if we shrink each budget by
   17%, all choices become rationalizable. The MPI of 0.21 means an
   arbitrageur could extract ~21% of total expenditure by cycling trades.

Batch Analysis
--------------

Scoring all 2,222 households:

.. code-block:: python

   from pyrevealed import BehaviorLog, validate_consistency, compute_integrity_score
   from pyrevealed import compute_confusion_metric
   from pyrevealed.algorithms.mpi import compute_houtman_maks_index

   results = []
   for hh_key, hh_data in households.items():
       log = hh_data.behavior_log
       garp = validate_consistency(log)

       if garp.is_consistent:
           ccei, mpi_val, hm_val = 1.0, 0.0, 0.0
       else:
           ccei = compute_integrity_score(log, tolerance=1e-4).efficiency_index
           mpi_val = compute_confusion_metric(log).mpi_value
           hm_val = compute_houtman_maks_index(log).fraction

       results.append({
           "hh": hh_key, "T": log.num_records,
           "garp": garp.is_consistent, "ccei": ccei,
           "mpi": mpi_val, "hm": hm_val,
       })

Score distributions across the panel:

.. list-table::
   :header-rows: 1
   :widths: 15 12 12 12 12 12 12

   * - Metric
     - Mean
     - Std
     - P10
     - P25
     - P50
     - P90
   * - CCEI
     - 0.839
     - 0.105
     - 0.698
     - 0.766
     - 0.852
     - 0.960
   * - MPI
     - 0.225
     - 0.112
     - 0.078
     - 0.142
     - 0.225
     - 0.371
   * - HM removed
     - 0.224
     - 0.118
     - 0.080
     - 0.143
     - 0.222
     - 0.369

Key statistics:

- **4.5%** of households are perfectly GARP-consistent (CCEI = 1.0)
- Mean CCEI = 0.839, meaning the average household wastes ~16% of budget
- The distribution is left-skewed: most households are moderately rational

Temporal Panel Analysis
-----------------------

Beyond a single snapshot, tracking CCEI over time reveals household
*dynamics*: who is consistently rational, who is deteriorating, and
who crosses between segments.

Rolling-window CCEI
~~~~~~~~~~~~~~~~~~~

For each household, compute CCEI over a sliding 20-week window:

.. code-block:: python

   def compute_rolling_ccei(log, window=20, step=5):
       results = []
       for start in range(0, log.num_records - window + 1, step):
           window_log = BehaviorLog(
               cost_vectors=log.cost_vectors[start:start+window],
               action_vectors=log.action_vectors[start:start+window],
           )
           ccei = compute_integrity_score(window_log).efficiency_index
           results.append((start, ccei))
       return results

Trajectory classification
~~~~~~~~~~~~~~~~~~~~~~~~~

Classify each household by the shape of their CCEI trajectory:

.. list-table::
   :header-rows: 1
   :widths: 18 12 40

   * - Trajectory
     - Criteria
     - Interpretation
   * - Stable
     - std < 0.03
     - Preferences don't change; reliable customer
   * - Improving
     - slope > +0.005
     - Learning better shopping habits over time
   * - Deteriorating
     - slope < -0.005
     - Choices becoming more erratic; possible life change
   * - Volatile
     - std > 0.03, |slope| < 0.005
     - Fluctuating consistency; context-dependent shopper

On Dunnhumby data (households with 30+ weeks):

.. code-block:: text

   Trajectory          N       %  Mean CCEI   Std CCEI  Avg Slope
   ────────────────  ───── ─────── ────────── ────────── ──────────
   stable              38%          0.897      0.010    -0.006
   improving           19%          0.838      0.070    +0.023
   deteriorating       23%          0.871      0.074    -0.044
   volatile            19%          0.931      0.049    +0.002

Crossover detection
~~~~~~~~~~~~~~~~~~~

Users whose first-half and second-half CCEI differ by more than 0.05
represent **crossovers** --- behavioral regime changes worth flagging:

.. code-block:: text

   HH-3       deteriorating    1st half: 0.969  →  2nd half: 0.729  (Δ = -0.24)
   HH-14      deteriorating    1st half: 0.947  →  2nd half: 0.775  (Δ = -0.17)
   HH-17      improving        1st half: 0.639  →  2nd half: 0.780  (Δ = +0.14)

A household dropping from CCEI 0.97 to 0.73 warrants investigation:
account sharing, life disruption, or response to a pricing change.

Interpretation
--------------

Customer segmentation
~~~~~~~~~~~~~~~~~~~~~

CCEI scores naturally segment customers into behavioral tiers:

.. list-table::
   :header-rows: 1
   :widths: 20 15 40

   * - Tier
     - CCEI Range
     - Interpretation
   * - Consistent optimizers
     - 0.95 -- 1.00
     - Price-sensitive, respond predictably to promotions
   * - Noisy maximizers
     - 0.80 -- 0.95
     - Preferences exist but imprecisely revealed
   * - Erratic shoppers
     - < 0.80
     - Choices hard to rationalize; may benefit from curation

Business applications
~~~~~~~~~~~~~~~~~~~~~

1. **Targeted pricing**: High-CCEI customers respond predictably to price
   changes. Choi et al. (2014) found a 1 SD increase in CCEI associates
   with 15--19% more household wealth.

2. **Fraud/anomaly detection**: A sudden CCEI drop for a previously
   consistent household signals account sharing or suspicious activity.

3. **Promotion evaluation**: If a new promotion strategy *increases*
   average CCEI, customers are making more coherent choices --- a sign
   of reduced cognitive load.

4. **Welfare measurement**: MPI quantifies the dollar value of welfare
   losses from inconsistent choices, enabling cost-benefit analysis
   of interventions (loyalty programs, simplified displays).

Limitations
~~~~~~~~~~~

- GARP tests existence of *any* utility function, not a specific one.
  A high CCEI doesn't mean the consumer is "smart" --- just consistent.
- CCEI is domain-specific: a consumer rational about groceries may be
  irrational about electronics (Chen et al. 2025, arXiv:2505.05275).
- With 10 product categories and 50+ observations, GARP has high power
  to detect violations even from slight preference noise.

References
----------

- Afriat, S. N. (1967). "The Construction of Utility Functions from
  Expenditure Data." *International Economic Review*, 8(1), 67--77.
  `doi:10.2307/2525382 <https://doi.org/10.2307/2525382>`_

- Varian, H. R. (1982). "The Nonparametric Approach to Demand Analysis."
  *Econometrica*, 50(4), 945--973.
  `doi:10.2307/1912771 <https://doi.org/10.2307/1912771>`_

- Dean, M. & Martin, D. (2016). "Measuring Rationality with the Minimum
  Cost of Revealed Preference Violations." *Review of Economics and
  Statistics*, 98(3), 524--534.
  `doi:10.1162/REST_a_00542 <https://doi.org/10.1162/REST_a_00542>`_

- Echenique, F., Lee, S., & Shum, M. (2011). "The Money Pump as a
  Measure of Revealed Preference Violations." *Journal of Political
  Economy*, 119(6), 1201--1223.
  `doi:10.1086/665011 <https://doi.org/10.1086/665011>`_

- Houtman, M. & Maks, J. (1985). "Determining All Maximal Data Subsets
  Consistent with Revealed Preference." *Kwantitatieve Methoden*, 19, 89--104.

- Choi, S., Kariv, S., Muller, W., & Silverman, D. (2014). "Who Is
  (More) Rational?" *American Economic Review*, 104(6), 1518--1550.
  `doi:10.1257/aer.104.6.1518 <https://doi.org/10.1257/aer.104.6.1518>`_

.. seealso::

   :doc:`theory_consistency` for the full mathematical treatment of GARP.
   :doc:`theory_efficiency` for proofs of CCEI and MPI properties.
   :doc:`api` for the complete function reference.
