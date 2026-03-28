E-commerce Benchmarks
=====================

*Last updated: 2026-03-28. Instacart now uses aisle-level menu construction.
H&M results with per-customer realized prices (v0.5.8).*

Eight public datasets, 217K users, 42 RP features. CatBoost (H&M) or LightGBM
(others) with 80/20 user holdout + bootstrap CI. Results split by data type:

- **Menu datasets**: RP features are **competitive with baselines**. Taobao
  RP-only AUC (0.925) beats the engagement baseline (0.913). Graph features
  (``menu_transitivity``, ``menu_pref_density``) and choice entropy carry
  real signal that engagement stats miss.
- **Budget datasets**: RP adds ~0% marginal lift over strong RFM baselines.
  Spending features already capture the signal; CCEI/MPI are correlated.
- **Instacart**: Aisle-level menu construction with trailing-3 menus.
  RP features show real structure (83.8% SARP violations) but add near-zero
  lift over baseline — consistent with habit-heavy grocery reordering.

All targets use top-tercile thresholds for consistency.

Assumptions
-----------

**Budget datasets** require prices. Dunnhumby and Open E-Commerce use global
median prices per category per period, shared across all households — individual
price exposure (coupons, regional variation) is not captured (Dean & Martin 2016).
H&M uses each customer's own average paid price per product group per month;
unpurchased groups are imputed via period-group median, then group median, then
global median. All budget datasets aggregate to 10--134 categories, so
within-category substitution is invisible. Dunnhumby's 10 commodity groups
capture ~$19/week of a ~$100--150 weekly grocery budget.

**Menu datasets** have no prices. REES46 uses server-defined session IDs
(gold standard), median menu size ~5 items. Taobao uses 30-minute inactivity
gaps to define session boundaries (84% of inter-event gaps < 30 min), median
menu size 4 items. Tenrec uses click-to-like windows with positional feedback
tracking, median ~5 clicks between likes; menus reflect algorithmic
recommendations, not organic browsing. For all three, menus contain only items
the user viewed or clicked — items shown but not engaged with are invisible.

**Instacart** is treated as menu-choice, not budget (no prices in raw data).
Observation = user × order × aisle with exactly one reordered SKU. Menu =
trailing-3 order products in the same aisle (familiarity set). Filters: menu
size ≥ 2, (user, aisle) pairs with ≥ 3 valid events. This yields 4.5M events
from 120K users across 715K user-aisle pairs. The data is habit-heavy: 58.6%
of repeated user-aisle pairs never switch products, and 83.8% of users have
SARP violations. RP features show real graph structure but near-zero predictive
lift, consistent with reorder-dominated behavior.

Results
-------

.. list-table::
   :header-rows: 1
   :widths: 18 8 15 10 10 8 8

   * - Dataset
     - N
     - Target
     - Baseline
     - +RP
     - Lift%
     - AUC-PR
   * - Dunnhumby
     - 2,222
     - High Spender
     - 0.960
     - 0.960
     - -0.0%
     - 0.931
   * - Dunnhumby
     - 2,222
     - Churn
     - 0.752
     - 0.740
     - -1.5%
     - 0.160
   * - Open E-Commerce
     - 4,694
     - High Spender
     - 0.950
     - 0.951
     - +0.0%
     - 0.914
   * - Open E-Commerce
     - 4,694
     - Churn
     - 0.846
     - 0.846
     - -0.0%
     - 0.297
   * - H&M
     - 46,757
     - High Spender
     - 0.784
     - 0.783
     - -0.1%
     - —
   * - H&M
     - 46,757
     - Future Spend (R²)
     - 0.337
     - 0.340
     - +0.003
     - —
   * - H&M
     - 46,757
     - Spend Change (R²)
     - 0.290
     - 0.295
     - +0.005
     - —
   * - Instacart
     - 50,000
     - Low Loyalty
     - 0.968
     - 0.969
     - +0.0%
     - —
   * - Instacart
     - 50,000
     - High Novelty
     - 0.765
     - 0.767
     - +0.3%
     - —
   * - REES46
     - 8,832
     - High Engagement
     - 0.996
     - 0.996
     - +0.0%
     - 0.966
   * - Taobao
     - 4,239
     - High Engagement
     - 0.913
     - **0.915**
     - **+0.2%**
     - 0.136
   * - Tenrec
     - 50,000
     - High Engagement
     - 0.993
     - 0.993
     - +0.0%
     - 0.983

*Baseline = LightGBM on 13 RFM features. +RP = same model with 42 RP features added (Engine scores + graph structure + utility recovery + choice entropy). Lift = (Combined - Baseline) / Baseline x 100.*

RP-Only Performance
~~~~~~~~~~~~~~~~~~~

RP features alone (no baseline) show where preference structure carries
independent signal:

.. list-table::
   :header-rows: 1
   :widths: 18 15 12 12

   * - Dataset
     - Target
     - RP-only
     - Baseline
   * - Taobao
     - High Engagement
     - **0.925**
     - 0.913
   * - Tenrec
     - High Engagement
     - **0.993**
     - 0.993
   * - REES46
     - High Engagement
     - 0.990
     - 0.996
   * - Instacart
     - High Novelty
     - 0.762
     - 0.765
   * - H&M
     - High Spender
     - 0.720
     - 0.784
   * - Open E-Commerce
     - Churn
     - 0.769
     - 0.846

On Taobao, RP-only **outperforms** the engagement baseline. Preference
graph transitivity and choice entropy capture patterns that session
counts and menu sizes miss.

Top Features
------------

Across all classification tasks (LightGBM feature importance, combined model):

.. list-table::
   :header-rows: 1
   :widths: 5 30 10 55

   * - #
     - Feature
     - Type
     - Interpretation
   * - 1
     - total_spend
     - Baseline
     - Total expenditure in training period
   * - 2
     - spend_slope
     - Baseline
     - Spending trend (increasing/decreasing)
   * - 3
     - n_sessions
     - Baseline
     - Number of menu presentations (menu datasets)
   * - 4
     - n_obs
     - Baseline
     - Number of observations (frequency)
   * - 5
     - mean_basket_size
     - Baseline
     - Average items per observation
   * - 6
     - max_choice_freq
     - Baseline
     - Most-chosen item frequency
   * - 7
     - util_gini
     - **RP**
     - Gini inequality of recovered Afriat utility values
   * - 8
     - spend_cv
     - Baseline
     - Spending variability (coefficient of variation)
   * - 9
     - items_per_session
     - Baseline
     - Item diversity per session
   * - 10
     - choice_entropy_norm
     - **RP**
     - Normalized Shannon entropy of choice distribution

Menu-dataset top features (Taobao + Tenrec):

.. list-table::
   :header-rows: 1
   :widths: 5 30 10 55

   * - #
     - Feature
     - Type
     - Interpretation
   * - 1
     - std_menu_size
     - Baseline
     - Variability of menu sizes
   * - 2
     - menu_transitivity
     - **RP**
     - Item graph transitivity ratio
   * - 3
     - n_sessions
     - Baseline
     - Number of sessions
   * - 4
     - menu_pref_density
     - **RP**
     - Item graph edge density
   * - 5
     - choice_entropy_norm
     - **RP**
     - Normalized choice entropy
   * - 8
     - menu_util_range
     - **RP**
     - Ordinal utility spread (max - min recovered rank)

Four of the top 8 menu features are RP-derived. Item graph
structure and choice entropy carry signal that engagement statistics do not capture.

Reproduce
---------

.. code-block:: bash

   pip install prefgraph lightgbm scikit-learn
   python case_studies/benchmarks/runner.py --datasets all

Datasets require ``kaggle`` CLI. See ``case_studies/benchmarks/`` for details.

----

Appendix: Pipeline
------------------

.. code-block:: text

   Raw CSV
     -> Loader (prefgraph.datasets)
     -> BehaviorPanel / MenuChoiceLog per user
     -> Temporal split: first 70% -> features, last 30% -> targets
     -> Feature extraction:
          Baseline (13): RFM, category concentration, temporal trends
          RP Engine (14): CCEI, MPI, HM, VEI, GARP, HARP, SCC, n_scc, harp_severity
          RP Extended (28): VEI distribution, utility recovery (Gini, CV),
              graph network (density, transitivity, cycles), MPI cycle costs,
              choice reversals, choice entropy, congruence, ordinal utility
     -> LightGBM (num_leaves=15, lr=0.03, reg_alpha=1.0, reg_lambda=5.0)
     -> 5-fold stratified CV
     -> Metrics: AUC-ROC, AUC-PR, Log Loss, F1

**Three models per target**: (a) Baseline only, (b) RP only, (c) Baseline + RP.

**Targets**: High Spender (top tercile spend), Future Spend (regression),
Spend Change (regression), High Engagement (top tercile sessions).

**Output**: ``case_studies/benchmarks/output/results.json`` (full metrics),
``summary_table.csv``, ``figures/``.

