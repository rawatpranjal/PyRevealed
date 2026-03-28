E-commerce Benchmarks
=====================

*Last updated: 2026-03-28. H&M results refreshed with per-customer realized
prices (v0.5.8).*

Seven public datasets, 167K users, 42 RP features. CatBoost (H&M) or LightGBM
(others) with 80/20 user holdout + bootstrap CI. Results split by data type:

- **Menu datasets**: RP features are **competitive with baselines**. Taobao
  RP-only AUC (0.925) beats the engagement baseline (0.913). Graph features
  (``menu_transitivity``, ``menu_pref_density``) and choice entropy carry
  real signal that engagement stats miss.
- **Budget datasets**: RP adds ~0% marginal lift over strong RFM baselines.
  Spending features already capture the signal; CCEI/MPI are correlated.

All targets use top-tercile thresholds for consistency.

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
     - Spend Drop
     - 0.666
     - 0.665
     - -0.0%
     - 0.197
   * - Instacart
     - 50,000
     - High Value
     - 0.966
     - 0.966
     - +0.0%
     - 0.941
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

Timing
------

Total wall time: **29 min** on M1 Mac (data on external USB drive via symlink).

.. list-table::
   :header-rows: 1
   :widths: 18 8 12 12

   * - Dataset
     - N
     - Data + Features
     - Model
   * - Dunnhumby
     - 2,222
     - 6s
     - 12s
   * - Open E-Commerce
     - 4,694
     - 16s
     - 12s
   * - H&M
     - 46,757
     - 1857s
     - 141s
   * - Instacart
     - 50,000
     - 92s
     - 27s
   * - REES46
     - 8,832
     - 709s
     - 4s
   * - Taobao
     - 4,239
     - 94s
     - 4s
   * - Tenrec
     - 50,000
     - 541s
     - 6s

REES46 and Taobao dominate wall time due to raw CSV parsing (110M and 100M
events). Pre-processing to Parquet eliminates this bottleneck entirely.

Parquet Speedup
~~~~~~~~~~~~~~~

Converting budget datasets to sorted Parquet and using the Rust Parquet
path (``Engine.analyze_parquet()``) shows **7--57x speedup** over
CSV load + in-memory analysis:

.. list-table::
   :header-rows: 1
   :widths: 20 10 12 12 10

   * - Dataset
     - N
     - CSV (s)
     - Parquet (s)
     - Speedup
   * - Dunnhumby
     - 2,222
     - 5.1
     - 0.7
     - 7x
   * - Open E-Commerce
     - 4,694
     - 13.3
     - 1.8
     - 7x
   * - H&M
     - 49,642
     - 151.2
     - 3.1
     - **49x**
   * - Instacart
     - 50,000
     - 90.1
     - 1.6
     - **57x**

*CSV = pandas load + Engine.analyze_arrays(). Parquet = Engine.analyze_parquet()
with Rust Parquet reader (end-to-end, includes I/O). One-time Parquet conversion
not included. Parquet files are 0.7--4.8 MB (zstd compressed).*

.. code-block:: python

   # One-time: convert CSV to sorted Parquet
   from prefgraph.io.parquet import prepare_parquet
   prepare_parquet("raw.csv", "users.parquet", user_col="user_id")

   # Fast repeated analysis
   results = rp.analyze("users.parquet", user_col="user_id",
                        cost_cols=["p1", "p2"], action_cols=["q1", "q2"])

Price Assumptions
-----------------

.. list-table::
   :header-rows: 1
   :widths: 18 20 12 50

   * - Dataset
     - Price Source
     - RP Type
     - Caveat
   * - Dunnhumby
     - Median oracle
     - Budget
     - Shared prices across households (Dean & Martin 2016)
   * - Open E-Commerce
     - Median per category/month
     - Budget
     - Forward-filled for missing periods
   * - H&M
     - Per-customer realized prices
     - Budget
     - Customer avg paid price if purchased; period-group median imputation otherwise
   * - Instacart
     - Uniform ($1/unit)
     - Budget
     - **Quantity-only consistency** — no price-quantity tradeoffs
   * - REES46
     - N/A
     - Menu
     - Click -> purchase sessions
   * - Taobao
     - N/A
     - Menu
     - Daily view -> purchase sessions (100M events)
   * - Tenrec
     - N/A
     - Menu
     - Click -> like sessions (493M events, NeurIPS 2022)

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

Data Assumptions
~~~~~~~~~~~~~~~~

Every dataset involves assumptions when mapping raw logs to RP inputs.
Here is what each loader does and what it cannot do.

**Budget datasets** (Dunnhumby, Open E-Commerce, H&M, Instacart):

- **Shared price oracle**: Dunnhumby and Open E-Commerce use global median
  prices per category per period, shared across all users. Individual price
  exposure (coupons, regional variation) is not captured. This follows
  Dean & Martin (2016).
- **Category-level aggregation**: 10--134 categories depending on dataset.
  Within-category substitution is invisible (e.g., "Dairy" includes milk,
  yogurt, and cheese at different prices). RP violations may reflect
  within-category product switching, not true preference inconsistency.
- **Instacart heuristic prices**: No prices in raw data. We assign per-aisle
  prices ($1.50--$14.00) via keyword matching on 134 aisle names. Yields
  $32/order average (real Instacart ~$35--50). Defensible but approximate.
- **H&M per-customer prices**: Each customer's own average paid price
  per product group per month. Unpurchased groups imputed via
  period-group median → group median → global median. Prices are
  normalized 0--1 (Kaggle); relative variation is real.
- **Dunnhumby coarse categories**: 10 commodity groups capture ~$19/week
  of a ~$100--150 weekly grocery budget. The RP analysis is valid within
  these categories but doesn't cover the full basket.

**Menu datasets** (REES46, Taobao, Tenrec):

- **Impression bias**: Menus contain only items the user viewed/clicked.
  Items shown but not clicked are invisible. The RP analysis is conditional
  on the user having engaged with these items, not the full catalog.
- **REES46 sessions**: Server-defined session IDs (gold standard).
  Median menu size ~5 items.
- **Taobao sessions**: 30-minute inactivity gap defines session boundaries
  (84% of inter-event gaps < 30 min). Median menu size 4 items.
- **Tenrec sessions**: Click-to-like windows with positional feedback
  tracking. Median ~5 clicks between likes. Menus reflect algorithmic
  recommendations, not organic browsing.
- **No budget constraint**: Menu datasets have no prices. Choices reveal
  preference orderings, not willingness-to-pay.
