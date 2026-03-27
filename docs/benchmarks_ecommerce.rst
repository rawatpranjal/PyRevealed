E-commerce Benchmarks
=====================

We test whether revealed preference features (CCEI, MPI, Houtman-Maks, VEI)
improve ML models on standard e-commerce prediction tasks. Six public
datasets, 113K users, LightGBM with 5-fold stratified CV. **Result: RP
features add 0--2% AUC over strong RFM baselines, with the clearest signal
on churn prediction (+0.7% Open E-Commerce, +1.8% Online Retail II).**
The lift is real but modest — most predictive power comes from standard
spending features. RP scores provide marginally independent signal about
behavioral consistency that spending statistics alone do not capture.

Results
-------

.. list-table::
   :header-rows: 1
   :widths: 18 7 15 12 12 8 10

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
     - 0.958
     - 0.959
     - +0.1%
     - 0.929
   * - Dunnhumby
     - 2,222
     - Churn
     - 0.748
     - 0.730
     - -2.4%
     - 0.254
   * - Open E-Commerce
     - 4,694
     - High Spender
     - 0.950
     - 0.951
     - +0.2%
     - 0.907
   * - Open E-Commerce
     - 4,694
     - Churn
     - 0.841
     - 0.847
     - +0.7%
     - 0.362
   * - Instacart
     - 50,000
     - Spend Drop
     - 0.665
     - 0.666
     - +0.2%
     - 0.210
   * - Instacart
     - 50,000
     - High Value
     - 0.967
     - 0.967
     - +0.0%
     - 0.932
   * - REES46
     - 8,832
     - High Engagement
     - 0.942
     - 0.941
     - -0.1%
     - 0.855
   * - Online Retail II
     - 443
     - Churn
     - 0.641
     - 0.652
     - +1.8%
     - 0.320
   * - H&M
     - 46,757
     - Churn
     - 0.778
     - 0.778
     - +0.0%
     - 0.293
   * - H&M
     - 46,757
     - High Spender
     - 0.763
     - 0.762
     - -0.1%
     - 0.667

*"Baseline" = LightGBM on RFM + spending features. "+RP" = same model with RP features added. Lift = (Combined − Baseline) / Baseline × 100.*

Price Assumptions
-----------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 15 45

   * - Dataset
     - Price Source
     - RP Type
     - Caveat
   * - Dunnhumby
     - Median price oracle
     - Budget
     - Shared prices across households (Dean & Martin 2016 standard)
   * - Open E-Commerce
     - Median per category/month
     - Budget
     - Forward-filled for missing periods
   * - Online Retail II
     - Actual transaction prices
     - Budget
     - Cleanest budget-based RP signal
   * - Instacart
     - Uniform ($1/unit)
     - Budget
     - **Quantity-only consistency** — CCEI/MPI do not test price-quantity tradeoffs
   * - REES46
     - N/A
     - Menu
     - Click → purchase sessions; no price assumptions
   * - H&M
     - Actual transaction prices
     - Budget
     - Normalized prices from competition data

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
     - spend_cv
     - Baseline
     - Spending variability (coefficient of variation)
   * - 4
     - herfindahl
     - Baseline
     - Category concentration (higher = fewer categories)
   * - 5
     - scc_ratio
     - **RP**
     - Fraction of observations in largest violation cycle
   * - 6
     - hm_ratio
     - **RP**
     - Houtman-Maks consistency fraction
   * - 7
     - mpi
     - **RP**
     - Money Pump Index (exploitability)
   * - 8
     - ccei
     - **RP**
     - Critical Cost Efficiency Index

Reproduce
---------

.. code-block:: bash

   pip install pyrevealed lightgbm scikit-learn
   python case_studies/benchmarks/runner.py --datasets all

Datasets download automatically via ``kaggle`` CLI. See ``case_studies/benchmarks/README.md``.

----

Appendix: Full Pipeline
-----------------------

**1. Data Ingestion**

Each dataset is loaded via a dedicated loader (``pyrevealed.datasets``) that
returns either a ``BehaviorPanel`` (budget data: prices × quantities per
user per time period) or a ``dict[str, MenuChoiceLog]`` (menu data: item
sets + chosen item per session per user). Loaders handle downloading,
filtering, category mapping, and temporal aggregation.

.. code-block:: text

   Raw CSV → Loader → BehaviorPanel / MenuChoiceLog per user

**2. Temporal Split (No Leakage)**

For each user, observations are split by time: first 70% → feature
extraction, last 30% → target computation. Features never see future data.

.. code-block:: text

   User observations: [obs_1, obs_2, ..., obs_T]
                       |← 70% features →|← 30% targets →|

**3. Feature Extraction**

Two feature sets per user:

*Baseline features* (13 features, no RP library needed):

- **RFM**: n_obs, total_spend, mean_spend, std_spend, max_spend, min_spend
- **Category**: herfindahl, top_category_share, n_active_categories
- **Temporal**: spend_slope, spend_cv, mean_abs_spend_change
- **Volume**: mean_basket_size

*RP features* (11 features, via ``Engine.analyze_arrays()`` batch API):

- **Consistency**: is_garp, is_harp
- **Efficiency**: ccei, mpi, vei_mean, vei_min
- **Noise**: hm_ratio (Houtman-Maks fraction), violation_density, n_violations
- **Structure**: max_scc, scc_ratio

For menu datasets, RP features come from ``Engine.analyze_menus()``:
is_sarp, is_warp, hm_ratio, sarp_violation_density, warp_violation_density,
scc_ratio.

**4. Target Construction**

From the held-out 30% of each user's observations:

- **High Spender**: Top tercile of test-period total spend (binary)
- **Churn**: Mean spend dropped >50% from train to test (binary)
- **Spend Change**: Difference in mean spend (regression)
- **High Engagement** (menu): Above-median sessions in test (binary)

**5. Model Training**

LightGBM with regularization to prevent overfitting:

.. code-block:: python

   {
       "num_leaves": 15,
       "learning_rate": 0.05,
       "n_estimators": 100,
       "min_child_samples": 20,
       "reg_alpha": 0.1,
       "reg_lambda": 1.0,
   }

Three models per target: (a) Baseline only, (b) RP only, (c) Baseline + RP.

**6. Evaluation**

- **Out-of-sample**: 5-fold stratified CV (splits users, not time)
- **In-sample**: Train on all, predict on all (overfitting diagnostic)
- **Metrics**: AUC-ROC, AUC-PR (average precision), Log Loss, F1, R², RMSE
- **Seed**: 42 for reproducibility

**7. Output**

.. code-block:: text

   case_studies/benchmarks/output/
   ├── results.json          # Full metrics per dataset × target
   ├── summary_table.csv     # One-row-per-task summary
   └── figures/
       ├── auc_comparison.png
       └── feature_importance.png
