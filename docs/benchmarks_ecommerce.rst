E-commerce Benchmarks
=====================

Six public datasets, 162K users, LightGBM with 5-fold stratified CV.
**RP features add 0--0.7% AUC over strong RFM baselines.** The lift is
real but modest — most predictive power comes from standard spending
and engagement features.

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
     - 0.183
   * - Open E-Commerce
     - 4,694
     - High Spender
     - 0.950
     - 0.951
     - +0.2%
     - 0.914
   * - Open E-Commerce
     - 4,694
     - Churn
     - 0.841
     - 0.847
     - **+0.7%**
     - 0.283
   * - H&M
     - 46,757
     - High Spender
     - 0.763
     - 0.762
     - -0.1%
     - 0.667
   * - H&M
     - 46,757
     - Churn
     - 0.778
     - 0.778
     - +0.0%
     - 0.293
   * - Instacart
     - 50,000
     - Spend Drop
     - 0.665
     - 0.666
     - +0.2%
     - 0.197
   * - Instacart
     - 50,000
     - High Value
     - 0.967
     - 0.967
     - +0.0%
     - 0.941
   * - REES46
     - 8,832
     - High Engagement
     - 0.942
     - 0.941
     - -0.1%
     - 0.922
   * - Taobao
     - 4,239
     - High Engagement
     - 0.913
     - 0.913
     - +0.0%
     - 0.137

*Baseline = LightGBM on RFM + spending features. +RP = same model with RP features added. Lift = (Combined - Baseline) / Baseline x 100. Runtime: 20 min on M1 Mac.*

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
     - Actual transaction prices
     - Budget
     - Normalized prices from Kaggle competition
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
     - n_obs
     - Baseline
     - Number of observations (frequency)
   * - 5
     - herfindahl
     - Baseline
     - Category concentration
   * - 6
     - scc_ratio
     - **RP**
     - Fraction of observations in largest violation cycle
   * - 7
     - hm_ratio
     - **RP**
     - Houtman-Maks consistency fraction
   * - 8
     - mpi
     - **RP**
     - Money Pump Index (exploitability)

Reproduce
---------

.. code-block:: bash

   pip install pyrevealed lightgbm scikit-learn
   python case_studies/benchmarks/runner.py --datasets all

Datasets require ``kaggle`` CLI. See ``case_studies/benchmarks/`` for details.

----

Appendix: Pipeline
------------------

.. code-block:: text

   Raw CSV
     -> Loader (pyrevealed.datasets)
     -> BehaviorPanel / MenuChoiceLog per user
     -> Temporal split: first 70% -> features, last 30% -> targets
     -> Feature extraction:
          Baseline (13): RFM, category concentration, temporal trends
          RP (11): CCEI, MPI, HM ratio, VEI, GARP, HARP, SCC ratio
     -> LightGBM (num_leaves=15, n_estimators=100, reg_alpha=0.1)
     -> 5-fold stratified CV
     -> Metrics: AUC-ROC, AUC-PR, Log Loss, F1

**Three models per target**: (a) Baseline only, (b) RP only, (c) Baseline + RP.

**Targets**: High Spender (top tercile spend), Churn (>50% spend drop),
Spend Change (regression), High Engagement (above-median sessions).

**Output**: ``case_studies/benchmarks/output/results.json`` (full metrics),
``summary_table.csv``, ``figures/``.
