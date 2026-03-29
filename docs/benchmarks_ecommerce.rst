Case Study 2: Predicting Customer Spend and Engagement
=======================================================

Across 11 datasets and 32 prediction targets, revealed preference features add near-zero marginal lift on average. Standard spend and engagement baselines already capture most of the signal. The one consistent exception is churn prediction on e-commerce data, where RP features detect declining purchase efficiency before spending drops.

Setup
-----

We test whether choice-consistency features improve predictions of spend, churn, engagement, and loyalty at the individual user level. Each user's purchase or click history is split in time. The early portion produces features. The later portion defines the prediction target. A regularized LightGBM and an L1-penalized logistic regression are each trained on a holdout sample and evaluated out of sample. All runs use SEED 42.

Features
^^^^^^^^

The baseline consists of 13 features that capture how much a user spends, how often they buy, and how concentrated their purchases are. These include total spend, number of observations, mean basket size, spend trend, category concentration, and recency. The revealed preference features consist of 42 measures that capture how consistently a user makes decisions. These include rationality scores such as CCEI and MPI, graph-structural measures such as transitivity and preference density, choice diversity measures such as entropy and reversal rate, and utility-recovery measures such as Afriat utility dispersion and efficiency trajectories.

Targets
^^^^^^^

Each dataset produces one or more prediction targets from the held-out future window. Spend Drop flags users whose average spending fell by more than half. High Spender identifies users in the top third of total expenditure. Future LTV and Spend Change predict continuous spend levels. High Engagement identifies the most active users by session count. Low Loyalty flags users whose choices became more dispersed over time. High CTR measures effective click-through rate on news impressions. All classification targets use top-tercile thresholds computed on the training set to prevent leakage.

.. _eco-results:

Results
-------

The table shows the percentage lift from adding 42 revealed preference features on top of a 13-feature baseline. Positive means RP helped. Negative means RP hurt. Near zero means no difference.

.. list-table::
   :header-rows: 1
   :widths: 12 16 7 10 10 10 10

   * - Dataset
     - Target
     - N
     - LGBM AUC-ROC
     - LGBM AUC-PR
     - Lasso AUC-ROC
     - Lasso AUC-PR
   * - Dunnhumby
     - Spend Drop
     - 2,222
     - +1.6%
     - -20.5%
     - -2.1%
     - -2.6%
   * - Dunnhumby
     - High Spender
     - 2,222
     - -0.1%
     - -0.2%
     - -0.1%
     - -0.1%
   * - Dunnhumby
     - Future LTV (R²)
     - 2,222
     - +0.9%
     -
     - -0.0%
     -
   * - Amazon
     - Spend Drop
     - 4,694
     - **+0.6%**
     - **+1.0%**
     - **+1.1%**
     - **+2.5%**
   * - Amazon
     - High Spender
     - 4,694
     - -0.1%
     - +0.2%
     - +0.0%
     - +0.0%
   * - H&M
     - High Spender
     - 46,757
     - -0.2%
     - -0.1%
     - +0.3%
     - +0.2%
   * - H&M
     - Spend Change (R²)
     - 46,757
     - +0.5%
     -
     - +0.0%
     -
   * - Instacart
     - Low Loyalty
     - 50,000
     - +0.1%
     - +0.1%
     - +0.1%
     - +0.2%
   * - REES46
     - Low Loyalty
     - 8,832
     - -0.0%
     - **+0.5%**
     - **+0.5%**
     - **+0.9%**
   * - Taobao
     - Engagement
     - 15,806
     - -0.0%
     - -0.0%
     - -0.0%
     - -0.2%
   * - Taobao BW
     - High Entropy
     - 590
     - -0.1%
     - +2.0%
     - +6.5%
     - +5.0%
   * - Tenrec
     - Engagement
     - 50,000
     - -0.0%
     - -0.0%
     - -0.0%
     - -0.0%
   * - MIND
     - High CTR
     - 5,091
     - -1.5%
     - -2.1%
     - -0.8%
     - -1.0%
   * - FINN
     - Low Loyalty
     - 1,869
     - +0.0%
     - +0.1%
     - -0.0%
     - -0.2%

All values are percentage lift from adding RP features to the baseline. Bold marks targets where all four columns agree on the direction. One representative target per dataset is shown. Full results for all 32 targets are in ``output/results.json`` and ``output/lasso_results.json``.

Findings
--------

Amazon Spend Drop is the one target where both models and both metrics agree that revealed preference features help. The lift ranges from 0.6 to 2.5 percent depending on the model and metric. On this target, per-observation budget efficiency captures declining rationality before spending actually drops.

REES46 Low Loyalty shows a smaller but consistent positive signal across three of four columns. On this target, choice entropy and preference graph structure detect users whose purchasing is becoming more dispersed.

Everywhere else the lift is within plus or minus one percent. The mean lift across all 32 targets is near zero. Revealed preference features are informative to the model but do not improve predictions when strong baselines are present.

.. _eco-features:

Feature Importance
------------------

.. list-table::
   :header-rows: 1
   :widths: 5 30 8 15 15

   * - Rank
     - Feature
     - Group
     - Mean importance
     - Top-10 in
   * - 1
     - ``n_sessions``
     - Base
     - 0.144
     - 15 of 19
   * - 2
     - ``std_menu_size``
     - Base
     - 0.112
     - 15 of 18
   * - 3
     - ``menu_transitivity``
     - **RP**
     - 0.101
     - 18 of 19
   * - 4
     - ``mean_basket_size``
     - Base
     - 0.092
     - 12 of 13
   * - 5
     - ``spend_slope``
     - Base
     - 0.080
     - 13 of 13
   * - 6
     - ``mean_menu_size``
     - Base
     - 0.072
     - 16 of 18
   * - 7
     - ``choice_entropy_norm``
     - **RP**
     - 0.072
     - 17 of 19
   * - 8
     - ``menu_pref_density``
     - **RP**
     - 0.072
     - 15 of 18
   * - 9
     - ``total_spend``
     - Base
     - 0.070
     - 11 of 13
   * - 10
     - ``mean_spend``
     - Base
     - 0.064
     - 11 of 13

Three of the top ten features measure choice consistency rather than volume or frequency.

.. _eco-replication:

Replication
-----------

.. code-block:: bash

   pip install prefgraph lightgbm scikit-learn

   python case_studies/benchmarks/runner.py --datasets validated --model both
   python case_studies/benchmarks/runner.py --datasets validated --max-users 250 --model both
   python case_studies/benchmarks/runner.py --datasets dunnhumby
   python case_studies/benchmarks/runner.py --replot

All results are deterministic. Per-dataset JSON files are saved to ``case_studies/benchmarks/output/``. For datasets on external drives, set ``PYREVEALED_DATA_DIR``.

.. _eco-appendix:

Dataset Summary
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 16 7 7 7 5 5 5 6 6 6

   * - Dataset
     - Type
     - N
     - Total obs
     - T med
     - K
     - pct T 10 or more
     - Repeat
     - Uniq pct
     - Domain
   * - Dunnhumby
     - Budget
     - 2,222
     - 55,388
     - 30
     - 10
     - 93
     - 28
     - 24
     - Grocery
   * - Amazon
     - Budget
     - 4,668
     - 54,356
     - 31
     - 15
     - 95
     - 73
     - 13
     - E-commerce
   * - H&M
     - Budget
     - 46,757
     - 29,132
     - 15
     - 9
     - 97
     - 68
     - 21
     - Fashion
   * - Instacart
     - Menu
     - 50,000
     - 59,649
     - 16
     - 18
     - 64
     - 57
     - 50
     - Grocery
   * - REES46
     - Menu
     - 8,832
     - 14,922
     - 7
     - 24
     - 15
     - 19
     - 86
     - E-commerce
   * - Taobao
     - Menu
     - 15,806
     - 11,061
     - 5
     - 30
     - 2
     - 2
     - 100
     - E-commerce
   * - Taobao BW
     - Menu
     - 590
     - 2,593
     - 4
     - 25
     - 3
     - 6
     - 100
     - E-commerce
   * - RetailRocket
     - Menu
     - 47
     - 356
     - 5
     - 57
     - 19
     - 0
     - 100
     - E-commerce
   * - Tenrec
     - Menu
     - 50,000
     - 12,847
     - 5
     - 24
     - 16
     - 0
     - 100
     - Video
   * - MIND
     - Menu
     - 5,091
     - 855
     - 4
     - 86
     - 2
     - 1
     - 100
     - News
   * - FINN
     - Menu
     - 46,858
     - 15,850
     - 9
     - 57
     - 39
     - 6
     - 100
     - Classifieds

Budget datasets have rich histories with 15 to 31 observations per user and repeat rates between 28 and 73 percent. Menu datasets are thinner with 4 to 9 observations and near-zero repeat rates on most platforms. Revealed preference features work best where users make repeated choices from overlapping sets.

Feature Correlation
-------------------------------

Revealed preference features are largely orthogonal to baseline features with a median cross-correlation of 0.12. The utility-recovery features are internally redundant with pairwise correlations above 0.95. The genuinely independent revealed preference features are choice entropy, menu transitivity, violation density, consistency ratio, and per-observation efficiency.

Null Rates
---------------------

50 of 59 revealed preference features are always populated on budget data. 25 of 27 are always populated on menu data. The exceptions are utility-recovery features that require a minimum number of intersecting choices. All missing values are imputed with training-set medians.
