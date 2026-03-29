Case Study 2: Predicting Customer Spend and Engagement
=======================================================

Across 11 datasets and 27 prediction targets, revealed preference features add between zero and two percent marginal lift on average. Standard spend and engagement baselines already capture most of the signal.

Setup
-----

We test whether choice-consistency features improve predictions of spend, churn, engagement, and loyalty at the individual user level. Each user's purchase or click history is split in time. The early portion produces features. The later portion defines the prediction target. Two models are trained on a holdout sample and evaluated out of sample.

.. _eco-results:

Results
-------

.. list-table::
   :header-rows: 1
   :widths: 14 7 7 16 8 8 7 7 7

   * - Dataset
     - Type
     - N
     - Target
     - Base
     - With RP
     - Lift
     - Lift %
     - Engine
   * - Dunnhumby
     - Budget
     - 2,222
     - Spend Drop
     - 0.164
     - 0.130
     - 0.034 lower
     - 20.5 lower
     - 6m 6s
   * - Dunnhumby
     -
     -
     - High Spender
     - 0.932
     - 0.929
     - 0.002 lower
     - 0.2 lower
     -
   * - Dunnhumby
     -
     -
     - Future LTV (R²)
     - 0.582
     - 0.587
     - 0.005 higher
     - 0.9 higher
     -
   * - Amazon
     - Budget
     - 4,668
     - Spend Drop
     - 0.196
     - 0.198
     - 0.002 higher
     - 1.0 higher
     - 12m
   * - Amazon
     -
     -
     - High Spender
     - 0.900
     - 0.902
     - 0.002 higher
     - 0.2 higher
     -
   * - H&M
     - Budget
     - 46,757
     - Spend Change (R²)
     - 0.299
     - 0.302
     - 0.003 higher
     - 1.0 higher
     - 16m
   * - H&M
     -
     -
     - High Spender
     - 0.673
     - 0.671
     - 0.002 lower
     - 0.3 lower
     -
   * - Instacart
     - Menu
     - 50,000
     - Low Loyalty
     - 0.935
     - 0.937
     - 0.001 higher
     - 0.1 higher
     - 8m
   * - REES46
     - Menu
     - 50,000
     - Low Loyalty
     - 0.670
     - 0.674
     - 0.004 higher
     - 0.5 higher
     - 8m
   * - Taobao
     - Menu
     - 50,000
     - Engagement
     - 0.822
     - 0.822
     - no change
     - no change
     - 3m
   * - Taobao BW
     - Menu
     - 29,519
     - Low Loyalty
     - 0.715
     - 0.822
     - 0.107 higher
     - 15.0 higher
     - 0.1s
   * - Tenrec
     - Menu
     - 50,000
     - Engagement
     - 0.982
     - 0.982
     - no change
     - no change
     - 15m
   * - MIND
     - Menu
     - 5,000
     - High CTR
     - 0.529
     - 0.518
     - 0.011 lower
     - 2.1 lower
     - 4m
   * - FINN
     - Menu
     - 1,869
     - Low Loyalty
     - 0.772
     - 0.773
     - 0.001 higher
     - 0.1 higher
     - 13m

All values are out-of-sample AUC-PR. One target per dataset is shown. Full results are in ``output/results.json``.

Findings
--------

On most targets the lift from adding revealed preference features is within one percent in either direction. The mean lift across all 27 targets is near zero. One outlier is Taobao Buy Window Low Loyalty at 15 percent higher, though this result comes from a small test set of 118 users and should be interpreted with caution.

The model finds revealed preference features informative. Three of the top ten features by importance measure choice consistency rather than volume or frequency. But this importance does not translate into better predictions because the baseline already covers similar ground through simpler measures.

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

Appendix: Dataset Summary
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
     - 50,000
     - 14,922
     - 7
     - 24
     - 15
     - 19
     - 86
     - E-commerce
   * - Taobao
     - Menu
     - 50,000
     - 11,061
     - 5
     - 30
     - 2
     - 2
     - 100
     - E-commerce
   * - Taobao BW
     - Menu
     - 29,519
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
     - 5,000
     - 855
     - 4
     - 86
     - 2
     - 1
     - 100
     - News
   * - FINN
     - Menu
     - 1,869
     - 15,850
     - 9
     - 57
     - 39
     - 6
     - 100
     - Classifieds

Budget datasets have rich histories with 15 to 31 observations per user and repeat rates between 28 and 73 percent. Menu datasets are thinner with 4 to 9 observations and near-zero repeat rates on most platforms. Revealed preference features work best where users make repeated choices from overlapping sets.

Appendix: Feature Correlation
-------------------------------

Revealed preference features are largely orthogonal to baseline features with a median cross-correlation of 0.12. The utility-recovery features are internally redundant with pairwise correlations above 0.95. The genuinely independent revealed preference features are choice entropy, menu transitivity, violation density, consistency ratio, and per-observation efficiency.

Appendix: Null Rates
---------------------

50 of 59 revealed preference features are always populated on budget data. 25 of 27 are always populated on menu data. The exceptions are utility-recovery features that require a minimum number of intersecting choices. All missing values are imputed with training-set medians.
