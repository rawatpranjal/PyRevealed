Case Study 2: Predicting Customer Spend & Engagement
=====================================================

Across 11 datasets and 27 prediction targets, revealed preference features add between zero and two percent marginal lift on average. Standard spend and engagement baselines already capture most of the signal.

Setup
-----

We test whether choice-consistency features improve predictions of spend, churn, engagement, and loyalty at the individual user level. Each user's purchase or click history is split in time. The early portion produces features. The later portion defines the prediction target. Two models are trained on an 80/20 user holdout and evaluated out of sample.

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
     - +RP
     - Δ
     - Lift %
     - Engine
   * - Dunnhumby
     - Budget
     - 2,222
     - Spend Drop
     - .164
     - .130
     - -.034
     - -20.5%
     - 6m 6s
   * - Dunnhumby
     -
     -
     - High Spender
     - .932
     - .929
     - -.002
     - -0.2%
     -
   * - Dunnhumby
     -
     -
     - Future LTV (R²)
     - .582
     - .587
     - +.005
     - +0.9%
     -
   * - Amazon
     - Budget
     - 4,668
     - Spend Drop
     - .196
     - .198
     - +.002
     - +1.0%
     - 12m
   * - Amazon
     -
     -
     - High Spender
     - .900
     - .902
     - +.002
     - +0.2%
     -
   * - H&M
     - Budget
     - 46,757
     - Spend Change (R²)
     - .299
     - .302
     - +.003
     - +1.0%
     - 16m
   * - H&M
     -
     -
     - High Spender
     - .673
     - .671
     - -.002
     - -0.3%
     -
   * - Instacart
     - Menu
     - 50,000
     - Low Loyalty
     - .935
     - .937
     - +.001
     - +0.1%
     - 8m
   * - REES46
     - Menu
     - 50,000
     - Low Loyalty
     - .670
     - .674
     - +.004
     - +0.5%
     - 8m
   * - Taobao
     - Menu
     - 50,000
     - Engagement
     - .822
     - .822
     - .000
     - 0.0%
     - 3m
   * - Taobao BW
     - Menu
     - 29,519
     - Low Loyalty
     - .715
     - **.822**
     - +.107
     - **+15.0%**
     - 0.1s
   * - Tenrec
     - Menu
     - 50,000
     - Engagement
     - .982
     - .982
     - .000
     - 0.0%
     - 15m
   * - MIND
     - Menu
     - 5,000
     - High CTR
     - .529
     - .518
     - -.011
     - -2.1%
     - 4m
   * - FINN
     - Menu
     - 1,869
     - Low Loyalty
     - .772
     - .773
     - +.001
     - +0.1%
     - 13m

All values are out-of-sample AUC-PR. One target per dataset is shown. Full results are in ``output/results.json``.

Findings
--------

On most targets the lift from adding RP features is within plus or minus one percent. The standout is Taobao Buy Window Low Loyalty at plus 15 percent, where preference graph features detect users whose choices are becoming more dispersed over time. The mean lift across all targets is near zero.

RP features rank highly in model importance but generally do not change predictions. The model finds them informative because they describe a different aspect of user behavior. They do not add lift because the baseline already covers most of the same ground through simpler measures.

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
     - 15/19 targets
   * - 2
     - ``std_menu_size``
     - Base
     - 0.112
     - 15/18
   * - 3
     - ``menu_transitivity``
     - **RP**
     - 0.101
     - 18/19
   * - 4
     - ``mean_basket_size``
     - Base
     - 0.092
     - 12/13
   * - 5
     - ``spend_slope``
     - Base
     - 0.080
     - 13/13
   * - 6
     - ``mean_menu_size``
     - Base
     - 0.072
     - 16/18
   * - 7
     - ``choice_entropy_norm``
     - **RP**
     - 0.072
     - 17/19
   * - 8
     - ``menu_pref_density``
     - **RP**
     - 0.072
     - 15/18
   * - 9
     - ``total_spend``
     - Base
     - 0.070
     - 11/13
   * - 10
     - ``mean_spend``
     - Base
     - 0.064
     - 11/13

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
     - %T≥10
     - Repeat
     - Uniq%
     - Domain
   * - Dunnhumby
     - Budget
     - 2,222
     - 55,388
     - 30
     - 10
     - 93%
     - 28%
     - 24%
     - Grocery
   * - Amazon
     - Budget
     - 4,668
     - 54,356
     - 31
     - 15
     - 95%
     - 73%
     - 13%
     - E-commerce
   * - H&M
     - Budget
     - 46,757
     - 29,132
     - 15
     - 9
     - 97%
     - 68%
     - 21%
     - Fashion
   * - Instacart
     - Menu
     - 50,000
     - 59,649
     - 16
     - 18
     - 64%
     - 57%
     - 50%
     - Grocery
   * - REES46
     - Menu
     - 50,000
     - 14,922
     - 7
     - 24
     - 15%
     - 19%
     - 86%
     - E-commerce
   * - Taobao
     - Menu
     - 50,000
     - 11,061
     - 5
     - 30
     - 2%
     - 2%
     - 100%
     - E-commerce
   * - Taobao BW
     - Menu
     - 29,519
     - 2,593
     - 4
     - 25
     - 3%
     - 6%
     - 100%
     - E-commerce
   * - RetailRocket
     - Menu
     - 47
     - 356
     - 5
     - 57
     - 19%
     - 0%
     - 100%
     - E-commerce
   * - Tenrec
     - Menu
     - 50,000
     - 12,847
     - 5
     - 24
     - 16%
     - 0%
     - 100%
     - Video
   * - MIND
     - Menu
     - 5,000
     - 855
     - 4
     - 86
     - 2%
     - 1%
     - 100%
     - News
   * - FINN
     - Menu
     - 1,869
     - 15,850
     - 9
     - 57
     - 39%
     - 6%
     - 100%
     - Classifieds

Budget datasets have rich histories with 15 to 31 observations per user and repeat rates between 28 and 73 percent. Menu datasets are thinner with 4 to 9 observations and near-zero repeat rates on most platforms. RP features work best where users make repeated choices from overlapping sets.

Appendix: Feature Correlation
-------------------------------

RP features are largely orthogonal to baseline features with a median cross-correlation of 0.12. The utility-recovery features are internally redundant with pairwise correlations above 0.95. The independent RP features are ``choice_entropy``, ``menu_transitivity``, ``sarp_violation_density``, ``hm_ratio``, and ``vei_mean``.

Appendix: Null Rates
---------------------

50 of 59 RP features are always populated on budget data. 25 of 27 are always populated on menu data. The exceptions are utility-recovery features that require a minimum number of intersecting choices. All nulls are imputed with train-set medians.
