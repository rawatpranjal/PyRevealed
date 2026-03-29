Case Study 2: Predicting Customer Spend & Engagement
=====================================================

**TL;DR.** Across 11 datasets and 27 prediction targets, RP features add 0–2% marginal lift. The signal is real but small — baseline spend and engagement features already capture most of the predictive power.

Setup
-----

We test whether revealed preference (RP) graph features improve user-level predictions of spend, churn, engagement, and loyalty. 11 datasets span grocery (Dunnhumby), e-commerce (Amazon, H&M, REES46, Taobao, RetailRocket), grocery menus (Instacart), video (Tenrec), news (MIND), and classifieds (FINN). Each user's first 70% of observations produce 42 RP features (GARP, CCEI, MPI, HM, VEI, graph density, transitivity, entropy) alongside a 13-feature RFM baseline. A regularized LightGBM (``max_depth=3``, ``num_leaves=8``) and L1-penalized logistic regression are trained on an 80/20 user holdout. All results are out-of-sample AUC-ROC; ``SEED=42`` throughout.

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
     - .683
     - **.694**
     - +.011
     - **+1.6%**
     - 6m 6s
   * - Dunnhumby
     -
     -
     - High Spender
     - .962
     - .961
     - -.001
     - -0.1%
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
     - .756
     - **.761**
     - +.005
     - **+0.7%**
     - 12m
   * - Amazon
     -
     -
     - High Spender
     - .940
     - .939
     - -.001
     - -0.1%
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
     - .787
     - .785
     - -.002
     - -0.3%
     -
   * - Instacart
     - Menu
     - 50,000
     - Low Loyalty
     - .969
     - .970
     - +.001
     - +0.1%
     - 8m
   * - REES46
     - Menu
     - 50,000
     - Low Loyalty
     - .883
     - .882
     - -.001
     - -0.1%
     - 8m
   * - Taobao
     - Menu
     - 50,000
     - Engagement
     - .938
     - .938
     - .000
     - 0.0%
     - 3m
   * - Taobao BW
     - Menu
     - 29,519
     - Low Loyalty
     - .984
     - .986
     - +.002
     - +0.2%
     - 0.1s
   * - Tenrec
     - Menu
     - 50,000
     - Engagement
     - .992
     - .992
     - .000
     - 0.0%
     - 15m
   * - MIND
     - Menu
     - 5,000
     - High CTR
     - .678
     - .667
     - -.011
     - -1.6%
     - 4m
   * - FINN
     - Menu
     - 1,869
     - Low Loyalty
     - .957
     - .957
     - .000
     - 0.0%
     - 13m

*All values are out-of-sample AUC-ROC (regularized LightGBM). Base = 13 RFM features. +RP = Base + 42 RP features. One representative target per dataset shown; full results in* ``output/results.json``. *Engine = Rust batch scoring time on Apple M-series.*

Findings
--------

The two Spend Drop targets are the only ones with consistent positive lift across sample sizes and models — Dunnhumby +1.6%, Amazon +0.7%. On these targets, VEI (per-observation budget efficiency) captures declining rationality before spending actually drops. Everywhere else, RP features are net-neutral: the mean lift across all 27 targets is +0.03%.

Despite near-zero lift, RP features rank highly in LGBM importance. ``menu_transitivity`` is the 3rd most important feature overall (top-10 in 18/19 menu targets), and ``choice_entropy_norm`` is 7th. The model *uses* RP features but they don't improve predictions because the baseline features already capture overlapping information through a different path.

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

*LGBM gain-based importance, normalized, averaged across all classification targets. 3 of the top 10 features are RP-derived — all graph-structural (transitivity, entropy, preference density).*

.. _eco-replication:

Replication
-----------

.. code-block:: bash

   pip install prefgraph lightgbm scikit-learn

   # Full run (all 11 validated datasets, both models)
   python case_studies/benchmarks/runner.py --datasets validated --model both

   # Quick smoke test (250 users)
   python case_studies/benchmarks/runner.py --datasets validated --max-users 250 --model both

   # Single dataset
   python case_studies/benchmarks/runner.py --datasets dunnhumby

   # Regenerate from cached results
   python case_studies/benchmarks/runner.py --replot

All results deterministic (``SEED=42``). Per-dataset JSON saved to ``case_studies/benchmarks/output/``. External datasets: set ``PYREVEALED_DATA_DIR=/path/to/datasets``.

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

Budget datasets have rich histories (T=15–31, repeat rate 28–73%) — ideal for RP testing. Menu datasets are thin (T=4–9, repeat rate 0–6%) because recommendation platforms surface novel items. The exceptions — Instacart (57% repeat) and REES46 (19%) — are where RP features show the most signal. RP needs repeated choices from overlapping sets; most menu platforms don't provide that.

Appendix: Feature Correlation
-------------------------------

RP features are largely orthogonal to baselines (median |r| = 0.12). The utility-recovery block (``util_*``, ``lambda_*``) is internally redundant (r > 0.95). The genuinely independent RP features — ``choice_entropy``, ``menu_transitivity``, ``sarp_violation_density``, ``hm_ratio``, ``vei_mean`` — are the ones that survive L1 selection and carry non-redundant signal.

Appendix: Null Rates
---------------------

50/59 RP features are always populated on budget data; 25/27 on menu data. The exceptions are utility-recovery features (``util_mean/std/range/cv/gini``) that require sufficient intersecting choices to solve the Afriat LP. Imputed with train-set medians.
