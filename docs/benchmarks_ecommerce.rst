E-commerce Benchmarks
=====================

**TL;DR.** RP features deliver a modest 0–1% gain on predictive tasks; strong engagement/spend baselines already capture most of the signal.

.. _eco-setup:

Setup
-----

We evaluate two data types: budgets with prices and menus without prices. The
targets include classification tasks such as High Spender, High Engagement, Low
Loyalty, and High Novelty, as well as regression tasks for Future Spend and
Spend Change. Targets use top tercile thresholds for consistency. Features
combine a 13 feature baseline that covers RFM, concentration, and trends with
42 revealed preference features, split into 14 RP Engine features and 28 RP
Extended features. These include GARP, CCEI, MPI, HARP, HM, VEI, graph density
and transitivity, utility recovery metrics such as Gini and CV, choice entropy,
and ordinal utility. Models use CatBoost with default hyperparameters. The split uses an 80 to 20
user holdout, five fold stratified cross validation, and bootstrap confidence
intervals on lift.

The hardest part is reconstructing the choice set and the observed choices.
For budgets, prices and quantities must reflect what the customer could have
afforded at each observation. For menus, the available alternatives must be
recovered from logs. This requires explicit assumptions about availability,
timing, and aggregation. We document those assumptions in the Appendix and,
honestly, we have to make good assumptions and hope they are right.

.. _eco-how-to-read:

How to read the results
-----------------------

Baseline uses CatBoost on 13 RFM features. +RP adds 42 revealed
preference features to that baseline, while RP only removes the baseline and
uses only the RP features. Lift percent is defined as (Combined minus Baseline)
divided by Baseline times 100. Classification tasks report AUC ROC, and some
menu tasks also show AUC PR. Regression tasks report R squared. As a rule of
thumb, RP adds signal when +RP is greater than Baseline and RP only is close to
Baseline. When RP only is smaller than Baseline and +RP is about the same as
Baseline, the baseline already captures the structure.

.. _eco-results:

Results
-------

.. list-table::
   :header-rows: 1
   :widths: 18 8 15 10 10 8 10

   * - Dataset
     - N
     - Target
     - Baseline
     - +RP
     - Lift%
     - RP-only
   * - Dunnhumby
     - 2,222
     - High Spender
     - 0.962
     - 0.965
     - +0.3%
     - 0.937
   * - Dunnhumby
     - 2,222
     - Churn
     - 0.711
     - 0.724
     - +1.8%
     - 0.622
   * - Dunnhumby
     - 2,222
     - Future LTV (R²)
     - 0.577
     - 0.589
     - +0.012
     - 0.246
   * - Open E-Commerce
     - 4,668
     - High Spender
     - 0.940
     - 0.942
     - +0.2%
     - 0.932
   * - Open E-Commerce
     - 4,668
     - Spend Drop
     - 0.784
     - 0.798
     - +1.8%
     - 0.684
   * - Open E-Commerce
     - 4,668
     - Spend Change (R²)
     - 0.144
     - 0.091
     - -0.053
     - -0.032
   * - Open E-Commerce
     - 4,668
     - Future LTV (R²)
     - 0.633
     - 0.622
     - -0.011
     - 0.387
   * - H&M
     - 46,757
     - High Spender
     - 0.784
     - 0.783
     - -0.1%
     - 0.720
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
   * - REES46
     - 8,832
     - High Engagement
     - 0.996
     - 0.996
     - +0.0%
     - 0.990
   * - Taobao (Buy Window)
     - 29,519
     - Pref Drift (AP)
     - 0.940
     - 0.938
     - -0.2%
     - —
   * - Taobao (Buy Window)
     - 29,519
     - High Entropy (AP)
     - 0.789
     - **0.790**
     - **+0.1%**
     - —
   * - Taobao (Buy Window)
     - 29,519
     - High Active Time (AUC)
     - 0.777
     - 0.778
     - +0.1%
     - —
   * - Taobao (Buy Window)
     - 29,519
     - High Click Volume (AUC)
     - 0.818
     - 0.818
     - +0.0%
     - —
   * - Taobao (Buy Window)
     - 29,519
     - Fast Conversion (AUC)
     - 0.561
     - 0.561
     - +0.0%
     - —

*Baseline = CatBoost on 13 RFM features. +RP = same model with 42 RP features
added. RP-only = RP features without baseline. On Taobao (buy‑anchored, 6h),
RP features contribute modest lift on structural targets; engagement/volume
targets remain baseline‑dominated.*


How The Taobao Choice Data Is Built (Buy‑Anchored)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Raw events
  - File: ``UserBehavior.csv`` with rows ``(user_id, item_id, category_id, behavior_type, timestamp)``.
  - We keep only ``pv`` (view) and ``buy`` events and sort by ``(user_id, timestamp)``.

Observation construction (6‑hour window)
  - For each buy at time ``t_buy``, define a trailing window ``[t_buy − 6h, t_buy)``.
  - Menu = unique items viewed in this window (pre‑buy only). Choice = the bought item.
  - Require the bought item was viewed in the window; drop otherwise.
  - Keep observations with menu size in ``[2, 50]``.

Per‑user logs and qualification
  - Collect all buy‑anchored observations for a user as one MenuChoiceLog.
  - Users must have at least 5 valid observations (``min_sessions=5``).
  - Item IDs are remapped to a compact per‑user space (0..N‑1) to ensure stable graph ops.

Why buy‑anchored windows (vs. whole sessions)
  - Whole‑session, pre‑buy menus (with a 30‑minute inactivity session boundary) are very clean but too sparse per user.
  - Extending the session gap to 60–180 minutes does not materially increase clean, pre‑buy observations.
  - Buy‑anchored windows preserve the economic idea (simultaneous availability right before purchase) and produce thousands of qualifying users while enforcing “buy was viewed”.

Train/test protocol (leakage‑safe)
  - Per‑user temporal split: first 70% of observations → features, last 30% → targets. Each half gets its own item remapping.
  - User holdout: 80/20 split across users (stratified for classification).
  - Thresholds (e.g., top/bottom tercile) are computed on TRAIN users only, then applied to TEST; labels are re‑binarized with the train‑only threshold in the evaluation routine.
  - Features: Baseline engagement stats + RP features from ``Engine.analyze_menus`` (SARP/WARP/HM, graph structure). Extended menu RP features are computed per user.

Target definitions (examples)
  - Pref Drift: 1 if the fraction of unique choices increases in the test window vs. train.
  - Choice Entropy: normalized entropy of test‑window choice distribution (top tercile = High Entropy).
  - Active Time: capped sum of inter‑event gaps within each buy window, summed over test windows (top tercile = High Active Time). Gaps are capped (e.g., 5 minutes) to avoid idle inflation.
  - Fast Conversion: bottom tercile of median latency from last pre‑buy view to the buy.

Notes and caveats
  - No dwell times are logged; “active seconds” uses capped inter‑event deltas as a practical proxy.
  - Novelty targets (e.g., any novel choice) can become degenerate on some splits; we report only targets with both classes present.
  - Results are reported as AUC‑PR (AP) when class imbalance is high and AUC otherwise; the bootstrap CI on lift is computed on the chosen metric.

.. _eco-findings:

Findings
--------

On menu datasets, RP features are competitive with, and sometimes exceed,
engagement baselines. The Taobao results show that item graph structure
such as transitivity and density, along with choice entropy, carry signal
that session counts and menu sizes do not capture.

On budget datasets, RP adds roughly zero marginal lift over strong RFM
baselines. Measures such as CCEI and MPI correlate with spending history,
so they contribute little independent predictive power when the baseline
already encodes that history.



Looking at feature importance, baseline spend features dominate in most
global models. RP features rise near the top on menu tasks. The menu
tables below show item graph structure and choice entropy among the most
informative signals.

.. _eco-top-features:

Top Features
------------

Across all classification tasks (CatBoost feature importance, combined model):

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

Menu-dataset top features (Taobao + REES46):

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

Four of the top 8 menu features are RP-derived. Item graph structure and choice
entropy carry signal that engagement statistics do not capture.

.. _eco-reproduce:

Reproduce
---------

.. code-block:: bash

   pip install prefgraph lightgbm scikit-learn
   python case_studies/benchmarks/runner.py --datasets all

Datasets require ``kaggle`` CLI. See ``case_studies/benchmarks/`` for details.

----

.. _eco-appendix:

Appendix: Datasets & Assumptions
--------------------------------

.. _eco-assumptions:

**Dunnhumby.** 2,222 households, 104 weeks, 10 staple commodity groups.
Budget-based RP. Each observation is one active household-week — weeks with
zero purchases in tracked categories are excluded, since they represent
spending outside the sub-basket, not zero demand. Quantities are total units
per commodity per week. Prices are a global median oracle per commodity per
week (Dean & Martin 2016), shared across all households — individual price
exposure (coupons, store variation) is not captured. RP outputs should be
interpreted as reduced-form consistency descriptors of conditional sub-basket
allocation, not structural preference parameters.

**Open E-Commerce.** 4,668 users, category-level quantities. Budget-based RP.
Median price per category per month, forward-filled for missing periods. Shared
oracle across users. Within-category product switching is invisible. While RP 
features underperform the RFM baseline on regression tasks (LTV, Spend Change), 
they provide a healthy boost (+1.8% lift) for predicting "Spend Drop", suggesting 
that shifting category allocations might be an early signal of churn before volume drops.

Polars fast-path (this repo):
- Users analyzed: 4,744 (goods=50, median T=34 months)
- Price variation: users with ≥3 distinct price vectors: ~100%
- GARP pass rate: ~2.9% (strong power)
- CCEI percentiles: p25≈0.300, p50≈0.424, p75≈0.567, p95≈0.855
- Tooling: ``tools/open_ecommerce_polars_variation.py``

Latest ML benchmark (Open E‑Commerce, ~5k users)
- Users: 4,668 (after train/test split constraints)

  - Targets and results
  - High Spender (classification)
    - AUC: RP=0.932, Base=0.940, Combined=0.942, Lift=+0.002
  - Spend Drop (classification)
    - AUC: RP=0.684, Base=0.784, Combined=0.798, Lift=+0.014
  - Spend Change (regression)
    - R2: RP=-0.032, Base=0.144, Combined=0.091
  - Future LTV (regression)
    - R2: RP=0.387, Base=0.633, Combined=0.622

- Read: Combined modestly improves “Spend Drop”; baseline dominates for LTV and
  Spend Change; RP alone underperforms baseline on those.

**H&M.** 46,757 customers, 31.8M transactions (2018‑09 to 2020‑09). Budget‑based
RP. Each customer’s purchases in a month define one choice occasion. Articles map
to 20 coarse product groups (first two digits of article_id). Quantity per group
is the article‑row count — each CSV row is one purchased unit. Price per group is
the customer’s own average paid price that month. Unpurchased groups are imputed
via period‑group median → group median → global median, because RP tests require
a full price vector to compare what a customer could have afforded across
observations. This per‑customer price construction preserves individual variation,
unlike the shared oracle used for Dunnhumby and Open E‑Commerce. Prices are
normalized 0–1 (Kaggle): relative variation is real, absolute dollar levels are
not. Filters: ≥ 6 active months, ≥ 10 total observations. Sales channel ignored.


**REES46.** 8,832 users, click-to-purchase sessions. Menu-based RP.
Server-defined session IDs (gold standard). Menus contain only items the user
clicked; unviewed items are invisible. Median menu size ~5 items. No prices —
choices reveal preference orderings only.

**Taobao.** 100M raw events (pv, buy). Buy‑anchored menu reconstruction:
for each buy at time ``t``, define a trailing window ``[t−6h, t)``;
menu = unique items viewed (pv) in that window; choice = the bought item.
Require the bought item was viewed; keep menus of size 2–50 only. Build per‑user
logs from all such observations; retain users with ≥5 observations. Item IDs are
remapped to a compact per‑user space. No prices — analysis is ordinal.

Assumptions: views approximate the considered set (impression bias: unseen
alternatives are unobserved); 6‑hour window is a pragmatic simultaneity proxy
(shorter/longer windows yield similar patterns); post‑purchase views are excluded;
exposure is observational (not randomized).

**Taobao (Session‑based).** 4,239 users, ~100M raw events. Menu‑based RP with sessions defined by 30‑minute inactivity gaps (84% of inter‑event gaps < 30 minutes). For each session, build a menu from the items the user viewed or purchased within that session; median menu size ≈ 4 items. No prices — choices reveal within‑session preference orderings only.

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
     -> CatBoost (random_seed=42, default hyperparameters)
     -> 5-fold stratified CV
     -> Metrics: AUC-ROC, AUC-PR, Log Loss, F1

**Three models per target**: (a) Baseline only, (b) RP only, (c) Baseline + RP.

**Targets**: High Spender (top tercile spend), Future Spend (regression),
Spend Change (regression), High Engagement (top tercile sessions).

**Output**: ``case_studies/benchmarks/output/results.json`` (full metrics),
``summary_table.csv``, ``figures/``.
