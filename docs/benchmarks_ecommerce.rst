E-commerce Benchmarks
=====================

Do revealed-preference (RP) features add independent predictive signal beyond
strong engagement/spend baselines on real e‑commerce data?

**TL;DR.** On menu datasets, RP captures structure that engagement stats miss
(Taobao RP‑only 0.925 > baseline 0.913). On budget datasets, RP adds ~0% over
RFM — spending history already carries the signal. Instacart shows heavy habit
structure (83.8% SARP violations) but near‑zero lift, consistent with
reordering.

*Last updated: 2026-03-28. Instacart now uses aisle-level menus. H&M uses
per-customer realized prices (v0.5.8).* 

Seven public datasets, 167K users, 42 RP features. CatBoost (H&M) or LightGBM
(others). 80/20 user holdout with bootstrap CIs.

Roadmap
-------

- :ref:`Setup <eco-setup>`: datasets, targets, models, and splits
- :ref:`Datasets & assumptions <eco-assumptions>`: what each dataset measures
- :ref:`How to read <eco-how-to-read>`: Baseline, +RP, RP‑only, Lift
- :ref:`Results <eco-results>`: full table with RP‑only highlights
- :ref:`RP‑only performance <eco-rp-only>`: where RP carries independent signal
- :ref:`Findings <eco-findings>`: practical takeaways
- :ref:`Top features <eco-top-features>`: what matters across tasks
- :ref:`Reproduce <eco-reproduce>` and :ref:`Appendix <eco-appendix>`: pipeline and code

.. _eco-setup:

Setup
-----

- **Data types**: budgets (with prices) and menus (no prices).
- **Targets**: classification (High Spender, High Engagement, Low Loyalty, High
  Novelty) and regression (Future Spend, Spend Change). Targets use top‑tercile
  thresholds for consistency.
- **Features**: Baseline (RFM + concentration + trends, 13); RP Engine (14) +
  RP Extended (28) for 42 total: GARP/CCEI/MPI/HARP/HM/VEI, graph density and
  transitivity, utility recovery (Gini/CV), choice entropy, ordinal utility.
- **Models**: LightGBM (menus, budgets except H&M); CatBoost (H&M). Default
  hyperparameters.
- **Split**: 80/20 user holdout; 5‑fold stratified CV; bootstrap CIs on lift.

.. _eco-assumptions:

Datasets & Assumptions
----------------------

**Dunnhumby.** 2,222 households, 104 weeks, 10 commodity groups (~$19/week of
a ~$100--150 weekly grocery basket). Budget-based RP. Global median price oracle
per commodity per week, shared across all households (Dean & Martin 2016).
Individual price exposure (coupons, regional variation) is not captured.
Within-commodity substitution is invisible.

**Open E-Commerce.** 4,694 users, category-level quantities. Budget-based RP.
Median price per category per month, forward-filled for missing periods. Shared
oracle across users. Within-category product switching is invisible.

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

**Instacart.** 206,209 users, 32.4M order-product rows across 3.4M prior orders
(Kaggle Market Basket Analysis). Menu‑based RP — the raw data contains no prices,
so budget‑based analysis is not possible. Each user's orders are broken into
user × order × aisle triples. Within each triple, a valid choice occasion requires
exactly one reordered SKU in that aisle; events with zero or multiple reorders are
dropped. The menu for each event is the set of distinct products the user bought in
that same aisle across their previous three orders, plus the current choice to
guarantee membership. This trailing‑3 window avoids inflating menus with stale
items from distant history. Events with menu size < 2 are dropped (trivial menus).
To ensure enough repeated structure for RP analysis, only user‑aisle pairs with at
least 3 valid events are retained. Each user's surviving events across all their
qualifying aisles are concatenated in order‑number sequence into a single
``MenuChoiceLog``, with product IDs remapped to a compact 0..N−1 space per user.
Filters: ≥ 5 total observations per user. Final dataset: 4.5M events, 120K users,
715K user‑aisle pairs.

**REES46.** 8,832 users, click-to-purchase sessions. Menu-based RP.
Server-defined session IDs (gold standard). Menus contain only items the user
clicked; unviewed items are invisible. Median menu size ~5 items. No prices —
choices reveal preference orderings only.

**Taobao.** 4,239 users, 100M raw events. Menu-based RP. Session boundaries
defined by 30-minute inactivity gaps (84% of inter-event gaps < 30 min). Median
menu size 4 items. Menus contain only items the user viewed or purchased within
a session. No prices.



.. _eco-how-to-read:

How to read the results
-----------------------

- **Baseline** = LightGBM/CatBoost on 13 RFM features.
- **+RP** = Baseline plus 42 RP features (Engine + Extended).
- **RP‑only** = RP features without the baseline.
- **Lift%** = (Combined − Baseline) / Baseline × 100.
- **AUC vs R²**: classification uses AUC‑ROC (AUC‑PR shown for some menu tasks);
  regression reports R².
- **Interpretation**: RP adds signal when +RP > Baseline and RP‑only ≈ Baseline.
  When RP‑only < Baseline and +RP ≈ Baseline, the baseline already captures the
  predictive structure.

.. _eco-results:

Results
-------

.. list-table::
   :header-rows: 1
   :widths: 18 8 15 10 10 10 8

   * - Dataset
     - N
     - Target
     - Baseline
     - +RP
     - RP-only
     - Lift%
   * - Dunnhumby
     - 2,222
     - High Spender
     - 0.960
     - 0.960
     - —
     - -0.0%
   * - Dunnhumby
     - 2,222
     - Churn
     - 0.752
     - 0.740
     - —
     - -1.5%
   * - Open E-Commerce
     - 4,694
     - High Spender
     - 0.950
     - 0.951
     - —
     - +0.0%
   * - Open E-Commerce
     - 4,694
     - Churn
     - 0.846
     - 0.846
     - 0.769
     - -0.0%
   * - H&M
     - 46,757
     - High Spender
     - 0.784
     - 0.783
     - 0.720
     - -0.1%
   * - H&M
     - 46,757
     - Future Spend (R²)
     - 0.337
     - 0.340
     - —
     - +0.003
   * - H&M
     - 46,757
     - Spend Change (R²)
     - 0.290
     - 0.295
     - —
     - +0.005
   * - Instacart
     - 50,000
     - Low Loyalty
     - 0.968
     - 0.969
     - —
     - +0.0%
   * - Instacart
     - 50,000
     - High Novelty
     - 0.765
     - 0.767
     - 0.762
     - +0.3%
   * - REES46
     - 8,832
     - High Engagement
     - 0.996
     - 0.996
     - 0.990
     - +0.0%
   * - Taobao
     - 4,239
     - High Engagement
     - 0.913
     - **0.915**
     - **0.925**
     - **+0.2%**

*Baseline = LightGBM on 13 RFM features. +RP = same model with 42 RP features
added. RP-only = RP features without baseline. On Taobao, RP-only (0.925)
outperforms the engagement baseline (0.913) — graph transitivity and choice
entropy capture patterns that session counts miss.*

.. _eco-rp-only:

RP-only performance
-------------------

RP features alone (no baseline) show where preference structure carries
independent signal.

.. list-table::
   :header-rows: 1
   :widths: 22 28 14 14

   * - Dataset
     - Target
     - RP-only
     - Baseline
   * - Taobao
     - High Engagement
     - 0.925
     - 0.913
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

On Taobao, RP‑only outperforms the engagement baseline. Preference graph
transitivity and choice entropy capture patterns that session counts and menu
sizes miss.

.. _eco-findings:

Findings
--------

- **Menu datasets**: RP is competitive with, and sometimes exceeds, engagement
  baselines (Taobao). Graph structure (transitivity, density) and choice entropy
  carry real signal beyond session counts and menu sizes.
- **Budget datasets**: RP adds ~0% marginal lift over strong RFM baselines.
  CCEI/MPI correlate with spend history; little independent predictive value.
- **Instacart**: Strong revealed structure (high SARP violations), yet +RP adds
  near‑zero lift — behavior is reorder‑dominated within aisles.
- **Feature importance**: Baseline spend features dominate globally; RP features
  rise to the top on menu tasks (see below).

.. _eco-top-features:

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
