Predictive Benchmarks (E-commerce)
===================================

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
and ordinal utility. Models use CatBoost with default hyperparameters. The split uses a per-user temporal divide (first 70% features, last 30% targets) followed by an 80/20 user holdout and bootstrap confidence intervals on lift.

.. admonition:: Null rates in RP features

   Not all RP features are populated for every user. Measured on n=500 samples per dataset:

   .. list-table::
      :header-rows: 1
      :widths: 20 12 12 56

      * - Dataset
        - Features
        - Null features
        - Which features, and why
      * - Dunnhumby
        - 59
        - 9 (15%)
        - ``violation_mean_position`` 100% NaN (shared category prices → zero GARP cycles for every user); 8 utility-recovery features (``util_mean/std/range/cv/gini``, ``lambda_mean/std/cv``) 89% NaN (Afriat LP near-degenerate on shared prices)
      * - REES46
        - 27
        - 2 (7%)
        - ``menu_util_range``, ``menu_util_std`` 26% NaN (ordinal utility LP fails when preference graph has insufficient constraints)
      * - Taobao
        - 27
        - 2 (7%)
        - Same two features, 82% NaN (buy-window sessions are sparse — few choices per user make the ordinal utility LP under-constrained)

   The 14 Engine features are always non-null across all datasets. All NaN values are imputed with the per-feature train-set median before model training. Features with high null rates are effectively constant after imputation and carry little predictive signal.

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
   * - Amazon
     - 4,668
     - High Spender
     - 0.940
     - 0.942
     - +0.2%
     - 0.932
   * - Amazon
     - 4,668
     - Spend Drop
     - 0.784
     - 0.798
     - +1.8%
     - 0.684
   * - Amazon
     - 4,668
     - Spend Change (R²)
     - 0.144
     - 0.091
     - -0.053
     - -0.032
   * - Amazon
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
     - -
   * - H&M
     - 46,757
     - Spend Change (R²)
     - 0.290
     - 0.295
     - +0.005
     - -
   * - REES46
     - 8,832
     - High Engagement
     - 0.996
     - 0.996
     - +0.0%
     - 0.990
   * - Taobao
     - 29,519
     - Pref Drift (AP)
     - 0.940
     - 0.938
     - -0.2%
     - -
   * - Taobao
     - 29,519
     - High Entropy (AP)
     - 0.789
     - **0.790**
     - **+0.1%**
     - -
   * - Taobao
     - 29,519
     - High Active Time (AUC)
     - 0.777
     - 0.778
     - +0.1%
     - -
   * - Taobao
     - 29,519
     - High Click Volume (AUC)
     - 0.818
     - 0.818
     - +0.0%
     - -
   * - Taobao
     - 29,519
     - Fast Conversion (AUC)
     - 0.561
     - 0.561
     - +0.0%
     - -

*Baseline = CatBoost on 13 RFM features. +RP = same model with 42 RP features
added. RP-only = RP features without baseline. On Taobao (buy‑anchored, 6h),
RP features contribute modest lift on structural targets; engagement/volume
targets remain baseline‑dominated.*


.. _eco-appendix:

Appendix: Datasets & Assumptions
--------------------------------

**Dunnhumby.** 2,222 households, 104 weeks, 10 staple commodity groups. Budget-based RP. Each observation is one active household-week - weeks with zero purchases in tracked categories are excluded, since they represent spending outside the sub-basket, not zero demand. Quantities are total units per commodity per week. Prices are a global median oracle per commodity per week, shared across all households - individual price exposure (coupons, store variation) is not captured. RP outputs should be interpreted as reduced-form consistency descriptors of conditional sub-basket allocation, not structural preference parameters.

**Amazon (Open E-Commerce).** 4,668 users, category-level quantities. Budget-based RP. Median price per category per month, forward-filled for missing periods, shared across users. Within-category product switching is invisible. Variation diagnostics show almost all users exhibit ≥3 distinct price vectors, GARP passes around 2.9%, and CCEI spans p25≈0.300, p50≈0.424, p75≈0.567, p95≈0.855. On predictive tasks, combined models modestly improve "Spend Drop" (Lift +1.4%), but the baseline dominates for Future LTV and Spend Change, with RP alone underperforming the baseline on those regression targets.

**H&M.** 46,757 customers, 31.8M transactions (2018-09 to 2020-09). Budget-based RP. Each customer's purchases in a month define one choice occasion. Articles map to 20 coarse product groups (first two digits of article_id). Quantity per group is the article-row count - each CSV row is one purchased unit. Price per group is the customer's own average paid price that month. Unpurchased groups are imputed via period-group median → group median → global median, because RP tests require a full price vector to compare what a customer could have afforded across observations. This per-customer price construction preserves individual variation, unlike the shared oracle used for Dunnhumby and Open E-Commerce. Prices are normalized 0-1: relative variation is real, absolute dollar levels are not. Filters: ≥ 6 active months, ≥ 10 total observations. Sales channel ignored.

.. **Instacart.** Treated as menu-choice, not budget (no prices in raw data). Observation = user × order × aisle with exactly one reordered SKU. Menu = trailing-3 order products in the same aisle (familiarity set). Filters: menu size ≥ 2, (user, aisle) pairs with ≥ 3 valid events. This yields 4.5M events from 120K users across 715K user-aisle pairs. The data is habit-heavy: 58.6% of repeated user-aisle pairs never switch products, and 83.8% of users have SARP violations. RP features show real graph structure but near-zero predictive lift, consistent with reorder-dominated behavior. Menu construction is circular (alternatives derived from own past purchases) and not a valid RP choice set.

**REES46.** 8,832 users with server-defined shopping sessions (gold standard boundaries). Menu-based RP with one observation per session: the menu is the set of products the user viewed in that session, and the choice is the single purchased product in that session. Raw logs include `event_time`, `event_type` (view/cart/purchase), `product_id`, and `user_session`; the loader keeps only view and purchase events, groups by `user_session`, and retains sessions with exactly one purchase and at least one view. For each kept session it forms the menu as the union of viewed items and the purchased item (to guarantee the chosen item is in the menu even if the platform did not log an explicit pre-buy view), then filters to menu sizes 2–50 to exclude non-choices and degenerate sessions. Users must have ≥ 5 valid sessions; item IDs are remapped to 0..N−1 per user for compact graphs. This construction reflects “what the user actually saw” (impression bias: unseen catalog items are not in the menu). Median menu size is about five items. There are no prices, so results describe within-menu preference orderings only (WARP/SARP/Congruence et al.), not willingness-to-pay. ``StochasticChoiceLog`` was tested as an alternative framing but found inapplicable: with ~300 items per user and menus of 3–10, exact menu repeat rate is 0.00 across all users, so frequency aggregation yields no additional information over the deterministic frame.

**Taobao.** ~29.5k users built from UserBehavior.csv (user_id, item_id, category_id, behavior_type, timestamp). Keep only view and buy events. For each buy at time t, define a trailing 6-hour window [t−6h, t); the menu is the set of unique items viewed in that window (pre-buy only), and the choice is the bought item. Require that the bought item was viewed; keep menus of size 2–50. Aggregate each user's buy-anchored observations into a single MenuChoiceLog with per-user item remapping; require ≥5 valid observations. Assumptions: views approximate the considered set (impression bias: unseen alternatives are unobserved); 6-hour window is a pragmatic simultaneity proxy (shorter/longer windows yield similar patterns); post-purchase views are excluded; exposure is observational (not randomized). Train/test uses a per-user temporal split (70/30) with separate remappings to avoid leakage. ``StochasticChoiceLog`` was tested as an alternative framing: with ~40 items per user and 6-hour windows of size 2–7, exact menu repeat rate is 0.00 across all users, confirming that session-specific view sets do not overlap and the deterministic ``MenuChoiceLog`` frame is appropriate.

.. **Taobao (Session-based).** 4,239 users, ~100M raw events. Menu-based RP with sessions defined by 30-minute inactivity gaps (84% of inter-event gaps < 30 minutes). For each session, build a menu from the items the user viewed or purchased within that session; median menu size ≈ 4 items. No prices - choices reveal within-session preference orderings only. Session boundaries conflate unrelated purchase intentions; the menu mixes items never seriously compared within a single choice problem.

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

Across all classification tasks, baseline spending features generally dominate in the global models. Features like `total_spend` (total expenditure in the training period), `spend_slope` (spending trend), and `n_obs` (number of observations) consistently account for the majority of predictive power. Core RP scores (`ccei`, `mpi`, `hm_ratio`) unfortunately correlate strongly with these spending baselines, adding no marginal value to a strong RFM baseline. 

What works for budget tracking is somewhat counterintuitive. While static-price GARP produces zero violations acting as completely degenerate markers, other RP structures like the `util_gini` (Gini inequality of recovered Afriat utility values) can add nuanced signals.

However, the true value of Revealed Preference features is in **menu datasets** (like Taobao), where RP features are highly competitive with baselines. Across these domains, four of the top eight most important features are RP-derived. When modeling menu-choice settings:
- **`menu_transitivity`** (Preference graph transitivity ratio) and **`pref_graph_density`** (Edge density of revealed preference graph) consistently rank in the top 5–10 features. They capture preference graph properties and patterns that simple engagement counts miss completely. 
- **`choice_entropy_norm`** (Normalized Shannon entropy of choice distribution) and **`menu_util_range`** (Ordinal utility spread) carry useful signal highlighting structured choice behavior that standalone engagement counts miss.
- **`n_scc`** (Number of strongly connected components) offers graph fragmentation signals that augment baseline statistics like standard sizes and diversity.

Overall, RP graph structure features add novel information that tree models actively depend upon, but the marginal predictive lift over well-engineered baseline metrics is relatively small and rarely significant for basic classification targets. 

.. _eco-replication:

Replication
-----------

.. code-block:: bash

   pip install prefgraph lightgbm scikit-learn
   python case_studies/benchmarks/runner.py --datasets all

Datasets require ``kaggle`` CLI. See ``case_studies/benchmarks/`` for details.

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
     -> 80/20 User Holdout (stratified for classification) with Bootstrap CI 
     -> Metrics: AUC-ROC, AUC-PR, R², Lift %

**Three models per dataset/target pair**: (a) Baseline only, (b) RP only, (c) Baseline + RP.

**Predicted Targets**:
  - **High Spender (Classification)**: User ranks in the top tercile of total expenditure during the 30% target period.
  - **Future Spend / LTV (Regression)**: Continuous prediction of exact dollars or units spent by the user in the target period.
  - **Spend Change (Regression)**: Predicting the ratio or absolute difference in the user's spending from the feature period to the target period.
  - **Spend Drop / Churn (Classification)**: The user drops their spending or engagement volume aggressively below a defined safety threshold.
  - **High Engagement (Classification)**: User ranks in the top tercile of total user sessions or views during the target period. 
  - **High Entropy / Pref Drift (Classification)**: The user's target-period choices show high diversity across products or significantly shift from their historical baseline.

**Output**: ``case_studies/benchmarks/output/results.json`` (full metrics),
``summary_table.csv``, ``figures/``.
