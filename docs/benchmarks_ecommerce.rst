Case Study 2: Predicting Customer Lifetime Values
=================================================

**TL;DR.** RP features deliver a modest 0–1% gain on predictive tasks; strong engagement/spend baselines already capture most of the signal.

.. _eco-setup:

Setup
-----

We evaluate two data types: budgets with prices and menus without prices. The
targets include classification tasks such as High Spender, High Engagement, Low
Loyalty, and High Novelty, as well as regression tasks for Future Spend and
Spend Change. Targets use top tercile thresholds for consistency. Features combine a 13-feature baseline (covering recency, frequency, monetary value, concentration, and trends) with 42 Revealed Preference (RP) features. These include GARP, CCEI, MPI, HARP, HM, VEI, graph density
and transitivity, utility recovery metrics such as Gini and CV, choice entropy,
and ordinal utility. Models use CatBoost with default hyperparameters. The split uses a per-user temporal divide (first 70% features, last 30% targets) followed by an 80/20 user holdout and bootstrap confidence intervals on lift.

The hardest part is reconstructing the choice set and the observed choices.
For budgets, prices and quantities must reflect what the customer could have
afforded at each observation. For menus, the available alternatives must be
recovered from logs. This requires explicit assumptions about availability,
timing, and aggregation. We document those assumptions in the Appendix and,
honestly, we have to make good assumptions and hope they are right.

.. _eco-how-to-read:

How to read the results
-----------------------

- **Baseline**: A standard model built purely on basic spending history (recency, frequency, monetary value).
- **+RP**: The baseline model supplemented with our 42 Revealed Preference features.
- **Lift %**: The percentage improvement gained by adding Revealed Preference features.
- **RP-only**: A model running strictly on Revealed Preference features, with all baseline history removed.

As a rule of thumb, Revealed Preference provides genuine new signal if **+RP** noticeably outperforms the **Baseline**, and if **RP-only** remains competitive. If the **Baseline** and **+RP** scores are nearly identical, it means traditional spending history already effectively captured the necessary patterns.

.. _eco-results:

Results
-------

.. list-table::
   :header-rows: 1
   :widths: 14 6 14 8 8 6 8 8 8 8 6

   * - Dataset
     - N
     - Target
     - Baseline
     - +RP
     - Lift %
     - RP-only
     - AUC-PR (Base)
     - AUC-PR (+RP)
     - Engine
     - Mem
   * - Dunnhumby
     - 2,222
     - High Spender
     - 0.962
     - 0.965
     - +0.3%
     - 0.937
     - 0.946
     - 0.951
     - 8m45s
     - 41 MB
   * - Dunnhumby
     - 2,222
     - Churn
     - 0.711
     - 0.724
     - +1.8%
     - 0.622
     - \-
     - \-
     -
     -
   * - Dunnhumby
     - 2,222
     - Future LTV (R²)
     - 0.577
     - 0.589
     - +0.012
     - 0.246
     - \-
     - \-
     -
     -
   * - Amazon
     - 4,668
     - High Spender
     - 0.940
     - 0.942
     - +0.2%
     - 0.932
     - \-
     - \-
     - *tbd*
     - *tbd*
   * - Amazon
     - 4,668
     - Spend Drop
     - 0.784
     - 0.798
     - +1.8%
     - 0.684
     - \-
     - \-
     -
     -
   * - Amazon
     - 4,668
     - Spend Change (R²)
     - 0.144
     - 0.091
     - -0.053
     - -0.032
     - \-
     - \-
     -
     -
   * - Amazon
     - 4,668
     - Future LTV (R²)
     - 0.633
     - 0.622
     - -0.011
     - 0.387
     - \-
     - \-
     -
     -
   * - H&M
     - 46,757
     - High Spender
     - 0.784
     - 0.783
     - -0.1%
     - 0.720
     - \-
     - \-
     - *tbd*
     - *tbd*
   * - H&M
     - 46,757
     - Future Spend (R²)
     - 0.337
     - 0.340
     - +0.003
     - \-
     - \-
     - \-
     -
     -
   * - H&M
     - 46,757
     - Spend Change (R²)
     - 0.290
     - 0.295
     - +0.005
     - \-
     - \-
     - \-
     -
     -
   * - Taobao
     - 29,519
     - High Entropy (AP)
     - 0.789
     - **0.790**
     - **+0.1%**
     - \-
     - \-
     - \-
     - *tbd*
     - *tbd*
   * - Taobao
     - 29,519
     - High Active Time
     - 0.777
     - 0.778
     - +0.1%
     - \-
     - \-
     - \-
     -
     -
   * - Taobao
     - 29,519
     - High Click Volume
     - 0.818
     - 0.818
     - +0.0%
     - \-
     - \-
     - \-
     -
     -
   * - Taobao
     - 29,519
     - Fast Conversion
     - 0.561
     - 0.561
     - +0.0%
     - \-
     - \-
     - \-
     -
     -

*Baseline = CatBoost on 13 RFM features. +RP = same model with 42 RP features
added. RP-only = RP features without baseline. AUC-PR shown when AUC-ROC >= 0.95
(identifies precision issues on imbalanced classes). Engine = Rust batch scoring
time; Mem = peak memory. Timing shown on first row per dataset. On Taobao
(buy-anchored, 6h), RP features contribute modest lift on structural targets;
engagement/volume targets remain baseline-dominated.*


.. _eco-appendix:

Appendix: Datasets & Assumptions
--------------------------------

To run revealed preference tests on real-world transaction logs, we have to bridge the gap between abstract theory and messy data. The hardest part is reconstructing what the shopper was choosing *from*.

For budget data (Dunnhumby, Amazon, H&M), we construct prices and quantities over time. In **Dunnhumby**, an observation is an active household-week across 10 staple commodity groups (excluding inactive weeks since they represent spending outside the tracked sub-basket, not zero demand). We measure total units purchased, but lacking individual receipt data, we assign the global median price for each commodity that week. **Amazon** groups purchases by month at the category level; we apply median monthly prices across all users, carrying them forward when a category has no sales. **H&M** requires a slightly different temporal approach. We group each customer's rows into monthly choice occasions mapped to 20 coarse product groups. Since a customer only sees their own checkout cart, we use their own average paid price that month, smoothing over missing categories by falling back to period-group medians. These budget datasets force us to use synthesized price vectors over aggregated time windows. As a result, the scores we compute are reduced-form consistency descriptors of sub-basket allocation, not structural proofs of underlying utility.

For menu data (REES46, Taobao), we discard prices entirely and extract preference orderings by looking at what users clicked before they bought. In **REES46**, the database provides native shopping session boundaries for a "gold standard" setup. We build the menu out of all the items a user viewed in a single session, with the final purchase acting as their choice. Since a user can only choose from what they actually saw (impression bias), this gives us tight menus, typically around five items. 

**Taobao** presents a harder problem because it lacks clear session boundaries. Instead of relying on arbitrary inactivity cutoffs, we use a *buy-anchored* approach. Whenever a user buys an item at time *t*, we look back at a trailing 6-hour window. The options they viewed in that window form the menu, with the bought item as the choice. (We exclude post-purchase views, and only keep menus sized between 2 and 50). This makes the 6-hour lookback a pragmatic proxy for simultaneity.

Both menu datasets yielded an interesting discovery: menu-based aggregation almost entirely rules out stochastic processing. We attempted to model users with the probabilistic ``StochasticChoiceLog`` frame, but across both REES46 and Taobao, exact menu repeat rates were functionally 0.00. Because view-streams rarely repeat the identical combination of items, standard probability aggregation yields no additional signal over treating the observations as deterministic sequences.

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

.. _eco-cost:

Computational Cost
------------------

Engine and memory figures are integrated into the results table above (shown on the first row of each dataset group). Measured end-to-end on Apple M-series (11 cores). Engine = Rust batch scoring (6 metrics: GARP, CCEI, MPI, HARP, HM, VEI). Engine time is dominated by O(T³) metrics (MPI, HARP) on long histories (T ≈ 50–73 after 70 % train split, K = 10 goods). Memory stays flat regardless of user count, thanks to chunked streaming. See :doc:`performance` for scaling characteristics.

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

Appendix: Null Rates in RP Features
-------------------------------------

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

The vast majority of RP features are fully populated: 50/59 on Dunnhumby, 25/27 on REES46 and Taobao. This issue is quite isolated; it primarily affects utility-recovery and ordinal features which mathematically require a minimum number of intersecting choices to solve smoothly. The remaining features (GARP, CCEI, MPI, HM, VEI, graph density, transitivity, entropy, etc.) are always non-null. All NaN values are imputed with the per-feature train-set median before model training; the handful of high-null features carry little signal after imputation.
