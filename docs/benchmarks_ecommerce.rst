Predicting Customer Spend and Engagement
=========================================

Across 11 datasets and 32 prediction targets, revealed preference features add near-zero marginal lift on average. Standard spend and engagement baselines already capture most of the signal.

Setup
-----

We test whether choice-consistency features improve predictions of spend, churn, engagement, and loyalty at the individual user level. Each user's purchase or click history is split in time. The early portion produces features. The later portion defines the prediction target. A regularized LightGBM is evaluated by 5-fold cross-validation. An L1-penalized logistic regression is used separately for feature selection and directional interpretation. All runs use SEED 42.

Features
^^^^^^^^

The baseline consists of 13 features that capture how much a user spends, how often they buy, and how concentrated their purchases are. These include total spend, number of observations, mean basket size, spend trend, category concentration, and recency. The revealed preference features consist of 42 measures that capture how consistently a user makes decisions. These include rationality scores such as CCEI and MPI, graph-structural measures such as transitivity and preference density, choice diversity measures such as entropy and reversal rate, and utility-recovery measures such as Afriat utility dispersion and efficiency trajectories.

Targets
^^^^^^^

Each dataset produces one or more prediction targets from the held-out future window. Spend Drop flags users whose average spending fell by more than half. High Spender identifies users in the top third of total expenditure. Future LTV and Spend Change predict continuous spend levels. High Engagement identifies the most active users by session count. Low Loyalty flags users whose choices became more dispersed over time. High CTR measures effective click-through rate on news impressions. All classification targets use top-tercile thresholds computed on the training set to prevent leakage.

.. _eco-results:

Results
-------

The table shows the lift from adding 42 revealed preference features on top of a 13-feature baseline, measured by 5-fold cross-validation with a regularized LightGBM. The standard deviation across folds is shown in parentheses.

.. list-table::
   :header-rows: 1
   :widths: 12 16 7 12 12 12 12

   * - Dataset
     - Target
     - N
     - Base AUC-ROC
     - +RP AUC-ROC
     - Base AUC-PR
     - +RP AUC-PR
   * - Dunnhumby
     - Spend Drop
     - 2,222
     - .730 (.020)
     - .706 (.016)
     - .180
     - .145
   * - Dunnhumby
     - High Spender
     - 2,222
     - .958 (.003)
     - .957 (.002)
     - .929
     - .925
   * - Dunnhumby
     - Future LTV (R²)
     - 2,222
     - .562 (.050)
     - .569 (.042)
     -
     -
   * - Amazon
     - Spend Drop
     - 4,694
     - .776 (.020)
     - **.789 (.012)**
     - .226
     - **.248**
   * - Amazon
     - High Spender
     - 4,694
     - .944 (.007)
     - .945 (.008)
     - .903
     - .905
   * - H&M
     - High Spender
     - 46,757
     - .792 (.004)
     - .791 (.003)
     - .683
     - .682
   * - H&M
     - Spend Change (R²)
     - 46,757
     - .285 (.013)
     - .287 (.013)
     -
     -
   * - Instacart
     - Low Loyalty
     - 50,000
     - .969 (.002)
     - .969 (.002)
     - .935
     - .935
   * - REES46
     - Low Loyalty
     - 8,832
     - .887 (.008)
     - .887 (.008)
     - .709
     - **.715**
   * - Taobao
     - Engagement
     - 15,806
     - .930 (.006)
     - .931 (.006)
     - .806
     - .807
   * - Taobao BW
     - Low Loyalty
     - 590
     - .989 (.006)
     - .991 (.005)
     - .868
     - .873
   * - Tenrec
     - Engagement
     - 50,000
     - .993 (.000)
     - .993 (.000)
     - .983
     - .983
   * - MIND
     - High CTR
     - 5,091
     - .657 (.022)
     - .653 (.018)
     - .514
     - .510
   * - FINN
     - Low Loyalty
     - 46,858
     - .958 (.001)
     - .958 (.001)
     - .780
     - .781

All values are 5-fold cross-validated means from a regularized LightGBM. Standard deviations across folds are in parentheses. Bold marks the one target where the lift is clearly above the fold-to-fold noise. Full results are in ``output/cv_results.json``.

Findings
--------

The standard deviations across folds are small relative to the means on most targets. This means the results are stable and not driven by a lucky train-test split. Earlier experiments with a single 80/20 holdout showed larger apparent lifts of 1 to 20 percent on some targets, but these did not replicate under cross-validation. The in-sample AUC on the holdout was 1.000 for most targets, confirming that the single-split estimates were inflated by overfitting.

Amazon Spend Drop is the only target where revealed preference features produce a lift that exceeds the fold-to-fold variation. The AUC-ROC improves from 0.776 to 0.789 and the AUC-PR from 0.226 to 0.248. On this target, per-observation budget efficiency detects declining rationality before spending drops. REES46 Low Loyalty shows a smaller positive on AUC-PR from 0.709 to 0.715. Everything else is within the noise.

Revealed preference features consistently rank among the most important features in the model. Menu transitivity appears in the top ten for 18 of 19 menu targets. Choice entropy appears in 17 of 19. These features describe how consistently a user makes decisions, which is different from what volume and frequency baselines measure.

The conclusion is that revealed preference features do not improve prediction accuracy over well-constructed baselines in most settings. They do contain valuable signal about decision-making behavior that is worth investigating. The directional patterns from the Lasso model below suggest interpretable relationships between choice consistency and future outcomes that standard features do not capture.

Suggestive Directions
---------------------

Users who choose from a wider variety of items tend to be more engaged and less loyal. Choice entropy is the most frequently selected revealed preference feature across all datasets. It is positive for engagement in every dataset where it appears, and positive for low loyalty in six of six cases. This suggests that when users explore broadly rather than repeating the same purchases, they are more likely to remain active but less likely to stick with any single product.

Users with more fragmented preference graphs tend to spend more. The number of strongly connected components in the preference graph is positive for high spending in all five budget targets where it appears. This suggests that when users have internally contradictory preferences across product categories, they end up buying more rather than less. Impulse buyers who do not optimize may be more valuable customers than deliberate ones.

Users whose per-observation purchase efficiency is declining are more likely to churn. On Amazon, per-observation efficiency is the only feature the Lasso selects for spend drop prediction. The coefficient is negative, meaning lower efficiency predicts higher churn. This suggests that when users start making less rational purchases over time, they are mentally disengaging before their spending actually falls.

Users with more transitive preference graphs tend to click more but stay more loyal. Menu transitivity is positive for engagement targets and negative for low loyalty targets. This suggests that when users have a clear and consistent preference ordering, they engage actively but do not wander across products.

These directions are suggestive rather than causal. The Lasso selects features under strong regularization, so only the most robust associations survive. But the consistency of directions across independent datasets and domains gives some confidence that these patterns reflect real behavioral differences rather than statistical artifacts.

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

Feature importances are computed from the full-scale single-holdout LGBM run across all datasets. The rankings are consistent with the cross-validated results above because gain-based importance aggregates across all trees and is less sensitive to a single split than prediction accuracy.

The three revealed preference features that appear in the top ten are:

- **Menu transitivity** measures how often a user's preferences form a consistent ordering. A user who prefers A over B and B over C but then picks C over A has low transitivity. High transitivity means the user knows what they want.

- **Choice entropy** measures how spread out a user's choices are across items. A user who always picks the same item has zero entropy. A user who picks many different items has high entropy. This captures exploration versus habit.

- **Menu preference density** measures how many pairwise comparisons can be inferred from a user's choices. A user who has been observed choosing between many different pairs of items has a dense preference graph. This captures how much evidence the data contains about that user's preferences.

.. _eco-replication:

Replication
-----------

.. code-block:: bash

   pip install prefgraph lightgbm scikit-learn

   # 5-fold CV results (main table)
   python case_studies/benchmarks/cv_benchmark.py --datasets validated

   # Single holdout with both models
   python case_studies/benchmarks/runner.py --datasets validated --model both

   # Quick smoke test
   python case_studies/benchmarks/runner.py --datasets validated --max-users 250 --model both

   # Regenerate from cached results
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

Revealed preference features are largely orthogonal to baseline features with a median cross-correlation of 0.12. Within the 59 revealed preference features on budget data, 45 pairs have absolute correlations above 0.80. These cluster into nine groups.

The largest cluster contains seven VEI and utility features that all measure the same underlying Afriat utility scale. The second largest contains six features linking violations, graph components, and VEI dispersion. These two clusters account for 13 of the 59 features and are heavily redundant. The third cluster is the one that matters for prediction. It contains preference density, strict preference density, and transitivity ratio, which correlate at 0.95 or above with each other. These are the graph-structural features that appear in the top ten importance list. CCEI and MPI form their own two-feature cluster at 0.92.

The three top-performing revealed preference features are only moderately correlated with the traditional consistency scores. Transitivity correlates with CCEI at about 0.5 and with MPI at about 0.3. Choice entropy is nearly uncorrelated with all traditional scores. This means the features that drive model importance are capturing different information from the classical rationality tests.

In the Lasso model, baseline features have a median standardized coefficient about five times larger than revealed preference features. However, the top revealed preference features reach magnitudes comparable to major baseline features. Choice entropy reaches 3.0 on REES46 Low Loyalty, which is in the same range as session count at 5.4. The number of strongly connected components reaches 3.8 on Instacart. When the Lasso does select a revealed preference feature, it gives it meaningful weight.

Null Rates
---------------------

50 of 59 revealed preference features are always populated on budget data. 25 of 27 are always populated on menu data. The exceptions are utility-recovery features that require a minimum number of intersecting choices. All missing values are imputed with training-set medians.

Dataset Descriptions
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 12 88

   * - Dataset
     - Description
   * - Dunnhumby
     - Household-level grocery scanner data from 2,222 households over two years. An observation is one active household-week across 10 staple commodity groups. Quantities are total units purchased per category. Prices are the global median price for each commodity that week, since individual receipt data is not available. Targets are Spend Drop, High Spender, and Future LTV.
   * - Amazon
     - Amazon purchase records from 4,694 consumers across 15 product categories. Purchases are grouped by month at the category level. Prices are median monthly prices across all users, carried forward when a category has no sales in a given month. Targets are Spend Drop, High Spender, Spend Change, and Future LTV.
   * - H&M
     - Fashion transactions from 46,757 customers over two years. Each customer's purchases are grouped into monthly choice occasions across 9 product groups. Prices are the customer's own average paid price that month, falling back to period-group medians for months with no purchase. Targets are High Spender, Spend Change, and Future Spend.
   * - Instacart
     - Grocery reorder data from 50,000 users. Each observation is a single reordered SKU in a given order and aisle. The menu is the set of distinct products the user purchased from that aisle in the trailing three orders. The choice is the product they actually reordered. Targets are High Engagement, Low Loyalty, and High Novelty.
   * - REES46
     - Multi-category e-commerce sessions from 8,832 users over two months. The platform provides native session boundaries. The menu is all items the user viewed in a single session. The choice is the purchased item. Targets are High Engagement, Low Loyalty, and High Novelty.
   * - Taobao
     - Click and purchase events from 15,806 users on China's largest e-commerce platform. Sessions are defined by a 30-minute inactivity gap. The menu is all items viewed in the session. The choice is the purchased item. Target is High Engagement.
   * - Taobao BW
     - A variant of Taobao using buy-anchored windows. Whenever a user purchases an item, the menu is defined as all items viewed in the preceding 6-hour window. This avoids arbitrary session boundaries. Only 590 users qualify after filtering. Targets are High Engagement, Low Loyalty, High Entropy, Pref Drift, High Click Volume, High Active Time, and Fast Conversion.
   * - RetailRocket
     - Click-stream data from a large e-commerce platform. Sessions are reconstructed using 30-minute inactivity windows. The menu is all items viewed in a session. The choice is the purchased item. Only 47 users have enough repeat sessions to qualify. Targets are High Engagement and Low Loyalty.
   * - Tenrec
     - Video recommendation data from 50,000 users on Tencent QQ Browser. The menu is all items clicked since the previous positive feedback event. The choice is the item the user liked or shared. Targets are High Engagement and Low Loyalty.
   * - MIND
     - News impression logs from 5,091 users on Microsoft News. Each impression is a directly logged slate of candidate articles. Only impressions with exactly one click are retained so the choice is unambiguous. The menu is the full set of articles shown. Targets are High Engagement, Low Loyalty, High Novelty, and High CTR.
   * - FINN
     - Directly logged recommendation slates from 46,858 users on Norway's largest classifieds marketplace. Each slate of up to 25 items is recorded by the platform along with which item was clicked. This is the strongest menu dataset because the choice set is observed rather than reconstructed. Targets are High Engagement, Low Loyalty, and High Search Ratio.
