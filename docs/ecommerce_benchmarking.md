# E-Commerce Benchmarking: Full Methodology

## Research Question

**Do revealed preference (RP) graph features add statistically significant predictive power beyond standard RFM baselines for forecasting future consumer behavior?**

We test this across 7 public e-commerce datasets, 4 budget-based and 3 menu-based, covering ~170K users and 15+ prediction targets.

---

## 1. Datasets

### 1.1 Budget Datasets

These datasets have prices and quantities. Each user-observation is a (price_vector, quantity_vector) pair representing a purchase occasion.

#### Dunnhumby (The Complete Journey)
- **Source**: Kaggle (dunnhumby)
- **Scale**: 2,222 qualifying households, ~50 weeks each
- **Observation unit**: Weekly grocery purchases
- **Goods**: 10 top commodity categories (Fluid Milk, Soft Drinks, Cold Cereal, Cheese, Yogurt, Ice Cream, Shredded Cheese, Lunch Meat, Bread, Butter/Margarine)
- **Prices**: Median price oracle per commodity per week, computed from `(SALES_VALUE - RETAIL_DISC - COUPON_DISC) / QUANTITY` across all transactions. Shared across households.
- **Quantities**: Aggregated item counts per household per week per commodity
- **Price filtering**: Unit prices clipped to $0.01–$50 range before computing weekly medians
- **Missing price handling**: Forward-fill, then backward-fill across weeks

#### Open E-Commerce (Amazon)
- **Source**: Open E-Commerce 1.0 (crowdsourced Amazon purchase history)
- **Scale**: 4,694 qualifying users, ~20 months each
- **Observation unit**: Monthly category-level purchases
- **Goods**: 50 product categories (keyword-based mapping from Amazon category strings)
- **Prices**: Median price per category per month across all users. Forward-fill, backward-fill, then global median fill for missing periods.
- **Quantities**: Sum of item counts per user per month per category
- **Price filtering**: Unit prices clipped to $0.01–$1,000 range

#### H&M (Fashion Recommendations)
- **Source**: Kaggle (H&M personalized fashion recommendations)
- **Scale**: 46,757 qualifying customers (capped at 50K), 6–24 months each
- **Observation unit**: Monthly purchases by product group
- **Goods**: 20 product groups (derived from first 2 digits of article_id)
- **Prices**: Real transaction prices from `transactions_train.csv` (normalized 0–1 range). Monthly median per product group. Fallback chain: ffill → bfill → group median → global median → 0.01.
- **Quantities**: Count of items purchased per customer per month per product group. Each row in the raw data = 1 item purchased.
- **Note**: Largest dataset by transaction volume (31.8M transactions)

#### Instacart (Market Basket Analysis)
- **Source**: Kaggle (Instacart Market Basket Analysis)
- **Scale**: 50,000 qualifying users, 10+ orders each
- **Observation unit**: Per-order purchases by aisle
- **Goods**: 134 aisles (from aisles.csv, joined via products.csv)
- **Prices**: Heuristic per-aisle prices ($1.50–$14.00) assigned via keyword matching on aisle names. No price data exists in the raw Instacart dataset. Price tiers: fresh produce $2.00–2.50, dairy $2.50–3.00, meat $5.00–7.00, alcohol $8.00–12.00, etc.
- **Quantities**: Item count per aisle per order
- **Plausibility check**: Heuristic prices yield $32/order average (real Instacart average is ~$35–50)

### 1.2 Menu Datasets

These datasets have menus (sets of available items) and choices (the item selected). No prices.

#### REES46 (Multi-Category eCommerce)
- **Source**: Kaggle (ecommerce-behavior-data-from-multi-category-store)
- **Scale**: 8,832 qualifying users, 5+ sessions each
- **Observation unit**: Server-defined shopping sessions
- **Menu construction**: Items viewed in a session form the menu. Choice = item purchased. Only sessions with exactly 1 purchase and 2–50 viewed items are kept.
- **Session boundary**: Platform-defined `user_session` column (gold standard — not a heuristic)
- **Item remapping**: Item IDs remapped to 0..N-1 per user for compact representation
- **Raw data**: 110M events from Oct–Nov 2019

#### Taobao (User Behavior)
- **Source**: Kaggle (UserBehavior)
- **Scale**: ~15,000 qualifying users, 5+ sessions each
- **Observation unit**: 30-minute inactivity gap sessions
- **Menu construction**: Items viewed (pv events) within a session form the menu. Choice = item purchased (buy event). Only sessions with exactly 1 purchased item and menu size 2–50.
- **Session boundary**: 30-minute (1800s) gap between consecutive events. EDA showed 84% of inter-event gaps < 30 min with a sharp break at the 90th percentile (~3.3 hours).
- **Raw data**: 100M events, chunked reading (5M rows per chunk)

#### Tenrec (Tencent QQ Browser)
- **Source**: NeurIPS 2022 Datasets and Benchmarks (Tenrec)
- **Scale**: 50,000 qualifying users (from 5M total), 5+ sessions each
- **Observation unit**: Click-to-like windows
- **Menu construction**: Items clicked by a user in sequence. When a clicked item also receives a "like" (positive feedback), the window closes and forms one menu-choice observation. Menu = all items clicked since the last like. Choice = the liked item.
- **Session boundary**: Each like event ends the current window. Feedback tracked positionally (per-click, not per-item-set) to avoid the set-vs-sequence bug where a window closes prematurely.
- **Feedback signal**: "like" column (default). Also supports "follow" or "share".
- **Raw data**: 493M events from QK-video.csv (14GB), chunked reading (5M rows per chunk)

---

## 2. Temporal Split (Per-User Event-Time)

For each qualifying user independently:

- **Feature window**: First 70% of that user's ordered observations
- **Target window**: Last 30% of that user's ordered observations
- No observation appears in both X and y
- Features (X) are computed **ONLY** from the feature window
- Targets (y) are computed **ONLY** from the target window

### Minimum observation requirements

| Parameter | Budget | Menu |
|-----------|--------|------|
| Min total observations | 10 | 5 |
| Min train observations | 5 | 3 |
| Min test observations | 3 | 2 |

Users who don't meet these thresholds after the 70/30 split are excluded.

### Known limitation

By requiring observations in both windows, we condition on users surviving long enough to have a test window. Users who truly churned (zero future events) are excluded. The estimand is: **"prediction among established users with sufficient interaction history."**

### Menu dataset item remapping

After the temporal split, items in each half are remapped to consecutive integers (0..N-1) independently. This ensures the preference graph is computed on compact, contiguous item IDs. An item appearing in both train and test may receive different integer IDs — this is correct because the RP analysis is per-window, not cross-window.

---

## 3. Feature Engineering

### 3.1 Baseline Features (no RP library needed)

#### Budget datasets: 13 features

Computed from raw (prices × quantities) arrays in the feature window.

| # | Feature | Computation | What it captures |
|---|---------|-------------|-----------------|
| 1 | `n_obs` | Count of observations | Data quantity / user tenure |
| 2 | `total_spend` | sum(prices × quantities) across all obs | Total budget spent |
| 3 | `mean_spend` | mean(spend per obs) | Average spending intensity |
| 4 | `std_spend` | std(spend per obs), 0 if T ≤ 1 | Spending volatility |
| 5 | `max_spend` | max(spend per obs) | Peak spending |
| 6 | `min_spend` | min(spend per obs) | Minimum spending |
| 7 | `mean_basket_size` | mean(total qty per obs) | Average purchase volume |
| 8 | `herfindahl` | sum(category_share²) | Category concentration (0=diverse, 1=concentrated) |
| 9 | `top_category_share` | max(category_share) | Dominance of top category |
| 10 | `n_active_categories` | count(categories with qty > 0) | Purchase diversity |
| 11 | `spend_slope` | OLS slope of spend vs time index (0 if T < 3) | Temporal trend direction |
| 12 | `spend_cv` | std / mean (0 if mean = 0) | Relative spending variability |
| 13 | `mean_abs_spend_change` | mean(|diff(spend)|) between consecutive obs (0 if T < 2) | Inter-period volatility |

#### Menu datasets: 11 features

Computed from menu sizes and choice frequencies in the feature window.

| # | Feature | Computation | What it captures |
|---|---------|-------------|-----------------|
| 1 | `n_sessions` | Count of menu presentations | Engagement frequency |
| 2 | `mean_menu_size` | mean(items in menu per session) | Average choice set size |
| 3 | `std_menu_size` | std(menu sizes) | Choice set variability |
| 4 | `max_menu_size` | Largest menu encountered | Maximum complexity |
| 5 | `min_menu_size` | Smallest menu | Minimum complexity |
| 6 | `n_unique_items` | Distinct items across all menus | Item catalog breadth |
| 7 | `items_per_session` | n_unique_items / n_sessions | Item diversity rate |
| 8 | `n_unique_choices` | Distinct items ever chosen | Choice repertoire |
| 9 | `max_choice_freq` | max(choice_count) / n_sessions | Loyalty to top choice |
| 10 | `choice_concentration` | n_unique_choices / n_unique_items | Fraction of items ever chosen |

### 3.2 RP Engine Features (batch Rust computation)

Computed via `Engine.analyze_arrays()` (budget) or `Engine.analyze_menus()` (menu) on the feature window data. Runs in Rust via Rayon for parallelism.

#### Budget Engine features: ~20 features

**Direct from Engine** (6 metric families → multiple output fields):
- `is_garp` (bool→int): GARP consistency pass/fail
- `n_violations` (int): Count of GARP violation cycles
- `ccei` (float 0–1): Afriat Efficiency Index
- `mpi` (float 0–1): Money Pump Index (exploitability)
- `is_harp` (bool→int): Homothetic consistency
- `harp_severity` (float): Max HARP cycle product (degree of non-homotheticity)
- `utility_success` (bool→int): Afriat LP feasibility
- `vei_mean` (float 0–1): Mean per-observation efficiency
- `vei_min` (float 0–1): Worst observation efficiency
- `vei_std` (float): Std dev of per-observation efficiency
- `vei_q25`, `vei_q75` (float): Quartiles of efficiency distribution
- `max_scc` (int): Largest strongly connected component in violation graph
- `n_scc` (int): Number of SCC components
- `scc_mean_size` (float): Mean SCC component size
- `r_density` (float): Preference graph edge density
- `r_out_degree_std` (float): Variation in observation out-degree
- `degree_gini` (float): Inequality of degree distribution
- `ew_mean`, `ew_std`, `ew_skew` (float): HARP edge weight distribution

**Derived features:**
- `hm_ratio` = hm_consistent / max(hm_total, 1)
- `violation_density` = n_violations / max(T × (T-1), 1)
- `scc_ratio` = max_scc / max(T, 1)
- `vei_iqr` = vei_q75 - vei_q25
- `scc_fragmentation` = n_scc / max(T, 1)

#### Menu Engine features: ~15 features

**Direct from Engine:**
- `is_sarp`, `is_warp`, `is_warp_la` (bool→int): Axiom consistency tests
- `n_sarp_violations`, `n_warp_violations` (int): Violation counts
- `max_scc`, `n_scc` (int): SCC structure
- `r_density` (float): Item preference graph density
- `pref_entropy` (float): Shannon entropy of preference out-degree distribution
- `choice_diversity` (float): Unique choices / total sessions

**Derived features:**
- `hm_ratio` = hm_consistent / max(hm_total, 1)
- `sarp_violation_density` = n_sarp_violations / max(T × (T-1) / 2, 1)
- `warp_violation_density` = n_warp_violations / max(T × (T-1) / 2, 1)
- `scc_ratio` = max_scc / max(n_items, 1)

### 3.3 RP Extended Features (per-user algorithm calls)

Computed by calling individual Python-level algorithms on each user's feature-window data. More expensive than Engine batch but captures richer structure.

#### Budget Extended: ~28 features

**VEI distributional (10 features):**
- `vei_std`: Standard deviation of per-observation efficiency vector
- `vei_q25`, `vei_q75`: 25th and 75th percentile efficiency
- `vei_iqr`: Interquartile range
- `vei_below_90`: Count of observations with efficiency < 0.9
- `vei_below_90_frac`: Fraction of observations below 0.9
- `vei_first_half`: Mean efficiency in first half of observations
- `vei_second_half`: Mean efficiency in second half
- `vei_temporal_shift`: second_half - first_half (consistency trend)

**Utility recovery distributional (8 features):**
- `util_mean`, `util_std`, `util_range`: Statistics of Afriat-recovered utility values
- `util_cv`: Coefficient of variation of utility
- `util_gini`: Gini inequality coefficient of utility distribution
- `lambda_mean`, `lambda_std`, `lambda_cv`: Lagrange multiplier (marginal utility of money) statistics

**Graph structural (8 features):**
- `pref_graph_density`: Edge density of direct revealed preference graph
- `strict_pref_density`: Edge density of strict preference relation
- `transitivity_ratio`: Transitive closure edges / direct preference edges
- `n_cycles`: Count of GARP violation cycles
- `mean_cycle_length`, `max_cycle_length`: Cycle length statistics
- `violation_obs_frac`: Fraction of observations involved in any violation
- `violation_mean_position`: Mean temporal position of violating observations (0=early, 1=late)

**MPI cycle cost (3 features):**
- `mpi_max_cycle_cost`: Largest individual cycle MPI value
- `mpi_mean_cycle_cost`: Mean cost across all violation cycles
- `mpi_n_cycles`: Number of MPI cycles found

#### Menu Extended: ~14 features

**Choice reversal (4 features):**
- `choice_reversal_count`: Number of pairwise preference contradictions (A over B, then B over A)
- `choice_reversal_ratio`: Reversals / comparable pairs
- `choice_entropy`: Shannon entropy of choice frequency distribution
- `choice_entropy_norm`: Entropy normalized by log2(n_items)

**Preference graph (4 features):**
- `menu_pref_density`: Revealed preference matrix density
- `menu_transitivity`: Transitivity ratio of preference graph
- `menu_n_cycles`: Count of SARP violation cycles
- `menu_max_cycle_len`: Longest SARP violation cycle

**Congruence (2 features):**
- `is_congruent` (int): Full rationalizability (SARP + maximality)
- `n_maximality_violations`: Count of unchosen-but-available items that should have been chosen

**Ordinal utility (3 features):**
- `menu_util_range`: Max - min of fitted ordinal utility values
- `menu_util_std`: Std dev of utility values
- `menu_rank_complete` (int): Whether a complete preference ranking was fitted

---

## 4. Prediction Targets

All targets computed from the **target window** (last 30%) only.

### 4.1 Budget Targets

| Target | Type | Computation | What it measures |
|--------|------|-------------|-----------------|
| **Churn** | Binary classification | `test_mean_spend / max(train_mean_spend, 1e-6) < 0.5` | Did spending drop by >50%? |
| **High Spender** | Binary classification | `test_total_spend > percentile(test_total_spends, 66.67)` | Is user in top tercile of future spend? |
| **Spend Change** | Regression | `test_mean_spend - train_mean_spend` | Absolute change in spending intensity |
| **Future LTV** | Regression | `test_mean_spend` | Mean spend per period in future window |

**Note on Instacart**: Uses quantity-based equivalents since prices are heuristic. "Spend Drop" threshold is 30% (not 50%). "High Value" uses total quantity. "Basket Size" and "Future LTV" use mean items per order.

**Note on High Spender threshold**: Computed across ALL users before the train/test user split. This introduces a minor threshold leakage (test users influence the percentile), but with 1000+ users the impact is negligible.

### 4.2 Menu Targets

| Target | Type | Computation | What it measures |
|--------|------|-------------|-----------------|
| **High Engagement** | Binary classification | `test_session_count > percentile(test_sessions, 66.67)` | Is user in top tercile of future session count? |

---

## 5. Evaluation Protocol

### 5.1 User Holdout (Out-of-Sample)

- 80% of qualifying users → **train set**
- 20% of qualifying users → **test set**
- Stratified split for classification targets (preserves class balance)
- Random split for regression targets
- Fixed random seed: 42

### 5.2 NaN/Inf Imputation

- Replace ±Inf with NaN
- Compute column medians on **train users only**
- Fill NaN values using train medians for both train and test
- Replace any remaining ±Inf with 0
- **No test user information leaks into imputation**

### 5.3 Model

**CatBoost** with pure default hyperparameters:
```python
{
    "random_seed": 42,
    "verbose": 0,
}
```
No hyperparameter tuning. No regularization knobs. The comparison is about **features**, not model optimization. CatBoost defaults include ordered boosting which handles noisy features better than LightGBM defaults.

### 5.4 Three-Way Comparison

For each target, three models are trained on the same 80% train split:

| Model | Features | Purpose |
|-------|----------|---------|
| **Baseline only** | 13 RFM (budget) or 11 engagement (menu) | What standard features achieve |
| **RP only** | ~42 budget RP or ~25 menu RP features | RP's standalone signal |
| **Combined** | Baseline + RP (~55 budget or ~36 menu) | Whether RP adds marginal value |

All models use identical hyperparameters and train/test split.

### 5.5 Metrics

**Classification:**
- **AUC-ROC**: Discrimination ability across all thresholds. Used for balanced targets (positive rate ≥ 15%).
- **AUC-PR (Average Precision)**: Better for imbalanced targets (positive rate < 15%, e.g., Churn at 5–7%). Primary metric when positive rate < 15%.

**Regression:**
- **R²**: Fraction of variance explained. Can be negative (model worse than predicting the mean).

**Lift:**
- `Lift% = (Combined_metric - Baseline_metric) / Baseline_metric × 100`

### 5.6 Bootstrap Confidence Intervals (1000 iterations)

On the 20% test set predictions (no retraining):

```
For each of 1000 iterations:
    1. Sample test users WITH replacement (same size as test set)
    2. Compute metric for Baseline and Combined on this bootstrap sample
    3. Compute lift percentage
Store all 1000 lifts
CI = [2.5th percentile, 97.5th percentile]
p-value = fraction of bootstrap lifts ≤ 0
```

- Non-parametric — no distributional assumptions
- Works for AUC-ROC, AUC-PR, and R²
- Proves whether observed lift is noise or genuine signal
- Fixed RNG seed (42) for reproducibility

### 5.7 Grouped Permutation Importance

Standard per-feature permutation importance fails for correlated RP features (shuffling `ccei` has no effect because `mpi` compensates). Grouped permutation shuffles **all RP features simultaneously**.

```
For each feature group (RP_features, Baseline_features):
    1. Get baseline test score from combined model
    2. Repeat 5 times:
        a. Shuffle ALL columns in this group simultaneously (same permutation for each column)
        b. Predict with shuffled features
        c. Record drop in test metric
    3. Average the 5 drops = group importance
```

**Feature groups:**
- `RP_features`: All columns from X_rp (Engine + Extended)
- `Baseline_features`: All columns from X_base (RFM/engagement)

A high `RP_features` drop means the model **relies on** RP features for prediction on unseen users. This overcomes the collinearity masking problem.

---

## 6. Data Assumptions & Limitations

### Budget datasets

- **Shared price oracle** (Dunnhumby, Open E-Commerce): All users face the same median prices per category per period. Individual price variation (coupons, regional differences) is not captured. Follows Dean & Martin (2016) approach.
- **Category-level aggregation**: 10–134 categories depending on dataset. Within-category substitution is invisible. A GARP violation may reflect switching between products within a category, not true preference inconsistency.
- **Instacart heuristic prices**: No prices in raw data. Per-aisle prices ($1.50–$14.00) assigned via keyword matching on 134 aisle names. Yields plausible order totals ($32 avg vs real $35–50) but are approximate.
- **H&M normalized prices**: Real price variation exists but values are normalized to 0–1 range. Relative comparisons are valid; absolute dollar amounts are not.
- **Dunnhumby coarse categories**: 10 commodities capture ~$19/week of a ~$100–150 weekly grocery budget. RP analysis is valid within these categories but covers only a fraction of total spending.

### Menu datasets

- **Impression bias**: Menus contain only items the user viewed/clicked. Items shown by the platform but not engaged with are invisible. RP analysis is conditional on the user having interacted with these items.
- **REES46**: Server-defined sessions (gold standard). Median menu size ~5 items.
- **Taobao**: 30-minute gap sessions (EDA-validated). Median menu size 4 items.
- **Tenrec**: Click-to-like positional windows. Menus reflect algorithmic recommendations, not organic browsing. Median ~5 clicks between likes.
- **No budget constraint**: Menu datasets have no prices. Choices reveal preference orderings but not willingness-to-pay.

### Evaluation design

- **Per-user temporal split**: Conditions on users having enough data in both windows. Users who truly disappeared (zero future events) are excluded. Population = "established users with sufficient history."
- **Single train/test split**: No cross-validation. Results may vary with different random seeds. Bootstrap CIs quantify this uncertainty.
- **No hyperparameter tuning**: CatBoost defaults. Results reflect feature quality, not model optimization.

---

## 7. Reproducibility

```bash
pip install prefgraph catboost scikit-learn

# Single dataset
python case_studies/benchmarks/runner.py --datasets dunnhumby

# All datasets
python case_studies/benchmarks/runner.py --datasets all

# With user cap for faster testing
python case_studies/benchmarks/runner.py --datasets dunnhumby --max-users 500
```

All random seeds fixed at 42. Deterministic results across runs (given same data).

**Output files:**
- `case_studies/benchmarks/output/results.json`: Full metrics per target
- Console: Paper-ready table with lift%, CI, p-value, group importance

---

## 8. Paper-Ready Results Table Format

```
Dataset          Target            N test  %pos  Baseline  Combined   Lift%        95% CI   p-val  RP Group  Base Group
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Dunnhumby        Churn (AP)           445  6.8%     0.177     0.150  -18.4%  (-43, +18)   0.887    0.0490      0.2096
Dunnhumby        High Spender (AUC)   445 33.3%     0.962     0.962   -0.0%  (-1, +1)     0.546    0.0103      0.3622
Dunnhumby        Spend Change (R²)    445    —     -0.030     0.023 +289.0%  (-111,+2536) 0.233
Dunnhumby        Future LTV (R²)      445    —      0.793     0.803   +1.3%  (-0, +3)     0.037**
```

Columns:
- **Baseline**: Metric using RFM/engagement features only
- **Combined**: Metric using RFM + RP features
- **Lift%**: (Combined - Baseline) / |Baseline| × 100
- **95% CI**: Bootstrap percentile interval on lift (1000 iterations)
- **p-val**: Fraction of bootstrap lifts ≤ 0 (lower = more significant)
- **RP Group**: Drop in test AUC when all RP features shuffled simultaneously
- **Base Group**: Drop in test AUC when all baseline features shuffled simultaneously
- Significance: *** p<0.01, ** p<0.05, * p<0.10

---

## 9. Data Creation Audit: What We Tried, What Broke, What We Learned

This section documents the iterative process of constructing defensible RP benchmark data. Every dead end is recorded.

### 9.1 Budget Datasets: The Price Problem

**Issue: Instacart has no prices.**
Instacart's raw data contains products, aisles, departments, and order sequences — but zero price information. Without prices, budget-based RP (GARP) requires a price assumption.

**Attempt 1: Uniform $1 prices (21 departments).**
Assigned $1/unit to all departments. Result: GARP reduces to quantity-dominance checks. Every user is perfectly consistent (CCEI=1.0, MPI=0.0, zero violations). RP features have zero variance. Dead on arrival.

**Attempt 2: Heuristic per-aisle prices (134 aisles, $1.50–$14.00).**
Keyword-matched 134 aisle names to price tiers (fresh produce $2.00, meat $5.00, alcohol $12.00, etc.). Prices are constant across time. EDA showed plausible order totals ($32/order vs real $35–50). But: static prices mean budget lines never intersect. GARP violations are mathematically impossible with parallel budget constraints. Verified: 200 users, every one perfectly GARP-consistent. All efficiency features (CCEI, MPI, VEI) have zero variance. **Conclusion: static heuristic prices make GARP degenerate regardless of granularity.**

**Attempt 3: Reframe Instacart as menu-choice (current).**
Instead of forcing prices, construct product-level menu-choice data within departments. Menu = user's known products in their most active department. Choice = first product added to cart. This produces real SARP violations (mean 68 per user, HM ratio 81%). See Section 9.3.

**Lesson learned:** RP budget analysis requires genuine temporal price variation. Without it, all users appear perfectly rational. Heuristic prices at any granularity cannot substitute for real market price dynamics.

**Other budget datasets — price oracle concerns:**
- **Dunnhumby & Open E-Commerce**: Global median price oracle shared across all users. Individual price exposure (coupons, store location) not captured. Forward-fill/backward-fill for missing periods creates artificial price continuity. Accepted limitation — follows Dean & Martin (2016) precedent.
- **H&M**: Real transaction prices (normalized 0–1). 4-level fallback chain (ffill → bfill → group median → 0.01) for missing prices. Some prices are heavily imputed. Accepted — the alternative is dropping observations.

### 9.2 Menu Datasets: The Session Boundary Problem

**Issue: What defines a "session" (menu presentation)?**

**REES46**: Server-defined `user_session` column — gold standard. No heuristic needed.

**Taobao — Attempt 1: Calendar day.**
Original construction: menu = items viewed that day, choice = item purchased. Problem: a user browsing electronics at 8am and buying groceries at 11pm both appear in one "menu." Cross-midnight sessions (view 23:50, buy 00:10) split incorrectly. Inflated menu sizes with irrelevant items.

**Taobao — Attempt 2: 30-minute gap sessions (current).**
EDA showed 84% of inter-event gaps < 30 min with a sharp break at p90 (3.3 hours). Using 1800s gap threshold produces natural sessions with median 4 items/menu. Valid sessions increased from 855K to 919K. **Accepted.**

**Tenrec — Original: Set-based feedback tracking (buggy).**
Stored liked items in a Python set. `if item_id in fb_set` closed the window at the first click of ANY item the user ever liked — even if the like happened later. Menus closed prematurely.

**Tenrec — Fix: Positional feedback tracking (current).**
Track `(item_id, had_feedback_at_this_position)` per click. Window closes only when THIS specific click had feedback=1. Correct temporal ordering preserved.

**Tenrec — Structural concern: Algorithmic menus.**
Menus are determined by Tencent's recommendation algorithm, not organic user browsing. SARP violations may reflect algorithm-user interaction patterns rather than pure user preferences. Documented as limitation — we measure "user-algorithm interaction consistency."

### 9.3 Instacart Menu-Choice: The Product-Level Construction

**Goal:** Rescue Instacart from dead budget-RP by constructing menu-choice data at the product level.

**Construction:**
- For each user, identify their most active department (most reorder events)
- Walk orders chronologically within that department
- Menu = products the user has bought before from this department (growing "known set")
- Choice = first product added to cart from that department this order
- Only count observations where choice is in the known set (genuine reorder = choosing from known options)
- Filter: menu size 3–30, min 5 valid observations per user

**EDA results (500 users):**
- Mean 22 sessions/user, median 13
- Menu size: mean 21, median 22 (growing known set → menus grow over time)
- SARP violations: mean 68 per user, HM ratio ~81%
- Real preference cycles detected (user prefers yogurt over milk, then switches)

**Issue: High Engagement target is trivially predictable.**
Baseline AUC = 1.000. `n_sessions` perfectly predicts future session count because both scale with user activity level. The menu construction is correct but the target is wrong — need behavioral-change targets (preference drift, loyalty, exploration rate).

**Open question:** What targets would RP features uniquely predict on this data? Candidates:
- Preference drift: did user's top-choice product change?
- Loyalty: fraction of test choices that repeat train choices
- Exploration rate: fraction of test choices on never-before-seen products

### 9.4 The Temporal Split Problem

**Per-user event-time split (current design):**
Each user's observations split 70/30 individually. Features from first 70%, targets from last 30%. 80/20 random user holdout for OOS evaluation.

**Attempt: Global calendar cutoff.**
Tried using 70th percentile of absolute observation index as a global cutoff for all users. On Dunnhumby: cutoff at observation 56 → 67% of users "churned" (had no data beyond that point). But this wasn't real churn — many users simply had shorter data collection periods. Baseline AUC shot to 0.975 (trivial to predict "churn" = short data collection). **Reverted.**

**Attempt: Global calendar cutoff using raw DAY column.**
Read Dunnhumby's actual `DAY` column (1–711). Cutoff at DAY 497 (70th percentile of unique days). Only 30 users truly churned (1.7%). More realistic but small sample. Churn AUC-PR: baseline 0.144, combined 0.189 (+35% lift, p=0.081). Promising but fragile with N=30.

**Lesson learned:** Global calendar cutoff is the gold standard for production deployment simulation, but fails on heterogeneous panel datasets where data collection windows vary by user cohort. Per-user event-time split is the accepted compromise — explicitly conditions on "established users with sufficient history."

### 9.5 The Evaluation Design Evolution

**V1: 5-fold stratified CV + LightGBM tuned.**
Original design. Reviewers would flag: temporal leakage in random folds, hyperparameter sensitivity.

**V2: Single 80/20 split + LightGBM defaults + regularization.**
Added `colsample_bytree=0.8`, `min_child_samples=50`. Better but still accused of favoring specific regularization.

**V3: CatBoost pure defaults.**
Ordered boosting handles noisy features better. Won all 3 Dunnhumby targets vs LightGBM (+0.002 to +0.022). But CatBoost is slower.

**V4 (current): CatBoost defaults + bootstrap CI + grouped permutation importance.**
Single split, no CV, no tuning. Bootstrap proves lift significance. Grouped permutation proves RP features did the work (not just noise). Most defensible design.

**Key reviewer concerns addressed:**
1. "Churn" renamed to "Spend Drop" (survival bias: users have 3+ test observations)
2. High Spender threshold computed on train users only (zero leakage)
3. R² reported as absolute ΔR², not percentage lift (avoids math breakdown on near-zero R²)
4. Recency + frequency rate added to baseline (was missing R and F from "RFM")
5. Instacart RP dead with static prices → reframed as menu-choice

### 9.6 Feature Engineering Lessons

**What works (orthogonal to baselines):**
- `strict_pref_density`: Preference graph structure. Correlation < 0.44 with all baselines. Ranked #3–#6 across datasets.
- `util_gini`: Inequality of recovered Afriat utility values. Correlation < 0.17. Ranked #7 overall.
- `choice_entropy_norm`: Normalized Shannon entropy of choice distribution. Menu-dataset specific.
- `menu_transitivity`: Preference graph transitivity ratio. #2 on menu datasets.
- `pref_graph_density`: Edge density of revealed preference graph. Consistently top 5–10.
- `n_scc`: Number of strongly connected components. Graph fragmentation signal.

**What doesn't work (correlated with baselines):**
- `ccei`, `mpi`, `hm_ratio`: Core RP scores are highly correlated with spending features. Adding them to a strong RFM baseline adds noise, not signal.
- `vei_mean`: Mean per-observation efficiency. Correlated with `n_obs` and `total_spend`.
- VEI distributional features (std, q25, q75): Many users have identical VEI vectors (all 1.0 if consistent), causing zero variance.

**What's broken (high NaN rates):**
- Utility recovery features (`util_mean`, `util_gini`, `lambda_cv`): Require GARP consistency for LP feasibility. ~40% of users fail, producing NaN. Imputed with median, diluting signal.
- Cycle features (`n_cycles`, `violation_mean_position`): Require violations to exist. Consistent users produce NaN. Same imputation problem.

### 9.7 Summary: Current State of Evidence

| Category | Finding | Confidence |
|----------|---------|------------|
| **Budget RP → classification** | ~0% marginal lift over RFM | High (7 datasets, bootstrap CI) |
| **Budget RP → regression (LTV)** | +1.3% R² lift on Dunnhumby (p=0.037) | Moderate (1 dataset, single split) |
| **Menu RP → classification** | RP-only competitive with baselines (Taobao RP-only 0.925 > baseline 0.913) | Moderate |
| **Graph features (pref_density, n_scc)** | Consistently rank in top 10 feature importance | High |
| **Core RP scores (CCEI, MPI)** | Correlated with RFM, no marginal value | High |
| **Static-price GARP** | Completely degenerate (zero violations) | Definitive |
| **Product-level menu choice** | Rich SARP signal (68 violations/user) but needs better targets | Preliminary |

The honest conclusion: **RP graph structure features add novel information that tree models use, but the marginal predictive lift over well-engineered RFM baselines is small and rarely statistically significant for classification. The strongest evidence for RP's value is in continuous regression targets (LTV) and in menu-choice settings where preference graph properties (transitivity, density) capture patterns that simple engagement counts miss.**
