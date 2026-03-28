# ML Validation Design: Revealed Preference Features as Predictive Signals

## Question

Do revealed preference (RP) graph features add predictive power beyond standard RFM baselines for forecasting future consumer behavior?

## Data

Seven public e-commerce datasets. Each user has a time-ordered sequence of observations (purchases or menu choices).

| Dataset | N users | Type | Observations | Price source |
|---------|---------|------|-------------|--------------|
| Dunnhumby | 2,222 | Budget | Weekly grocery (10 commodities) | Median oracle |
| Open E-Commerce | 4,694 | Budget | Monthly Amazon (50 categories) | Median oracle |
| H&M | 46,757 | Budget | Monthly fashion (20 product groups) | Real (normalized) |
| Instacart | 50,000 | Budget | Per-order grocery (134 aisles) | Heuristic per-aisle |
| REES46 | 8,832 | Menu | Server sessions (view → purchase) | N/A |
| Taobao | ~15,000 | Menu | 30-min sessions (view → purchase) | N/A |
| Tenrec | 50,000 | Menu | Click-to-like windows (493M events) | N/A |

## Temporal Split (no leakage)

For each user independently:
- **Feature window**: First 70% of observations (chronologically)
- **Target window**: Last 30% of observations (chronologically)
- Features (X) are computed ONLY from the feature window
- Targets (y) are computed ONLY from the target window
- No observation appears in both X and y

## Feature Engineering

### Baseline features (13 per user, budget datasets)

Computed from raw prices × quantities in the feature window. No RP library needed.

| Feature | Computation |
|---------|------------|
| n_obs | Number of observations |
| total_spend | sum(prices × quantities) |
| mean_spend | mean(spend per obs) |
| std_spend | std(spend per obs) |
| max_spend, min_spend | Extrema |
| mean_basket_size | mean(total quantity per obs) |
| herfindahl | sum(category_share²) — concentration |
| top_category_share | max(category_share) |
| n_active_categories | count(share > 0) |
| spend_slope | Linear trend coefficient |
| spend_cv | std / mean |
| mean_abs_spend_change | mean(|diff(spend)|) |

### Baseline features (11 per user, menu datasets)

| Feature | Computation |
|---------|------------|
| n_sessions | Number of menu presentations |
| mean/std/max/min_menu_size | Menu size statistics |
| n_unique_items | Distinct items across all menus |
| items_per_session | n_items / n_sessions |
| n_unique_choices | Distinct items ever chosen |
| max_choice_freq | Most-chosen item / n_sessions |
| choice_concentration | n_unique_choices / n_unique_items |

### RP Engine features (14 per user, budget)

From `Engine.analyze_arrays()` — batch Rust computation:

| Feature | Source | What it captures |
|---------|--------|-----------------|
| is_garp | GARP test | Binary consistency |
| ccei | Afriat index | Efficiency 0-1 |
| mpi | Money Pump | Exploitability 0-1 |
| is_harp | HARP test | Homothetic consistency |
| n_violations | GARP cycles | Violation count |
| max_scc | SCC decomposition | Largest cycle size |
| vei_mean, vei_min | Per-obs LP | Observation-level efficiency |
| vei_std, vei_q25, vei_q75 | VEI distribution | Efficiency spread |
| n_scc | SCC count | Graph fragmentation |
| harp_severity | HARP cycle product | Degree of non-homotheticity |
| scc_mean_size | T / n_scc | Average component size |

Derived:
- hm_ratio = hm_consistent / hm_total
- violation_density = n_violations / T(T-1)
- scc_ratio = max_scc / T
- vei_iqr = vei_q75 - vei_q25
- scc_fragmentation = n_scc / T

### RP Engine features (menu)

| Feature | Source |
|---------|--------|
| is_sarp, is_warp, is_warp_la | Axiom tests |
| n_sarp_violations, n_warp_violations | Violation counts |
| max_scc, n_scc | SCC structure |
| r_density | Item graph edge density |
| pref_entropy | Out-degree entropy |
| choice_diversity | Unique choices / sessions |

Derived: hm_ratio, sarp/warp_violation_density, scc_ratio

### RP Extended features (~28 per user, per-user algorithm calls)

**Budget — distributional:**
- VEI temporal: vei_first_half, vei_second_half, vei_temporal_shift
- VEI shape: vei_below_90, vei_below_90_frac
- Utility recovery: util_mean, util_std, util_range, util_cv, util_gini
- Marginal utility: lambda_mean, lambda_std, lambda_cv

**Budget — graph structural:**
- Preference graph: pref_graph_density, strict_pref_density, transitivity_ratio
- Cycles: n_cycles, mean_cycle_length, max_cycle_length
- Temporal: violation_obs_frac, violation_mean_position
- MPI detail: mpi_max_cycle_cost, mpi_mean_cycle_cost, mpi_n_cycles

**Menu — behavioral:**
- Reversals: choice_reversal_count, choice_reversal_ratio
- Entropy: choice_entropy, choice_entropy_norm
- Graph: menu_pref_density, menu_transitivity, menu_n_cycles, menu_max_cycle_len
- Rationality: is_congruent, n_maximality_violations
- Ordinal utility: menu_util_range, menu_util_std, menu_rank_complete

## Targets

Computed from the target window (future data) only.

**Budget datasets:**
- High Spender: test_total_spend > percentile(66.67) — top tercile
- Churn: test_mean_spend / train_mean_spend < 0.5 — spend dropped >50%
- Spend Change: test_mean_spend - train_mean_spend — regression

**Menu datasets:**
- High Engagement: test_session_count > percentile(66.67) — top tercile

## Evaluation Protocol

### Out-of-Sample (OOS)

Random 80/20 stratified split of USERS (not time). Train on 80% of users' past features → predict 20% of users' future targets.

- This tests generalization to unseen users
- The temporal split within each user prevents future leakage
- NaN imputation uses train-user medians only (no test leakage)

### Metrics

**Classification:**
- AUC-ROC: discrimination across all thresholds
- AUC-PR: average precision (primary for imbalanced targets like Churn at 5-7%)

**Regression:**
- R²: explained variance

### Model

LightGBM with default hyperparameters. No tuning. `random_state=42, verbose=-1`.

No hyperparameter optimization ensures the comparison is about FEATURES, not model fitting.

### Three-way comparison

For each target, train three models on the same train split:
1. **Baseline only**: 13 RFM features
2. **RP only**: ~42 RP features (Engine + Extended)
3. **Combined**: Baseline + RP (~55 features)

Lift = (Combined AUC - Baseline AUC) / Baseline AUC × 100

### Feature importance

Permutation-based LightGBM feature importance on the OOS train split. Reports top 15 features by importance score.

## Data Assumptions & Limitations

**Budget:**
- Shared price oracle across users (Dunnhumby, Open E-Commerce)
- Category-level aggregation hides within-category substitution
- Instacart uses heuristic per-aisle prices ($1.50-$14.00), not observed prices
- H&M prices normalized (0-1), not absolute dollar values

**Menu:**
- Menus = viewed/clicked items only (impression bias — unclicked items invisible)
- REES46: server-defined sessions (gold standard)
- Taobao: 30-min inactivity gap sessions (EDA-validated)
- Tenrec: click-to-like positional windows (algorithmic exposure bias)

## Reproducibility

```bash
pip install prefgraph lightgbm scikit-learn
python case_studies/benchmarks/runner.py --datasets dunnhumby
python case_studies/benchmarks/runner.py --datasets all
```

All random seeds fixed at 42. LightGBM defaults. No stochastic variation across runs.
