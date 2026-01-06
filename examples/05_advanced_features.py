"""Example: Advanced Features - Separability, data loading, and temporal analysis.

This module demonstrates advanced features:
- Separability testing (are feature groups independent?)
- Auto-discovery of independent groups
- Cross-impact/cannibalization metrics
- Data loading from pandas DataFrames
- Temporal window analysis
- Alternative consistency metrics (Houtman-Maks)
"""

import numpy as np
import pandas as pd
from pyrevealed import (
    BehaviorLog,
    test_feature_independence,  # or check_separability
    discover_independent_groups,  # or find_separable_partition
    compute_cross_impact,  # or compute_cannibalization
    compute_minimal_outlier_fraction,  # or compute_houtman_maks_index
    validate_consistency,
    compute_integrity_score,
)

# =============================================================================
# Example 1: Test Feature Independence (Separability)
# =============================================================================

print("=" * 60)
print("Example 1: Test Feature Independence (Separability)")
print("=" * 60)

# Simulate a superapp user with 4 products:
# Group A: Rides (products 0, 1)
# Group B: Food delivery (products 2, 3)
#
# If separable: user's spending on Rides doesn't affect Food choices

np.random.seed(42)
T = 20  # 20 observations

# Prices vary independently for each group
prices_rides = np.abs(np.random.randn(T, 2)) + 1.0
prices_food = np.abs(np.random.randn(T, 2)) + 1.5
prices = np.hstack([prices_rides, prices_food])

# Quantities respond only to own-group prices (separable behavior)
# When ride prices go up, ride quantities go down, but food unchanged
qty_rides = 5.0 / prices_rides
qty_food = 4.0 / prices_food
quantities = np.hstack([qty_rides, qty_food])

log = BehaviorLog(
    cost_vectors=prices,
    action_vectors=quantities,
    user_id="superapp_user"
)

result = test_feature_independence(
    log,
    group_a=[0, 1],  # Rides
    group_b=[2, 3],  # Food
)

print(f"User: {log.user_id}")
print(f"Is separable: {result.is_separable}")
print(f"Cross-effect strength: {result.cross_effect_strength:.3f}")
print(f"  (0 = independent, 1 = strongly coupled)")
print(f"Within-group A (Rides) consistency: {result.within_group_a_consistency:.3f}")
print(f"Within-group B (Food) consistency: {result.within_group_b_consistency:.3f}")
print(f"Recommendation: {result.recommendation}")
print(f"Computation time: {result.computation_time_ms:.2f} ms")

# =============================================================================
# Example 2: Non-Separable Groups (Substitutes)
# =============================================================================

print("\n" + "=" * 60)
print("Example 2: Non-Separable Groups (Substitutes)")
print("=" * 60)

# User treats Rides and Food as substitutes
# When food prices go up, they take more rides instead
qty_rides_sub = 5.0 / prices_rides + 0.5 * prices_food.mean(axis=1, keepdims=True)
qty_food_sub = 4.0 / prices_food - 0.3 * prices_rides.mean(axis=1, keepdims=True)
qty_food_sub = np.maximum(qty_food_sub, 0.1)  # Ensure positive
quantities_sub = np.hstack([qty_rides_sub, qty_food_sub])

log_sub = BehaviorLog(
    cost_vectors=prices,
    action_vectors=quantities_sub,
    user_id="substitute_user"
)

result_sub = test_feature_independence(log_sub, group_a=[0, 1], group_b=[2, 3])

print(f"User: {log_sub.user_id}")
print(f"Is separable: {result_sub.is_separable}")
print(f"Cross-effect strength: {result_sub.cross_effect_strength:.3f}")
print(f"Recommendation: {result_sub.recommendation}")

# =============================================================================
# Example 3: Auto-Discover Independent Groups
# =============================================================================

print("\n" + "=" * 60)
print("Example 3: Auto-Discover Independent Groups")
print("=" * 60)

# Create data with 6 goods that naturally cluster into 2 groups
# Goods 0-2 are consumed together, Goods 3-5 are consumed together
prices_6 = np.abs(np.random.randn(15, 6)) + 1.0
quantities_6 = np.zeros((15, 6))

for t in range(15):
    if t % 2 == 0:
        # Even observations: consume group 1
        quantities_6[t, 0:3] = [3.0, 2.0, 1.0]
    else:
        # Odd observations: consume group 2
        quantities_6[t, 3:6] = [2.0, 3.0, 2.0]

log_6 = BehaviorLog(cost_vectors=prices_6, action_vectors=quantities_6)

groups = discover_independent_groups(log_6, max_groups=2)

print(f"Discovered {len(groups)} independent groups:")
for i, group in enumerate(groups):
    print(f"  Group {i+1}: goods {group}")

# =============================================================================
# Example 4: Compute Cross-Impact (Cannibalization)
# =============================================================================

print("\n" + "=" * 60)
print("Example 4: Compute Cross-Impact (Cannibalization)")
print("=" * 60)

impact = compute_cross_impact(log_sub, group_a=[0, 1], group_b=[2, 3])

print("Cross-impact metrics:")
print(f"  A->B (Rides cannibalizing Food): {impact['a_to_b']:.3f}")
print(f"  B->A (Food cannibalizing Rides): {impact['b_to_a']:.3f}")
print(f"  Symmetric cannibalization: {impact['symmetric']:.3f}")
print(f"  Net direction: {impact['net_direction']:.3f}")
print(f"    (positive = A cannibalizes B more)")

# =============================================================================
# Example 5: Load Data from Pandas DataFrame
# =============================================================================

print("\n" + "=" * 60)
print("Example 5: Load Data from Pandas DataFrame")
print("=" * 60)

# Wide format DataFrame
df_wide = pd.DataFrame({
    'price_A': [1.0, 2.0, 1.5],
    'price_B': [2.0, 1.0, 1.5],
    'qty_A': [3.0, 1.0, 2.0],
    'qty_B': [1.0, 3.0, 2.0],
})

log_wide = BehaviorLog.from_dataframe(
    df_wide,
    cost_cols=['price_A', 'price_B'],
    action_cols=['qty_A', 'qty_B'],
    user_id='df_user'
)

print("Created from wide-format DataFrame:")
print(f"  Records: {log_wide.num_records}")
print(f"  Features: {log_wide.num_features}")
print(f"  Consistent: {validate_consistency(log_wide).is_consistent}")

# Long format DataFrame (SQL-style transactions)
df_long = pd.DataFrame({
    'time': [0, 0, 1, 1, 2, 2],
    'item_id': ['A', 'B', 'A', 'B', 'A', 'B'],
    'price': [1.0, 2.0, 2.0, 1.0, 1.5, 1.5],
    'quantity': [3.0, 1.0, 1.0, 3.0, 2.0, 2.0],
})

print("\nLong-format DataFrame:")
print(df_long)

log_long = BehaviorLog.from_long_format(
    df_long,
    time_col='time',
    item_col='item_id',
    cost_col='price',
    action_col='quantity'
)

print(f"\nCreated from long-format DataFrame:")
print(f"  Records: {log_long.num_records}")
print(f"  Features: {log_long.num_features}")

# =============================================================================
# Example 6: Temporal Window Analysis
# =============================================================================

print("\n" + "=" * 60)
print("Example 6: Temporal Window Analysis")
print("=" * 60)

# Create behavior log with 12 observations (e.g., 12 months)
np.random.seed(123)
prices_12 = np.abs(np.random.randn(12, 3)) + 1.0
quantities_12 = np.abs(np.random.randn(12, 3))

log_12 = BehaviorLog(
    cost_vectors=prices_12,
    action_vectors=quantities_12,
    user_id="annual_user"
)

# Split into quarterly windows
windows = log_12.split_by_window(window_size=3)

print(f"Original log: {log_12.num_records} observations")
print(f"Split into {len(windows)} windows of 3 observations each")
print()

for i, window_log in enumerate(windows):
    integrity = compute_integrity_score(window_log).efficiency_index
    print(f"  Window {i+1} ({window_log.user_id}): integrity = {integrity:.3f}")

# Detect structural breaks (sudden changes in consistency)
integrity_scores = [compute_integrity_score(w).efficiency_index for w in windows]
print(f"\nIntegrity trend: {integrity_scores}")

# =============================================================================
# Example 7: Houtman-Maks Index (Minimal Outlier Fraction)
# =============================================================================

print("\n" + "=" * 60)
print("Example 7: Minimal Outlier Fraction (Houtman-Maks Index)")
print("=" * 60)

# Create data with some violations (at equal prices, preferences flip)
prices_hm = np.array([
    [1.0, 1.0],
    [1.0, 1.0],
    [1.0, 1.0],
    [1.0, 1.0],
])
quantities_hm = np.array([
    [4.0, 1.0],  # Prefers A
    [1.0, 4.0],  # Prefers B (violation!)
    [3.0, 2.0],  # Prefers A
    [2.0, 3.0],  # Prefers B (violation!)
])

log_hm = BehaviorLog(cost_vectors=prices_hm, action_vectors=quantities_hm)

# compute_minimal_outlier_fraction returns (fraction_removed, list_of_removed_indices)
outlier_fraction, removed_indices = compute_minimal_outlier_fraction(log_hm)

is_originally_consistent = validate_consistency(log_hm).is_consistent
print(f"Is originally GARP consistent: {is_originally_consistent}")
print(f"Outlier fraction: {outlier_fraction:.3f}")
print(f"  (fraction of observations that need to be REMOVED)")
print(f"Removed observation indices: {removed_indices}")
print(f"Remaining observations: {1 - outlier_fraction:.3f}")
print(f"  (fraction of observations that ARE consistent)")

# Compare to Afriat Efficiency Index
aei = compute_integrity_score(log_hm).efficiency_index
print(f"\nComparison:")
print(f"  Afriat Efficiency Index: {aei:.3f} (budget efficiency)")
print(f"  Houtman-Maks removal fraction: {outlier_fraction:.3f} (observation count)")

# =============================================================================
# Practical Applications
# =============================================================================

print("\n" + "=" * 60)
print("Practical Applications")
print("=" * 60)

print("""
1. SUPERAPP PRODUCT STRATEGY:
   result = test_feature_independence(user_log, rides_goods, food_goods)
   if result.is_separable:
       print("Can price Rides and Food independently")
   else:
       print("Need unified pricing strategy - products are substitutes")

2. DETECT CANNIBALIZATION:
   impact = compute_cross_impact(user_log, new_product, existing_product)
   if impact['a_to_b'] > 0.3:
       print("New product is cannibalizing existing product!")

3. STRUCTURAL BREAK DETECTION:
   windows = user_log.split_by_window(window_size=10)
   scores = [compute_integrity_score(w).efficiency_index for w in windows]
   for i in range(1, len(scores)):
       if abs(scores[i] - scores[i-1]) > 0.2:
           print(f"Structural break at window {i}")

4. DATA PIPELINE INTEGRATION:
   # Load from database
   df = pd.read_sql("SELECT * FROM transactions", conn)
   log = BehaviorLog.from_long_format(df, ...)

5. CATEGORY MANAGEMENT:
   groups = discover_independent_groups(store_data, max_groups=5)
   for group in groups:
       print(f"Category {group}: can optimize independently")
""")
