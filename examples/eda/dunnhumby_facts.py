#!/usr/bin/env python3
"""
Dunnhumby EDA - Factual Summary
================================

Loads dunnhumby household shopping data and reports:
- Dataset dimensions (transactions, households, commodities, time span)
- Price statistics by commodity
- Quantity statistics by commodity
- Household shopping patterns
- Budget-choice matrix properties

No interpretation, opinions, or analysis—only facts.
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl

# Add repo and dunnhumby to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "case_studies" / "dunnhumby"))

from data_loader import load_filtered_data, get_data_summary
from price_oracle import get_master_price_grid
from session_builder import build_all_sessions, get_session_summary
from config import TOP_COMMODITIES, MIN_SHOPPING_WEEKS

# =============================================================================
# LOAD DATA
# =============================================================================

print("=" * 80)
print("DUNNHUMBY BUDGET-BASED CHOICE DATA: FACTUAL SUMMARY")
print("=" * 80)
print()

# Load filtered transactions
print("Loading filtered transaction data...")
filtered_df = load_filtered_data(use_cache=True)
print()

# Get transaction-level summary
summary = get_data_summary(filtered_df)
print("TRANSACTION-LEVEL FACTS:")
print(f"  Total transactions: {summary['n_transactions']:,}")
print(f"  Unique households: {summary['n_households']:,}")
print(f"  Weeks covered: {summary['n_weeks']} (1-104)")
print(f"  Commodities in analysis: {summary['n_commodities']}")
print(f"  Store locations: {summary['n_stores']}")
print(f"  Total units purchased: {summary['total_quantity']:,.0f}")
print(f"  Total spend across all transactions: ${summary['total_spend']:,.2f}")
print()

# Convert to polars for efficient analysis
print("Converting to Polars for efficient analysis...")
df = pl.from_pandas(filtered_df)
print()

# =============================================================================
# PRICE FACTS
# =============================================================================

print("UNIT PRICE FACTS BY COMMODITY:")
print("-" * 80)

price_stats = df.group_by("commodity").agg([
    pl.col("unit_price").min().alias("min_price"),
    pl.col("unit_price").quantile(0.25).alias("q25_price"),
    pl.col("unit_price").median().alias("median_price"),
    pl.col("unit_price").mean().alias("mean_price"),
    pl.col("unit_price").quantile(0.75).alias("q75_price"),
    pl.col("unit_price").max().alias("max_price"),
    pl.col("unit_price").std().alias("std_price"),
    pl.col("unit_price").count().alias("n_obs"),
]).sort("commodity")

for row in price_stats.rows(named=True):
    print(f"\n{row['commodity']}:")
    print(f"  Min:     ${row['min_price']:.4f}")
    print(f"  Q1:      ${row['q25_price']:.4f}")
    print(f"  Median:  ${row['median_price']:.4f}")
    print(f"  Mean:    ${row['mean_price']:.4f}")
    print(f"  Q3:      ${row['q75_price']:.4f}")
    print(f"  Max:     ${row['max_price']:.4f}")
    print(f"  Std Dev: ${row['std_price']:.4f}")
    print(f"  Obs:     {row['n_obs']:,}")

print()

# =============================================================================
# QUANTITY FACTS
# =============================================================================

print("QUANTITY FACTS BY COMMODITY:")
print("-" * 80)

qty_stats = df.group_by("commodity").agg([
    pl.col("quantity").min().alias("min_qty"),
    pl.col("quantity").quantile(0.25).alias("q25_qty"),
    pl.col("quantity").median().alias("median_qty"),
    pl.col("quantity").mean().alias("mean_qty"),
    pl.col("quantity").quantile(0.75).alias("q75_qty"),
    pl.col("quantity").max().alias("max_qty"),
    pl.col("quantity").std().alias("std_qty"),
]).sort("commodity")

for row in qty_stats.rows(named=True):
    print(f"\n{row['commodity']}:")
    print(f"  Min:     {row['min_qty']:.2f} units")
    print(f"  Q1:      {row['q25_qty']:.2f} units")
    print(f"  Median:  {row['median_qty']:.2f} units")
    print(f"  Mean:    {row['mean_qty']:.2f} units")
    print(f"  Q3:      {row['q75_qty']:.2f} units")
    print(f"  Max:     {row['max_qty']:.2f} units")
    print(f"  Std Dev: {row['std_qty']:.2f} units")

print()

# =============================================================================
# HOUSEHOLD FACTS
# =============================================================================

print("HOUSEHOLD-LEVEL SHOPPING FACTS:")
print("-" * 80)

# Use pandas for household aggregation (simpler)
hh_stats_pd = filtered_df.groupby("household_key").agg({
    "quantity": "count",
    "week": "nunique",
}).rename(columns={"quantity": "n_transactions", "week": "active_weeks"})

# Calculate total spend
hh_spend = (filtered_df["quantity"] * filtered_df["unit_price"]).groupby(filtered_df["household_key"]).sum()
hh_stats_pd["total_spend"] = hh_spend
hh_stats_pd["spend_per_week"] = hh_stats_pd["total_spend"] / hh_stats_pd["active_weeks"]

hh_stats = hh_stats_pd

print(f"  Minimum transactions per household: {hh_stats['n_transactions'].min()}")
print(f"  Q1 transactions per household:      {hh_stats['n_transactions'].quantile(0.25)}")
print(f"  Median transactions per household:  {hh_stats['n_transactions'].median()}")
print(f"  Mean transactions per household:    {hh_stats['n_transactions'].mean():.1f}")
print(f"  Q3 transactions per household:      {hh_stats['n_transactions'].quantile(0.75)}")
print(f"  Maximum transactions per household: {hh_stats['n_transactions'].max()}")
print()

print(f"  Minimum active weeks per household: {hh_stats['active_weeks'].min()}")
print(f"  Median active weeks per household:  {hh_stats['active_weeks'].median()}")
print(f"  Mean active weeks per household:    {hh_stats['active_weeks'].mean():.1f}")
print(f"  Maximum active weeks per household: {hh_stats['active_weeks'].max()}")
print()

print(f"  Minimum total spend per household:  ${hh_stats['total_spend'].min():.2f}")
print(f"  Median total spend per household:   ${hh_stats['total_spend'].median():.2f}")
print(f"  Mean total spend per household:     ${hh_stats['total_spend'].mean():.2f}")
print(f"  Maximum total spend per household:  ${hh_stats['total_spend'].max():.2f}")
print()

print(f"  Minimum spend per week:             ${hh_stats['spend_per_week'].min():.2f}")
print(f"  Median spend per week:              ${hh_stats['spend_per_week'].median():.2f}")
print(f"  Mean spend per week:                ${hh_stats['spend_per_week'].mean():.2f}")
print(f"  Maximum spend per week:             ${hh_stats['spend_per_week'].max():.2f}")
print()

# =============================================================================
# BUDGET-CHOICE MATRIX FACTS
# =============================================================================

print("BUDGET-CHOICE MATRIX PROPERTIES:")
print("-" * 80)

# Build price grid
print("  Building price grid (104 weeks × 10 commodities)...")
price_grid = get_master_price_grid(filtered_df)
print(f"    Price grid shape: {price_grid.shape}")
print(f"    Min price across grid: ${price_grid.min():.4f}")
print(f"    Max price across grid: ${price_grid.max():.4f}")
print(f"    Mean price across grid: ${price_grid.mean():.4f}")
print()

# Build sessions
print("  Building household behavior logs (10+ active weeks minimum)...")
households = build_all_sessions(filtered_df, price_grid,
                                min_weeks=MIN_SHOPPING_WEEKS)
print()

session_summary = get_session_summary(households)
print("QUALIFYING HOUSEHOLD FACTS (10+ weeks shopping):")
print(f"  Qualifying households: {session_summary['n_households']:,}")
print(f"  Total observations (weeks): {session_summary['total_observations']:,}")
print(f"  Min observations per household: {session_summary['min_observations']}")
print(f"  Median observations per household: {session_summary['median_observations']:.0f}")
print(f"  Mean observations per household: {session_summary['mean_observations']:.1f}")
print(f"  Max observations per household: {session_summary['max_observations']}")
print()

print(f"  Total spending (qualifying households): ${session_summary['total_spend']:,.2f}")
print(f"  Mean spending per household: ${session_summary['mean_spend']:.2f}")
print()

# Matrix properties
print("BUDGET-CHOICE MATRIX DIMENSIONS:")
obs_counts = np.array([h.num_observations for h in households.values()])
print(f"  Each household has (q, p) matrices:")
print(f"    q (quantities): {obs_counts.min()}-{obs_counts.max()} observations × 10 commodities")
print(f"    p (prices):     {obs_counts.min()}-{obs_counts.max()} observations × 10 commodities")
print()

spend_per_obs = [h.total_spend / h.num_observations for h in households.values()]
print(f"  Spending per observation:")
print(f"    Min:  ${min(spend_per_obs):.2f}")
print(f"    Median: ${np.median(spend_per_obs):.2f}")
print(f"    Mean:   ${np.mean(spend_per_obs):.2f}")
print(f"    Max:  ${max(spend_per_obs):.2f}")
print()

# =============================================================================
# SUMMARY
# =============================================================================

print("=" * 80)
print("DATASET READY FOR REVEALED PREFERENCE ANALYSIS")
print("=" * 80)
print(f"  Input: {filtered_df.shape[0]:,} transactions")
print(f"  Output: {session_summary['n_households']:,} households × {obs_counts.min()}-{obs_counts.max()} weeks")
print(f"  Commodities: {len(TOP_COMMODITIES)} (Soda, Milk, Bread, Cheese, Chips, Soup, Yogurt, Beef, Pizza, Lunch)")
print(f"  Time span: 104 weeks (2 years)")
print()
