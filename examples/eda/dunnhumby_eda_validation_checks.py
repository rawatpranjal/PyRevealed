#!/usr/bin/env python3
"""
Dunnhumby EDA Validation Checks (Tier 1)
=========================================
Three decision-critical sensitivity tests before paper submission:

1. Store-week vs chain-week price sensitivity (HIGHEST PRIORITY)
   - Question: Do RP metrics stabilize across price specifications?
   - Risk: If RP scores diverge → price mismeasurement is driving results

2. Violation concentration by commodity (SECOND)
   - Question: Are GARP violations concentrated in high-error categories?
   - Risk: If violations cluster in (beef, soda) → likely oracle artifacts

3. Null model (permutation test) (THIRD)
   - Question: Is observed RP structure meaningful vs randomized demand?
   - Risk: If null model has similar violation rates → no genuine RP signal

Run: python3 examples/eda/dunnhumby_eda_validation_checks.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "case_studies" / "dunnhumby"))

from data_loader import load_filtered_data
from price_oracle import get_master_price_grid
from config import TOP_COMMODITIES, NUM_WEEKS, COMMODITY_SHORT_NAMES


def compute_garp_consistency(hh_data, price_grid, weeks_observed):
    """
    Compute GARP consistency for a household.

    Returns: (n_violations, n_comparable_pairs, consistency_rate)
    """
    # Build (q, p) matrices
    hh_pivot = hh_data.pivot_table(
        index="week", columns="commodity", values="quantity", aggfunc="sum"
    )
    T = len(weeks_observed)

    # Ensure all commodities present
    for commodity in TOP_COMMODITIES:
        if commodity not in hh_pivot.columns:
            hh_pivot[commodity] = 0.0
    q = hh_pivot[TOP_COMMODITIES].fillna(0.0).values  # T x 10

    # Get prices
    p_list = [price_grid[week - 1, :] for week in weeks_observed]
    p = np.array(p_list)  # T x 10

    # Compute GARP: x_i RP x_j iff p_i . q_j <= budget_i AND p_i . q_i <= p_j . q_j
    # (revealed preference test with Afriat cycling)
    budgets = (p * q).sum(axis=1)  # T
    afford = p @ q.T  # T x T: [i,j] = p_i . q_j

    # i reveals preference to j if i is affordable at j's prices
    # Build direct preference matrix
    pref = afford <= budgets[:, None]  # T x T
    np.fill_diagonal(pref, False)

    # Count violations: cycles in preference graph (simplified: any i→j→i where both true)
    violations = 0
    comparable = 0

    for i in range(T):
        for j in range(i + 1, T):
            if pref[i, j] or pref[j, i]:
                comparable += 1
                # Check for 2-cycle (simplest violation)
                if pref[i, j] and pref[j, i]:
                    violations += 1

    consistency_rate = 1.0 - (violations / comparable) if comparable > 0 else np.nan

    return violations, comparable, consistency_rate


def main():
    print("=" * 80)
    print("DUNNHUMBY EDA VALIDATION CHECKS (Tier 1)")
    print("=" * 80)
    print()

    # Load data
    print("Loading data...")
    filtered_pd = load_filtered_data(use_cache=True)
    price_grid_chain = get_master_price_grid(filtered_pd, use_cache=True)

    print(f"  Households: {filtered_pd['household_key'].nunique():,}")
    print(f"  Transactions: {len(filtered_pd):,}")
    print()

    # Get qualifying households
    weekly_counts = filtered_pd.groupby("household_key")["week"].nunique()
    qualifying_hhs = weekly_counts[weekly_counts >= 10].index.tolist()[:100]  # Sample 100 for speed

    print(f"  Analyzing {len(qualifying_hhs)} qualifying households (sampled for speed)")
    print()

    # =========================================================================
    # CHECK 1: STORE-WEEK VS CHAIN-WEEK PRICE SENSITIVITY
    # =========================================================================
    print("=" * 80)
    print("CHECK 1: STORE-WEEK vs CHAIN-WEEK PRICE SENSITIVITY")
    print("=" * 80)
    print()

    print("Computing store-week median prices (this may take 10-20 seconds)...")

    # Build store-week price grid: 477 stores × 104 weeks × 10 commodities
    store_week_prices = {}

    for commodity in TOP_COMMODITIES:
        comm_data = filtered_pd[filtered_pd["commodity"] == commodity]

        # Group by (store, week) and compute median unit price
        store_week = comm_data.groupby(["store_id", "week"])["unit_price"].median()
        store_week_prices[commodity] = store_week

    # Build store-week oracle: for each (week, store, commodity), use median
    # For missing cells, fall back to chain-week median
    def get_store_week_price(week, commodity, store_id):
        """Get price for (week, commodity, store) with chain-week fallback."""
        try:
            return store_week_prices[commodity][(store_id, week)]
        except KeyError:
            # Fall back to chain-week median
            return price_grid_chain[week - 1, TOP_COMMODITIES.index(commodity)]

    # Recompute GARP metrics for sample HHs under both price specs
    results_chain = []
    results_store = []

    for hh_key in qualifying_hhs:
        hh_data = filtered_pd[filtered_pd["household_key"] == hh_key]
        weeks_observed = sorted(hh_data["week"].unique())

        T = len(weeks_observed)
        if T < 5:
            continue

        # Chain-week prices
        v_chain, c_chain, rate_chain = compute_garp_consistency(
            hh_data, price_grid_chain, weeks_observed
        )

        # Store-week prices (requires building per-household price matrix)
        hh_pivot = hh_data.pivot_table(
            index="week", columns="commodity", values="quantity", aggfunc="sum"
        )
        for commodity in TOP_COMMODITIES:
            if commodity not in hh_pivot.columns:
                hh_pivot[commodity] = 0.0
        q = hh_pivot[TOP_COMMODITIES].fillna(0.0).values

        # Build store-specific price matrix
        p_store_list = []
        for week in weeks_observed:
            week_data = hh_data[hh_data["week"] == week]
            store_ids = week_data["store_id"].unique()

            # Use most common store for this household-week
            if len(store_ids) > 0:
                store_id = week_data["store_id"].mode()[0]
            else:
                store_id = 0

            p_week = [
                get_store_week_price(week, comm, store_id)
                for comm in TOP_COMMODITIES
            ]
            p_store_list.append(p_week)

        p_store = np.array(p_store_list)

        # Compute GARP on store prices
        budgets = (p_store * q).sum(axis=1)
        afford = p_store @ q.T
        pref = afford <= budgets[:, None]
        np.fill_diagonal(pref, False)

        violations = 0
        comparable = 0
        for i in range(T):
            for j in range(i + 1, T):
                if pref[i, j] or pref[j, i]:
                    comparable += 1
                    if pref[i, j] and pref[j, i]:
                        violations += 1

        v_store = violations
        c_store = comparable
        rate_store = 1.0 - (violations / comparable) if comparable > 0 else np.nan

        results_chain.append((v_chain, c_chain, rate_chain))
        results_store.append((v_store, c_store, rate_store))

    # Aggregate results
    chain_rates = np.array([r[2] for r in results_chain if not np.isnan(r[2])])
    store_rates = np.array([r[2] for r in results_store if not np.isnan(r[2])])

    print(f"\nSample size: {len(chain_rates)} households")
    print(f"\nChain-week oracle:")
    print(f"  Mean consistency rate: {chain_rates.mean():.1%}")
    print(f"  Median:               {np.median(chain_rates):.1%}")
    print(f"  Std:                  {chain_rates.std():.1%}")

    print(f"\nStore-week oracle:")
    print(f"  Mean consistency rate: {store_rates.mean():.1%}")
    print(f"  Median:               {np.median(store_rates):.1%}")
    print(f"  Std:                  {store_rates.std():.1%}")

    print(f"\nDifference (store - chain):")
    print(f"  Mean difference:      {(store_rates.mean() - chain_rates.mean()):+.1%}")
    print(f"  Correlation (Pearson): {np.corrcoef(chain_rates, store_rates)[0, 1]:.3f}")

    # Statistical test
    t_stat, p_val = stats.ttest_rel(store_rates, chain_rates)
    print(f"  Paired t-test p-value: {p_val:.4f} {'(SIGNIFICANT)' if p_val < 0.05 else '(not significant)'}")

    print("\nVERDICT (Check 1):")
    if abs(store_rates.mean() - chain_rates.mean()) < 0.02 and np.corrcoef(chain_rates, store_rates)[0, 1] > 0.9:
        print("  ✅ RP results are STABLE across price specifications.")
        print("     Chain-week oracle is trustworthy for this analysis.")
    else:
        print("  ⚠️  RP results SHIFT under store-week prices.")
        print("     Price mismeasurement is a first-order threat to conclusions.")
    print()

    # =========================================================================
    # CHECK 2: VIOLATION CONCENTRATION BY COMMODITY
    # =========================================================================
    print("=" * 80)
    print("CHECK 2: VIOLATION CONCENTRATION BY COMMODITY")
    print("=" * 80)
    print()

    print("Computing commodity-level violation contributions...")

    # For each household, compute which commodities drive violations
    violation_by_commodity = {c: 0 for c in TOP_COMMODITIES}
    total_violations = 0

    for hh_key in qualifying_hhs[:30]:  # Use smaller sample for speed
        hh_data = filtered_pd[filtered_pd["household_key"] == hh_key]
        weeks_observed = sorted(hh_data["week"].unique())

        T = len(weeks_observed)
        if T < 5:
            continue

        # Build matrices
        hh_pivot = hh_data.pivot_table(
            index="week", columns="commodity", values="quantity", aggfunc="sum"
        )
        for commodity in TOP_COMMODITIES:
            if commodity not in hh_pivot.columns:
                hh_pivot[commodity] = 0.0
        q = hh_pivot[TOP_COMMODITIES].fillna(0.0).values

        p = np.array([price_grid_chain[w - 1, :] for w in weeks_observed])

        budgets = (p * q).sum(axis=1)
        afford = p @ q.T
        pref = afford <= budgets[:, None]
        np.fill_diagonal(pref, False)

        # For each violation (2-cycle), attribute to commodities with highest variance
        for i in range(T):
            for j in range(i + 1, T):
                if pref[i, j] and pref[j, i]:
                    total_violations += 1
                    # Attribute to commodity with max price/quantity difference
                    diff = np.abs(p[i] - p[j]) * np.abs(q[i] - q[j])
                    top_comm_idx = np.argmax(diff)
                    violation_by_commodity[TOP_COMMODITIES[top_comm_idx]] += 1

    print(f"\nTotal violations analyzed: {total_violations}")
    print(f"\nViolations by commodity:")
    print(f"{'Commodity':<25} {'Count':<10} {'% of Total':<12} {'Oracle MAE':<12}")
    print("-" * 60)

    oracle_maes = {
        "SOFT DRINKS": 1.44,
        "FLUID MILK PRODUCTS": 0.38,
        "BAKED BREAD/BUNS/ROLLS": 0.60,
        "CHEESE": 0.81,
        "BAG SNACKS": 0.80,
        "SOUP": 0.62,
        "YOGURT": 0.64,
        "BEEF": 2.51,
        "FROZEN PIZZA": 1.41,
        "LUNCHMEAT": 0.81,
    }

    high_error_comms = set()
    for comm, v_count in sorted(violation_by_commodity.items(), key=lambda x: x[1], reverse=True):
        pct = v_count / total_violations * 100 if total_violations > 0 else 0
        mae = oracle_maes.get(comm, "?")
        short_name = COMMODITY_SHORT_NAMES.get(comm, comm[:15])

        # High error = MAE > 1.0
        if mae > 1.0:
            high_error_comms.add(comm)
            marker = " ← HIGH ERROR"
        else:
            marker = ""

        print(f"{short_name:<25} {v_count:<10} {pct:>10.1f}% ${mae:<10} {marker}")

    print("\nVERDICT (Check 2):")

    high_error_violations = sum(
        violation_by_commodity[c] for c in high_error_comms if c in violation_by_commodity
    )
    high_error_pct = high_error_violations / total_violations * 100 if total_violations > 0 else 0

    if high_error_pct > 50:
        print(f"  ⚠️  {high_error_pct:.0f}% of violations are in high-error categories (beef, soda).")
        print("     These violations are likely oracle artifacts, not real preference inconsistency.")
    else:
        print(f"  ✅ Only {high_error_pct:.0f}% of violations in high-error categories.")
        print("     Violations appear distributed across quality tiers-suggests real RP signal.")
    print()

    # =========================================================================
    # CHECK 3: NULL MODEL (PERMUTATION TEST)
    # =========================================================================
    print("=" * 80)
    print("CHECK 3: NULL MODEL (PERMUTATION TEST)")
    print("=" * 80)
    print()

    print("Computing violations under randomized quantities (this takes 20-30 seconds)...")

    observed_violations = []
    null_violations = []

    np.random.seed(42)

    for hh_key in qualifying_hhs[:50]:  # Use sample for speed
        hh_data = filtered_pd[filtered_pd["household_key"] == hh_key]
        weeks_observed = sorted(hh_data["week"].unique())

        T = len(weeks_observed)
        if T < 5:
            continue

        # Build true quantities
        hh_pivot = hh_data.pivot_table(
            index="week", columns="commodity", values="quantity", aggfunc="sum"
        )
        for commodity in TOP_COMMODITIES:
            if commodity not in hh_pivot.columns:
                hh_pivot[commodity] = 0.0
        q_true = hh_pivot[TOP_COMMODITIES].fillna(0.0).values

        p = np.array([price_grid_chain[w - 1, :] for w in weeks_observed])

        # Compute observed violations
        budgets = (p * q_true).sum(axis=1)
        afford = p @ q_true.T
        pref = afford <= budgets[:, None]
        np.fill_diagonal(pref, False)

        v_obs = 0
        for i in range(T):
            for j in range(i + 1, T):
                if pref[i, j] and pref[j, i]:
                    v_obs += 1

        observed_violations.append(v_obs)

        # Compute violations under permuted quantities (null model)
        # Permute within household: randomize week ordering for quantities
        perm_idx = np.random.permutation(T)
        q_perm = q_true[perm_idx, :]

        budgets_perm = (p * q_perm).sum(axis=1)
        afford_perm = p @ q_perm.T
        pref_perm = afford_perm <= budgets_perm[:, None]
        np.fill_diagonal(pref_perm, False)

        v_null = 0
        for i in range(T):
            for j in range(i + 1, T):
                if pref_perm[i, j] and pref_perm[j, i]:
                    v_null += 1

        null_violations.append(v_null)

    print(f"\nObserved violations:")
    print(f"  Mean: {np.mean(observed_violations):.2f}")
    print(f"  Median: {np.median(observed_violations):.1f}")
    print(f"  Std: {np.std(observed_violations):.2f}")

    print(f"\nNull model (permuted quantities):")
    print(f"  Mean: {np.mean(null_violations):.2f}")
    print(f"  Median: {np.median(null_violations):.1f}")
    print(f"  Std: {np.std(null_violations):.2f}")

    print(f"\nDifference:")
    print(f"  Observed - Null: {np.mean(observed_violations) - np.mean(null_violations):+.2f}")
    print(f"  Effect size: {(np.mean(observed_violations) - np.mean(null_violations)) / np.std(null_violations):.2f} std")

    t_stat, p_val = stats.ttest_rel(
        np.array(observed_violations), np.array(null_violations)
    )
    print(f"  Paired t-test p-value: {p_val:.4f} {'(SIGNIFICANT)' if p_val < 0.05 else '(not significant)'}")

    print("\nVERDICT (Check 3):")
    if np.mean(observed_violations) > np.mean(null_violations) * 1.5:
        print("  ✅ Observed violations significantly exceed null model.")
        print("     Dunnhumby has genuine RP structure beyond randomized demand.")
    elif np.mean(observed_violations) > np.mean(null_violations):
        print("  ⚠️  Observed violations modestly exceed null model.")
        print("     RP signal is present but not overwhelming.")
    else:
        print("  ❌ Observed violations ≤ null model.")
        print("     No genuine RP signal-violations may be noise.")
    print()

    print("=" * 80)
    print("VALIDATION CHECKS COMPLETE")
    print("=" * 80)
    print()
    print("SUMMARY:")
    print("--------")
    print("Check 1 (Price sensitivity): Assess correlation of RP metrics across specs")
    print("Check 2 (Violation source): Determine if violations cluster in high-error categories")
    print("Check 3 (Null model): Verify RP structure exceeds randomized baseline")
    print()


if __name__ == "__main__":
    main()
