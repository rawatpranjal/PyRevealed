#!/usr/bin/env python3
"""
Dunnhumby Sample Observations & RP Scores
===========================================

For 3 random households:
1. Show actual (quantities, prices) matrices
2. Compute RP scores using Engine batch API
3. Display detailed result objects

Shows what revealed preference analysis looks like on real data.
"""

import sys
from pathlib import Path

import numpy as np
import random

# Add repo and dunnhumby to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "case_studies" / "dunnhumby"))

from data_loader import load_filtered_data
from price_oracle import get_master_price_grid
from session_builder import build_all_sessions
from config import TOP_COMMODITIES, MIN_SHOPPING_WEEKS

# PrefGraph imports
from prefgraph import Engine

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

print("=" * 80)
print("DUNNHUMBY SAMPLE OBSERVATIONS: 3 RANDOM HOUSEHOLDS")
print("=" * 80)
print()

# Load data
print("Loading data...")
filtered_df = load_filtered_data(use_cache=True)
price_grid = get_master_price_grid(filtered_df)
households = build_all_sessions(filtered_df, price_grid, min_weeks=MIN_SHOPPING_WEEKS)
print(f"Loaded {len(households)} qualifying households")
print()

# Pick 3 random households
hh_keys = list(households.keys())
sample_keys = random.sample(hh_keys, 3)

# Prepare data for Engine batch scoring
batch_data = []
sample_info = {}

for hh_key in sample_keys:
    hh = households[hh_key]
    batch_data.append((
        hh.behavior_log.cost_vectors,
        hh.behavior_log.action_vectors
    ))
    sample_info[hh_key] = {
        'obs_count': hh.num_observations,
        'total_spend': hh.total_spend,
        'active_weeks': hh.active_weeks,
    }

# Run Engine batch analysis
print("Running Engine batch analysis on 3 households...")
engine = Engine()
results = engine.analyze_arrays(batch_data)
print()

# Display each household
for i, hh_key in enumerate(sample_keys):
    hh = households[hh_key]
    result = results[i]

    print("=" * 80)
    print(f"HOUSEHOLD {i+1}: household_{hh_key}")
    print("=" * 80)
    print()

    # Basic info
    info = sample_info[hh_key]
    print(f"Shopping History:")
    print(f"  Observations: {info['obs_count']}")
    print(f"  Active weeks: {info['active_weeks']}")
    print(f"  Total spending: ${info['total_spend']:,.2f}")
    print()

    # Quantities matrix (first 5 and last 5 observations shown)
    q = hh.behavior_log.action_vectors
    p = hh.behavior_log.cost_vectors

    print("QUANTITIES (q) - Units purchased")
    print("-" * 80)
    print(f"Shape: {q.shape[0]} observations × {q.shape[1]} commodities")
    print()

    # Show header
    print(f"{'Week':<6}", end="")
    for j, comm in enumerate(TOP_COMMODITIES):
        print(f"{comm[:8]:<10}", end="")
    print()
    print("-" * 80)

    # Show first 5
    for obs_idx in range(min(5, q.shape[0])):
        print(f"{obs_idx+1:<6}", end="")
        for j in range(q.shape[1]):
            print(f"{q[obs_idx, j]:.1f}        ", end="")
        print()

    if q.shape[0] > 10:
        print("...")

    # Show last 5
    for obs_idx in range(max(5, q.shape[0] - 5), q.shape[0]):
        print(f"{obs_idx+1:<6}", end="")
        for j in range(q.shape[1]):
            print(f"{q[obs_idx, j]:.1f}        ", end="")
        print()

    print()

    # Prices matrix
    print("PRICES (p) - Unit prices from grid")
    print("-" * 80)
    print(f"Shape: {p.shape[0]} observations × {p.shape[1]} commodities")
    print()

    # Show header
    print(f"{'Week':<6}", end="")
    for j, comm in enumerate(TOP_COMMODITIES):
        print(f"{comm[:8]:<10}", end="")
    print()
    print("-" * 80)

    # Show first 5
    for obs_idx in range(min(5, p.shape[0])):
        print(f"{obs_idx+1:<6}", end="")
        for j in range(p.shape[1]):
            print(f"${p[obs_idx, j]:.2f}     ", end="")
        print()

    if p.shape[0] > 10:
        print("...")

    # Show last 5
    for obs_idx in range(max(5, p.shape[0] - 5), p.shape[0]):
        print(f"{obs_idx+1:<6}", end="")
        for j in range(p.shape[1]):
            print(f"${p[obs_idx, j]:.2f}     ", end="")
        print()

    print()

    # Expenditure per observation
    spend_per_obs = np.sum(q * p, axis=1)
    print("EXPENDITURE PER OBSERVATION:")
    print("-" * 80)
    print(f"  Min:    ${spend_per_obs.min():.2f}")
    print(f"  Mean:   ${spend_per_obs.mean():.2f}")
    print(f"  Median: ${np.median(spend_per_obs):.2f}")
    print(f"  Max:    ${spend_per_obs.max():.2f}")
    print(f"  Total:  ${spend_per_obs.sum():.2f}")
    print()

    # RP Scores
    print("REVEALED PREFERENCE SCORES:")
    print("-" * 80)

    print(f"  GARP (Generalized Axiom of Revealed Preference):")
    print(f"    Consistent: {result.is_garp}")
    print(f"    Violations: {result.n_violations}")
    print(f"    (True = no cycles, False = preference cycles detected)")
    print()

    print(f"  CCEI (Critical Cost Efficiency Index / Afriat Efficiency Index):")
    print(f"    Score: {result.ccei:.4f}")
    print(f"    (1.0 = perfectly rational, 0.0 = worst possible)")
    print(f"    Interpretation: {result.ccei*100:.1f}% of budget is used efficiently")
    print()

    print(f"  MPI (Money Pump Index):")
    print(f"    Score: {result.mpi:.4f}")
    print(f"    (0.0 = no exploitable cycles, 1.0 = maximum exploitability)")
    if result.mpi == 0.0:
        print(f"    Interpretation: No money pump violations found (consistent with GARP)")
    else:
        print(f"    Interpretation: {result.mpi*100:.1f}% average budget loss to cycles")
    print()

    if result.hm_total is not None:
        hm_frac = result.hm_consistent / result.hm_total if result.hm_total > 0 else 0
        print(f"  HM (Houtman-Maks):")
        print(f"    Consistent obs: {result.hm_consistent}/{result.hm_total} ({hm_frac*100:.1f}%)")
        print(f"    (fraction of observations that are consistent)")
        print()

    if result.is_harp is not None:
        print(f"  HARP (Homothetic axiom):")
        print(f"    Passes: {result.is_harp}")
        print(f"    (True = homothetic preferences, False = violated)")
        print()

    # Full summary
    print("SUMMARY:")
    print("-" * 80)
    print(result.summary())
    print()
    print()

print("=" * 80)
print("END OF SAMPLE OBSERVATIONS")
print("=" * 80)
