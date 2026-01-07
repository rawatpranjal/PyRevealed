#!/usr/bin/env python3
"""Quick validation script for Open E-Commerce 1.0 dataset.

Usage:
    python run_validation.py [--quick]  # Run on sample
    python run_validation.py            # Run on all users
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add parent directories for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pyrevealed import validate_consistency, compute_integrity_score

from session_builder import load_sessions, get_session_summary


def run_validation(quick: bool = False, sample_size: int = 100):
    """
    Run GARP validation on user sessions.

    Args:
        quick: If True, only validate a sample
        sample_size: Number of users to validate in quick mode
    """
    print("=" * 60)
    print("Open E-Commerce 1.0 - PyRevealed Validation")
    print("=" * 60)

    # Load sessions
    users = load_sessions(use_cache=True)

    if not users:
        print("\nNo qualifying users found!")
        print("Make sure to run download.py first.")
        return

    print("\n" + "-" * 60)
    print("Session Summary")
    print("-" * 60)
    for key, value in get_session_summary(users).items():
        if isinstance(value, float):
            print(f"  {key}: {value:,.2f}")
        else:
            print(f"  {key}: {value:,}")

    # Select users to validate
    user_ids = list(users.keys())
    if quick:
        user_ids = user_ids[:sample_size]
        print(f"\n[Quick mode: validating {len(user_ids)} users]")

    print("\n" + "-" * 60)
    print("Running GARP Validation")
    print("-" * 60)

    n_consistent = 0
    n_violations = 0
    aei_scores = []

    for i, user_id in enumerate(user_ids):
        user = users[user_id]
        log = user.behavior_log

        # Check GARP consistency
        result = validate_consistency(log)

        if result.is_consistent:
            n_consistent += 1
            aei_scores.append(1.0)
        else:
            n_violations += 1
            # Compute AEI for inconsistent users
            aei_result = compute_integrity_score(log)
            aei_scores.append(aei_result.efficiency_index)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(user_ids)} users...")

    print("\n" + "-" * 60)
    print("Results")
    print("-" * 60)
    print(f"  Users validated: {len(user_ids)}")
    print(f"  GARP consistent: {n_consistent} ({100*n_consistent/len(user_ids):.1f}%)")
    print(f"  GARP violations: {n_violations} ({100*n_violations/len(user_ids):.1f}%)")

    if aei_scores:
        import numpy as np
        print(f"\n  AEI (Afriat Efficiency Index):")
        print(f"    Mean: {np.mean(aei_scores):.4f}")
        print(f"    Median: {np.median(aei_scores):.4f}")
        print(f"    Min: {np.min(aei_scores):.4f}")
        print(f"    Max: {np.max(aei_scores):.4f}")

    print("\n" + "=" * 60)
    print("Validation complete!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Validate Open E-Commerce data")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick validation on sample (100 users)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=100,
        help="Sample size for quick mode (default: 100)",
    )
    args = parser.parse_args()

    run_validation(quick=args.quick, sample_size=args.sample)


if __name__ == "__main__":
    main()
