#!/usr/bin/env python3
"""Quick validation script for UCI Online Retail dataset.

Usage:
    python run_validation.py [--quick]  # Run on sample
    python run_validation.py            # Run on all customers
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add parent directories for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pyrevealed import validate_consistency, compute_integrity_score

from session_builder import load_sessions, get_session_summary


def run_validation(quick: bool = False, sample_size: int = 50):
    """
    Run GARP validation on customer sessions.

    Args:
        quick: If True, only validate a sample
        sample_size: Number of customers to validate in quick mode
    """
    print("=" * 60)
    print("UCI Online Retail - PyRevealed Validation")
    print("=" * 60)

    # Load sessions
    customers = load_sessions(use_cache=True)

    if not customers:
        print("\nNo qualifying customers found!")
        print("Make sure to run download.py first.")
        return

    print("\n" + "-" * 60)
    print("Session Summary")
    print("-" * 60)
    for key, value in get_session_summary(customers).items():
        if isinstance(value, float):
            print(f"  {key}: {value:,.2f}")
        else:
            print(f"  {key}: {value:,}")

    # Select customers to validate
    customer_ids = list(customers.keys())
    if quick:
        customer_ids = customer_ids[:sample_size]
        print(f"\n[Quick mode: validating {len(customer_ids)} customers]")

    print("\n" + "-" * 60)
    print("Running GARP Validation")
    print("-" * 60)

    n_consistent = 0
    n_violations = 0
    aei_scores = []

    for i, customer_id in enumerate(customer_ids):
        customer = customers[customer_id]
        log = customer.behavior_log

        # Check GARP consistency
        result = validate_consistency(log)

        if result.is_consistent:
            n_consistent += 1
            aei_scores.append(1.0)
        else:
            n_violations += 1
            # Compute AEI for inconsistent customers
            aei_result = compute_integrity_score(log)
            aei_scores.append(aei_result.efficiency_index)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(customer_ids)} customers...")

    print("\n" + "-" * 60)
    print("Results")
    print("-" * 60)
    print(f"  Customers validated: {len(customer_ids)}")
    print(f"  GARP consistent: {n_consistent} ({100*n_consistent/len(customer_ids):.1f}%)")
    print(f"  GARP violations: {n_violations} ({100*n_violations/len(customer_ids):.1f}%)")

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
    parser = argparse.ArgumentParser(description="Validate UCI Online Retail data")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick validation on sample (50 customers)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=50,
        help="Sample size for quick mode (default: 50)",
    )
    args = parser.parse_args()

    run_validation(quick=args.quick, sample_size=args.sample)


if __name__ == "__main__":
    main()
