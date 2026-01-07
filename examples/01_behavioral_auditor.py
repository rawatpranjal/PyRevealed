"""Example: BehavioralAuditor - Linter-style API for behavioral consistency.

The BehavioralAuditor provides a high-level, "linter-style" API for validating
user behavior consistency using revealed preference theory.

This example shows how to use the auditor to:
- Check if behavior is consistent with utility maximization
- Measure behavioral integrity (Afriat Efficiency Index)
- Measure exploitability of inconsistencies (Money Pump Index)
"""

import numpy as np
from pyrevealed import BehaviorLog, BehavioralAuditor

# =============================================================================
# Example 1: Consistent User
# =============================================================================

print("=" * 60)
print("Example 1: Consistent User Behavior")
print("=" * 60)

# This user responds rationally to price changes:
# - When good A is cheap, they buy more A
# - When good B is cheap, they buy more B
consistent_log = BehaviorLog(
    cost_vectors=np.array([
        [1.0, 2.0],   # A is cheap, B is expensive
        [2.0, 1.0],   # A is expensive, B is cheap
        [1.5, 1.5],   # Equal prices
    ]),
    action_vectors=np.array([
        [4.0, 1.0],   # Bought more A (it was cheap)
        [1.0, 4.0],   # Bought more B (it was cheap)
        [2.5, 2.5],   # Bought equal amounts
    ]),
    user_id="user_consistent"
)

auditor = BehavioralAuditor()
report = auditor.full_audit(consistent_log)

print(f"User: {consistent_log.user_id}")
print(f"  Is Consistent: {report.is_consistent}")
print(f"  Integrity Score: {report.integrity_score:.2f}")
print(f"  Confusion Score: {report.confusion_score:.2f}")
print()

# =============================================================================
# Example 2: Inconsistent User
# =============================================================================

print("=" * 60)
print("Example 2: Inconsistent User Behavior")
print("=" * 60)

# This user violates revealed preference (WARP violation):
# - At prices [1, 2]: spent $7 on [3, 2], could have bought [5, 1] for $7
#   Reveals: prefers [3, 2] over [5, 1]
# - At prices [2, 1]: spent $12 on [5, 1], could have bought [3, 2] for $8
#   But they spent MORE to get [5, 1], which was revealed WORSE before!
inconsistent_log = BehaviorLog(
    cost_vectors=np.array([
        [1.0, 2.0],   # Price of A=1, B=2
        [2.0, 1.0],   # Price of A=2, B=1
    ]),
    action_vectors=np.array([
        [3.0, 2.0],   # Spent $7, chose more B (expensive)
        [5.0, 1.0],   # Spent $11, chose more A (expensive) - CONTRADICTION!
    ]),
    user_id="user_inconsistent"
)

report = auditor.full_audit(inconsistent_log)

print(f"User: {inconsistent_log.user_id}")
print(f"  Is Consistent: {report.is_consistent}")
print(f"  Integrity Score: {report.integrity_score:.2f}")
print(f"  Confusion Score: {report.confusion_score:.2f}")
print()

# =============================================================================
# Example 3: Using Individual Methods
# =============================================================================

print("=" * 60)
print("Example 3: Individual Methods")
print("=" * 60)

# You can also call methods individually for more control
print("Quick consistency check:")
is_ok = auditor.validate_history(consistent_log)
print(f"  validate_history() -> {is_ok}")

print("\nIntegrity score only:")
integrity = auditor.get_integrity_score(consistent_log)
print(f"  get_integrity_score() -> {integrity:.3f}")

print("\nConfusion score only:")
confusion = auditor.get_confusion_score(consistent_log)
print(f"  get_confusion_score() -> {confusion:.3f}")

# =============================================================================
# Example 4: Detailed Results
# =============================================================================

print("\n" + "=" * 60)
print("Example 4: Detailed Consistency Results")
print("=" * 60)

details = auditor.get_consistency_details(inconsistent_log)
print(f"Is Consistent: {details.is_consistent}")
print(f"Violations Found: {len(details.violations)}")
print(f"Computation Time: {details.computation_time_ms:.2f} ms")

if details.violations:
    print("\nViolation cycles (preference contradictions):")
    for cycle in details.violations[:3]:  # Show first 3
        print(f"  Cycle: {cycle}")

# =============================================================================
# Practical Use Cases
# =============================================================================

print("\n" + "=" * 60)
print("Practical Use Cases")
print("=" * 60)

print("""
1. DATA QUALITY ASSESSMENT:
   if report.integrity_score < 0.85:
       flag_for_review(user_id)

2. A/B TESTING:
   control_confusion = np.mean([audit(u).confusion_score for u in control])
   variant_confusion = np.mean([audit(u).confusion_score for u in variant])
   if variant_confusion < control_confusion:
       print("New UX reduces user confusion!")

3. SEGMENT ANALYSIS:
   for segment in user_segments:
       scores = [audit(u).integrity_score for u in segment]
       print(f"{segment.name}: mean integrity = {np.mean(scores):.2f}")
""")
