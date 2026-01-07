"""Example: Spatial Preferences - Ideal point analysis in embedding space.

This module finds a user's "ideal point" in feature space based on their choices.
The model assumes users prefer items closer to their ideal point (Euclidean
preference model from Chambers & Echenique Chapter 11).

Use cases:
- Recommendation explainability ("You prefer items near this anchor")
- Personalization (recommend items close to user's ideal point)
- Preference heterogeneity analysis
- Embedding-based preference analysis
"""

import numpy as np
from pyrevealed import (
    EmbeddingChoiceLog,  # or SpatialSession (legacy name)
    find_preference_anchor,  # or find_ideal_point (legacy name)
    validate_embedding_consistency,  # or check_euclidean_rationality
    compute_signal_strength,  # or compute_preference_strength
    find_multiple_anchors,  # or find_multiple_ideal_points
)

# =============================================================================
# Example 1: Finding User's Ideal Point
# =============================================================================

print("=" * 60)
print("Example 1: Find User's Preference Anchor (Ideal Point)")
print("=" * 60)

# 6 items in 2D feature space (e.g., product embeddings)
# Features could be: [sweetness, crunchiness], [price_tier, quality], etc.
item_features = np.array([
    [0.0, 0.0],   # Item 0: low on both dimensions
    [1.0, 0.0],   # Item 1: high on dim 0, low on dim 1
    [0.0, 1.0],   # Item 2: low on dim 0, high on dim 1
    [1.0, 1.0],   # Item 3: high on both
    [0.5, 0.5],   # Item 4: middle
    [0.2, 0.3],   # Item 5: near origin
])

# User's choice history: given choice sets, which item did they pick?
# This user seems to prefer items near the origin
choice_sets = [
    [0, 1, 2],     # Chose from items 0, 1, 2
    [1, 3, 4],     # Chose from items 1, 3, 4
    [0, 4, 5],     # Chose from items 0, 4, 5
    [2, 3, 5],     # Chose from items 2, 3, 5
]
choices = [0, 4, 5, 5]  # User consistently chose items near origin

session = EmbeddingChoiceLog(
    item_features=item_features,
    choice_sets=choice_sets,
    choices=choices,
    session_id="user_origin_lover"
)

result = find_preference_anchor(session)

print(f"User: {session.session_id}")
print(f"Ideal point (anchor): {result.ideal_point}")
print(f"Is Euclidean rational: {result.is_euclidean_rational}")
print(f"Number of violations: {result.num_violations}")
print(f"Explained variance: {result.explained_variance:.1%}")
print(f"Mean distance to chosen: {result.mean_distance_to_chosen:.4f}")
print(f"Computation time: {result.computation_time_ms:.2f} ms")

# =============================================================================
# Example 2: User Preferring Different Location
# =============================================================================

print("\n" + "=" * 60)
print("Example 2: User with Different Preferences")
print("=" * 60)

# This user prefers items near (1, 1)
choices_high = [2, 3, 4, 3]  # Chose items closer to (1, 1)

session_high = EmbeddingChoiceLog(
    item_features=item_features,
    choice_sets=choice_sets,
    choices=choices_high,
    session_id="user_premium_lover"
)

result_high = find_preference_anchor(session_high)

print(f"User: {session_high.session_id}")
print(f"Ideal point: {result_high.ideal_point}")
print(f"Explained variance: {result_high.explained_variance:.1%}")

# =============================================================================
# Example 3: Validate Embedding Consistency
# =============================================================================

print("\n" + "=" * 60)
print("Example 3: Validate Embedding Consistency")
print("=" * 60)

is_rational, violations = validate_embedding_consistency(session)

print(f"Is Euclidean rational: {is_rational}")
if violations:
    print("Violations (chosen item was farther than alternative):")
    for choice_idx, alt_item in violations:
        print(f"  Choice {choice_idx}: Item {alt_item} was closer but not chosen")
else:
    print("No violations - choices consistent with single ideal point")

# =============================================================================
# Example 4: Compute Signal Strength
# =============================================================================

print("\n" + "=" * 60)
print("Example 4: Signal Strength per Choice")
print("=" * 60)

strengths = compute_signal_strength(session, result.ideal_point)

print("Signal strength per choice:")
print("  Positive = chosen was clearly closer to ideal")
print("  Negative = violation (chose farther item)")
print()
for i, s in enumerate(strengths):
    status = "strong preference" if s > 0.3 else ("clear" if s > 0 else "VIOLATION")
    print(f"  Choice {i}: strength={s:.3f} ({status})")

# =============================================================================
# Example 5: Analyzing Preference Heterogeneity (Multiple Anchors)
# =============================================================================

print("\n" + "=" * 60)
print("Example 5: Analyze Preference Heterogeneity with Multiple Anchors")
print("=" * 60)

# Simulate heterogeneous preferences: some choices favor origin,
# some favor (1, 1)
mixed_choices = [0, 3, 5, 3]  # Inconsistent: some near origin, some near (1,1)

mixed_session = EmbeddingChoiceLog(
    item_features=item_features,
    choice_sets=choice_sets,
    choices=mixed_choices,
    session_id="mixed_preferences"
)

# Find multiple anchors
anchors = find_multiple_anchors(mixed_session, n_points=2)

print(f"User: {mixed_session.session_id}")
print(f"\nMultiple anchors found:")
for i, (anchor, explained) in enumerate(anchors):
    print(f"  Anchor {i+1}: {anchor}")
    print(f"    Explains: {explained:.1%} of choices")

# Single anchor check
single_result = find_preference_anchor(mixed_session)
print(f"\nSingle anchor explained variance: {single_result.explained_variance:.1%}")
print(f"Violations with single anchor: {single_result.num_violations}")

if len(anchors) > 1 and anchors[1][1] > 0.2:
    print("\nHETEROGENEOUS PREFERENCES: Multiple distinct preference profiles detected!")

# =============================================================================
# Example 6: Recommendation Use Case
# =============================================================================

print("\n" + "=" * 60)
print("Example 6: Personalized Recommendations")
print("=" * 60)

# New items to potentially recommend
new_items = np.array([
    [0.1, 0.1],   # Close to origin
    [0.9, 0.9],   # Far from origin
    [0.3, 0.2],   # Moderately close to origin
])

# For user who prefers origin
ideal = result.ideal_point
print(f"User's ideal point: {ideal}")
print(f"\nRecommendation scores (lower distance = better match):")

distances = [np.linalg.norm(item - ideal) for item in new_items]
ranked = sorted(enumerate(distances), key=lambda x: x[1])

for rank, (idx, dist) in enumerate(ranked, 1):
    print(f"  {rank}. New item {idx} (features={new_items[idx]}): distance={dist:.3f}")

print(f"\nRecommend: New item {ranked[0][0]} (closest to user's ideal point)")

# =============================================================================
# Practical Applications
# =============================================================================

print("\n" + "=" * 60)
print("Practical Applications")
print("=" * 60)

print("""
1. RECOMMENDATION EXPLAINABILITY:
   result = find_preference_anchor(user_choices)
   print(f"We recommend this because it's similar to items you liked")
   print(f"Your preference anchor: {result.ideal_point}")

2. PERSONALIZATION WITH EMBEDDINGS:
   # Use any embedding (word2vec, image embeddings, etc.)
   user_anchor = find_preference_anchor(user_choices).ideal_point
   # Score new items by distance to anchor
   scores = [-np.linalg.norm(item_emb - user_anchor) for item_emb in catalog]

3. PREFERENCE HETEROGENEITY ANALYSIS:
   anchors = find_multiple_anchors(choices, n_points=3)
   if len(anchors) > 1 and anchors[1][1] > 0.25:
       print("Multiple distinct preference profiles detected")

4. PREFERENCE DRIFT DETECTION:
   old_anchor = find_preference_anchor(old_choices).ideal_point
   new_anchor = find_preference_anchor(new_choices).ideal_point
   drift = np.linalg.norm(new_anchor - old_anchor)
   if drift > threshold:
       trigger_preference_refresh()

5. A/B TEST ANALYSIS:
   # Do different UIs lead to different preference profiles?
   control_anchors = [find_preference_anchor(u) for u in control_users]
   variant_anchors = [find_preference_anchor(u) for u in variant_users]
   # Compare explained_variance distributions
""")
