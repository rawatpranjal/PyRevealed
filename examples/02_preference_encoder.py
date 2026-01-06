"""Example: PreferenceEncoder - sklearn-style API for ML integration.

The PreferenceEncoder follows the scikit-learn pattern (fit/transform) to extract
latent preference values from user behavior. Use it for:
- Feature extraction for ML models (user embeddings)
- Counterfactual predictions (what would user buy at different prices?)
- Personalization and recommendation systems
"""

import numpy as np
from pyrevealed import BehaviorLog, PreferenceEncoder

# =============================================================================
# Example 1: Basic Fit and Extract
# =============================================================================

print("=" * 60)
print("Example 1: Basic Fit and Extract Latent Values")
print("=" * 60)

# User behavior over 5 observations
# This data represents a rational consumer responding to price changes
log = BehaviorLog(
    cost_vectors=np.array([
        [1.0, 3.0],   # Good A cheap, B expensive
        [3.0, 1.0],   # Good A expensive, B cheap
        [2.0, 2.0],   # Equal prices
        [1.5, 2.5],   # A slightly cheaper
        [2.5, 1.5],   # B slightly cheaper
    ]),
    action_vectors=np.array([
        [6.0, 1.0],   # Bought more A (it was cheap)
        [1.0, 6.0],   # Bought more B (it was cheap)
        [3.0, 3.0],   # Equal amounts at equal prices
        [5.0, 2.0],   # More A when A cheaper
        [2.0, 5.0],   # More B when B cheaper
    ]),
    user_id="user_001"
)

# Fit encoder to behavior
encoder = PreferenceEncoder()
encoder.fit(log)

print(f"Encoder fitted: {encoder.is_fitted}")
print(f"Solver status: {encoder.solver_status}")

# Extract latent values (one per observation)
latent_values = encoder.extract_latent_values()
print(f"\nLatent values per observation:")
for i, val in enumerate(latent_values):
    print(f"  Observation {i}: {val:.4f}")

# Extract marginal weights (sensitivity to cost changes)
marginal_weights = encoder.extract_marginal_weights()
print(f"\nMarginal weights (price sensitivity):")
for i, w in enumerate(marginal_weights):
    print(f"  Observation {i}: {w:.4f}")

print(f"\nMean marginal weight: {encoder.mean_marginal_weight:.4f}")

# =============================================================================
# Example 2: Value Function for Counterfactuals
# =============================================================================

print("\n" + "=" * 60)
print("Example 2: Value Function for Any Action")
print("=" * 60)

# Get a callable value function
value_fn = encoder.get_value_function()

# Estimate value of hypothetical bundles
test_bundles = [
    [3.0, 3.0],   # Equal amounts
    [10.0, 0.0],  # Only good A
    [0.0, 10.0],  # Only good B
    [5.0, 5.0],   # More of both
]

print("Estimated values for hypothetical bundles:")
for bundle in test_bundles:
    value = value_fn(np.array(bundle))
    print(f"  Bundle {bundle}: value = {value:.4f}")

# =============================================================================
# Example 3: Counterfactual Predictions
# =============================================================================

print("\n" + "=" * 60)
print("Example 3: Predict Choice Under New Conditions")
print("=" * 60)

# What would the user buy if prices changed?
new_prices = np.array([0.5, 2.0])  # Good A becomes very cheap
budget = 10.0

predicted = encoder.predict_choice(new_prices, budget)
if predicted is not None:
    print(f"New prices: {new_prices}")
    print(f"Budget: ${budget}")
    print(f"Predicted purchase: {predicted}")
    print(f"Predicted spend: ${np.dot(new_prices, predicted):.2f}")
else:
    print("Prediction failed (behavior may be too inconsistent)")

# Another scenario
print("\nScenario 2: Good B becomes cheap")
new_prices_2 = np.array([3.0, 0.5])
predicted_2 = encoder.predict_choice(new_prices_2, budget)
if predicted_2 is not None:
    print(f"New prices: {new_prices_2}")
    print(f"Budget: ${budget}")
    print(f"Predicted purchase: {predicted_2}")

# =============================================================================
# Example 4: Using Latent Values as ML Features
# =============================================================================

print("\n" + "=" * 60)
print("Example 4: ML Feature Extraction")
print("=" * 60)

# Simulate multiple users
np.random.seed(42)
users = []
features_list = []

for user_idx in range(5):
    # Generate synthetic user data
    T = np.random.randint(5, 10)
    prices = np.abs(np.random.randn(T, 3)) + 0.5
    quantities = np.abs(np.random.randn(T, 3))

    user_log = BehaviorLog(
        cost_vectors=prices,
        action_vectors=quantities,
        user_id=f"user_{user_idx:03d}"
    )

    enc = PreferenceEncoder()
    enc.fit(user_log)

    if enc.is_fitted:
        # Use mean latent value as user-level feature
        mean_latent = np.mean(enc.extract_latent_values())
        mean_marginal = enc.mean_marginal_weight
        users.append(user_log.user_id)
        features_list.append([mean_latent, mean_marginal])

print("Extracted features per user (for ML models):")
print(f"{'User':<12} {'Mean Latent':<14} {'Mean Marginal':<14}")
print("-" * 40)
for user, feats in zip(users, features_list):
    print(f"{user:<12} {feats[0]:<14.4f} {feats[1]:<14.4f}")

print("""
These features can be used for:
- User clustering (segment users by preference patterns)
- Churn prediction (users with unstable preferences)
- Personalization (similar preference = similar recommendations)
""")

# =============================================================================
# Example 5: Fit Details for Diagnostics
# =============================================================================

print("=" * 60)
print("Example 5: Diagnostic Details")
print("=" * 60)

details = encoder.get_fit_details()
print(f"LP solver success: {details.success}")
print(f"Solver status: {details.lp_status}")
print(f"Computation time: {details.computation_time_ms:.2f} ms")
print(f"Mean marginal utility: {details.mean_marginal_utility:.4f}")

if details.residuals is not None:
    print(f"Max constraint residual: {np.max(np.abs(details.residuals)):.6f}")
