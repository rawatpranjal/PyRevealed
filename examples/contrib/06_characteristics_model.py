"""Example: Lancaster Characteristics Model for attribute-level analysis.

This module demonstrates the Lancaster Characteristics Model, which transforms
product-space behavioral data into characteristics-space. This enables:

1. "Rationality Rescue" - Users who seem irrational may be rational about attributes
2. Shadow Prices - Discover implicit valuations for each product attribute
3. New Product Prediction - Predict demand for new product configurations
4. Product Efficiency - Identify overpriced products on the efficient frontier

Use Cases:
- SaaS tier analysis (users valuing storage, seats, features)
- Grocery/CPG analysis (users valuing nutrients, convenience)
- Cloud computing (users valuing vCPU, RAM, storage)
- Subscription services (users valuing content hours, downloads, quality)
"""

import numpy as np

from pyrevealed import (
    BehaviorLog,
    LancasterLog,
    transform_to_characteristics,
    validate_consistency,
    compute_integrity_score,
)

# =============================================================================
# Example 1: Basic Lancaster Transformation
# =============================================================================

print("=" * 70)
print("Example 1: Basic Lancaster Transformation")
print("=" * 70)

# Scenario: Grocery shopping with 4 products
# Products: Chicken breast, Eggs, Greek yogurt, Almonds
# Characteristics: Protein (g), Fat (g), Carbs (g) per serving

attribute_matrix = np.array([
    # Protein, Fat, Carbs per serving
    [31.0, 3.6, 0.0],    # Chicken breast (100g)
    [6.0, 5.0, 0.6],     # Eggs (1 large)
    [10.0, 0.7, 3.6],    # Greek yogurt (100g)
    [6.0, 14.0, 6.0],    # Almonds (28g)
])

# Observed prices and quantities over 5 shopping trips
prices = np.array([
    [3.50, 0.25, 1.20, 0.75],
    [4.00, 0.30, 1.00, 0.80],
    [3.20, 0.28, 1.30, 0.70],
    [3.80, 0.22, 1.15, 0.85],
    [3.60, 0.27, 1.10, 0.72],
])

quantities = np.array([
    [2.0, 6.0, 3.0, 2.0],
    [1.5, 4.0, 4.0, 1.0],
    [2.5, 8.0, 2.0, 3.0],
    [1.0, 6.0, 5.0, 1.5],
    [2.0, 5.0, 3.5, 2.0],
])

# Create Lancaster log with characteristic names
lancaster_log = LancasterLog(
    cost_vectors=prices,
    action_vectors=quantities,
    attribute_matrix=attribute_matrix,
    user_id="health_conscious_shopper",
    metadata={"characteristic_names": ["protein", "fat", "carbs"]},
)

print(f"Products: {lancaster_log.num_products}")
print(f"Characteristics: {lancaster_log.num_characteristics}")
print(f"Observations: {lancaster_log.num_observations}")

print("\nSample characteristics consumed (first observation):")
z = lancaster_log.characteristics_quantities[0]
print(f"  Protein: {z[0]:.1f}g, Fat: {z[1]:.1f}g, Carbs: {z[2]:.1f}g")

print("\nSample shadow prices (observation 1):")
pi = lancaster_log.shadow_prices[0]
print(f"  $/g protein: ${pi[0]:.4f}")
print(f"  $/g fat: ${pi[1]:.4f}")
print(f"  $/g carbs: ${pi[2]:.4f}")

# =============================================================================
# Example 2: Run Algorithms on Characteristics Space
# =============================================================================

print("\n" + "=" * 70)
print("Example 2: Revealed Preference Analysis on Characteristics")
print("=" * 70)

# Get the characteristics-space BehaviorLog
char_log = lancaster_log.behavior_log

# Run standard algorithms
consistency = validate_consistency(char_log)
integrity = compute_integrity_score(char_log)

print(f"Consistency (at characteristic level): {consistency.is_consistent}")
print(f"Integrity score: {integrity.efficiency_index:.4f}")

if consistency.is_consistent:
    print("=> User has consistent preferences over nutrients!")
else:
    print(f"=> Found {consistency.num_violations} violation(s) in nutrient preferences")

# =============================================================================
# Example 3: Valuation Report
# =============================================================================

print("\n" + "=" * 70)
print("Example 3: Valuation Report (Business Insights)")
print("=" * 70)

report = lancaster_log.valuation_report()

print("Mean shadow prices ($/g):")
names = report.characteristic_names or ["protein", "fat", "carbs"]
for i, name in enumerate(names):
    print(f"  {name}: ${report.mean_shadow_prices[i]:.4f} "
          f"(std: ${report.shadow_price_std[i]:.4f}, CV: {report.shadow_price_cv[i]:.2f})")

print(f"\nSpend shares:")
for i, name in enumerate(names):
    print(f"  {name}: {report.spend_shares[i]*100:.1f}%")

print(f"\nModel diagnostics:")
print(f"  Attribute matrix rank: {report.attribute_matrix_rank}")
print(f"  Well-specified: {report.is_well_specified}")
print(f"  Mean NNLS residual: {report.mean_nnls_residual:.4f}")
print(f"  Problematic observations: {report.problematic_observations or 'None'}")

print(f"\nKey insights:")
print(f"  Most valued characteristic: {names[report.most_valued_characteristic]}")
print(f"  Most volatile characteristic: {names[report.most_volatile_characteristic]}")

# =============================================================================
# Example 4: Compare Product vs Characteristics Consistency
# =============================================================================

print("\n" + "=" * 70)
print("Example 4: Product Space vs Characteristics Space")
print("=" * 70)

# Create product-space BehaviorLog for comparison
product_log = BehaviorLog(cost_vectors=prices, action_vectors=quantities)

product_consistency = validate_consistency(product_log)
product_integrity = compute_integrity_score(product_log)

print("Product-space analysis:")
print(f"  Consistent: {product_consistency.is_consistent}")
print(f"  Integrity: {product_integrity.efficiency_index:.4f}")

print("\nCharacteristics-space analysis:")
print(f"  Consistent: {consistency.is_consistent}")
print(f"  Integrity: {integrity.efficiency_index:.4f}")

print("""
Interpretation:
- If characteristics-space is MORE consistent than product-space:
  User cares about underlying attributes, not specific products
  (Good candidate for store-brand substitution)

- If product-space is MORE consistent:
  User has product-specific preferences (brand loyalty, etc.)
""")

# =============================================================================
# Example 5: Transform Existing BehaviorLog
# =============================================================================

print("=" * 70)
print("Example 5: Transform Existing BehaviorLog")
print("=" * 70)

# Imagine you already have a BehaviorLog from another analysis
existing_log = BehaviorLog(
    cost_vectors=prices,
    action_vectors=quantities,
    user_id="existing_user",
)

# Transform it to characteristics space
transformed = transform_to_characteristics(
    existing_log,
    attribute_matrix,
    characteristic_names=["protein", "fat", "carbs"],
)

print(f"Original user_id: {existing_log.user_id}")
print(f"Transformed user_id: {transformed.user_id}")
print(f"Ready for characteristics analysis: {transformed.num_features} features")

# =============================================================================
# Example 6: Cloud Computing Instance Selection
# =============================================================================

print("\n" + "=" * 70)
print("Example 6: Cloud Computing Instance Selection")
print("=" * 70)

# Scenario: User selecting AWS-like instances
# Products: t3.small, m5.large, c5.xlarge
# Characteristics: vCPU, RAM (GB)

cloud_attributes = np.array([
    [2, 2],    # t3.small: 2 vCPU, 2 GB RAM
    [2, 8],    # m5.large: 2 vCPU, 8 GB RAM
    [4, 8],    # c5.xlarge: 4 vCPU, 8 GB RAM
])

cloud_prices = np.array([
    [0.02, 0.09, 0.17],  # Month 1
    [0.02, 0.08, 0.16],  # Month 2 (RAM got cheaper)
    [0.03, 0.10, 0.18],  # Month 3
])

cloud_quantities = np.array([
    [10, 0, 0],   # Month 1: Bought cheap t3s
    [0, 5, 0],    # Month 2: Switched to m5s (more RAM per dollar)
    [0, 0, 2],    # Month 3: Switched to c5s (needed CPU)
])

cloud_log = LancasterLog(
    cost_vectors=cloud_prices,
    action_vectors=cloud_quantities,
    attribute_matrix=cloud_attributes,
    user_id="devops_team",
    metadata={"characteristic_names": ["vCPU", "RAM_GB"]},
)

print("Cloud instance analysis:")
print(f"  Total observations: {cloud_log.num_observations}")
print(f"  Characteristics: {cloud_log.metadata['characteristic_names']}")

cloud_report = cloud_log.valuation_report()
print(f"\nImplied valuations:")
print(f"  $/vCPU: ${cloud_report.mean_shadow_prices[0]:.4f}")
print(f"  $/GB RAM: ${cloud_report.mean_shadow_prices[1]:.4f}")

cloud_char = cloud_log.behavior_log
cloud_consistency = validate_consistency(cloud_char)
print(f"\nIs compute usage rational? {cloud_consistency.is_consistent}")

print("""
Business insight:
- The user is switching instances but may be rational about compute resources
- Check if they're optimizing for vCPU vs RAM based on workload needs
- Use shadow prices to recommend cost-optimal instance types
""")

print("\n" + "=" * 70)
print("Examples complete!")
print("=" * 70)
