"""Example: Risk Analysis - Profiling risk attitudes from choices under uncertainty.

This module analyzes choices between safe options (certain payoffs) and risky
options (lotteries) to classify users as:
- Risk-averse ("investors"): Prefer certainty, need risk premium
- Risk-neutral: Maximize expected value
- Risk-seeking ("gamblers"): Prefer uncertainty, pay for risk

Based on Constant Relative Risk Aversion (CRRA) utility model.
"""

import numpy as np
from pyrevealed import (
    RiskChoiceLog,  # or RiskSession (legacy name)
    compute_risk_profile,
    check_expected_utility_axioms,
    classify_risk_type,
)

# =============================================================================
# Example 1: Risk-Averse User ("Investor")
# =============================================================================

print("=" * 60)
print("Example 1: Risk-Averse User (Investor Profile)")
print("=" * 60)

# Risk-averse person: prefers certain $50 over 50/50 chance of $100/$0
# Even though expected value of lottery ($50) equals the safe option
safe_values = np.array([50.0, 45.0, 40.0, 35.0, 30.0])
risky_outcomes = np.array([
    [100.0, 0.0],   # 50/50 chance of $100 or $0
    [100.0, 0.0],
    [100.0, 0.0],
    [100.0, 0.0],
    [100.0, 0.0],
])
risky_probabilities = np.array([
    [0.5, 0.5],
    [0.5, 0.5],
    [0.5, 0.5],
    [0.5, 0.5],
    [0.5, 0.5],
])
# Only takes gamble when safe option drops to $30
choices = np.array([False, False, False, False, True])

risk_averse_session = RiskChoiceLog(
    safe_values=safe_values,
    risky_outcomes=risky_outcomes,
    risky_probabilities=risky_probabilities,
    choices=choices,
    session_id="investor_user"
)

result = compute_risk_profile(risk_averse_session)

print(f"User: {risk_averse_session.session_id}")
print(f"Risk aversion coefficient (rho): {result.risk_aversion_coefficient:.3f}")
print(f"  (rho > 0 = risk averse, rho < 0 = risk seeking)")
print(f"Risk category: {result.risk_category}")
print(f"Model consistency: {result.consistency_score:.1%}")
print(f"Utility curvature: {result.utility_curvature:.4f}")
print(f"Computation time: {result.computation_time_ms:.2f} ms")

print("\nCertainty equivalents (what certain amount equals each lottery):")
for i, ce in enumerate(result.certainty_equivalents):
    ev = np.sum(risky_outcomes[i] * risky_probabilities[i])
    print(f"  Lottery {i}: CE=${ce:.2f} vs EV=${ev:.2f}")

# =============================================================================
# Example 2: Risk-Seeking User ("Gambler")
# =============================================================================

print("\n" + "=" * 60)
print("Example 2: Risk-Seeking User (Gambler Profile)")
print("=" * 60)

# Risk-seeking person: prefers lottery even when EV is lower than safe option
safe_values_seek = np.array([60.0, 55.0, 50.0, 45.0, 40.0])
# Takes the gamble even at $60 safe (EV of lottery is only $50)
choices_seek = np.array([True, True, True, True, True])

risk_seeking_session = RiskChoiceLog(
    safe_values=safe_values_seek,
    risky_outcomes=risky_outcomes,
    risky_probabilities=risky_probabilities,
    choices=choices_seek,
    session_id="gambler_user"
)

result_seek = compute_risk_profile(risk_seeking_session)

print(f"User: {risk_seeking_session.session_id}")
print(f"Risk aversion coefficient (rho): {result_seek.risk_aversion_coefficient:.3f}")
print(f"Risk category: {result_seek.risk_category}")
print(f"Model consistency: {result_seek.consistency_score:.1%}")

# =============================================================================
# Example 3: Risk-Neutral User
# =============================================================================

print("\n" + "=" * 60)
print("Example 3: Risk-Neutral User")
print("=" * 60)

# Risk-neutral: chooses based purely on expected value
safe_values_neutral = np.array([60.0, 55.0, 50.0, 45.0, 40.0])
# Takes safe when EV < safe, takes risky when EV > safe
# EV of lottery = $50
choices_neutral = np.array([False, False, False, True, True])

neutral_session = RiskChoiceLog(
    safe_values=safe_values_neutral,
    risky_outcomes=risky_outcomes,
    risky_probabilities=risky_probabilities,
    choices=choices_neutral,
    session_id="neutral_user"
)

result_neutral = compute_risk_profile(neutral_session)

print(f"User: {neutral_session.session_id}")
print(f"Risk aversion coefficient (rho): {result_neutral.risk_aversion_coefficient:.3f}")
print(f"Risk category: {result_neutral.risk_category}")
print(f"Model consistency: {result_neutral.consistency_score:.1%}")

# =============================================================================
# Example 4: Quick Classification
# =============================================================================

print("\n" + "=" * 60)
print("Example 4: Quick Classification")
print("=" * 60)

# Use classify_risk_type for simple labeling
print("Quick classifications:")
print(f"  Risk-averse user: {classify_risk_type(risk_averse_session)}")
print(f"  Risk-seeking user: {classify_risk_type(risk_seeking_session)}")
print(f"  Risk-neutral user: {classify_risk_type(neutral_session)}")

# =============================================================================
# Example 5: Check Expected Utility Axioms
# =============================================================================

print("\n" + "=" * 60)
print("Example 5: Check Expected Utility Axioms")
print("=" * 60)

# Create a session with axiom violations
# Violation: choosing dominated option (safe < min(risky))
violating_session = RiskChoiceLog(
    safe_values=np.array([30.0, 50.0]),
    risky_outcomes=np.array([
        [40.0, 35.0],   # Min is 35, which is > safe of 30
        [100.0, 0.0],
    ]),
    risky_probabilities=np.array([
        [0.5, 0.5],
        [0.5, 0.5],
    ]),
    choices=np.array([False, True]),  # Chose safe $30 over guaranteed $35+
    session_id="violating_user"
)

is_consistent, violations = check_expected_utility_axioms(violating_session)

print(f"Axiom check: {'PASS' if is_consistent else 'FAIL'}")
if violations:
    print("Violations found:")
    for v in violations:
        print(f"  {v}")

# Check a consistent session
is_consistent_ok, _ = check_expected_utility_axioms(risk_averse_session)
print(f"\nRisk-averse session: {'PASS' if is_consistent_ok else 'FAIL'}")

# =============================================================================
# Example 6: RiskChoiceLog Properties
# =============================================================================

print("\n" + "=" * 60)
print("Example 6: RiskChoiceLog Convenience Properties")
print("=" * 60)

session = risk_averse_session

print(f"Number of observations: {session.num_observations}")
print(f"Number of outcomes per lottery: {session.num_outcomes}")
print(f"\nExpected values of lotteries:")
for i, ev in enumerate(session.expected_values):
    print(f"  Lottery {i}: EV = ${ev:.2f}")

print(f"\nRisk-neutral reference choices (EV > safe):")
print(f"  {session.risk_neutral_choices}")

print(f"\nRisk-seeking choices (chose risky despite lower EV): {session.num_risk_seeking_choices}")
print(f"Risk-averse choices (chose safe despite lower EV): {session.num_risk_averse_choices}")

# =============================================================================
# Practical Applications
# =============================================================================

print("\n" + "=" * 60)
print("Practical Applications")
print("=" * 60)

print("""
1. INVESTMENT PRODUCT RECOMMENDATIONS:
   result = compute_risk_profile(user_choices)
   if result.risk_category == "risk_averse":
       recommend_bonds_and_savings()
   elif result.risk_category == "risk_seeking":
       recommend_high_volatility_stocks()

2. INSURANCE PRICING:
   # Risk-averse users pay more for certainty
   if result.risk_aversion_coefficient > 0.5:
       premium_multiplier = 1.2  # They'll pay 20% more

3. GAME DESIGN (Loot boxes, gacha):
   user_type = classify_risk_type(player_choices)
   if user_type == "gambler":
       show_rarer_high_variance_offers()

4. FRAUD DETECTION:
   # Sudden risk preference changes may indicate account compromise
   historical_rho = compute_risk_profile(old_choices).risk_aversion_coefficient
   current_rho = compute_risk_profile(new_choices).risk_aversion_coefficient
   if abs(historical_rho - current_rho) > 1.0:
       flag_account_for_review()
""")
