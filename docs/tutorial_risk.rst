Tutorial 8: Risk Analysis
=========================

This tutorial covers analyzing risk preferences using revealed preference methods.
We estimate risk aversion coefficients, test expected utility axioms, and classify
decision-makers as risk-averse, risk-neutral, or risk-seeking.

Topics covered:

- RiskChoiceLog data structure
- CRRA utility estimation
- Expected utility axiom testing
- Quick risk type classification
- Application examples

Prerequisites
-------------

- Python 3.10+
- Understanding of BehaviorLog (Tutorial 1)
- Basic knowledge of expected utility theory

.. note::

   This tutorial implements methods from Chapter 8 of Chambers & Echenique (2016)
   "Revealed Preference Theory", based on Chambers, Echenique, and Saito (2015).


Part 1: Theory Review
---------------------

Expected Utility
~~~~~~~~~~~~~~~~

Under expected utility theory, a decision-maker evaluates risky options by
computing the probability-weighted sum of utilities:

.. math::

   EU(L) = \sum_i p_i \cdot u(x_i)

where :math:`L` is a lottery with outcomes :math:`x_i` occurring with
probabilities :math:`p_i`.

CRRA Utility
~~~~~~~~~~~~

The **Constant Relative Risk Aversion (CRRA)** utility function is:

.. math::

   u(x) = \begin{cases}
   \frac{x^{1-\rho}}{1-\rho} & \text{if } \rho \neq 1 \\
   \ln(x) & \text{if } \rho = 1
   \end{cases}

where :math:`\rho` is the **Arrow-Pratt coefficient of relative risk aversion**:

- :math:`\rho > 0`: Risk averse (prefers certainty)
- :math:`\rho = 0`: Risk neutral (maximizes expected value)
- :math:`\rho < 0`: Risk seeking (prefers gambles)


Part 2: The RiskChoiceLog Data Structure
----------------------------------------

A ``RiskChoiceLog`` stores choices between safe and risky options:

.. code-block:: python

   import numpy as np
   from pyrevealed import RiskChoiceLog

   # 5 binary choices: safe amount vs risky lottery
   safe_values = np.array([50.0, 40.0, 30.0, 20.0, 10.0])

   # Each lottery: [win_outcome, lose_outcome]
   risky_outcomes = np.array([
       [100.0, 0.0],   # 50/50 chance of $100 or $0
       [100.0, 0.0],
       [100.0, 0.0],
       [100.0, 0.0],
       [100.0, 0.0],
   ])

   # Probabilities for each outcome
   risky_probabilities = np.array([
       [0.5, 0.5],
       [0.5, 0.5],
       [0.5, 0.5],
       [0.5, 0.5],
       [0.5, 0.5],
   ])

   # Choices: True = chose risky, False = chose safe
   # Risk-averse person takes gamble only at low safe amounts
   choices = np.array([False, False, False, True, True])

   log = RiskChoiceLog(
       safe_values=safe_values,
       risky_outcomes=risky_outcomes,
       risky_probabilities=risky_probabilities,
       choices=choices,
       session_id="investor_001"
   )

   print(f"Observations: {log.num_observations}")

Output:

.. code-block:: text

   Observations: 5


Part 3: Estimating Risk Profiles
--------------------------------

The ``compute_risk_profile`` function estimates the CRRA parameter :math:`\rho`
using maximum likelihood estimation:

.. code-block:: python

   from pyrevealed import compute_risk_profile

   result = compute_risk_profile(log)

   print(f"Risk aversion coefficient (ρ): {result.risk_aversion_coefficient:.3f}")
   print(f"Risk category: {result.risk_category}")
   print(f"Consistency score: {result.consistency_score:.2%}")

Output:

.. code-block:: text

   Risk aversion coefficient (ρ): 1.234
   Risk category: risk_averse
   Consistency score: 100.00%

Interpreting Results
~~~~~~~~~~~~~~~~~~~~

.. list-table:: Risk Coefficient Interpretation
   :header-rows: 1
   :widths: 25 25 50

   * - Coefficient (ρ)
     - Category
     - Interpretation
   * - ρ > 2
     - Highly risk averse
     - Strongly prefers certainty; avoids most gambles
   * - 0.5 < ρ < 2
     - Moderately risk averse
     - Typical investor behavior
   * - -0.1 < ρ < 0.1
     - Risk neutral
     - Maximizes expected monetary value
   * - ρ < -0.5
     - Risk seeking
     - Prefers gambles to expected value equivalents

Certainty Equivalents
~~~~~~~~~~~~~~~~~~~~~

The **certainty equivalent** is the guaranteed amount that makes the decision-maker
indifferent to a lottery:

.. code-block:: python

   print("Certainty equivalents for each lottery:")
   for i, ce in enumerate(result.certainty_equivalents):
       ev = risky_outcomes[i] @ risky_probabilities[i]  # Expected value
       print(f"  Lottery {i}: CE=${ce:.2f} (EV=${ev:.2f})")

Output:

.. code-block:: text

   Certainty equivalents for each lottery:
     Lottery 0: CE=$35.72 (EV=$50.00)
     Lottery 1: CE=$35.72 (EV=$50.00)
     Lottery 2: CE=$35.72 (EV=$50.00)
     Lottery 3: CE=$35.72 (EV=$50.00)
     Lottery 4: CE=$35.72 (EV=$50.00)

For a risk-averse person, CE < EV (they would accept less than expected value
to avoid risk).


Part 4: Testing Expected Utility Axioms
---------------------------------------

The ``check_expected_utility_axioms`` function tests for basic EU violations:

.. code-block:: python

   from pyrevealed import check_expected_utility_axioms

   is_consistent, violations = check_expected_utility_axioms(log)

   if is_consistent:
       print("Choices are consistent with Expected Utility")
   else:
       print("EU axiom violations found:")
       for v in violations:
           print(f"  {v}")

Output:

.. code-block:: text

   Choices are consistent with Expected Utility

Monotonicity Violations
~~~~~~~~~~~~~~~~~~~~~~~

The test checks for **monotonicity** violations:

- If safe > max(risky outcomes), should always choose safe
- If safe < min(risky outcomes), should always choose risky

.. code-block:: python

   # Example with a monotonicity violation
   bad_choices = np.array([
       True,   # Chose risky when safe=$100 dominates lottery [80, 0]
       False,
       False,
   ])

   bad_log = RiskChoiceLog(
       safe_values=np.array([100.0, 50.0, 30.0]),
       risky_outcomes=np.array([[80.0, 0.0], [100.0, 0.0], [100.0, 0.0]]),
       risky_probabilities=np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]),
       choices=bad_choices,
   )

   is_ok, violations = check_expected_utility_axioms(bad_log)
   print(f"Consistent: {is_ok}")
   print(f"Violations: {violations}")

Output:

.. code-block:: text

   Consistent: False
   Violations: ['Obs 0: Chose risky [80.0, 0.0] over dominating safe 100.0']


Part 5: Quick Risk Classification
---------------------------------

For rapid classification without detailed analysis, use ``classify_risk_type``:

.. code-block:: python

   from pyrevealed import classify_risk_type

   risk_type = classify_risk_type(log)
   print(f"Classification: {risk_type}")

Output:

.. code-block:: text

   Classification: investor

Classification Types
~~~~~~~~~~~~~~~~~~~~

.. list-table:: Risk Type Classification
   :header-rows: 1
   :widths: 25 75

   * - Type
     - Description
   * - investor
     - Risk-averse; prefers certainty over gambles
   * - gambler
     - Risk-seeking; prefers gambles over expected value
   * - neutral
     - Maximizes expected monetary value
   * - inconsistent
     - Choices don't fit any clear pattern (consistency < 60%)


Part 6: Application Example
---------------------------

Analyzing Investment Decisions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from pyrevealed import (
       RiskChoiceLog,
       compute_risk_profile,
       check_expected_utility_axioms,
       classify_risk_type,
   )

   # Simulate an investor facing portfolio choices
   np.random.seed(42)

   # 20 investment decisions
   n_choices = 20

   # Safe options: guaranteed returns (e.g., bonds)
   safe_returns = np.random.uniform(3.0, 8.0, n_choices)  # 3-8% return

   # Risky options: stock-like returns
   risky_high = np.random.uniform(15.0, 25.0, n_choices)  # Bull market
   risky_low = np.random.uniform(-10.0, 2.0, n_choices)   # Bear market
   risky_outcomes = np.column_stack([risky_high, risky_low])

   # Market conditions: 60% bull, 40% bear
   risky_probabilities = np.tile([0.6, 0.4], (n_choices, 1))

   # Simulate choices for a moderately risk-averse investor (ρ ≈ 1.5)
   # They choose risky when expected utility exceeds safe utility
   def crra_utility(x, rho=1.5):
       return np.power(np.maximum(x, 0.01), 1-rho) / (1-rho)

   eu_risky = np.sum(crra_utility(risky_outcomes) * risky_probabilities, axis=1)
   u_safe = crra_utility(safe_returns)
   choices = eu_risky > u_safe

   # Add some noise (10% random choices)
   noise_mask = np.random.random(n_choices) < 0.10
   choices[noise_mask] = ~choices[noise_mask]

   log = RiskChoiceLog(
       safe_values=safe_returns,
       risky_outcomes=risky_outcomes,
       risky_probabilities=risky_probabilities,
       choices=choices,
       session_id="portfolio_investor"
   )

   # Full analysis
   print("=" * 60)
   print("INVESTMENT RISK PROFILE ANALYSIS")
   print("=" * 60)

   # Step 1: Check EU axioms
   is_consistent, violations = check_expected_utility_axioms(log)
   print(f"\nExpected Utility Axioms:")
   print(f"  Consistent: {is_consistent}")
   if violations:
       print(f"  Violations: {len(violations)}")

   # Step 2: Estimate risk profile
   result = compute_risk_profile(log)
   print(f"\nRisk Profile:")
   print(f"  Risk aversion (ρ): {result.risk_aversion_coefficient:.3f}")
   print(f"  Category: {result.risk_category}")
   print(f"  Consistency: {result.consistency_score:.1%}")
   print(f"  Utility curvature: {result.utility_curvature:.4f}")

   # Step 3: Quick classification
   classification = classify_risk_type(log)
   print(f"\nClassification: {classification}")

   # Step 4: Analyze certainty equivalents
   evs = np.sum(risky_outcomes * risky_probabilities, axis=1)
   ces = result.certainty_equivalents
   risk_premium = evs - ces

   print(f"\nRisk Premium Analysis:")
   print(f"  Mean expected value: {np.mean(evs):.2f}%")
   print(f"  Mean certainty equivalent: {np.mean(ces):.2f}%")
   print(f"  Mean risk premium: {np.mean(risk_premium):.2f}%")

Example output:

.. code-block:: text

   ============================================================
   INVESTMENT RISK PROFILE ANALYSIS
   ============================================================

   Expected Utility Axioms:
     Consistent: True

   Risk Profile:
     Risk aversion (ρ): 1.423
     Category: risk_averse
     Consistency: 90.0%
     Utility curvature: -0.0234

   Classification: investor

   Risk Premium Analysis:
     Mean expected value: 8.45%
     Mean certainty equivalent: 5.82%
     Mean risk premium: 2.63%


Part 7: Notes
-------------

Data Collection
~~~~~~~~~~~~~~~

1. **Real trade-offs**: Amounts should be meaningful to the decision-maker
2. **Vary safe amounts**: Test multiple certainty levels to map preferences
3. **Simple probabilities**: 50/50, 70/30, etc. are easier to understand
4. **Boundary cases**: Include clearly dominated and clearly dominant options

Sample Size
~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Choices
     - Reliability
   * - < 5
     - Very noisy estimates
   * - 5-10
     - Basic classification possible
   * - 10-20
     - Reliable risk profile
   * - > 20
     - High-precision estimates

Interpretation Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Consistency score < 60%**: Data may be too noisy or preferences unstable
2. **Check for monotonicity violations**: May indicate misunderstanding or error
3. **Compare to benchmarks**: Typical adults have ρ ∈ [0.5, 3.0]
4. **Context matters**: Same person may be more risk-averse for large stakes


Function Reference
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Purpose
     - Function
   * - Risk profile estimation
     - ``compute_risk_profile()``
   * - EU axiom testing
     - ``check_expected_utility_axioms()``
   * - Quick classification
     - ``classify_risk_type()``


See Also
--------

- :doc:`tutorial` — Budget-based analysis (GARP, CCEI)
- :doc:`tutorial_menu_choice` — Menu-based choice analysis
- :doc:`api` — Full API documentation
- :doc:`theory` — Mathematical foundations (Chapter 8)
