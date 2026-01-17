Tutorial 10: Intertemporal Choice
==================================

This tutorial covers analyzing time preferences through revealed preference
tests. These methods detect present bias, estimate discount factors, and
test whether choices are consistent with exponential or hyperbolic discounting.

Topics covered:

- The DatedChoice data structure
- Testing exponential discounting (time-consistent preferences)
- Detecting present bias
- Quasi-hyperbolic (beta-delta) model
- Recovering discount factors

Prerequisites
-------------

- Python 3.10+
- Completed Tutorial 1 (Budget-Based Analysis)
- Basic understanding of discounting and time preferences

.. note::

   **Key insight**: Time inconsistency (present bias) is a major driver of
   suboptimal decisions in savings, health, and subscription services. Detecting
   present bias helps design better defaults, commitment devices, and pricing
   strategies for time-delayed consumption.


Part 1: Theory - Time Preferences
---------------------------------

Exponential Discounting
~~~~~~~~~~~~~~~~~~~~~~~

The standard economic model assumes **exponential discounting**:

.. math::

   U = \sum_{t=0}^{T} \delta^t \cdot u(c_t)

where:

- :math:`\delta \in (0, 1)` is the discount factor
- :math:`u(c_t)` is utility from consumption at time t
- A higher :math:`\delta` means more patience

This implies **time-consistent** preferences: if you prefer $100 today over
$110 tomorrow, you also prefer $100 in 30 days over $110 in 31 days.

Quasi-Hyperbolic (Beta-Delta) Discounting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Many people exhibit **present bias** - they discount the immediate future
more heavily. The beta-delta model captures this:

.. math::

   U = u(c_0) + \beta \sum_{t=1}^{T} \delta^t \cdot u(c_t)

where:

- :math:`\beta < 1` captures present bias (over-valuing immediate rewards)
- :math:`\delta` is the long-run discount factor

Classic pattern:

- Prefer $100 today over $110 tomorrow (impatient)
- Prefer $110 in 31 days over $100 in 30 days (patient)

This reversal violates exponential discounting but is explained by beta-delta.


Part 2: The DatedChoice Data Structure
--------------------------------------

Intertemporal choices are represented using ``DatedChoice``:

.. code-block:: python

   import numpy as np
   from pyrevealed.algorithms.intertemporal import DatedChoice

   # A choice between $100 today or $110 in 30 days
   # User chose the $100 today (index 0)
   choice1 = DatedChoice(
       amounts=np.array([100.0, 110.0]),
       dates=np.array([0, 30]),
       chosen=0,  # Chose the immediate option
   )

   # A choice between $100 in 30 days or $110 in 60 days
   # User chose $110 in 60 days (index 1)
   choice2 = DatedChoice(
       amounts=np.array([100.0, 110.0]),
       dates=np.array([30, 60]),
       chosen=1,  # Chose to wait
   )

   print(f"Choice 1: Chose ${choice1.amounts[choice1.chosen]} at day {choice1.dates[choice1.chosen]}")
   print(f"Choice 2: Chose ${choice2.amounts[choice2.chosen]} at day {choice2.dates[choice2.chosen]}")

Output:

.. code-block:: text

   Choice 1: Chose $100.0 at day 0
   Choice 2: Chose $110.0 at day 60

Multi-Option Choices
~~~~~~~~~~~~~~~~~~~~

Choices can have more than two options:

.. code-block:: python

   # Multiple payout options
   complex_choice = DatedChoice(
       amounts=np.array([100.0, 105.0, 115.0, 130.0]),
       dates=np.array([0, 30, 60, 90]),
       chosen=2,  # Chose $115 in 60 days
   )

   print(f"Options available:")
   for i, (amt, date) in enumerate(zip(complex_choice.amounts, complex_choice.dates)):
       marker = " <-- CHOSEN" if i == complex_choice.chosen else ""
       print(f"  ${amt:.0f} in {date} days{marker}")

Output:

.. code-block:: text

   Options available:
     $100 in 0 days
     $105 in 30 days
     $115 in 60 days <-- CHOSEN
     $130 in 90 days


Part 3: Testing Exponential Discounting
---------------------------------------

Test whether choices are consistent with time-consistent preferences:

.. code-block:: python

   from pyrevealed.algorithms.intertemporal import test_exponential_discounting, DatedChoice
   import numpy as np

   # Create a set of consistent choices
   # Higher interest rates for longer waits (patient behavior)
   consistent_choices = [
       DatedChoice(amounts=np.array([100.0, 105.0]), dates=np.array([0, 30]), chosen=1),
       DatedChoice(amounts=np.array([100.0, 110.0]), dates=np.array([0, 60]), chosen=1),
       DatedChoice(amounts=np.array([100.0, 103.0]), dates=np.array([30, 60]), chosen=1),
   ]

   result = test_exponential_discounting(consistent_choices)

   print(f"Consistent with exponential discounting: {result.is_consistent}")
   print(f"Delta range: [{result.delta_lower:.3f}, {result.delta_upper:.3f}]")
   print(f"Number of violations: {len(result.violations)}")

Output:

.. code-block:: text

   Consistent with exponential discounting: True
   Delta range: [0.833, 0.997]
   Number of violations: 0

Interpreting Delta Bounds
~~~~~~~~~~~~~~~~~~~~~~~~~

The result provides bounds on the discount factor:

.. code-block:: python

   print(f"\nDiscount factor interpretation:")
   print(f"  Delta lower bound: {result.delta_lower:.3f}")
   print(f"  Delta upper bound: {result.delta_upper:.3f}")

   if result.has_tight_bounds:
       mid = (result.delta_lower + result.delta_upper) / 2
       annual_rate = (1/mid - 1) * 12  # Monthly to annual
       print(f"  Estimated monthly delta: ~{mid:.3f}")
       print(f"  Implied annual interest rate: ~{annual_rate*100:.1f}%")

Output:

.. code-block:: text

   Discount factor interpretation:
     Delta lower bound: 0.833
     Delta upper bound: 0.997
     Estimated monthly delta: ~0.915
     Implied annual interest rate: ~11.1%

Full Summary Report
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   print(result.summary())

.. code-block:: text

   ================================================================================
                        EXPONENTIAL DISCOUNTING TEST REPORT
   ================================================================================

   Status: CONSISTENT

   Metrics:
   -------
     Is Consistent ........................ Yes
     Violations ............................. 0
     Delta Lower Bound .................. 0.8333
     Delta Upper Bound .................. 0.9970
     Observations ........................... 3

   Interpretation:
   --------------
     Choices are time-consistent (exponential discounting).
     Estimated discount factor: ~0.915

   Computation Time: 0.12 ms
   ================================================================================


Part 4: Detecting Present Bias
------------------------------

Present bias is detected when people are impatient for immediate rewards
but patient for future tradeoffs:

.. code-block:: python

   from pyrevealed.algorithms.intertemporal import test_present_bias, DatedChoice
   import numpy as np

   # Classic present bias pattern
   present_biased_choices = [
       # Impatient NOW: prefer $100 today over $110 tomorrow
       DatedChoice(amounts=np.array([100.0, 110.0]), dates=np.array([0, 1]), chosen=0),

       # Patient LATER: prefer $110 in 31 days over $100 in 30 days
       DatedChoice(amounts=np.array([100.0, 110.0]), dates=np.array([30, 31]), chosen=1),

       # More examples
       DatedChoice(amounts=np.array([50.0, 55.0]), dates=np.array([0, 7]), chosen=0),
       DatedChoice(amounts=np.array([50.0, 55.0]), dates=np.array([60, 67]), chosen=1),
   ]

   result = test_present_bias(present_biased_choices, threshold=0.1)

   print(f"Present bias detected: {result['has_present_bias']}")
   print(f"Bias magnitude: {result['bias_magnitude']:.2f}")
   print(f"Patience when immediate: {result['immediate_patience']:.1%}")
   print(f"Patience when future: {result['future_patience']:.1%}")

Output:

.. code-block:: text

   Present bias detected: True
   Bias magnitude: 1.00
   Patience when immediate: 0.0%
   Patience when future: 100.0%

Interpreting Present Bias
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   print("\nPresent Bias Interpretation:")
   print(f"  Immediate choices: {result['num_immediate_choices']}")
   print(f"  Future choices: {result['num_future_choices']}")

   if result['has_present_bias']:
       print("\n  Pattern: User is IMPATIENT when rewards are immediate")
       print("           but PATIENT when both options are in the future.")
       print("  This indicates hyperbolic/quasi-hyperbolic preferences.")

Output:

.. code-block:: text

   Present Bias Interpretation:
     Immediate choices: 2
     Future choices: 2

     Pattern: User is IMPATIENT when rewards are immediate
              but PATIENT when both options are in the future.
     This indicates hyperbolic/quasi-hyperbolic preferences.


Part 5: Quasi-Hyperbolic (Beta-Delta) Model
-------------------------------------------

The beta-delta model captures present bias with two parameters:

.. code-block:: python

   from pyrevealed.algorithms.intertemporal import test_quasi_hyperbolic, DatedChoice
   import numpy as np

   # Choices exhibiting present bias
   choices = [
       # Immediate vs near future - impatient
       DatedChoice(amounts=np.array([100.0, 115.0]), dates=np.array([0, 30]), chosen=0),
       DatedChoice(amounts=np.array([100.0, 108.0]), dates=np.array([0, 14]), chosen=0),

       # Far future vs further future - patient
       DatedChoice(amounts=np.array([100.0, 105.0]), dates=np.array([60, 90]), chosen=1),
       DatedChoice(amounts=np.array([100.0, 103.0]), dates=np.array([90, 120]), chosen=1),
   ]

   result = test_quasi_hyperbolic(choices)

   print(f"Beta-delta consistent: {result.is_consistent}")
   print(f"Beta range: [{result.beta_lower:.3f}, {result.beta_upper:.3f}]")
   print(f"Delta range: [{result.delta_lower:.3f}, {result.delta_upper:.3f}]")
   print(f"Present bias detected: {result.has_present_bias}")

Output:

.. code-block:: text

   Beta-delta consistent: True
   Beta range: [0.010, 0.870]
   Delta range: [0.000, 1.000]
   Present bias detected: True

Interpreting Beta-Delta Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   print("\nParameter Interpretation:")

   # Beta interpretation
   if result.has_present_bias:
       beta_mid = (result.beta_lower + result.beta_upper) / 2
       print(f"  Beta (present bias): ~{beta_mid:.2f}")
       print(f"    - Beta = 1.0 means no present bias")
       print(f"    - Beta < 1.0 means immediate rewards are over-weighted")
       print(f"    - Lower beta = stronger present bias")

   # Delta interpretation
   delta_mid = (result.delta_lower + result.delta_upper) / 2
   print(f"\n  Delta (long-run patience): ~{delta_mid:.2f}")
   print(f"    - Higher delta = more patient in the long run")

Output:

.. code-block:: text

   Parameter Interpretation:
     Beta (present bias): ~0.44
       - Beta = 1.0 means no present bias
       - Beta < 1.0 means immediate rewards are over-weighted
       - Lower beta = stronger present bias

     Delta (long-run patience): ~0.50
       - Higher delta = more patient in the long run

Full Summary Report
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   print(result.summary())

.. code-block:: text

   ================================================================================
                    QUASI-HYPERBOLIC DISCOUNTING TEST REPORT
   ================================================================================

   Status: CONSISTENT

   Parameters:
   ----------
     Beta Range: [0.010, 0.870]
     Delta Range: [0.000, 1.000]
     Present Bias Detected: True

   Interpretation:
   --------------
     Choices exhibit present bias (beta < 1).
     Immediate rewards are over-weighted.

   Computation Time: 0.45 ms
   ================================================================================


Part 6: Recovering Discount Factors
-----------------------------------

Bound the discount factor implied by observed choices:

.. code-block:: python

   from pyrevealed.algorithms.intertemporal import recover_discount_factor, DatedChoice
   import numpy as np

   # Choices that reveal discount rate preferences
   choices = [
       DatedChoice(amounts=np.array([100.0, 102.0]), dates=np.array([0, 30]), chosen=1),
       DatedChoice(amounts=np.array([100.0, 105.0]), dates=np.array([0, 60]), chosen=1),
       DatedChoice(amounts=np.array([100.0, 101.0]), dates=np.array([0, 30]), chosen=0),
   ]

   bounds = recover_discount_factor(choices)

   print(f"Discount factor bounds: [{bounds.delta_lower:.4f}, {bounds.delta_upper:.4f}]")
   print(f"Is identified: {bounds.is_identified}")
   print(f"Midpoint estimate: {bounds.midpoint:.4f}")

Output:

.. code-block:: text

   Discount factor bounds: [0.9804, 0.9901]
   Is identified: True
   Midpoint estimate: 0.9852

Implied Interest Rates
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   print(f"\nImplied Interest Rates (per period):")
   print(f"  Lower bound: {bounds.implied_interest_rate_lower:.2%}")
   print(f"  Upper bound: {bounds.implied_interest_rate_upper:.2%}")

   # Convert to annual if periods are months
   if bounds.is_identified:
       annual_lower = (1 + bounds.implied_interest_rate_lower) ** 12 - 1
       annual_upper = (1 + bounds.implied_interest_rate_upper) ** 12 - 1
       print(f"\nAnnualized (assuming monthly periods):")
       print(f"  Lower bound: {annual_lower:.1%}")
       print(f"  Upper bound: {annual_upper:.1%}")

Output:

.. code-block:: text

   Implied Interest Rates (per period):
     Lower bound: 1.00%
     Upper bound: 2.00%

   Annualized (assuming monthly periods):
     Lower bound: 12.7%
     Upper bound: 26.8%


Part 7: Application - Subscription Service Design
-------------------------------------------------

Use intertemporal analysis to optimize subscription pricing:

.. code-block:: python

   import numpy as np
   from pyrevealed.algorithms.intertemporal import (
       DatedChoice,
       test_exponential_discounting,
       test_present_bias,
       recover_discount_factor,
   )

   np.random.seed(42)

   # Simulate user choices between payment plans
   # Monthly: $10/month, Annual: $100/year ($8.33/month)

   def simulate_subscription_choices(n_users, beta=0.7, delta=0.95):
       """Simulate choices with beta-delta preferences."""
       choices = []

       for _ in range(n_users):
           # Choice 1: Pay $10 now vs $100 in one lump sum (annual)
           # Effective: $10 today vs $8.33/month for 12 months

           # Monthly option: pay now
           amounts = np.array([10.0, 8.33])
           dates = np.array([0, 0])  # Both "now" but different commitment

           # For simplicity, model as: $10 now vs $100 now (with annual value)
           # User with present bias prefers smaller immediate payment

           # Choice: monthly ($10) vs annual ($100) upfront
           choice = DatedChoice(
               amounts=np.array([10.0, 100.0]),
               dates=np.array([0, 0]),
               chosen=0 if np.random.random() > 0.3 else 1,  # 70% choose monthly
           )
           choices.append(choice)

           # Choice 2: Renew in 11 months - same tradeoff
           choice2 = DatedChoice(
               amounts=np.array([10.0, 100.0]),
               dates=np.array([330, 330]),  # 11 months from now
               chosen=1 if np.random.random() > 0.4 else 0,  # 60% would choose annual
           )
           choices.append(choice2)

       return choices

   # Simulate choices
   choices = simulate_subscription_choices(50)

   # Analyze present bias
   bias_result = test_present_bias(choices, threshold=0.1)

   print("=== Subscription Pricing Analysis ===")
   print(f"\nSample size: {len(choices)} choices from 50 users")

   print(f"\nPresent Bias Analysis:")
   print(f"  Present bias detected: {bias_result['has_present_bias']}")
   print(f"  Bias magnitude: {bias_result['bias_magnitude']:.2f}")

   # Recommendations
   print(f"\nPricing Recommendations:")
   if bias_result['has_present_bias']:
       print("  - Users show present bias: they prefer monthly despite higher total cost")
       print("  - Consider: offer commitment devices (annual with early exit penalty)")
       print("  - Consider: frame annual savings as 'X% off' rather than total savings")
       print("  - Consider: auto-renewal to reduce decision points")
   else:
       print("  - Users show time-consistent preferences")
       print("  - Standard discount for annual plans should work")

Example output:

.. code-block:: text

   === Subscription Pricing Analysis ===

   Sample size: 100 choices from 50 users

   Present Bias Analysis:
     Present bias detected: True
     Bias magnitude: 0.30

   Pricing Recommendations:
     - Users show present bias: they prefer monthly despite higher total cost
     - Consider: offer commitment devices (annual with early exit penalty)
     - Consider: frame annual savings as 'X% off' rather than total savings
     - Consider: auto-renewal to reduce decision points


Part 8: Notes
-------------

When to Use Intertemporal Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Good Applications
     - Less Suitable For
   * - Subscription vs one-time pricing
     - Impulse purchases
   * - Savings program design
     - Single-period decisions
   * - Loan/credit product analysis
     - Non-monetary choices
   * - Health behavior interventions
     - Choices without timing component
   * - Retirement planning tools
     - Real-time recommendations

Typical Parameter Values
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Empirical Benchmarks
   :header-rows: 1
   :widths: 30 70

   * - Parameter
     - Typical Range
   * - Delta (monthly)
     - 0.95-0.99 (patient) to 0.80-0.90 (impatient)
   * - Beta
     - 0.7-0.9 (mild bias) to 0.4-0.6 (strong bias)
   * - Annual interest
     - 5-15% (patient) to 50-200% (very impatient)


Function Reference
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Purpose
     - Function
   * - Test exponential discounting
     - ``test_exponential_discounting()``
   * - Test quasi-hyperbolic model
     - ``test_quasi_hyperbolic()``
   * - Detect present bias
     - ``test_present_bias()``
   * - Recover discount factor bounds
     - ``recover_discount_factor()``


See Also
--------

- :doc:`tutorial` - Budget-based analysis fundamentals
- :doc:`tutorial_risk` - Risk preferences under uncertainty
- :doc:`api` - Full API documentation
