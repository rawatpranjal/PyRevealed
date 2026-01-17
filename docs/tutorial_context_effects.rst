Tutorial 9: Context Effects
===========================

This tutorial covers detecting context effects in stochastic choice data. Context
effects are violations of rational choice theory where adding or removing irrelevant
alternatives changes choice probabilities - critical for e-commerce, A/B testing,
and menu design.

Topics covered:

- Decoy (attraction) effects
- Compromise effects
- Combined context effect analysis
- Practical applications

Prerequisites
-------------

- Python 3.10+
- Completed Tutorial 5 (Stochastic Choice)
- Understanding of StochasticChoiceLog

.. note::

   **Key insight**: Context effects reveal how choice architecture influences
   decisions. A "decoy" option can boost sales of a target product, while
   "extremeness aversion" drives consumers toward middle options. Understanding
   these effects is essential for product positioning and pricing strategy.


Part 1: Theory - What Are Context Effects?
------------------------------------------

Context effects violate the **Independence of Irrelevant Alternatives (IIA)**
axiom. IIA states that adding or removing options that aren't chosen shouldn't
change the relative preference between other options.

Two main types of context effects:

**Decoy Effect (Attraction Effect)**

Adding a dominated option D ("decoy") increases choice probability of the
dominating option T ("target"):

.. math::

   P(T \mid \{T, C\}) < P(T \mid \{T, C, D\})

where D is dominated by T (worse on all attributes).

**Compromise Effect**

Adding extreme options increases choice probability of middle options:

.. math::

   P(M \mid \{A, M\}) < P(M \mid \{A, M, B\})

where M is between A and B on relevant attributes.


Part 2: Creating Stochastic Choice Data
---------------------------------------

Context effects require stochastic choice data - choice frequencies across
different menu configurations.

.. code-block:: python

   from pyrevealed import StochasticChoiceLog

   # Subscription pricing experiment
   # Items: 0=Basic ($5), 1=Standard ($10), 2=Premium ($15), 3=Decoy ($12)
   log = StochasticChoiceLog(
       menus=[
           frozenset({0, 1, 2}),      # Without decoy
           frozenset({0, 1, 2, 3}),   # With decoy (dominated by Premium)
           frozenset({0, 1}),         # Just Basic vs Standard
           frozenset({1, 2}),         # Just Standard vs Premium
       ],
       choice_frequencies=[
           {0: 30, 1: 45, 2: 25},           # Without decoy
           {0: 22, 1: 38, 2: 35, 3: 5},     # With decoy - Premium boosted!
           {0: 40, 1: 60},                  # Basic vs Standard
           {1: 55, 2: 45},                  # Standard vs Premium
       ],
       total_observations_per_menu=[100, 100, 100, 100],
       item_labels=["Basic", "Standard", "Premium", "Decoy"],
   )

   print(f"Number of menus: {log.num_menus}")
   print(f"All items: {log.all_items}")

Output:

.. code-block:: text

   Number of menus: 4
   All items: {0, 1, 2, 3}


Part 3: Detecting Decoy Effects
-------------------------------

The decoy effect occurs when adding a dominated alternative boosts the
choice probability of the dominating option.

.. code-block:: python

   from pyrevealed import detect_decoy_effect

   result = detect_decoy_effect(log, threshold=0.05)

   print(f"Decoy effect detected: {result.has_decoy_effect}")
   print(f"Number of decoy triples: {len(result.decoy_triples)}")
   print(f"Average magnitude: {result.magnitude:.1%}")
   print(f"Menus tested: {result.num_menus_tested}")

Output:

.. code-block:: text

   Decoy effect detected: True
   Number of decoy triples: 1
   Average magnitude: 10.0%
   Menus tested: 4

Interpreting Decoy Triples
~~~~~~~~~~~~~~~~~~~~~~~~~~

Each decoy triple is ``(target, competitor, decoy)``:

.. code-block:: python

   print("\nDecoy relationships found:")
   for target, competitor, decoy in result.decoy_triples:
       t_label = log.item_labels[target] if log.item_labels else f"Item {target}"
       c_label = log.item_labels[competitor] if log.item_labels else f"Item {competitor}"
       d_label = log.item_labels[decoy] if log.item_labels else f"Item {decoy}"
       print(f"  {d_label} acts as decoy, boosting {t_label} vs {c_label}")

Output:

.. code-block:: text

   Decoy relationships found:
     Decoy acts as decoy, boosting Premium vs Standard

Vulnerability Analysis
~~~~~~~~~~~~~~~~~~~~~~

The ``vulnerabilities`` dict shows which items lose share to decoy effects:

.. code-block:: python

   print("\nVulnerability to decoy manipulation:")
   for item, boost in sorted(result.vulnerabilities.items(), key=lambda x: -x[1]):
       label = log.item_labels[item] if log.item_labels else f"Item {item}"
       print(f"  {label}: loses up to {boost:.1%} share")

Output:

.. code-block:: text

   Vulnerability to decoy manipulation:
     Standard: loses up to 10.0% share


Part 4: Detecting Compromise Effects
------------------------------------

The compromise effect occurs when consumers prefer "middle" options,
avoiding extremes.

.. code-block:: python

   import numpy as np
   from pyrevealed import detect_compromise_effect

   # Create data with compromise effect
   # Items: 0=Economy, 1=Standard, 2=Luxury
   # Attributes: [Price (lower=better), Quality (higher=better)]
   attributes = np.array([
       [1, 3],  # Economy: cheap but low quality
       [2, 5],  # Standard: middle on both
       [3, 7],  # Luxury: expensive but high quality
   ])

   compromise_log = StochasticChoiceLog(
       menus=[
           frozenset({0, 1}),      # Economy vs Standard
           frozenset({1, 2}),      # Standard vs Luxury
           frozenset({0, 1, 2}),   # All three
       ],
       choice_frequencies=[
           {0: 45, 1: 55},          # Standard slightly preferred
           {1: 48, 2: 52},          # Luxury slightly preferred
           {0: 25, 1: 50, 2: 25},   # Standard gets boost in full menu!
       ],
       total_observations_per_menu=[100, 100, 100],
       item_labels=["Economy", "Standard", "Luxury"],
   )

   result = detect_compromise_effect(
       compromise_log,
       attribute_vectors=attributes,
       threshold=0.05
   )

   print(f"Compromise effect detected: {result.has_compromise_effect}")
   print(f"Compromise items: {result.compromise_items}")
   print(f"Magnitude: {result.magnitude:.1%}")

Output:

.. code-block:: text

   Compromise effect detected: True
   Compromise items: [1]
   Magnitude: 8.5%

When to Provide Attribute Vectors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **With attributes**: More precise identification of "middle" items
- **Without attributes**: Infers from choice patterns (less accurate)

.. code-block:: python

   # Without explicit attributes
   result_no_attrs = detect_compromise_effect(compromise_log, threshold=0.05)

   print(f"Without attributes - detected: {result_no_attrs.has_compromise_effect}")

Output:

.. code-block:: text

   Without attributes - detected: True

Extreme Pairs Analysis
~~~~~~~~~~~~~~~~~~~~~~

The result includes which extreme pairs drive the compromise effect:

.. code-block:: python

   print("\nExtreme pairs (A, B, Middle):")
   for a, b, middle in result.extreme_pairs:
       labels = compromise_log.item_labels
       print(f"  {labels[a]} and {labels[b]} are extremes, {labels[middle]} is middle")

Output:

.. code-block:: text

   Extreme pairs (A, B, Middle):
     Economy and Luxury are extremes, Standard is middle


Part 5: Combined Analysis
-------------------------

Test for all context effects at once:

.. code-block:: python

   from pyrevealed import test_context_effects

   result = test_context_effects(log, threshold=0.05)

   print(f"Any context effects: {result['has_context_effects']}")
   print(f"Strongest effect: {result['strongest_effect']}")
   print(f"Overall magnitude: {result['overall_magnitude']:.1%}")

Output:

.. code-block:: text

   Any context effects: True
   Strongest effect: decoy
   Overall magnitude: 10.0%

Accessing Individual Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Access detailed results
   decoy = result["decoy_result"]
   compromise = result["compromise_result"]

   print(f"\nDecoy effect:")
   print(f"  Found: {decoy.has_decoy_effect}")
   print(f"  Triples: {len(decoy.decoy_triples)}")

   print(f"\nCompromise effect:")
   print(f"  Found: {compromise.has_compromise_effect}")
   print(f"  Items: {compromise.compromise_items}")


Part 6: Practical Example - Product Pricing Strategy
----------------------------------------------------

Use context effects to optimize pricing tiers:

.. code-block:: python

   import numpy as np
   from pyrevealed import StochasticChoiceLog, detect_decoy_effect, detect_compromise_effect

   np.random.seed(42)

   # Simulate A/B test: pricing page with/without decoy tier
   # Control: Basic ($9), Pro ($29)
   # Treatment: Basic ($9), Plus ($25, decoy), Pro ($29)

   # Control group choices (1000 users)
   control_basic = 420
   control_pro = 580

   # Treatment group choices (1000 users)
   treatment_basic = 350
   treatment_plus = 80   # Few choose the decoy
   treatment_pro = 570   # Pro choice rate stays similar

   log = StochasticChoiceLog(
       menus=[
           frozenset({0, 1}),      # Control: Basic, Pro
           frozenset({0, 1, 2}),   # Treatment: Basic, Plus (decoy), Pro
       ],
       choice_frequencies=[
           {0: control_basic, 1: control_pro},
           {0: treatment_basic, 1: treatment_pro, 2: treatment_plus},
       ],
       total_observations_per_menu=[1000, 1000],
       item_labels=["Basic $9", "Pro $29", "Plus $25"],
   )

   # Analyze context effects
   decoy_result = detect_decoy_effect(log, threshold=0.01)

   print("=== Pricing Strategy Analysis ===")
   print(f"\nControl (2 tiers):")
   print(f"  Basic: {control_basic/10:.1f}%")
   print(f"  Pro: {control_pro/10:.1f}%")

   print(f"\nTreatment (3 tiers with decoy):")
   print(f"  Basic: {treatment_basic/10:.1f}%")
   print(f"  Plus (decoy): {treatment_plus/10:.1f}%")
   print(f"  Pro: {treatment_pro/10:.1f}%")

   # Revenue comparison
   control_revenue = control_basic * 9 + control_pro * 29
   treatment_revenue = treatment_basic * 9 + treatment_plus * 25 + treatment_pro * 29

   print(f"\nRevenue per 1000 users:")
   print(f"  Control: ${control_revenue:,.0f}")
   print(f"  Treatment: ${treatment_revenue:,.0f}")
   print(f"  Lift: {(treatment_revenue/control_revenue - 1)*100:.1f}%")

   # Context effect analysis
   print(f"\nContext Effect Analysis:")
   print(f"  Decoy effect detected: {decoy_result.has_decoy_effect}")
   if decoy_result.has_decoy_effect:
       print(f"  Magnitude: {decoy_result.magnitude:.1%}")

Example output:

.. code-block:: text

   === Pricing Strategy Analysis ===

   Control (2 tiers):
     Basic: 42.0%
     Pro: 58.0%

   Treatment (3 tiers with decoy):
     Basic: 35.0%
     Plus (decoy): 8.0%
     Pro: 57.0%

   Revenue per 1000 users:
     Control: $20,600
     Treatment: $21,680
     Lift: 5.2%

   Context Effect Analysis:
     Decoy effect detected: True
     Magnitude: 7.0%

The decoy tier (Plus at $25) doesn't cannibalize Pro much but pulls users
away from Basic, increasing overall revenue.


Part 7: Notes
-------------

When to Test for Context Effects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Test for Context Effects When
     - Standard Models Suffice When
   * - Pricing tier optimization
     - Single product pricing
   * - Menu/recommendation design
     - Fixed assortment
   * - A/B testing choice architecture
     - Testing product features
   * - Understanding competitive dynamics
     - Monopoly pricing
   * - Users seem "irrational"
     - Clear preference patterns

Effect Magnitudes
~~~~~~~~~~~~~~~~~

.. list-table:: Typical Effect Sizes
   :header-rows: 1
   :widths: 30 70

   * - Magnitude
     - Interpretation
   * - < 5%
     - Weak effect, may be noise
   * - 5-10%
     - Moderate effect, practically significant
   * - 10-20%
     - Strong effect, major design consideration
   * - > 20%
     - Very strong, unusual in practice


Function Reference
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Purpose
     - Function
   * - Detect decoy/attraction effects
     - ``detect_decoy_effect()``
   * - Detect compromise effects
     - ``detect_compromise_effect()``
   * - Test all context effects
     - ``test_context_effects()``
   * - Test regularity axiom
     - ``test_regularity()``


See Also
--------

- :doc:`tutorial_advanced` - Stochastic choice fundamentals (RUM, IIA)
- :doc:`tutorial_attention` - Limited attention models
- :doc:`api` - Full API documentation
