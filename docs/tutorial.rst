Tutorial
========

What can 2 years of grocery data tell us about human decision-making? In this tutorial, you'll analyze real shopping behavior from 2,222 households and discover that most people aren't perfectly "rational"—but that's not necessarily a bad thing.

By the end of this tutorial, you'll be able to:

- Load and prepare behavioral data for analysis
- Test whether choices are internally consistent
- Measure how exploitable a user's inconsistencies are
- Discover hidden preference structures in the data
- Identify substitute and complement relationships
- Transform product-level behavior into characteristic-level insights

Prerequisites
-------------

- Python 3.8+
- Basic familiarity with NumPy and pandas
- Understanding of basic statistics (means, correlations)

.. note::

   The full code for this tutorial is available in the ``dunnhumby/`` directory of the PyRevealed repository.


Part 1: Setup and Dataset Overview
----------------------------------

First, let's install PyRevealed and download the dataset:

.. code-block:: bash

   pip install pyrevealed[viz]

   # Download the Dunnhumby dataset (requires Kaggle API)
   cd dunnhumby && ./download_data.sh

The **Dunnhumby "The Complete Journey"** dataset contains 2 years of grocery transactions from approximately 2,500 households. We'll focus on 10 product categories: Soda, Milk, Bread, Cheese, Chips, Soup, Yogurt, Beef, Pizza, and Lunchmeat.

.. list-table:: Dataset Overview
   :header-rows: 1
   :widths: 40 60

   * - Metric
     - Value
   * - Households analyzed
     - 2,222
   * - Product categories
     - 10
   * - Time period
     - 104 weeks (2 years)
   * - Total transactions
     - 645,288


Part 2: Loading and Preparing the Data
--------------------------------------

In this section, you'll learn how to transform raw transaction data into a ``BehaviorLog`` that PyRevealed can analyze.

**What is a BehaviorLog?**

A ``BehaviorLog`` captures a series of choices made by a user. Each observation consists of:

- **Prices**: What each option cost at the time of the choice
- **Quantities**: How much of each option the user chose

.. code-block:: python
   :caption: Building a BehaviorLog from transaction data

   import numpy as np
   from pyrevealed import BehaviorLog

   # For a single household, we aggregate transactions into weekly observations
   # Each row is a week; each column is a product category

   # Example: 52 weeks, 10 products
   prices = np.array([
       [2.50, 3.20, 2.10, 4.50, 3.00, 1.80, 2.90, 8.50, 5.00, 6.20],  # Week 1
       [2.45, 3.30, 2.15, 4.40, 2.90, 1.85, 3.00, 8.20, 5.10, 6.00],  # Week 2
       # ... more weeks
   ])

   quantities = np.array([
       [2.0, 1.5, 3.0, 0.5, 1.0, 2.0, 1.0, 0.5, 0.0, 0.5],  # Week 1
       [1.5, 2.0, 2.5, 0.5, 1.5, 1.5, 1.0, 0.5, 1.0, 0.0],  # Week 2
       # ... more weeks
   ])

   # Create the behavior log
   log = BehaviorLog(
       cost_vectors=prices,
       action_vectors=quantities,
       user_id="household_123"
   )

   print(f"Observations: {log.num_records}")
   print(f"Products: {log.num_goods}")

The Dunnhumby analysis scripts handle this data preparation automatically. Let's load pre-built sessions:

.. code-block:: python
   :caption: Loading pre-built sessions from the Dunnhumby analysis

   import pickle
   from pathlib import Path

   # Load cached sessions (built by run_all.py)
   cache_file = Path("dunnhumby/cache/sessions.pkl")
   with open(cache_file, 'rb') as f:
       sessions = pickle.load(f)

   print(f"Loaded {len(sessions)} household sessions")

   # Each session contains:
   # - behavior_log: The BehaviorLog object
   # - household_id: Unique identifier
   # - metadata: Additional household info


Part 3: Testing Behavioral Consistency
--------------------------------------

In this section, you'll learn what it means for behavior to be "consistent" and how to test for it.

**What is GARP?**

The Generalized Axiom of Revealed Preference (GARP) is a test for behavioral consistency. The idea is simple: if you chose bundle A when bundle B was affordable, you're revealing that you prefer A to B. GARP checks whether these revealed preferences form a consistent ordering.

A violation occurs when your choices contradict each other—for example, if you reveal that you prefer A to B, and also that you prefer B to A.

.. code-block:: python
   :caption: Testing consistency for a single household

   from pyrevealed import validate_consistency

   # Pick a household
   household_id = list(sessions.keys())[0]
   log = sessions[household_id].behavior_log

   # Test consistency
   result = validate_consistency(log)

   if result.is_consistent:
       print("This household's choices are perfectly consistent!")
       print("Their behavior can be explained by a single utility function.")
   else:
       print(f"Found {result.num_violations} preference contradictions.")
       print("This doesn't mean they're irrational—just that their")
       print("choices don't fit a simple utility-maximization model.")

**Running the test on all households**

Let's see how many households pass the consistency test:

.. code-block:: python
   :caption: Testing all 2,222 households

   consistent_count = 0

   for household_id, session_data in sessions.items():
       result = validate_consistency(session_data.behavior_log)
       if result.is_consistent:
           consistent_count += 1

   total = len(sessions)
   pct = 100 * consistent_count / total

   print(f"Consistent households: {consistent_count} / {total} ({pct:.1f}%)")

**Result**: Only **100 households (4.5%)** are perfectly consistent.

.. image:: images/showcase_a_rationality_histogram.png
   :alt: Distribution of rationality scores across households
   :width: 600px

.. note::

   **Does this mean 95% of people are irrational?**

   Not necessarily! There are many reasons for apparent inconsistencies:

   - **Measurement noise**: Prices and quantities aren't perfectly observed
   - **Changing preferences**: Tastes evolve over 2 years
   - **Context effects**: A birthday party changes what you buy
   - **Multiple decision-makers**: Different family members shop differently

   The consistency test tells us whether a *single, stable utility function* can explain the data—not whether people are "rational."


Part 4: Measuring Integrity and Exploitability
----------------------------------------------

In this section, you'll learn about two key metrics: the **integrity score** (how consistent is the behavior?) and the **confusion metric** (how exploitable are the inconsistencies?).

**The Integrity Score (Afriat Efficiency Index)**

The integrity score measures what fraction of behavior is consistent with utility maximization. A score of 1.0 means perfectly consistent; lower scores indicate more inconsistency.

.. code-block:: python
   :caption: Computing integrity scores

   from pyrevealed import compute_integrity_score

   integrity_scores = []

   for household_id, session_data in sessions.items():
       result = compute_integrity_score(session_data.behavior_log)
       integrity_scores.append(result.efficiency_index)

   import numpy as np
   print(f"Mean integrity: {np.mean(integrity_scores):.3f}")
   print(f"Median integrity: {np.median(integrity_scores):.3f}")
   print(f"Low integrity (<0.7): {sum(s < 0.7 for s in integrity_scores)} households")

**Result**: Mean integrity is **0.839**—most behavior is about 84% consistent.

**The Confusion Metric (Money Pump Index)**

The confusion metric measures how exploitable a user's inconsistencies are. If someone prefers A to B and B to C but C to A, a clever seller could "pump" money from them by cycling through these preferences.

.. code-block:: python
   :caption: Computing confusion metrics

   from pyrevealed import compute_confusion_metric

   confusion_scores = []

   for household_id, session_data in sessions.items():
       result = compute_confusion_metric(session_data.behavior_log)
       confusion_scores.append(result.mpi_value)

   print(f"Mean confusion: {np.mean(confusion_scores):.3f}")

**Result**: Mean confusion is **0.225**. There's a strong negative correlation with integrity (r=-0.89)—more consistent users are less exploitable.

.. image:: images/analysis_mpi_distribution.png
   :alt: Distribution of confusion scores
   :width: 600px

.. tip::

   **Practical applications**:

   - **Bot detection**: Bots often have very low integrity scores (random behavior)
   - **UX confusion**: High confusion scores may indicate UI problems
   - **A/B testing**: Compare confusion scores between interface variants


Part 5: Discovering Preference Structures
-----------------------------------------

In this section, you'll learn how to discover hidden structures in preferences—like whether people treat different product groups as separate "mental budgets."

**Mental Accounting: Do People Compartmentalize?**

Some theories suggest people maintain separate mental budgets for different categories. Let's test whether grocery shoppers treat product groups independently:

.. code-block:: python
   :caption: Testing feature independence between product groups

   from pyrevealed import test_feature_independence

   # Define product groups
   DAIRY = [1, 3, 6]      # Milk, Cheese, Yogurt (indices)
   PROTEIN = [7, 9]       # Beef, Lunchmeat
   STAPLES = [2, 5]       # Bread, Soup
   SNACKS = [0, 4, 8]     # Soda, Chips, Pizza

   # Test if Dairy choices are independent of Protein choices
   independence_scores = []

   for household_id, session_data in sessions.items():
       result = test_feature_independence(
           session_data.behavior_log,
           group_a=DAIRY,
           group_b=PROTEIN
       )
       independence_scores.append(result.is_separable)

   pct_independent = 100 * sum(independence_scores) / len(independence_scores)
   print(f"Dairy vs Protein independent: {pct_independent:.1f}%")

**Result**: Only **Protein vs Staples** shows strong evidence of separate budgeting (62%). Most category pairs show pooled budgets (<35%).

.. image:: images/showcase_e_mental_accounting.png
   :alt: Mental accounting heatmap
   :width: 600px

**Auto-Discovering Product Groups**

Instead of defining groups manually, let PyRevealed discover them from the data:

.. code-block:: python
   :caption: Auto-discovering independent groups

   from pyrevealed import discover_independent_groups

   # Run on a sample household
   log = sessions[list(sessions.keys())[0]].behavior_log
   groups = discover_independent_groups(log)

   print("Discovered groups:")
   for i, group in enumerate(groups):
       print(f"  Group {i+1}: {group}")

**Result**: The algorithm confirms 3 of our 4 manual groupings (Dairy, Snacks, Staples). Protein products don't cluster as strongly as expected.


Part 6: Cross-Price Analysis
----------------------------

In this section, you'll learn how to identify substitute and complement relationships between products.

**Substitutes vs Complements**

- **Substitutes**: When the price of A goes up, demand for B goes up (people switch)
- **Complements**: When the price of A goes up, demand for B goes down (bought together)

.. code-block:: python
   :caption: Computing the cross-price matrix

   from pyrevealed import compute_cross_price_matrix

   # Aggregate across households
   PRODUCT_NAMES = ['Soda', 'Milk', 'Bread', 'Cheese', 'Chips',
                    'Soup', 'Yogurt', 'Beef', 'Pizza', 'Lunch']

   # Compute for a sample of households
   sample_keys = list(sessions.keys())[:200]

   for key in sample_keys:
       result = compute_cross_price_matrix(sessions[key].behavior_log)
       # Aggregate the relationship_matrix...

   # Top complement pairs:
   # Milk & Bread: -0.31
   # Soda & Pizza: -0.29
   # Milk & Cheese: -0.29

**Result**: Most product pairs show **complementary** relationships—they're bought together. This makes sense for grocery shopping: you buy milk when you buy bread, soda when you buy pizza.

.. image:: images/showcase_o_cross_price.png
   :alt: Cross-price relationship matrix
   :width: 600px


Part 7: Advanced Analysis - The Lancaster Model
-----------------------------------------------

In this section, you'll learn how to transform product-level behavior into characteristic-level insights, revealing hidden rationality.

**The Problem with Product-Level Analysis**

When we analyze at the product level, only 4.5% of households appear consistent. But people don't think about "products"—they think about what products *deliver*.

A household isn't really choosing between "Beef" and "Yogurt"—they're choosing between protein sources, balancing macronutrients, or optimizing calories per dollar.

**Transforming to Characteristics Space**

The Lancaster model transforms choices from product-space to characteristics-space using product attributes:

.. code-block:: python
   :caption: Applying the Lancaster transformation

   from pyrevealed import transform_to_characteristics, LancasterLog

   # Define nutritional characteristics per unit of each product
   # Columns: Protein (g), Carbs (g), Fat (g), Sodium (mg)
   characteristics = np.array([
       [0, 39, 0, 15],      # Soda
       [8, 12, 8, 120],     # Milk
       [9, 49, 3, 490],     # Bread
       [25, 1, 33, 620],    # Cheese
       [7, 52, 35, 470],    # Chips
       [4, 20, 2, 890],     # Soup
       [6, 7, 0, 80],       # Yogurt
       [26, 0, 15, 75],     # Beef
       [12, 36, 10, 640],   # Pizza
       [10, 4, 8, 1050],    # Lunchmeat
   ])

   # Transform a household's behavior log
   log = sessions[list(sessions.keys())[0]].behavior_log
   lancaster_log = transform_to_characteristics(log, characteristics)

   # Now test consistency in characteristics-space
   result = validate_consistency(lancaster_log)

**The "Rationality Rescue" Effect**

When we retest consistency in characteristics-space:

.. list-table:: Product Space vs Characteristics Space
   :header-rows: 1
   :widths: 30 35 35

   * - Metric
     - Product Space
     - Characteristics Space
   * - Mean integrity
     - 0.839
     - 0.890 (+5.1%)
   * - Consistent households
     - 100 (4.5%)
     - 166 (7.5%) (+66%)

**120 households (5.4%)** are "rescued"—they looked irrational in product-space but are perfectly consistent about *nutrients*.

.. image:: images/showcase_p_lancaster.png
   :alt: Lancaster analysis visualization
   :width: 600px

.. image:: images/showcase_p_rationality_rescue.png
   :alt: Rationality rescue visualization
   :width: 600px

**Shadow Prices: Implicit Valuations**

The Lancaster model also reveals implicit valuations—how much households are willing to pay per unit of each characteristic:

.. list-table:: Implied Shadow Prices
   :header-rows: 1
   :widths: 30 30 40

   * - Nutrient
     - $/unit
     - Spend Share
   * - Protein
     - $0.105/g
     - 34.0%
   * - Fat
     - $0.129/g
     - 32.3%
   * - Carbs
     - $0.032/g
     - 28.7%
   * - Sodium
     - $0.0004/mg
     - 5.1%

**Key insight**: Fat and Protein command the highest implicit prices. Households optimize for calorie-dense, satiating macronutrients.

.. tip::

   **Practical implication**: Those 120 "rescued" households are prime candidates for store-brand substitution—they care about nutrition, not labels. The households whose consistency *decreased* are the opposite: brand loyalists who'll pay premiums regardless of nutritional equivalence.


Part 8: Putting It All Together
-------------------------------

Let's summarize what we've learned from analyzing 2,222 households over 2 years:

.. list-table:: Key Findings
   :header-rows: 1
   :widths: 35 65

   * - Category
     - Finding
   * - **Consistency**
     - 4.5% perfectly consistent, mean integrity = 0.839
   * - **Exploitability**
     - Mean confusion = 0.225 (inversely correlated with integrity)
   * - **Mental Accounting**
     - Only Protein vs Staples shows separate budgets (62%)
   * - **Cross-Price**
     - Mostly complements (Milk+Bread, Soda+Pizza)
   * - **Lancaster Model**
     - 5.4% "rescued" in characteristics-space
   * - **Shadow Prices**
     - Fat and Protein most valued ($0.11-0.13/g)

**What Have We Learned?**

1. **Most people aren't "perfectly rational"**—but that doesn't mean they're irrational. Context, noise, and changing preferences all contribute to apparent inconsistencies.

2. **Consistency and exploitability are related**—more consistent users are harder to manipulate. This has implications for UX design and fraud detection.

3. **People don't compartmentalize as much as we thought**—mental accounting is weaker than expected for grocery categories.

4. **The level of analysis matters**—switching from products to characteristics reveals hidden rationality. People optimize for *what products deliver*, not the products themselves.

**Next Steps**

- Explore the :doc:`api` for detailed function documentation
- Try the :doc:`quickstart` with your own data
- Check out the ``examples/`` directory for more advanced use cases:

  - ``01_behavioral_auditor.py``: Bot and fraud detection
  - ``02_preference_encoder.py``: ML feature extraction
  - ``03_risk_analysis.py``: Risk profiling
  - ``06_characteristics_model.py``: Lancaster model deep dive

**Try It Yourself**

Here's a challenge: Run the analysis on your own behavioral data. You'll need:

1. A series of choices (what the user selected)
2. The prices/costs of each option at each choice
3. Convert to ``BehaviorLog`` format

.. code-block:: python

   from pyrevealed import (
       BehaviorLog,
       BehavioralAuditor,
   )

   # Your data here
   log = BehaviorLog(
       cost_vectors=your_prices,      # Shape: (n_observations, n_options)
       action_vectors=your_quantities  # Shape: (n_observations, n_options)
   )

   # Run the full audit
   auditor = BehavioralAuditor()
   report = auditor.full_audit(log)

   print(f"Consistent: {report.is_consistent}")
   print(f"Integrity: {report.integrity_score:.3f}")
   print(f"Confusion: {report.confusion_score:.3f}")
   print(f"Bot risk: {report.bot_risk:.2f}")
