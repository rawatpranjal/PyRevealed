Tutorial 11: Ranking & Pairwise Analysis
=========================================

This tutorial covers analyzing pairwise comparisons and ranking data. These
methods are essential for RLHF preference learning, recommendation systems,
surveys, and voting analysis.

Topics covered:

- Bradley-Terry model for pairwise comparisons
- Ranking comparison metrics (Kendall tau, Spearman footrule, RBO)
- Aggregating multiple rankings
- Application to RLHF preference data

Prerequisites
-------------

- Python 3.10+
- Completed Tutorial 2 (Menu-Based Choice)
- Basic understanding of preference orderings

.. note::

   **Key insight**: Pairwise comparisons are the foundation of many modern
   AI preference learning systems (RLHF). The Bradley-Terry model provides a
   principled way to aggregate comparisons into rankings, while ranking
   metrics help evaluate agreement between human raters or model outputs.


Part 1: Theory - Pairwise Preferences
-------------------------------------

Bradley-Terry Model
~~~~~~~~~~~~~~~~~~~

The Bradley-Terry model represents choice probabilities as:

.. math::

   P(i \text{ beats } j) = \frac{\pi_i}{\pi_i + \pi_j}

where :math:`\pi_i > 0` is the "strength" parameter for item i.

Equivalently, using log-scores :math:`s_i = \log(\pi_i)`:

.. math::

   P(i \text{ beats } j) = \frac{1}{1 + e^{s_j - s_i}} = \sigma(s_i - s_j)

This is the logistic function on the score difference - exactly the form
used in RLHF reward modeling.

Ranking Metrics
~~~~~~~~~~~~~~~

For comparing two rankings:

- **Kendall tau**: Correlation based on concordant/discordant pairs
- **Spearman footrule**: Sum of absolute rank differences
- **Rank-Biased Overlap (RBO)**: Top-weighted similarity


Part 2: Bradley-Terry Model
---------------------------

Fitting the Model
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pyrevealed import fit_bradley_terry

   # Pairwise comparison data: (winner, loser, count)
   comparisons = [
       (0, 1, 15),  # Item 0 beat Item 1 fifteen times
       (1, 0, 5),   # Item 1 beat Item 0 five times
       (0, 2, 18),  # Item 0 beat Item 2 eighteen times
       (2, 0, 2),   # etc.
       (1, 2, 12),
       (2, 1, 8),
   ]

   result = fit_bradley_terry(comparisons, method="mle")

   print(f"Converged: {result.converged}")
   print(f"Log-likelihood: {result.log_likelihood:.2f}")
   print(f"Number of items: {result.num_items}")
   print(f"Total comparisons: {result.num_comparisons}")

Output:

.. code-block:: text

   Converged: True
   Log-likelihood: -34.21
   Number of items: 3
   Total comparisons: 60

Interpreting Scores and Rankings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   print("\nItem Scores (normalized, higher = better):")
   for item, score in sorted(result.scores.items(), key=lambda x: -x[1]):
       print(f"  Item {item}: {score:.3f}")

   print(f"\nRanking (best first): {result.ranking}")

Output:

.. code-block:: text

   Item Scores (normalized, higher = better):
     Item 0: 1.523
     Item 1: 0.412
     Item 2: 0.000

   Ranking (best first): [0, 1, 2]

Predicting Pairwise Probabilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pyrevealed import predict_pairwise_probability

   # Predict probability of each outcome
   for i in range(3):
       for j in range(3):
           if i != j:
               prob = predict_pairwise_probability(result, i, j)
               print(f"  P(Item {i} beats Item {j}): {prob:.3f}")

Output:

.. code-block:: text

     P(Item 0 beats Item 1): 0.751
     P(Item 0 beats Item 2): 0.821
     P(Item 1 beats Item 0): 0.249
     P(Item 1 beats Item 2): 0.601
     P(Item 2 beats Item 0): 0.179
     P(Item 2 beats Item 1): 0.399

Full Summary Report
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   print(result.summary())

.. code-block:: text

   ================================================================================
                            BRADLEY-TERRY MODEL REPORT
   ================================================================================

   Model Fit:
   ---------
     Converged ............................ Yes
     Log-likelihood ................... -34.21
     Items ................................. 3
     Comparisons .......................... 60

   Rankings (best first):
   ---------------------
     1. Item 0 (score: 1.523)
     2. Item 1 (score: 0.412)
     3. Item 2 (score: 0.000)

   Pairwise Win Probabilities:
   --------------------------
     P(0 beats 1): 75.1%
     P(0 beats 2): 82.1%
     P(1 beats 2): 60.1%

   Computation Time: 2.34 ms
   ================================================================================


Part 3: Ranking Comparison Metrics
----------------------------------

Comparing two rankings using multiple metrics:

Kendall Tau Correlation
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pyrevealed import compute_kendall_tau

   ranking1 = [0, 1, 2, 3, 4]  # Original ranking
   ranking2 = [0, 2, 1, 3, 4]  # One swap (positions 1 and 2)

   tau = compute_kendall_tau(ranking1, ranking2)

   print(f"Kendall tau: {tau:.3f}")
   print(f"Interpretation: {tau:.0%} correlation")

Output:

.. code-block:: text

   Kendall tau: 0.800
   Interpretation: 80% correlation

**Interpretation**:

- tau = 1.0: Perfect agreement (identical rankings)
- tau = 0.0: No correlation (random)
- tau = -1.0: Perfect disagreement (reversed rankings)

Spearman Footrule Distance
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pyrevealed import compute_spearman_footrule

   ranking1 = [0, 1, 2, 3]
   ranking2 = [3, 2, 1, 0]  # Completely reversed

   footrule = compute_spearman_footrule(ranking1, ranking2, normalize=True)

   print(f"Spearman footrule (normalized): {footrule:.3f}")
   print(f"Interpretation: {footrule:.0%} of maximum possible displacement")

Output:

.. code-block:: text

   Spearman footrule (normalized): 1.000
   Interpretation: 100% of maximum possible displacement

**Interpretation**:

- 0.0: Identical rankings
- 1.0: Maximally different

Rank-Biased Overlap (RBO)
~~~~~~~~~~~~~~~~~~~~~~~~~

RBO emphasizes agreement at the top of rankings:

.. code-block:: python

   from pyrevealed import compute_rank_biased_overlap

   # Same top-2, different rest
   ranking1 = [0, 1, 2, 3, 4]
   ranking2 = [0, 1, 4, 3, 2]

   rbo_high = compute_rank_biased_overlap(ranking1, ranking2, p=0.9)
   rbo_low = compute_rank_biased_overlap(ranking1, ranking2, p=0.5)

   print(f"RBO (p=0.9, weights deep): {rbo_high:.3f}")
   print(f"RBO (p=0.5, weights top): {rbo_low:.3f}")

Output:

.. code-block:: text

   RBO (p=0.9, weights deep): 0.847
   RBO (p=0.5, weights top): 0.906

**The p parameter**:

- p close to 1: More weight to all positions (like Jaccard)
- p close to 0: Only top positions matter
- p = 0.9 (default): Top ~10 positions get most weight

Comprehensive Comparison
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pyrevealed import compare_rankings

   ranking1 = [0, 1, 2, 3, 4]
   ranking2 = [0, 2, 1, 4, 3]

   result = compare_rankings(ranking1, ranking2, p=0.9)

   print(f"Kendall tau: {result.kendall_tau:.3f}")
   print(f"Footrule: {result.spearman_footrule:.3f}")
   print(f"RBO: {result.rank_biased_overlap:.3f}")
   print(f"Common items: {result.num_common_items}")

Output:

.. code-block:: text

   Kendall tau: 0.600
   Footrule: 0.250
   RBO: 0.823
   Common items: 5


Part 4: Aggregating Multiple Rankings
-------------------------------------

Combine rankings from multiple judges/raters:

.. code-block:: python

   from pyrevealed import aggregate_rankings

   # Three judges rank 5 items
   judge_rankings = [
       [0, 1, 2, 3, 4],  # Judge 1
       [0, 2, 1, 3, 4],  # Judge 2
       [1, 0, 2, 3, 4],  # Judge 3
   ]

   # Borda count aggregation (fast, good default)
   consensus_borda = aggregate_rankings(judge_rankings, method="borda")

   print(f"Consensus ranking (Borda): {consensus_borda}")

Output:

.. code-block:: text

   Consensus ranking (Borda): [0, 1, 2, 3, 4]

Kemeny Optimal Aggregation
~~~~~~~~~~~~~~~~~~~~~~~~~~

For small item sets, Kemeny aggregation minimizes total disagreement:

.. code-block:: python

   # Kemeny optimal (exact, but slow for n > 10)
   consensus_kemeny = aggregate_rankings(judge_rankings, method="kemeny")

   print(f"Consensus ranking (Kemeny): {consensus_kemeny}")

   # Compare methods
   print(f"\nAgreement between methods:")
   tau = compute_kendall_tau(consensus_borda, consensus_kemeny)
   print(f"  Kendall tau: {tau:.3f}")

Output:

.. code-block:: text

   Consensus ranking (Kemeny): [0, 1, 2, 3, 4]

   Agreement between methods:
     Kendall tau: 1.000


Part 5: Application - RLHF Preference Aggregation
-------------------------------------------------

Use Bradley-Terry to aggregate human preferences for AI training:

.. code-block:: python

   import numpy as np
   from pyrevealed import fit_bradley_terry, predict_pairwise_probability

   np.random.seed(42)

   # Simulate RLHF preference data
   # 4 model responses to rank
   n_responses = 4
   response_labels = ["Response A", "Response B", "Response C", "Response D"]

   # True quality scores (unknown to us)
   true_quality = np.array([0.9, 0.7, 0.5, 0.3])

   # Simulate 100 pairwise comparisons with noise
   comparisons = []
   n_comparisons_per_pair = 10

   for i in range(n_responses):
       for j in range(i + 1, n_responses):
           # Probability i wins based on true quality (with noise)
           p_i_wins = 1 / (1 + np.exp(-(true_quality[i] - true_quality[j]) * 3))

           # Simulate comparisons
           i_wins = np.random.binomial(n_comparisons_per_pair, p_i_wins)
           j_wins = n_comparisons_per_pair - i_wins

           if i_wins > 0:
               comparisons.append((i, j, i_wins))
           if j_wins > 0:
               comparisons.append((j, i, j_wins))

   # Fit Bradley-Terry model
   result = fit_bradley_terry(comparisons)

   print("=== RLHF Preference Aggregation ===\n")
   print(f"Total comparisons: {result.num_comparisons}")
   print(f"Model converged: {result.converged}")

   print("\nRecovered Response Ranking:")
   for rank, item in enumerate(result.ranking, 1):
       print(f"  {rank}. {response_labels[item]} (score: {result.scores[item]:.3f})")

   print("\nPredicted Win Rates (for reward model):")
   for i in result.ranking[:2]:  # Top 2 vs others
       for j in result.ranking[2:]:  # Bottom 2
           p = predict_pairwise_probability(result, i, j)
           print(f"  P({response_labels[i]} > {response_labels[j]}): {p:.1%}")

Example output:

.. code-block:: text

   === RLHF Preference Aggregation ===

   Total comparisons: 60
   Model converged: True

   Recovered Response Ranking:
     1. Response A (score: 1.847)
     2. Response B (score: 0.982)
     3. Response C (score: 0.456)
     4. Response D (score: 0.000)

   Predicted Win Rates (for reward model):
     P(Response A > Response C): 80.2%
     P(Response A > Response D): 86.3%
     P(Response B > Response C): 63.2%
     P(Response B > Response D): 72.7%

Inter-Rater Agreement
~~~~~~~~~~~~~~~~~~~~~

Evaluate consistency between human raters:

.. code-block:: python

   from pyrevealed import compare_rankings

   # Two raters rank the same responses
   rater1_ranking = [0, 1, 2, 3]  # A > B > C > D
   rater2_ranking = [0, 2, 1, 3]  # A > C > B > D (swapped B and C)

   agreement = compare_rankings(rater1_ranking, rater2_ranking)

   print("=== Inter-Rater Agreement ===")
   print(f"Kendall tau: {agreement.kendall_tau:.3f}")
   print(f"RBO (top-weighted): {agreement.rank_biased_overlap:.3f}")

   # Interpretation
   if agreement.kendall_tau > 0.8:
       print("\nInterpretation: Strong agreement")
   elif agreement.kendall_tau > 0.5:
       print("\nInterpretation: Moderate agreement")
   else:
       print("\nInterpretation: Weak agreement - consider more training")

Example output:

.. code-block:: text

   === Inter-Rater Agreement ===
   Kendall tau: 0.667
   RBO (top-weighted): 0.875

   Interpretation: Moderate agreement


Part 6: Notes
-------------

When to Use Each Method
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Method
     - Best For
     - Limitations
   * - Bradley-Terry
     - Aggregating pairwise comparisons
     - Assumes transitivity
   * - Kendall tau
     - Overall rank correlation
     - Doesn't weight top items
   * - Spearman footrule
     - Measuring displacement
     - Sensitive to small changes
   * - RBO
     - Top-k comparisons
     - Requires tuning p parameter
   * - Borda aggregation
     - Fast consensus ranking
     - Sensitive to irrelevant alternatives
   * - Kemeny aggregation
     - Optimal consensus
     - Slow for n > 10 items

Typical Agreement Levels
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Benchmark Interpretations
   :header-rows: 1
   :widths: 30 70

   * - Kendall Tau
     - Interpretation
   * - 0.9+
     - Excellent agreement (well-defined task)
   * - 0.7-0.9
     - Good agreement (usable for training)
   * - 0.5-0.7
     - Moderate (consider clearer guidelines)
   * - < 0.5
     - Poor (task may be too subjective)


Function Reference
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Purpose
     - Function
   * - Fit Bradley-Terry model
     - ``fit_bradley_terry()``
   * - Predict pairwise probability
     - ``predict_pairwise_probability()``
   * - Kendall tau correlation
     - ``compute_kendall_tau()``
   * - Spearman footrule distance
     - ``compute_spearman_footrule()``
   * - Rank-biased overlap
     - ``compute_rank_biased_overlap()``
   * - Comprehensive comparison
     - ``compare_rankings()``
   * - Aggregate multiple rankings
     - ``aggregate_rankings()``


See Also
--------

- :doc:`tutorial_menu_choice` - Menu-based choice fundamentals
- :doc:`tutorial_advanced` - Stochastic choice models
- :doc:`api` - Full API documentation
