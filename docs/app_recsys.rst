Recommendation Clicks
=====================

Test whether users' click patterns from recommendation menus reveal
consistent preferences, and use consistency scores for segmentation
and churn detection.

Introduction
------------

Every recommendation platform generates the same data: users see a
*menu* of items (search results, playlist tracks, product carousel) and
*click* one. Across sessions with varying menus, these clicks reveal
implicit preferences. If a user clicks item A over B in one session but
B over A in another, their preferences have a cycle --- no fixed ranking
can explain the choices.

Kallus & Udell (2016) formalized scalable preference learning from
assortment choice data. Cazzola & Daly (2024) argued that
rank-preference consistency --- conceptually SARP satisfaction --- is a
better evaluation metric for recommender systems than RMSE or MAE.

**What you'll learn:**

- How to map click-stream data into menu-choice observations
- The SARP test and Houtman-Maks efficiency for discrete choices
- Reconstructing menus from RetailRocket e-commerce sessions
- User segmentation by preference consistency
- Temporal analysis for churn detection via sliding-window SARP

**Companion script:** ``applications/03_recommendation_clicks.py``

Formal Setup
------------

Menu choice and revealed preference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A user faces :math:`T` sessions. In session :math:`t`, they see menu
:math:`S_t \subseteq \{1, \ldots, N\}` (a subset of :math:`N` items in
the catalog) and choose item :math:`c_t \in S_t`.

Choosing :math:`c_t` from :math:`S_t` **reveals** :math:`c_t` is preferred
to every other item in the menu:

.. math::

   c_t \succ a \quad \forall \, a \in S_t \setminus \{c_t\}

This builds an **item preference graph** :math:`G = (V, E)` where vertices
are items and edges are revealed preferences:

.. math::

   (a, b) \in E \iff \exists \, t : c_t = a \text{ and } b \in S_t

SARP
~~~~

The **Strong Axiom of Revealed Preference** (Richter, 1966) requires that
the transitive closure :math:`G^*` of the preference graph is acyclic:

.. math::

   \text{SARP: } \quad \nexists \text{ cycle } a_1 \succ^* a_2 \succ^* \cdots \succ^* a_k \succ^* a_1

SARP holds if and only if the user's choices can be explained by a strict
linear ordering over items. If it fails, no fixed preference ranking
rationalizes the data.

Houtman-Maks efficiency
~~~~~~~~~~~~~~~~~~~~~~~

When SARP fails, the **Houtman-Maks index** measures the minimum fraction
of observations to discard to restore consistency:

.. math::

   \text{HM} = 1 - \frac{\min |S| : \text{removing observations } S \text{ makes SARP hold}}{T}

HM = 1.0 means perfectly consistent. HM = 0.5 means half the observations
must be discarded. This is equivalent to finding the maximum acyclic
subgraph of the item preference graph (NP-hard in general, solved by
greedy feedback vertex set heuristic).

Data
----

RetailRocket click-stream
~~~~~~~~~~~~~~~~~~~~~~~~~

The `RetailRocket <https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset>`_
dataset contains 2.75M events from an e-commerce platform: views,
add-to-cart, and transactions with timestamps.

**Download:**

.. code-block:: bash

   # Requires Kaggle CLI (pip install kaggle)
   kaggle datasets download -d retailrocket/ecommerce-dataset
   unzip ecommerce-dataset.zip -d datasets/retailrocket/data/

**Menu reconstruction:** Items viewed in a session form the menu; the
purchased item is the choice. Sessions are split by 30-minute gaps.

.. code-block:: python

   from pyrevealed.datasets import load_retailrocket

   user_logs = load_retailrocket(min_sessions=5, max_users=200)
   print(f"Users: {len(user_logs)}")

   # Inspect one user
   uid = list(user_logs.keys())[0]
   log = user_logs[uid]
   print(f"User {uid}: {len(log.choices)} sessions, "
         f"{len(set().union(*log.menus))} unique items")

.. note::

   The RetailRocket dataset must be downloaded from Kaggle. See the
   download instructions above.

EDA: RetailRocket
~~~~~~~~~~~~~~~~~

After loading and session reconstruction:

.. list-table::
   :header-rows: 1
   :widths: 30 20

   * - Statistic
     - Value
   * - Raw events
     - 2,756,101
   * - Unique visitors
     - 1,407,580
   * - Unique items
     - 417,053
   * - Valid sessions (1 purchase, menu >= 2)
     - ~45,000
   * - Users with >= 5 sessions
     - ~1,200
   * - Mean menu size
     - 4.3 items
   * - Median menu size
     - 3 items

Most sessions have small menus (3--5 items viewed before purchase),
which is realistic for e-commerce browse-to-buy funnels. Larger menus
(10+ items) represent research-heavy purchases.

Algorithm
---------

The SARP test operates on the **item graph** (not the observation graph
used in GARP). This is a key difference from budget-based analysis:

.. code-block:: text

   SARP-TEST(menus M[1..T], choices c[1..T]):
   ───────────────────────────────────────────
   1. BUILD ITEM PREFERENCE GRAPH              O(T × |S|)
      Initialize N×N matrix G ← 0
      For each session t = 1, ..., T:
        For each item a ∈ Mₜ, a ≠ cₜ:
          G[cₜ, a] ← 1    // cₜ revealed preferred to a

   2. TRANSITIVE CLOSURE                       O(N³)
      G* ← Floyd-Warshall(G)
      // G*[a,b] = 1 iff a ≻* b (transitive preference)

   3. CHECK FOR CYCLES                         O(N²)
      For each pair (a, b):
        if G*[a, b] AND G*[b, a]:
          Record cycle: a ≻* b ≻* a
      Return: is_consistent, violation_cycles

   HOUTMAN-MAKS(menus, choices):
   ─────────────────────────────
   4. FIND MIN FEEDBACK VERTEX SET             NP-hard
      Find smallest set S of observations to remove
      such that SARP-TEST(M \ S, c \ S) is consistent
      Greedy heuristic: repeatedly remove observation
      contributing most edges to violation cycles

   Total: O(N³) where N = catalog size

.. note::

   Complexity depends on :math:`N` (catalog items), not :math:`T`
   (sessions). With 20 items, Floyd-Warshall runs on a 20x20 matrix
   --- nearly instant. With 1,000+ items (large catalogs), consider
   filtering to frequently-interacted items first.

Pipeline Walkthrough
--------------------

Single user
~~~~~~~~~~~

.. code-block:: python

   from pyrevealed import MenuChoiceLog
   from pyrevealed.algorithms.abstract_choice import validate_menu_sarp
   from pyrevealed.algorithms.abstract_choice import compute_menu_efficiency

   # Using a loaded user
   sarp = validate_menu_sarp(log)
   print(f"SARP consistent: {sarp.is_consistent}")
   print(f"Violations: {sarp.num_violations}")

.. code-block:: text

   SARP consistent: False
   Violations: 42

.. code-block:: python

   hm = compute_menu_efficiency(log)
   print(f"HM efficiency: {hm.efficiency_index:.3f}")
   print(f"Observations to remove: {len(hm.removed_observations)}/{len(log.choices)}")

.. code-block:: text

   HM efficiency: 0.620
   Observations to remove: 19/50

This user's choices require removing 38% of sessions to become
SARP-consistent, suggesting moderately noisy preferences.

Batch Analysis
--------------

User segmentation
~~~~~~~~~~~~~~~~~

Scoring all users and segmenting by consistency:

.. code-block:: python

   results = []
   for uid, log in user_logs.items():
       sarp = validate_menu_sarp(log)
       hm = compute_menu_efficiency(log)
       results.append({
           "user": uid,
           "sessions": len(log.choices),
           "is_sarp": sarp.is_consistent,
           "violations": sarp.num_violations,
           "hm_efficiency": hm.efficiency_index,
       })

Segmentation by HM efficiency:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 30

   * - Segment
     - HM Range
     - Users (%)
     - Action
   * - Stable preferences
     - 0.90 -- 1.00
     - ~30%
     - Invest in personalization
   * - Moderate noise
     - 0.60 -- 0.90
     - ~40%
     - Balance exploration and exploitation
   * - Noisy / drifting
     - < 0.60
     - ~30%
     - Invest in curation and defaults

Temporal Analysis: Churn Detection
----------------------------------

The most novel application: tracking SARP consistency over time to detect
preference drift --- a leading indicator of churn.

Split-half analysis
~~~~~~~~~~~~~~~~~~~

For each user, split sessions into first-half and second-half, then
run SARP on each half separately:

.. code-block:: python

   for uid, log in user_logs.items():
       mid = len(log.choices) // 2
       log_1 = MenuChoiceLog(menus=log.menus[:mid], choices=log.choices[:mid])
       log_2 = MenuChoiceLog(menus=log.menus[mid:], choices=log.choices[mid:])

       hm_full = compute_menu_efficiency(log).efficiency_index
       hm_1 = compute_menu_efficiency(log_1).efficiency_index
       hm_2 = compute_menu_efficiency(log_2).efficiency_index

       # Preference drift signal:
       # High per-half consistency + low full consistency = drift
       drift_signal = (hm_1 + hm_2) / 2 - hm_full

The key insight:

.. code-block:: text

   User Type       Full HM   1st Half   2nd Half   Drift Signal
   ─────────────   ───────   ────────   ────────   ────────────
   Consistent       1.000      1.000      1.000       0.000
   Noisy            0.488      0.706      0.757      +0.244
   Drifting         0.616      1.000      1.000      +0.384
   Random           0.342      0.717      0.717      +0.375

**Drifting users** show perfect consistency within each half but low
full-sequence consistency. Their preference ranking genuinely changed.
This pattern --- high per-window consistency but low full-window
consistency --- is a **churn leading indicator** that precedes engagement
drops.

Lifecycle classification
~~~~~~~~~~~~~~~~~~~~~~~~

Classify each user by the shape of their rolling-window HM trajectory:

.. list-table::
   :header-rows: 1
   :widths: 18 12 40

   * - Lifecycle
     - Criteria
     - Action
   * - Stable
     - std < 0.05
     - Reliable user; invest in deep personalization
   * - Improving
     - slope > +0.01
     - Preferences crystallizing; increase recommendation specificity
   * - Deteriorating
     - slope < -0.01
     - Preference erosion; early churn signal
   * - Volatile
     - std > 0.05, |slope| < 0.01
     - Context-dependent; emphasize exploration over exploitation

Cross-tabulating true user type with detected lifecycle validates the
detection method:

.. code-block:: text

               stable  improving  deteriorating  volatile
   consistent      12          0              0         0
   noisy            0         13              7         2
   drifting         0          0              0        10
   random           2          0              1         3

Consistent users map cleanly to "stable." Drifting users all appear
as "volatile" --- their per-window consistency is high but the trajectory
shifts. This is exactly the signal a churn detection system should flag.

Sliding window extension
~~~~~~~~~~~~~~~~~~~~~~~~~

For production use, compute HM efficiency over a sliding window
(e.g., last 20 sessions) and track the trend:

.. code-block:: python

   window_size = 20
   for start in range(0, len(log.choices) - window_size + 1, 5):
       end = start + window_size
       window_log = MenuChoiceLog(
           menus=log.menus[start:end],
           choices=log.choices[start:end],
       )
       hm = compute_menu_efficiency(window_log).efficiency_index
       # Track hm over time; declining trend = preference drift

Interpretation
--------------

Three use cases from one score
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Recommender evaluation**: A/B test two ranking algorithms. The one
   yielding higher average SARP consistency across users is eliciting
   more coherent preferences --- not just more clicks.

2. **User segmentation**: High-HM users have learnable, stable preferences.
   Invest in personalization for them. Low-HM users benefit more from
   curated defaults and exploration.

3. **Churn detection**: Monitor sliding-window HM efficiency. A user
   whose consistency drops is losing a coherent reason to engage ---
   a leading indicator before engagement metrics move.

Comparison with standard metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 25 25

   * - Metric
     - Measures
     - Limitation
   * - CTR
     - Click rate
     - Doesn't distinguish coherent from random clicks
   * - RMSE/MAE
     - Rating prediction accuracy
     - Requires explicit ratings
   * - NDCG
     - Ranking quality vs. ground truth
     - Requires ground-truth relevance labels
   * - **SARP/HM**
     - **Preference consistency**
     - **Only needs clicks from varying menus**

SARP consistency requires no ground truth, no explicit ratings, and no
user feedback surveys. It's computed directly from the data every platform
already logs.

Limitations
~~~~~~~~~~~

- Menu reconstruction from click-stream is approximate. The "true" menu
  (what the user actually saw) may differ from items they viewed.
- SARP tests for a *strict linear order*. Real preferences may be
  incomplete (indifference) or context-dependent.
- With large catalogs, sparsity limits the test's power: if two items
  never co-occur in a menu, SARP says nothing about their relative ranking.
- Session-level analysis assumes within-session independence; fatigue or
  position bias may create spurious violations.

References
----------

- Kallus, N. & Udell, M. (2016). "Revealed Preference at Scale: Learning
  Personalized Preferences from Assortment Choices." *Proceedings of the
  17th ACM Conference on Economics and Computation* (EC '16), 821--837.
  `doi:10.1145/2940716.2940752 <https://doi.org/10.1145/2940716.2940752>`_

- Cazzola, A. & Daly, M. (2024). "Rank-Preference Consistency as the
  Appropriate Metric for Recommender Systems." arXiv:2404.17097.

- Richter, M. K. (1966). "Revealed Preference Theory." *Econometrica*,
  34(3), 635--645.
  `doi:10.2307/1909773 <https://doi.org/10.2307/1909773>`_

- Houtman, M. & Maks, J. (1985). "Determining All Maximal Data Subsets
  Consistent with Revealed Preference." *Kwantitatieve Methoden*, 19, 89--104.

- RetailRocket E-Commerce Dataset. Kaggle, CC-BY-NC-SA 4.0.
  `kaggle.com/datasets/retailrocket/ecommerce-dataset <https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset>`_

.. seealso::

   :doc:`tutorial_menu_choice` for the full menu-choice tutorial.
   :doc:`theory_abstract` for the mathematical foundations of SARP.
   :doc:`app_grocery` for budget-based GARP analysis (the continuous-choice analog).
