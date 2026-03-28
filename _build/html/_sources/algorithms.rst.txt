Algorithms
==========

.. admonition:: Design Philosophy

   Algorithms are chosen to be provably
   optimal or best-in-class. The Rust engine (``rpt-core``) handles all graph and LP
   computation; Python is I/O only. Rayon thread-pool parallelism gives linear
   scaling across cores.

This page documents the algorithmic choices, complexity analysis, and the reasoning
behind each implementation decision.

Complexity
--------------------

The complexity classification for preference-graph acyclicity testing is due to
Smeulders, Cherchye, De Rock & Spieksma (2014, *ACM TEAC* 2(1)). A comprehensive survey of
the algorithmic landscape is provided by Smeulders, Crama & Spieksma (2019, *EJOR* 272(3)).

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Problem
     - Complexity
     - Economic Significance
   * - **GARP / SARP / WARP**
     - :math:`O(T^2)`
     - Fundamental test of utility maximization.
   * - **CCEI (Afriat index)**
     - :math:`O(T^2 \log T)`
     - Measure of "near-rationality" via budget deflation.
   * - **MPI (Money Pump)**
     - :math:`O(T^3)`
     - Direct measure of welfare loss from inconsistency.
   * - **HARP (Homothetic)**
     - :math:`O(T^3)`
     - Test for homothetic (scale-invariant) preferences.
   * - **Houtman-Maks**
     - NP-hard
     - Max subset of rational observations (Outlier detection).
   * - **VEI (Varian Index)**
     - NP-hard
     - Observation-specific efficiency (Precision diagnostics).
   * - **Stochastic RUM**
     - NP-hard
     - Population-level rationality (Random Utility Models).

Budget-Based Methods
--------------------

GARP — SCC Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Definition**: A dataset :math:`\{(p_t, x_t)\}_{t=1}^T` satisfies the Generalized
Axiom of Revealed Preference (GARP) if for every sequence of observations
:math:`(t_1, t_2, \dots, t_k)`, the condition :math:`p_{t_1}x_{t_1} \geq p_{t_1}x_{t_2},
\dots, p_{t_k}x_{t_k} \geq p_{t_k}x_{t_1}` implies that all inequalities are
actually equalities.

**Intuition**: If an agent selects bundle :math:`A` when :math:`B` was less expensive,
this reveals :math:`A \succeq B`. If the agent subsequently selects :math:`B` when :math:`A` was
strictly less expensive, this produces a contradiction (:math:`B \succ A`), implying no stable
utility function can rationalize the observed behavior.

**Traditional approach** (pre-2015): Build the direct observation graph
:math:`G_{R_0}`, compute its transitive closure :math:`R^*` via Floyd-Warshall in
:math:`O(T^3)`, then check :math:`\neg(i R^* j \wedge j P_0 i)` for all pairs.

.. raw:: html

   <div style="margin: 2em 0;">
       <div style="text-align: center;">
           <img src="_static/floyd_warshall.gif" style="width: 100%; max-width: 550px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
           <p style="font-size: 0.9em; color: #666; margin-top: 8px;">Floyd-Warshall builds the full transitive closure R* in O(T³)</p>
       </div>
   </div>

**Our approach**: Talla Nobibon, Smeulders & Spieksma (2015, *JOTA* 166(3)) proved
that transitive closure is unnecessary. Instead, we use **Strongly Connected
Components (SCCs)**.

.. raw:: html

   <div style="margin: 2em 0;">
       <div style="text-align: center;">
           <img src="_static/scc_tarjan.gif" style="width: 100%; max-width: 550px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
           <p style="font-size: 0.9em; color: #666; margin-top: 8px;">Tarjan's SCC detects GARP violations in O(T²) — no closure needed</p>
       </div>
   </div>

.. admonition:: Theorem (Talla Nobibon et al., 2015)

   GARP is violated if and only if some strongly connected component (SCC)
   of the direct weak preference graph :math:`G_{R_0}` contains a strict
   preference arc :math:`P_0`.

**Why this works**: If observations :math:`i` and :math:`j` are in the same SCC
of :math:`R_0`, then :math:`i R^* j` (there exists a directed path of weak
preferences from :math:`i` to :math:`j`). A GARP violation occurs if :math:`i R^* j`
and :math:`p_j x_j > p_j x_i`. This is exactly what the SCC check detects: a cycle
containing at least one "strictly more expensive" edge.

**Example**:
Suppose at :math:`t=1`, you buy :math:`x_1` at prices :math:`p_1`. You could have
bought :math:`x_2` (:math:`p_1 x_1 \geq p_1 x_2`).
At :math:`t=2`, you buy :math:`x_2` at prices :math:`p_2`. You could have bought
:math:`x_1` and it was **strictly cheaper** (:math:`p_2 x_2 > p_2 x_1`).
This forms a 2-cycle :math:`1 \xrightarrow{R_0} 2 \xrightarrow{P_0} 1`. Both
observations are in the same SCC, and there is a strict preference arc :math:`P_0`
between them. **GARP fails.**

**Algorithm**:

1. Build :math:`R_0` and :math:`P_0` from expenditure data — :math:`O(T^2)`
2. Tarjan's SCC decomposition on :math:`R_0` — :math:`O(T + |A|) \leq O(T^2)`
3. For each arc :math:`(i,j)` where :math:`\text{scc}[i] = \text{scc}[j]`, check
   :math:`P_0[i,j]` — :math:`O(T^2)`

**Total**: :math:`O(T^2)` — provably tight. For :math:`T = 10{,}000`, this is
:math:`1{,}000\times` faster than Floyd-Warshall.

.. rubric:: Implementation

- **Rust**: ``rpt-core/src/garp.rs`` — ``garp_check()`` uses Tarjan's SCC (no closure).
- **Batch dispatch**: ``batch.rs`` auto-selects :math:`O(T^2)` when only GARP is
  requested.


CCEI (Afriat Efficiency Index)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Definition**: The Critical Cost Efficiency Index (CCEI) is the supremum of all
:math:`e \in (0,1]` such that the deflated data :math:`\{(e \cdot p_t, x_t)\}_{t=1}^T`
satisfies GARP.

**Intuition**: If you fail GARP, how much would we have to "shrink" your budget
to make your choices look rational? A CCEI of 0.95 means that if you had 5% less
money at each step, your observed choices would no longer be seen as "wasteful"
relative to other options, as those other options would have been outside your
budget.

**Example**:
Suppose :math:`p_1 x_1 = 100` and :math:`p_1 x_2 = 105`. You bought :math:`x_1`
even though :math:`x_2` was only slightly more expensive. If you also have a
preference revealing :math:`x_2 \succ x_1`, you have a violation. By setting
:math:`e = 100/105 \approx 0.952`, the cost of :math:`x_2` at :math:`t=1` becomes
:math:`0.952 \times 105 = 100`. Now :math:`x_2` is exactly as expensive as :math:`x_1`,
so choosing :math:`x_1` no longer reveals a strict preference over :math:`x_2`.

**Algorithm**:
The CCEI is found by a discrete binary search over the :math:`T^2` critical
efficiency ratios :math:`\{E_{ij} / E_{ii}\}`.

1. Collect all pairwise ratios :math:`e_{ij} = p_i x_j / p_i x_i` where :math:`e_{ij} < 1`.
2. Sort and deduplicate these :math:`\leq T^2` values.
3. Binary search: for a candidate :math:`e`, check GARP on the deflated data.

**Total**: :math:`O(T^2 \log T)`.

.. admonition:: Optimization: SCC vs Closure

   Previous implementations often called Floyd-Warshall (:math:`O(T^3)`) inside the
   binary search. Since we only need a pass/fail result, the :math:`O(T^2)` SCC check
   is sufficient, saving a factor of :math:`T` in the inner loop.

.. rubric:: Implementation

- **Rust**: ``rpt-core/src/ccei.rs`` — ``ccei_search()`` performs the discrete binary
  search.

.. raw:: html

   <div style="margin: 2em 0;">
       <div style="text-align: center;">
           <img src="_static/ccei_algorithm.gif" style="width: 100%; max-width: 550px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
           <p style="font-size: 0.9em; color: #666; margin-top: 8px;">CCEI shrinks budget sets until preference cycles disappear</p>
       </div>
   </div>


MPI (Money Pump Index) — Karp's Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Definition**: The Money Pump Index (MPI) measures the maximum average budget
savings per step in a preference cycle.

**Intuition**: If your preferences are :math:`A \succ B \succ C \succ A`, an
arbitrageur could trade you :math:`A` for :math:`B` (and charge a small fee),
then :math:`B` for :math:`C`, then :math:`C` for :math:`A`, ending up with their
original goods plus your fees. The MPI quantifies how much "money" can be
pumped out of you this way.

**Example (Money Pump Cycle)**:
1. At :math:`t=1`, you buy :math:`x_1` for $10. You could have bought :math:`x_2`
   for $8. (Savings = 20%)
2. At :math:`t=2`, you buy :math:`x_2` for $10. You could have bought :math:`x_1`
   for $8. (Savings = 20%)
By trading back and forth, 20% of the budget is "wasted" in each round of the cycle.
The MPI for this cycle is 0.20.

**Algorithm**:
We model this as finding the **Maximum Mean-Weight Cycle** in a directed graph
where edge weights are relative savings :math:`w_{ij} = (E_{ii} - E_{ij})/E_{ii}`.

PrefGraph uses **Karp's Algorithm**, which uses dynamic programming to find the
optimal cycle in :math:`O(VE)` time, which is :math:`O(T^3)` here.

.. math::

   \text{MPI} = \max_C \frac{1}{|C|} \sum_{(i,j) \in C} \frac{E_{ii} - E_{ij}}{E_{ii}}

.. rubric:: Implementation

- **Rust**: ``rpt-core/src/mpi.rs`` — ``mpi_karp()`` implements the exact DP.

**References**: Echenique, Lee & Shum (2011); Smeulders et al. (2013).


HARP (Homothetic Axiom) — Max-Product Paths
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Definition**: The Homothetic Axiom of Revealed Preference (HARP) tests if
choices are consistent with a utility function :math:`u(x)` that is
linearly homogeneous (:math:`u(\alpha x) = \alpha u(x)`).

**Intuition**: Homothetic preferences imply that your relative choices between
goods don't change as your income increases; you just scale everything up.
This imposes a much stricter requirement: not just "no cycles", but "the product
of expenditure ratios along any cycle cannot exceed 1."

**Algorithm**:
We use a log-transform to turn the product check into a sum check.
1. Define weights :math:`W_{ij} = \log(E_{ii} / E_{ij})`.
2. Find the maximum-weight path between all pairs using a modified Floyd-Warshall.
3. HARP holds if no diagonal entry is positive (no positive-sum cycle).

**Complexity**: :math:`O(T^3)` due to the all-pairs shortest (longest) path
requirement.

.. rubric:: Implementation

- **Rust**: ``rpt-core/src/harp.rs`` — ``harp_check()``.


Houtman-Maks Index — Greedy + ILP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Definition**: The Houtman-Maks index is the size of the largest subset of
observations that is consistent with GARP.

**Intuition**: If you have 100 shopping trips and 5 of them are completely
unusual (e.g., buying for a large party), GARP might fail because of those 5
outliers. Houtman-Maks asks: "What is the maximum number of observations we
can keep such that they are perfectly rational?"

**Complexity**: This is NP-hard. Formally, it is equivalent to the **Maximum
Weight Independent Set** on a conflict graph, or more directly, the **Minimum
Directed Feedback Vertex Set (DFVS)** on the preference graph.

**Algorithm**:
1. **Greedy (Default)**: We use an SCC-aware greedy heuristic. Following
   Heufer & Hjertstrand (2015), the SCC decomposition reduces the problem to
   independent subproblems per strongly connected component. In each SCC, we
   repeatedly remove the node with the highest degree (participation in violations).
   This is extremely fast and usually within 1-2% of the optimal.
2. **Exact (ILP)**: We solve the problem using Integer Linear Programming (ILP).
   Binary variables :math:`z_t \in \{0,1\}` indicate whether observation :math:`t`
   is kept. The objective is to maximize :math:`\sum z_t` subject to GARP.

**Total**: NP-hard, but practical for :math:`T \leq 500` using SCC decomposition.

.. admonition:: Mononen (2023) correction

   Demuynck & Rehbeck's original formulation can report incorrect values because
   strict inequality constraints are evaluated as weak in the LP relaxation.
   Our implementation handles this via the binary threshold (``z < 0.5``), which
   is robust to this issue since the variables are constrained to be integer.

.. rubric:: Implementation

- **Rust**: ``rpt-core/src/houtman_maks.rs`` — ``houtman_maks()`` (greedy) and
  ``houtman_maks_exact()`` (ILP via HiGHS).
- **ILP solver**: ``rpt-core/src/lp.rs`` — ``solve_hm_ilp()``.

**References**: Houtman & Maks (1985); Heufer & Hjertstrand (2015).

.. raw:: html

   <div style="margin: 2em 0;">
       <div style="text-align: center;">
           <img src="_static/hm_algorithm.gif" style="width: 100%; max-width: 550px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
           <p style="font-size: 0.9em; color: #666; margin-top: 8px;">Houtman-Maks removes the minimum observations to break all preference cycles</p>
       </div>
   </div>


VEI (Varian Efficiency Index) — Exact MILP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Definition**: The VEI assigns an individual efficiency level :math:`e_t \in [0,1]`
to each observation such that the vector :math:`(e_t)_{t=1}^T` maximizes some
objective (usually :math:`\sum e_t`) subject to GARP.

**Intuition**: Unlike CCEI, which applies a single "penalty" to every observation,
VEI allows us to say: "Trip #14 was extremely irrational (e=0.7), but Trip #1 was
perfect (e=1.0)." This provides much higher diagnostic resolution for identifying
*when* behavior became inconsistent.

**Algorithm (Mononen, 2023)**:
PrefGraph implements the state-of-the-art **Row Generation** algorithm.
1. Formulate the problem as a **Weighted Minimum Feedback Arc Set (WFAS)** — find the minimum-cost set of strict revealed preferences to remove so that no directed cycle remains.
2. Initialize with all 2-cycles (WARP violations).
3. Solve the MILP with the current constraint set.
4. Run a separation oracle (DFS) to find any remaining violated cycles in the residual graph.
5. If cycles are found, add new cycle constraints and re-solve; otherwise, terminate.

**Complexity**: NP-hard, but this reformulation is :math:`10{,}000\times` faster than
previous naive ILP formulations. The LP relaxation (``compute_vei``) is available
as a fast polynomial-time heuristic.

.. admonition:: Demuynck & Rehbeck (2023) bug

   Mononen (2023) documents a 15–62% error rate in the Demuynck & Rehbeck MILP
   formulation, caused by treating strict inequality constraints as weak in the
   LP relaxation. The WFAS reformulation used in PrefGraph avoids this entirely.

.. rubric:: Implementation

- **Rust**: ``rpt-core/src/vei.rs`` — ``compute_vei()`` (LP relaxation) and
  ``compute_vei_exact()`` (MILP with row generation).

**References**: Varian (1990, *J Econometrics*); Mononen (2023).


GAPP (Generalized Axiom of Price Preference)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Definition**: GAPP tests whether prices (not quantities) reveal consistent
preferences. This is the **dual of GARP**.

**Intuition**: While GARP asks "is the bundle you chose better than other affordable
bundles?", GAPP asks "is the price you paid lower than other prices that would
have made that bundle affordable?" It tests for utility maximization when consumers
respond primarily to price signals rather than quantity constraints.

**Algorithm**:
The price preference matrices are defined as:

.. math::

   R_p[s,t] = (p^s \cdot x^t \leq p^t \cdot x^t), \qquad
   P_p[s,t] = (p^s \cdot x^t < p^t \cdot x^t)

A violation occurs if : :math:`R_p^*[s,t] \wedge P_p[t,s]`. This is the same structure as GARP but on the transposed price-expenditure graph.

.. rubric:: Implementation

- **Rust**: ``rpt-core/src/gapp.rs`` — ``gapp_check()`` uses SCC-optimized transitive
  closure on the price preference graph.

**Reference**: Deb, Kitamura, Quah & Stoye (2023, *RES*).


Stochastic Choice and RUM
-------------------------

**Definition**: Stochastic choice models analyze the probability :math:`P(x|A)` of
choosing item :math:`x` from a menu :math:`A`. A Random Utility Model (RUM) assumes
the consumer has a utility :math:`U_i = V_i + \epsilon_i` where :math:`\epsilon_i` is
a random error term.

**Key Axioms**:
- **Regularity**: :math:`P(x|A) \geq P(x|B)` whenever :math:`A \subseteq B`. Removing
  options from a menu should not decrease the probability of choosing an existing item.
- **IIA (Independence of Irrelevant Alternatives)**: The ratio of probabilities
  between two items :math:`P(x|A)/P(y|A)` should be constant across all menus
  containing both :math:`x` and :math:`y`.

**Algorithms**:
PrefGraph implements maximum likelihood estimation for:
- **Multinomial Logit (MNL)**: Type‑I Extreme Value (Gumbel) errors; satisfies IIA.
- **Luce choice rule**: :math:`P(x\mid A) = \dfrac{w_x}{\sum_{y\in A} w_y}`; equivalent to MNL under a log‑weight parameterization.

.. rubric:: Implementation

- **Python**: ``src/prefgraph/contrib/stochastic.py`` — ``fit_random_utility_model()``.

**References**: McFadden (1974); Chambers & Echenique (2016), Chapter 13.


Practical Usage: Code Examples
------------------------------

The following examples demonstrate how to call the primary algorithms using the
Python API.

**GARP and CCEI (Budget Data)**

.. code-block:: python

   from prefgraph import BehaviorLog, validate_consistency, compute_integrity_score

   # 1. Create a log (Prices p and Quantities x)
   log = BehaviorLog(cost_vectors=p, action_vectors=x)

   # 2. Test GARP (O(T²))
   result = validate_consistency(log)
   print(f"Consistent: {result.is_consistent}")

   # 3. Compute CCEI (O(T² log T))
   ccei = compute_integrity_score(log)
   print(f"CCEI: {ccei.efficiency_index:.4f}")

**SARP and WARP (Menu Data)**

.. code-block:: python

   from prefgraph import MenuChoiceLog, validate_menu_sarp

   # 1. Create a log (Menus and chosen item indices)
   log = MenuChoiceLog(menus=menus, choices=choices)

   # 2. Test SARP (O(T²))
   result = validate_menu_sarp(log)
   print(f"SARP Consistent: {result.is_consistent}")

**Money Pump Index (MPI)**

.. code-block:: python

   from prefgraph import compute_confusion_metric

   # Calculate the max average savings per cycle (O(T³))
   mpi = compute_confusion_metric(log)
   print(f"Money Pump Index: {mpi.mpi_value:.4f}")

**Stochastic Choice (RUM)**

.. code-block:: python

   from prefgraph import StochasticChoiceLog, fit_random_utility_model

   # 1. Create a log (Menus and choice frequencies)
   log = StochasticChoiceLog(menus=menus, choice_frequencies=choice_frequencies)

   # 2. Fit Logit model
   result = fit_random_utility_model(log, model_type="logit")
   print(f"Log-Likelihood: {result.log_likelihood:.2f}")
   print(f"Satisfies IIA: {result.satisfies_iia}")


Solver Stack
------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Solver
     - Status
     - Notes
   * - **HiGHS**
     - Default
     - MIT-licensed. Best open-source LP/MILP solver (Machado, 2024). Used by SciPy
       since v1.6.0. Competitive with Gurobi for LP; ~10× slower for MIP.
   * - **Gurobi**
     - Optional
     - Commercial license required. Enable via ``cargo build --features gurobi``.
       ~10× faster for Houtman-Maks ILP. Functions: ``solve_afriat_lp_gurobi()``,
       ``solve_hm_ilp_gurobi()`` in ``lp.rs``.

To build with Gurobi support:

.. code-block:: bash

   # Requires GUROBI_HOME set and gurobi shared library available
   cd rust && cargo build --release --features gurobi
