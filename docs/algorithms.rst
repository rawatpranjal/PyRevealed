Algorithms
==========

.. admonition:: Design Philosophy

   Every default in PyRevealed is paper-led. Algorithms are chosen to be provably
   optimal or best-in-class. The Rust engine (``rpt-core``) handles all graph and LP
   computation; Python is I/O only. Rayon thread-pool parallelism gives linear
   scaling across cores.

This page documents the algorithmic choices, complexity analysis, and the reasoning
behind each implementation decision. Focus: budget-based and menu-based methods.

Complexity Landscape
--------------------

The definitive complexity classification for revealed preference testing is due to
Smeulders, Cherchye, De Rock & Spieksma (2014, *ACM TEAC* 2(1)), which established the
computational hardness of various goodness-of-fit measures. A comprehensive survey of
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

GARP — :math:`O(T^2)` SCC Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Definition**: A dataset :math:`\{(p_t, x_t)\}_{t=1}^T` satisfies the Generalized
Axiom of Revealed Preference (GARP) if for every sequence of observations
:math:`(t_1, t_2, \dots, t_k)`, the condition :math:`p_{t_1}x_{t_1} \geq p_{t_1}x_{t_2},
\dots, p_{t_k}x_{t_k} \geq p_{t_k}x_{t_1}` implies that all inequalities are
actually equalities.

**Intuition**: If you choose bundle :math:`A` when :math:`B` was cheaper, you
reveal :math:`A \succeq B`. If you then choose :math:`B` when :math:`A` was
strictly cheaper, you have a contradiction (:math:`B \succ A`), implying no stable
utility function can explain your behavior.

.. raw:: html

   <div style="display: flex; gap: 40px; margin-top: 20px; margin-bottom: 24px; align-items: flex-start; justify-content: center; flex-wrap: wrap;">
       <div style="flex: 1; min-width: 300px; text-align: center;">
           <img src="_static/floyd_warshall.gif" style="width: 100%; max-width: 400px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
           <p style="font-size: 0.9em; color: #666; margin-top: 8px;">Floyd-Warshall (O(T³))</p>
       </div>
       <div style="flex: 1; min-width: 300px; text-align: center;">
           <img src="_static/scc_tarjan.gif" style="width: 100%; max-width: 400px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
           <p style="font-size: 0.9em; color: #666; margin-top: 8px;">Tarjan's SCC (O(T²))</p>
       </div>
   </div>

**Traditional approach** (pre-2015): Build the direct revealed preference graph
:math:`G_{R_0}`, compute its transitive closure :math:`R^*` via Floyd-Warshall in
:math:`O(T^3)`, then check :math:`\neg(i R^* j \wedge j P_0 i)` for all pairs.

**Our approach**: Talla Nobibon, Smeulders & Spieksma (2015, *JOTA* 166(3)) proved
that transitive closure is unnecessary. Instead, we use **Strongly Connected
Components (SCCs)**.

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


CCEI (Afriat Efficiency Index) — :math:`O(T^2 \log T)`
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


MPI (Money Pump Index) — :math:`O(T^3)` Karp's Algorithm
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

PyRevealed uses **Karp's Algorithm**, which uses dynamic programming to find the
optimal cycle in :math:`O(VE)` time, which is :math:`O(T^3)` here.

.. math::

   \text{MPI} = \max_C \frac{1}{|C|} \sum_{(i,j) \in C} \frac{E_{ii} - E_{ij}}{E_{ii}}

.. rubric:: Implementation

- **Rust**: ``rpt-core/src/mpi.rs`` — ``mpi_karp()`` implements the exact DP.

**References**: Echenique, Lee & Shum (2011); Smeulders et al. (2013).


HARP (Homothetic Axiom) — :math:`O(T^3)` Max-Product Paths
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


Houtman-Maks Index — NP-hard; Greedy + ILP
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


VEI (Varian Efficiency Index) — NP-hard; Exact MILP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Definition**: The VEI assigns an individual efficiency level :math:`e_t \in [0,1]`
to each observation such that the vector :math:`(e_t)_{t=1}^T` maximizes some
objective (usually :math:`\sum e_t`) subject to GARP.

**Intuition**: Unlike CCEI, which applies a single "penalty" to every observation,
VEI allows us to say: "Trip #14 was extremely irrational (e=0.7), but Trip #1 was
perfect (e=1.0)." This provides much higher diagnostic resolution for identifying
*when* behavior became inconsistent.

**Algorithm (Mononen, 2023)**:
PyRevealed implements the state-of-the-art **Row Generation** algorithm.
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
   LP relaxation. The WFAS reformulation used in PyRevealed avoids this entirely.

.. rubric:: Implementation

- **Rust**: ``rpt-core/src/vei.rs`` — ``compute_vei()`` (LP relaxation) and
  ``compute_vei_exact()`` (MILP with row generation).

**References**: Varian (1990, *J Econometrics*); Mononen (2023).


GAPP (Generalized Axiom of Price Preference) — :math:`O(T^3)`
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


Menu-Based Methods
------------------

SARP / WARP — :math:`O(T^2)` SCC on Item Graph
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Definition**: For discrete choice data (menus + single picks), the Weak Axiom (WARP)
and Strong Axiom (SARP) test the consistency of selection.

**Intuition**:
If you chose Apple when Orange was available, you reveal Apple :math:`\succeq` Orange.
- **WARP**: No direct reversals. If you reveal Apple :math:`\succ` Orange, you
  cannot later choose Orange when Apple is available.
- **SARP**: No cycles. If Apple :math:`\succ` Orange and Orange :math:`\succ` Banana,
  you cannot choose Banana when Apple is available.

**Algorithm**:
We construct a **Directed Preference Graph** where nodes are the **items** (not
the observations). An edge :math:`i \to j` exists if item :math:`i` was chosen
from a menu containing item :math:`j`.
1. Build the choice-graph — :math:`O(T \times \text{menu\_size})`.
2. Find SCCs using Tarjan's algorithm — :math:`O(K + E)` where :math:`K` is items.
3. SARP is violated if any SCC contains a strict preference (choosing :math:`i`
   over :math:`j` when :math:`j` was available).

.. rubric:: Implementation

- **Rust**: ``rpt-core/src/menu.rs`` — ``menu_sarp_check()`` and ``menu_warp_check()``.


WARP-LA (Limited Attention) — Consideration Sets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Definition**: WARP with Limited Attention (Masatlioglu et al., 2012) tests
whether choices are consistent with a preference order and an **attention filter**.

**Intuition**: In large menus (like Amazon or Netflix), you don't actually see
every item. If you choose :math:`x` even though :math:`y` is better, it may not
be "irrational"—it may be that you didn't even *consider* :math:`y`.
WARP-LA allows for this by requiring only that your attention is "consistent":
removing an item you *didn't* choose shouldn't change which items you *do* consider.

**Example**:
Suppose from menu {A, B, C}, you choose B.
If you later choose A from menu {A, B}, you have violated WARP.
However, this is consistent with **Limited Attention** if, in the first case,
having C in the menu "distracted" you from seeing A. But if we remove C and you
*still* don't choose A, then the contradiction remains.

.. rubric:: Implementation

- **Rust**: ``rpt-core/src/attention.rs`` — ``warp_la_check()``.

**Reference**: Masatlioglu, Nakajima & Ozbay (2012, *AER*).


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


Open Research Directions
------------------------

These represent genuine open problems where no published work exists:

- **GPU-accelerated GARP**: The :math:`O(T^2)` arc-building step (all pairwise
  :math:`p_i \cdot q_j`) is embarrassingly parallel and GPU-suitable. SCC
  decomposition (Tarjan) is inherently sequential (DFS-based). A hybrid
  GPU-parallel construction + CPU SCC approach could push throughput for very
  large :math:`T`.
- **Streaming / online GARP**: Incremental consistency checking as observations
  arrive, without re-checking from scratch.
- **Warm-starting CCEI**: Reusing SCC structure from previous binary search
  iterations to speed up subsequent checks.
- **Randomized approximate GARP**: Trading exactness for sub-quadratic runtime.


References
----------

.. [Afriat1967] Afriat, S. N. (1967). "The Construction of Utility Functions from
   Expenditure Data." *International Economic Review* 8(1), 67-77.

.. [TallaNobibon2015] Talla Nobibon, F., Smeulders, B., & Spieksma, F. C. R. (2015).
   "A Note on Testing Axioms of Revealed Preference." *Journal of Optimization Theory
   and Applications* 166(3), 1063-1070.

.. [Shiozawa2016] Shiozawa, K. (2016). "Revealed Preference Test and Shortest Path
   Problem." *Journal of Mathematical Economics* 67, 38-48.

.. [Smeulders2014] Smeulders, B., Cherchye, L., De Rock, B., & Spieksma, F. C. R.
   (2014). "Goodness-of-Fit Measures for Revealed Preference Tests: Complexity Results
   and Algorithms." *ACM Transactions on Economics and Computation* 2(1), Article 3.

.. [Smeulders2013] Smeulders, B., Spieksma, F. C. R., Cherchye, L., & De Rock, B.
   (2013). "The Money Pump as a Measure of Revealed Preference Violations: A Comment."
   *Journal of Political Economy* 121(6), 1248-1258.

.. [Smeulders2019] Smeulders, B., Crama, Y., & Spieksma, F. C. R. (2019). "Revealed
   Preference Theory: An Algorithmic Outlook." *European Journal of Operational Research*
   272(3), 803-815.

.. [Smeulders2021] Smeulders, B. (2021). "Nonparametric Analysis of Random Utility Models:
   Computational Tools for Statistical Testing." *Econometrica* 89(5), 2227-2250.

.. [DemuynckRehbeck2023] Demuynck, T., & Rehbeck, J. (2023). "Computing Revealed
   Preference Goodness-of-Fit Measures with Integer Programming." *Economic Theory*
   76(4), 1175-1195.

.. [Mononen2023] Mononen, L. (2023). "Computing and Comparing Measures of Rationality."
   University of Zurich Working Paper 437.

.. [HeuferHjertstrand2015] Heufer, J., & Hjertstrand, P. (2015). "Consistent Subsets:
   Computationally Feasible Methods to Compute the Houtman-Maks-Index." *Economics
   Letters* 128, 87-89.

.. [EcheniqueLeeShum2011] Echenique, F., Lee, S., & Shum, M. (2011). "The Money Pump
   as a Measure of Revealed Preference Violations." *Journal of Political Economy*
   119(6), 1201-1223.

.. [Varian1983] Varian, H. R. (1983). "Non-parametric Tests of Consumer Behaviour."
   *Review of Economic Studies* 50(1), 99-110.

.. [Varian1990] Varian, H. R. (1990). "Goodness-of-Fit in Optimizing Models." *Journal
   of Econometrics* 46(1-2), 125-140.

.. [DebKitamuraQuahStoye2023] Deb, R., Kitamura, Y., Quah, J., & Stoye, J. (2023).
   "Revealed Price Preference: Theory and Stochastic Testing." *Review of Economic
   Studies* 90(2), 707-743.

.. [Richter1966] Richter, M. K. (1966). "Revealed Preference Theory." *Econometrica*
   34(3), 635-645.

.. [Masatlioglu2012] Masatlioglu, Y., Nakajima, D., & Ozbay, E. Y. (2012). "Revealed
   Attention." *American Economic Review* 102(5), 2183-2205.

.. [Machado2024] Machado, D. (2024). "A Benchmark of Optimization Solvers." *mSystems*
   9(2).

.. [DziewulskiLanierQuah2024] Dziewulski, P., Lanier, J., & Quah, J. (2024).
   "Revealed Preference and Revealed Preference Cycles: A Survey." *Journal of
   Mathematical Economics* 113(C).

.. [KitamuraStoye2018] Kitamura, Y., & Stoye, J. (2018). "Nonparametric Analysis of
   Random Utility Models." *Econometrica* 86(6), 1883-1909.

.. [ChambersEchenique2016] Chambers, C. P., & Echenique, F. (2016). *Revealed
   Preference Theory*. Cambridge University Press.
