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

The definitive complexity classification is due to Smeulders, Cherchye, De Rock
& Spieksma (2014, *ACM TEAC* 2(1)), with the survey by Smeulders, Crama &
Spieksma (2019, *EJOR* 272(3)) providing the algorithmic overview.

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Problem
     - Complexity
     - Reference
   * - GARP / SARP / WARP testing
     - :math:`O(T^2)` [tight]
     - Talla Nobibon, Smeulders & Spieksma (2015)
   * - HARP testing
     - :math:`O(T^3)`
     - Product-weight cycle detection
   * - CCEI (Afriat index)
     - :math:`O(T^2 \log T)`
     - Binary search over :math:`T^2` ratios × GARP
   * - Varian efficiency index
     - NP-hard, inapprox.
     - Smeulders et al. (2014); exact via row generation (Mononen, 2023)
   * - Houtman-Maks index
     - NP-hard, inapprox.
     - Smeulders et al. (2014); SCC decomposition (Heufer & Hjertstrand, 2015)
   * - MPI (min/max)
     - :math:`O(T^3)` polynomial
     - Smeulders et al. (2013)
   * - MPI (mean/median)
     - NP-hard
     - Smeulders et al. (2013)
   * - Stochastic rationality (RUM)
     - NP-hard
     - Smeulders (2021)


Budget-Based Methods
--------------------

GARP — :math:`O(T^2)` SCC Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. raw:: html

   <div style="display: flex; gap: 40px; margin-top: 20px; margin-bottom: 24px; align-items: flex-start; justify-content: center; flex-wrap: wrap;">
       <div style="flex: 1; min-width: 300px; text-align: center;">
           <img src="_static/floyd_warshall.gif" style="width: 100%; max-width: 400px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
       </div>
       <div style="flex: 1; min-width: 300px; text-align: center;">
           <img src="_static/scc_tarjan.gif" style="width: 100%; max-width: 400px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
       </div>
   </div>

**Traditional approach** (pre-2015): Build the direct revealed preference graph
:math:`G_{R_0}`, compute its transitive closure :math:`R^*` via Floyd-Warshall in
:math:`O(T^3)`, then check :math:`\neg(i R^* j \wedge j P_0 i)` for all pairs.

**Our approach**: Talla Nobibon, Smeulders & Spieksma (2015, *JOTA* 166(3)) proved
that transitive closure is unnecessary:

.. admonition:: Theorem (Talla Nobibon et al., 2015)

   GARP is violated if and only if some strongly connected component (SCC)
   of the direct weak preference graph :math:`G_{R_0}` contains a strict
   preference arc :math:`P_0`.

**Why this works**: If observations :math:`i` and :math:`j` are in the same SCC
of :math:`R_0`, then :math:`i R^* j` (there exists a directed path of weak
preferences from :math:`i` to :math:`j`). So checking :math:`R^*[i,j] \wedge P_0[j,i]`
reduces to checking whether any same-SCC pair :math:`(j,i)` has :math:`P_0[j,i]`.

**Algorithm**:

1. Build :math:`R_0` and :math:`P_0` from expenditure data — :math:`O(T^2)`
2. Tarjan's SCC decomposition on :math:`R_0` — :math:`O(T + |A|) \leq O(T^2)`
3. For each arc :math:`(i,j)` where :math:`\text{scc}[i] = \text{scc}[j]`, check
   :math:`P_0[i,j]` — :math:`O(T^2)`

**Total**: :math:`O(T^2)` — provably tight (Ω(T²) lower bound from input size).

**Speedup**: For :math:`T = 10{,}000`, this is :math:`1{,}000\times` faster than
Floyd-Warshall.

Shiozawa (2016, *JME* 67) independently provides an alternative :math:`O(T^2)` algorithm
via the shortest-path problem (SPP):

.. admonition:: Shortest-Path Connection (Shiozawa, 2016)

   Afriat's inequalities :math:`U_i \leq U_j + \lambda_j(E_{ji} - E_{jj})` are equivalent
   to a shortest-path system with edge weights :math:`w(j \to i) = E_{ji} - E_{jj}`.
   GARP holds iff no negative-weight cycle exists.  This unifies rationalizability
   tests (budget, homothetic, quasilinear) under one framework.  For integer data,
   Bellman-Ford directly recovers integer-valued utility (see ``recover_utility_bellman_ford()``
   in ``utility.rs``).

.. rubric:: Implementation

- **Rust**: ``rpt-core/src/garp.rs`` — ``garp_check()`` uses Tarjan's SCC (no closure).
  ``garp_check_with_closure()`` computes full :math:`R^*` only when downstream
  algorithms (MPI, VEI) need the closure matrix.
- **Batch dispatch**: ``batch.rs`` auto-selects :math:`O(T^2)` when only GARP is
  requested, :math:`O(T^3)` when MPI/VEI also need the closure.


CCEI (Afriat Efficiency Index) — :math:`O(T^2 \log T)`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The CCEI finds the largest efficiency level :math:`e^* \in (0,1]` such that the
deflated data :math:`\{(e \cdot p_i \cdot x_i, x_i)\}` satisfies GARP.

**Method**: Discrete binary search over the :math:`\leq T^2` critical efficiency
ratios :math:`\{E_{ij} / E_{ii}\}` (Afriat, 1967). At each candidate :math:`e`:

1. Rebuild :math:`R_0` and :math:`P_0` at efficiency :math:`e` — :math:`O(T^2)`
2. Check GARP via SCC — :math:`O(T^2)`

Binary search requires :math:`\sim 2 \log_2 T` iterations over the sorted candidate
list.

**Total**: :math:`O(T^2 \log T)`.

.. admonition:: Key optimization

   Previous implementations (including our earlier version) called ``ensure_closure()``
   (:math:`O(T^3)`) inside each binary search step, giving :math:`O(T^3 \log T)` total.
   Since only a GARP pass/fail is needed per step, the :math:`O(T^2)` SCC check
   suffices — a :math:`\sim T / \log T` speedup in the inner loop.

The discrete method finds the **exact analytical CCEI** with zero floating-point error,
unlike continuous bisection which converges to a tolerance.

.. rubric:: Implementation

- **Rust**: ``rpt-core/src/ccei.rs`` — ``ccei_search()`` collects :math:`T^2` ratios,
  sorts, deduplicates, then binary-searches with ``garp_check()`` per step.
  Full closure is computed only once at the end (for downstream consumers).

**References**: Afriat (1967, *IER*); Smeulders et al. (2014, *ACM TEAC*).


MPI (Money Pump Index) — :math:`O(T^3)` Karp's Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The MPI measures the maximum fraction of expenditure that could be "pumped" from a
consumer who violates GARP, via a sequence of trades exploiting preference cycles.

**Algorithm**: Karp's max-mean-weight cycle algorithm on the expenditure savings
graph. Edge weight from :math:`i` to :math:`j`:

.. math::

   w_{ij} = \frac{E_{ii} - E_{ij}}{E_{ii}}

The MPI is the maximum mean weight over all directed cycles:

.. math::

   \text{MPI} = \max_C \frac{1}{|C|} \sum_{(i,j) \in C} w_{ij}

**Complexity**: :math:`O(T^3)` — optimal for this formulation. Karp's algorithm
builds a :math:`(T+1) \times T` dynamic programming table of shortest :math:`k`-edge
paths.

.. admonition:: Complexity note (Smeulders et al., 2013)

   Min/max MPI are polynomial (:math:`O(T^3)`). Mean and median MPI are NP-hard.
   PyRevealed computes the max MPI (theory-correct per Chambers & Echenique, 2016).

.. rubric:: Implementation

- **Rust**: ``rpt-core/src/mpi.rs`` — ``mpi_karp()`` implements the exact algorithm.
  ``mpi_fast()`` provides a faster upper-bound estimate (max per-edge savings).

**References**: Echenique, Lee & Shum (2011, *JPE*); Smeulders, Spieksma, Cherchye
& De Rock (2013, *JPE*).


HARP (Homothetic Axiom) — :math:`O(T^3)` Max-Product Paths
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

HARP tests whether observed choices are consistent with homothetic preferences
(utility functions of the form :math:`u(x) = f(h(x))` where :math:`h` is
linearly homogeneous).

**Algorithm**: Modified Floyd-Warshall that maximizes the log-sum of expenditure
ratios along any path, equivalent to finding the maximum-product path:

.. math::

   W_{ij} = \log\frac{E_{ii}}{E_{ij}}, \qquad
   M_{ij} = \max_{\text{paths } i \to j} \sum_{(s,t) \in \text{path}} W_{st}

HARP is violated iff any diagonal entry :math:`M_{ii} > 0` (a positive-product
cycle exists).

**Complexity**: :math:`O(T^3)` — unavoidable since we need all-pairs max-product
paths to detect violations and reconstruct cycles.

.. rubric:: Implementation

- **Rust**: ``rpt-core/src/harp.rs`` — ``harp_check()`` builds log-ratio weights,
  runs max-product Floyd-Warshall, checks diagonal.

**Reference**: Varian (1983, *RES*).


Houtman-Maks Index — NP-hard; Greedy + ILP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Houtman-Maks index is the fraction of observations in the maximum subset
that satisfies GARP. Computing this exactly is NP-hard — equivalent to minimum
directed feedback vertex set (DFVS).

**Default (greedy FVS)**: Repeatedly identify non-trivial SCCs, remove the
highest-degree node, recompute SCCs. This is a 2-approximation.  Following
Heufer & Hjertstrand (2015), the SCC decomposition reduces the DFVS problem to
independent subproblems per strongly connected component, improving practical
runtime substantially.

**Exact (ILP)**: Big-M Afriat formulation. Binary variables :math:`z_i \in \{0,1\}`
indicate which observations to keep. The constraint system ensures the kept
subset satisfies Afriat's inequalities:

.. math::

   U_i - U_j - \lambda_j (E_{ji} - E_{jj}) \leq M(2 - z_i - z_j) \quad \forall i \neq j

Maximize :math:`\sum z_i`. Solved via HiGHS MILP (or Gurobi if available).

For :math:`T \leq 200`, the ILP typically solves in under 3 seconds
(Demuynck & Rehbeck, 2023).

.. admonition:: Mononen (2023) correction

   Demuynck & Rehbeck's original formulation can report incorrect values because
   strict inequality constraints are evaluated as weak in the LP relaxation.
   Our implementation handles this via the binary threshold (``z < 0.5``), which
   is robust to this issue since the variables are constrained to be integer.

.. rubric:: Implementation

- **Rust**: ``rpt-core/src/houtman_maks.rs`` — ``houtman_maks()`` (greedy, default),
  ``houtman_maks_exact()`` (ILP via HiGHS).
- **ILP solver**: ``rpt-core/src/lp.rs`` — ``solve_hm_ilp()``.

**References**: Houtman & Maks (1985); Heufer & Hjertstrand (2015, *Econ Letters*);
Demuynck & Rehbeck (2023, *Econ Theory*); Mononen (2023, UZH WP 437).


VEI (Varian Efficiency Index) — NP-hard; LP Relaxation + Exact MILP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The VEI assigns per-observation efficiency levels :math:`e_i \in [0,1]` that
make each observation "as rational as possible." Unlike CCEI (one global scalar),
VEI provides a vector.

**LP Relaxation** (``compute_vei``): Single LP with :math:`T` variables and :math:`O(T^2)` constraints:

.. math::

   \max \sum_i e_i \quad \text{s.t.} \quad
   e_i \geq \frac{E_{ij}}{E_{ii}} \quad \forall (i,j) \text{ where } i R^* j

This is polynomial (requires :math:`O(T^3)` transitive closure then a standard LP),
but it is a **relaxation**: it only constrains efficiency via the existing preference
structure :math:`R^*`, without accounting for how lowering :math:`e_i` changes which
preferences are revealed.

**Exact Algorithm** (``compute_vei_exact``): Mononen (2023) reformulates the exact VEI as
a **Weighted Minimum Feedback Arc Set (WFAS)** problem — find the minimum-cost set
of strict revealed preferences to remove so that no directed cycle remains:

.. math::

   \min \sum_{(i,j) \in P_{\text{strict}}} \theta_{ij} \cdot \frac{E_{ii} - E_{ij}}{E_{ii}}
   \quad \text{s.t.} \quad
   \sum_{(i,j) \in C \cap P_{\text{strict}}} \theta_{ij} \geq 1 \quad \forall \text{ cycles } C

where :math:`\theta_{ij} \in \{0,1\}` indicates whether strict preference
:math:`(i,j)` is removed, and the cost of removing :math:`(i,j)` is the fraction of
observation :math:`i`'s budget that is "wasted."

.. admonition:: Row Generation (Mononen, 2023)

   The exponential number of cycle constraints is handled via row generation:

   1. Initialize with all 2-cycles (WARP violations)
   2. Solve binary LP with current constraint set
   3. Run DFS separation oracle (Algorithm 1) to find violated cycles in the
      residual graph
   4. Add new cycle constraints and re-solve
   5. Terminate when no cycles remain

   Runtime: 5.83s for 343 subjects (Mononen) vs 65,820s (Demuynck & Rehbeck).

.. admonition:: Complexity

   The general VEI is NP-hard with no polynomial :math:`O(n^{1-\delta})`
   approximation (Smeulders et al., 2014). The LP relaxation above is polynomial
   but not exact for general violations. The MILP row generation is exact and
   practical for :math:`T \leq 200`.

.. admonition:: Demuynck & Rehbeck (2023) bug

   Mononen (2023) documents a 15–62% error rate in the Demuynck & Rehbeck MILP
   formulation, caused by treating strict inequality constraints as weak in the
   LP relaxation. The WFAS reformulation avoids this entirely.

.. rubric:: Implementation

- **Rust**: ``rpt-core/src/vei.rs`` — ``compute_vei()`` (LP relaxation, fast),
  ``compute_vei_exact()`` (binary LP with row generation via HiGHS MILP).

**References**: Varian (1990, *J Econometrics*); Smeulders et al. (2014, *ACM TEAC*);
Mononen (2023, UZH WP 437).


GAPP (Generalized Axiom of Price Preference) — :math:`O(T^3)`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

GAPP tests whether prices (not quantities) reveal consistent preferences. The
price preference matrices are:

.. math::

   R_p[s,t] = (p^s \cdot x^t \leq p^t \cdot x^t), \qquad
   P_p[s,t] = (p^s \cdot x^t < p^t \cdot x^t)

Violation: :math:`R_p^*[s,t] \wedge P_p[t,s]` — same structure as GARP but on
the transposed price graph.

.. rubric:: Implementation

- **Rust**: ``rpt-core/src/gapp.rs`` — SCC-optimized transitive closure on price
  preference graph.

**Reference**: Deb, Kitamura, Quah & Stoye (2023, *RES*).


Menu-Based Methods
------------------

SARP / WARP — :math:`O(T^2)` SCC on Item Graph
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For discrete choice data (menus + picks), WARP and SARP are tested on the
item-level preference graph:

- **WARP**: No direct reversals — if :math:`x` chosen over :math:`y` from some menu,
  then :math:`y` is never chosen over :math:`x` from any menu containing both.
- **SARP**: No preference cycles of any length (transitive WARP).

Both use Tarjan's SCC on the revealed preference graph over items.

.. rubric:: Implementation

- **Rust**: ``rpt-core/src/menu.rs`` — ``menu_sarp_check()``, ``menu_warp_check()``,
  ``menu_houtman_maks()``.

**Reference**: Richter (1966, *Econometrica*).


WARP-LA (Limited Attention) — Consideration Sets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Tests whether violations of WARP can be explained by limited attention — the
consumer considers only a subset of the menu. WARP-LA checks whether there
exists an attention filter :math:`\Gamma` and a preference order such that
choices are rational given limited consideration.

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
