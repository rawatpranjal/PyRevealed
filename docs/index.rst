PyRevealed
==========

You have data on what people chose. PyRevealed tests whether those choices are rational — whether any coherent preference ordering explains them. Each user gets a score from 0 (incoherent) to 1 (perfectly consistent). Rust engine, Python API.

.. raw:: html

   <div class="feature-grid">
     <div class="feature-card">
       <h3>Axiomatic Scoring</h3>
       <p>Quantify consistency via GARP, SARP, and WARP. Compute efficiency indices including CCEI (0–1), Money Pump Index (MPI), and Houtman-Maks noise fractions.</p>
     </div>
     <div class="feature-card">
       <h3>Heterogeneous Data</h3>
       <p>Support for budget-constrained choices (price-quantity pairs) and discrete menu-based selections (item-set pairs) across large-scale longitudinal datasets.</p>
     </div>
     <div class="feature-card">
       <h3>Computational Engine</h3>
       <p>Optimized Rust backend utilizing graph-theoretic algorithms and HiGHS linear programming, parallelized via Rayon for high-throughput processing.</p>
     </div>
     <div class="feature-card">
       <h3>Dual API</h3>
       <p>Every method has two names — pick whichever reads naturally in your code. Results are dataclasses with <code>.to_dict()</code> and <code>.summary()</code>, ready for pandas or sklearn pipelines.</p>
     </div>
     <div class="feature-card">
       <h3>Algorithms</h3>
       <p>Graph closure, cycle detection, linear programming, and combinatorial search — all implemented in Rust. Covers 60% of the methods surveyed in Chambers &amp; Echenique (2016).</p>
     </div>
     <div class="feature-card">
       <h3>Post-Estimation</h3>
       <p>Once you have a consistency score, go further: recover the underlying utility function, compute welfare bounds, test for separability, or measure statistical power.</p>
     </div>
   </div>

.. code-block:: bash

   pip install pyrevealed

Why Revealed Preference?
------------------------

Most approaches assume a model of preferences first, then fit parameters. Revealed preference works backwards: take raw choices and ask "could any rational model have produced these?" No assumptions about what people want — just a consistency check on what they did. Afriat (1967), Varian (1982).

.. raw:: html

   <div style="margin: 1.5em 0;"></div>

.. raw:: html

   <div style="display: flex; gap: 20px; justify-content: space-between; flex-wrap: wrap;">
     <div style="width: 48%;">
       <img src="_static/budget_hero.gif" style="width: 100%; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);" alt="Budget Choices">
       <p class="gif-caption"><strong>Budget choices.</strong> A shopper buys goods at given prices. Budget lines show what was affordable. When chosen bundles sit inside each other's budget lines, that's a contradiction — CCEI measures how much you'd need to shrink budgets to fix it. Afriat (1967).</p>
     </div>
     <div style="width: 48%;">
       <img src="_static/menu_hero.gif" style="width: 100%; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);" alt="Menu Choices">
       <p class="gif-caption"><strong>Menu choices.</strong> A user picks one option from a set. Picking Laptop over Tablet in one menu, then Tablet over Laptop in another, is a contradiction — HM counts how many choices to throw out to fix it. Houtman &amp; Maks (1985).</p>
     </div>
   </div>

Two Workflows
-------------

.. code-block:: text

   Budget Data (prices + quantities)          Menu Data (sets + picks)
   ───────────────────────────────            ──────────────────────────
   1. Load → BehaviorLog                     1. Load → MenuChoiceLog
   2. Rational? → validate_consistency()     2. Rational? → validate_menu_sarp()
   3. How much? → compute_integrity_score()  3. How much? → compute_menu_efficiency()
   4. Segment users by score                 4. Segment users by score

Performance
-----------

Benchmarked on synthetic data, T=15 observations, 10 goods, M1 Mac:

.. list-table::
   :header-rows: 1
   :widths: 30 25 25 20

   * - Configuration
     - Throughput
     - Latency (per agent)
     - Complexity
   * - **GARP only**
     - ~49,000 agents/sec
     - 20 μs
     - O(T²)
   * - **GARP + CCEI**
     - ~2,400 agents/sec
     - 420 μs
     - O(T² log T)
   * - **Full suite** (GARP, CCEI, MPI, HARP)
     - ~2,000 agents/sec
     - 500 μs
     - O(T³)
   * - **Menu** (SARP + WARP + HM)
     - ~19,000 agents/sec
     - 50 μs
     - O(N³)

Empirical benchmarks: CCEI ≈ 0.88 in lab experiments (Choi et al., 2014); HM ≈ 0.70–0.85 in recommendation click data.

----

Based on Chambers & Echenique (2016) `Revealed Preference Theory <https://www.amazon.com/Revealed-Preference-Econometric-Society-Monographs/dp/1107087805>`_. See :doc:`theory_landscape` for the full method map.

.. toctree::
   :maxdepth: 2
   :hidden:

   quickstart
   budget/index
   menu/index
   applications
   algorithms
   performance
   theory_landscape
   api
   references
