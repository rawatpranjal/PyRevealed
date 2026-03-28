Preference Graphs
==================

**PrefGraph** translates raw user choices into directed preference networks to detect inconsistencies and measure behavioral coherence at scale. Without assuming prior models or making parametric guesses, it tests whether choices follow a stable ranking—making it possible to score the rationality of millions of users or evaluate LLM decision-making instantly.

.. raw:: html

   <div class="feature-grid">
     <div class="feature-card">
       <h3>1. Build &amp; Test</h3>
       <p>Map budget choices (prices × quantities) and menu selections to directed graphs. Instantly test models like GARP, SARP, and WARP for cyclic contradictions.</p>
     </div>
     <div class="feature-card">
       <h3>2. Score &amp; Recover</h3>
       <p>Assign a 0-to-1 rationality score using indices like CCEI or Houtman-Maks. Recover utility bound estimates and identify specific choice violations.</p>
     </div>
     <div class="feature-card">
       <h3>3. Scale Out</h3>
       <p>Process 49k+ users per second with a Rayon/Rust backend. Use the <strong>Engine API</strong> for batch processing and ML pipelines, or <strong>Functions</strong> for single-user audits.</p>
     </div>
   </div>

.. code-block:: bash

   pip install prefgraph

Why Preference Graphs?
-----------------------

Most behavioral analysis assumes a utility model first and tries to fit parameters to it. Preference graphs work in the exact opposite direction: they start with raw choices, build the revealed preference graph, and ask, "Is it acyclic?" By testing whether observed actions follow a logically valid ranking, PrefGraph evaluates choice quality directly from the data, without making assumptions about underlying tastes or functional forms. 

In a preference graph, a cycle (A > B > C > A) represents a logical contradiction where no coherent ranking can explain the choices. While inconsistency isn't inherently bad—it can simply reflect changing tastes, exploration, or random noise—we often need to know when decisions are inconsistent. Using fast algorithms like Tarjan's SCC, PrefGraph detects these cycles to quantify consistency.

.. raw:: html

   <div style="margin: 1.5em 0;"></div>

.. raw:: html

   <div style="display: flex; gap: 20px; justify-content: space-between; flex-wrap: wrap;">
     <div style="width: 48%;">
       <img src="_static/budget_hero.gif" style="width: 100%; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);" alt="Budget Choices">
       <p class="gif-caption"><strong>Budget choices.</strong> A shopper buys goods at given prices. Budget lines show what was affordable. When chosen bundles sit inside each other's budget lines, that's a contradiction — CCEI measures how much you'd need to shrink budgets to fix it.</p>
     </div>
     <div style="width: 48%;">
       <img src="_static/menu_hero.gif" style="width: 100%; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);" alt="Menu Choices">
       <p class="gif-caption"><strong>Menu choices.</strong> A user picks one option from a set. Picking Laptop over Tablet in one menu, then Tablet over Laptop in another, is a contradiction — HM counts how many choices to throw out to fix it. Houtman &amp; Maks (1985).</p>
     </div>
   </div>

Two Core Data Types
-------------------

PrefGraph is designed to handle two fundamentally different types of choice environments out of the box: **Budget** (e.g. retail shopping where users buy quantities given prices and a budget constraint) and **Menu** (e.g. search pages or LLM prompting where users pick one discrete item from an available set). Both follow the exact same unified workflow.

.. code-block:: text

   Budget Data (prices + quantities)          Menu Data (menus + choices)   ───────────────────────────────            ──────────────────────────
   1. Load → BehaviorLog                     1. Load → MenuChoiceLog
   2. Rational? → validate_consistency()     2. Rational? → validate_menu_sarp()
   3. How much? → compute_integrity_score()  3. How much? → compute_menu_efficiency()
   4. Segment users by score                 4. Segment users by score

LLM Consistency Benchmark
--------------------------

Do LLMs have stable action rankings? We build preference graphs from
gpt-4o-mini decisions across 5 enterprise scenarios (support triage,
alert routing, content moderation, job screening, procurement) and test
for cycles. Full results: :doc:`budget/app_llm_benchmark`.

We find that LLMs are mostly consistent: they usually pick the same thing even if you change the options; only a small share of menus make them suffer from decoy effects. When they do switch, it’s predictable, extreme options nudge them to the middle (jobs) and lenient options make them stricter (content), and the best instructions depend on the task (no one-size-fits-all).

.. list-table::
   :header-rows: 1
   :widths: 18 20 18 16 18

   * - Scenario
     - SARP pass (det/stoch)
     - IIA (det/stoch)
     - Mixed menus (%)
     - Det↔Stoch agree (%)
   * - Support
     - 88 / 90
     - 3 / 3
     - 11
     - 95.8
   * - Alert
     - 92 / 90
     - 2 / 3
     - 8
     - 96.6
   * - Content
     - 82 / 76
     - 9 / 12
     - 12
     - 95.5
   * - Jobs
     - 74 / 78
     - 15 / 14
     - 8
     - 97.6
   * - Procurement
     - 84 / 83
     - 8 / 6
     - 12
     - 97.7

Preference graphs reveal what accuracy benchmarks miss: decoy/compromise
effects (jobs), scenario‑dependent prompt effects (decision‑tree 100% on
procurement but weak on jobs), and severity anchoring even on “clear” content
inputs.

*Det = temp=0 deterministic; Stoch = majority‑vote at temp=0.7 with K=20.
Mixed menus = percent of menus with non‑unanimous responses across K reps.*

E-commerce Benchmarks
---------------------

Does measuring behavioral consistency actually improve machine learning models? We evaluated PrefGraph across seven public datasets to predict concrete business outcomes like future spend, churn, and engagement. Using CatBoost, we compared strong baselines against models augmented with revealed preference (RP) features. 
Full results: :doc:`benchmarks_ecommerce`.

.. list-table::
   :header-rows: 1
   :widths: 18 8 15 10 10 8 10

   * - Dataset
     - N
     - Target
     - Baseline
     - +RP
     - Lift%
     - RP-only
   * - Dunnhumby
     - 2,222
     - High Spender
     - 0.962
     - 0.965
     - +0.3%
     - 0.937
   * - Dunnhumby
     - 2,222
     - Churn
     - 0.711
     - 0.724
     - +1.8%
     - 0.622
   * - Dunnhumby
     - 2,222
     - Future LTV (R²)
     - 0.577
     - 0.589
     - +0.012
     - 0.246
   * - Open E-Commerce
     - 4,694
     - High Spender
     - 0.950
     - 0.951
     - +0.0%
     - —
   * - Open E-Commerce
     - 4,694
     - Churn
     - 0.846
     - 0.846
     - -0.0%
     - 0.769
   * - H&M
     - 46,757
     - High Spender
     - 0.784
     - 0.783
     - -0.1%
     - 0.720
   * - H&M
     - 46,757
     - Future Spend (R²)
     - 0.337
     - 0.340
     - +0.003
     - —
   * - H&M
     - 46,757
     - Spend Change (R²)
     - 0.290
     - 0.295
     - +0.005
     - —
   * - REES46
     - 8,832
     - High Engagement
     - 0.996
     - 0.996
     - +0.0%
     - 0.990
   * - Taobao (Buy Window)
     - 29,519
     - High Entropy (AP)
     - 0.789
     - **0.790**
     - **+0.1%**
     - —

Overall, incorporating preference graph features provides a modest but consistent lift over strong baseline models (such as traditional RFM features).

Performance
-----------

PrefGraph achieves throughput up to two orders of magnitude faster than naive Python implementations. By combining PyArrow for memory-efficient Parquet data loading with a Rust core powered by Rayon (parallelism) and algorithm optimizations like Tarjan's SCC, PrefGraph easily scales to datasets with millions of users.

Benchmarked on synthetic data, T=15 observations, 10 goods, M1 Mac:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Configuration
     - Throughput
     - Latency (per agent)
   * - **GARP only**
     - ~49,000 agents/sec
     - 20 μs
   * - **GARP + CCEI**
     - ~2,400 agents/sec
     - 420 μs
   * - **Full suite** (GARP, CCEI, MPI, HARP)
     - ~2,000 agents/sec
     - 500 μs
   * - **Menu** (SARP + WARP + HM)
     - ~19,000 agents/sec
     - 50 μs
 
..
   Archived: homepage book blurb moved to docs/archive/homepage_extras.rst

.. toctree::
   :maxdepth: 2
   :hidden:

   quickstart
   budget/index
   menu/index
   benchmarks
   algorithms
   performance
   api
   references
