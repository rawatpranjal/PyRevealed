Preference Graphs
==================

**PrefGraph** translates raw user choices into directed preference networks to detect inconsistencies and measure behavioral coherence at scale.

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

Most behavioral analysis assumes a utility model first and tries to fit parameters to it. Preference graphs work in the exact opposite direction: they start with raw choices, build the revealed preference graph, and ask, "Is it acyclic?"

In a preference graph, a cycle (A > B > C > A) represents a logical contradiction. Using fast algorithms like Tarjan's SCC, PrefGraph detects these cycles to quantify consistency and evaluate choice quality directly from the data.

.. raw:: html

   <div style="margin: 1.5em 0;"></div>

.. raw:: html

   <div style="display: flex; gap: 20px; justify-content: space-between; flex-wrap: wrap;">
     <div style="width: 48%;">
       <img src="_static/budget_hero.gif" style="width: 100%; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);" alt="Budget Choices">
       <p class="gif-caption"><strong>Budget choices.</strong> A shopper buys goods at given prices. Budget lines show what was affordable. When chosen bundles sit inside each other's budget lines, that's a contradiction - CCEI measures how much you'd need to shrink budgets to fix it.</p>
     </div>
     <div style="width: 48%;">
       <img src="_static/menu_hero.gif" style="width: 100%; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);" alt="Menu Choices">
       <p class="gif-caption"><strong>Menu choices.</strong> A user picks one option from a set. Picking Laptop over Tablet in one menu, then Tablet over Laptop in another, is a contradiction - HM counts how many choices to throw out to fix it. Houtman &amp; Maks (1985).</p>
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

LLM Consistency Benchmarks
--------------------------

Do LLMs have stable action rankings? We build preference graphs from
gpt-4o-mini decisions across 5 enterprise scenarios (support triage,
alert routing, content moderation, job screening, procurement) and test
for cycles. Full results: :doc:`budget/app_llm_benchmark`.

We find that LLMs are mostly consistent: they usually pick the same thing even if you change the options; only a small share of menus make them suffer from decoy effects. When they do switch, it’s predictable, extreme options nudge them to the middle (jobs) and lenient options make them stricter (content), and the best instructions depend on the task (no one-size-fits-all).

.. list-table::
   :header-rows: 1
   :widths: 14 14 14 14 14 16 16 14

   * - Scenario
     - SARP pass (%) — deterministic
     - SARP pass (%) — stochastic
     - IIA violations — deterministic
     - IIA violations — stochastic
     - Menus with mixed responses (%)
     - Deterministic–stochastic agreement (%)
     - RUM pass rate (%)
   * - Support
     - 88
     - 90
     - 3
     - 3
     - 11
     - 95.8
     - 54
   * - Alert
     - 92
     - 90
     - 2
     - 3
     - 8
     - 96.6
     - 74
   * - Content
     - 82
     - 76
     - 9
     - 12
     - 12
     - 95.5
     - 60
   * - Jobs
     - 74
     - 78
     - 15
     - 14
     - 8
     - 97.6
     - 62
   * - Procurement
     - 84
     - 83
     - 8
     - 6
     - 12
     - 97.7
     - 61

Preference graphs reveal what accuracy benchmarks miss: decoy/compromise
effects (jobs), scenario‑dependent prompt effects (decision‑tree 100% on
procurement but weak on jobs), and severity anchoring even on “clear” content
inputs.

*Definitions: SARP pass (deterministic) = percent of vignette–prompt pairs
with an acyclic preference graph at temperature 0. SARP pass (stochastic)
uses majority vote over K=20 samples at temperature 0. IIA violations count
pairwise preference flips when a third option is added. Menus with mixed
responses = percent of menus where the K stochastic responses are not all the
same. Deterministic–stochastic agreement = percent of menus where the
deterministic choice matches the stochastic majority. RUM pass rate aggregates
frequencies over K samples and tests Random Utility Model consistency.*

Predictive Benchmarks (E-commerce)
-----------------------------------

Do revealed-preference (RP) features improve predictive models? We add GARP/CCEI/MPI/HM/VEI-based features to strong spend/engagement baselines and evaluate lift on churn, high-spender, novelty, and LTV across multiple public datasets.
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
   * - Amazon
     - 4,694
     - High Spender
     - 0.950
     - 0.951
     - +0.0%
     - -
   * - Amazon
     - 4,694
     - Churn
     - 0.784
     - 0.798
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
     - -
   * - H&M
     - 46,757
     - Spend Change (R²)
     - 0.290
     - 0.295
     - +0.005
     - -
   * - REES46
     - 8,832
     - High Engagement
     - 0.996
     - 0.996
     - +0.0%
     - 0.990
   * - Taobao
     - 29,519
     - Pref Drift (AP)
     - 0.940
     - 0.938
     - -0.2%
     - -
   * - Taobao
     - 29,519
     - High Entropy (AP)
     - 0.789
     - **0.790**
     - **+0.1%**
     - -
   * - Taobao
     - 29,519
     - High Active Time (AUC)
     - 0.777
     - 0.778
     - +0.1%
     - -
   * - Taobao
     - 29,519
     - High Click Volume (AUC)
     - 0.818
     - 0.818
     - +0.0%
     - -
   * - Taobao
     - 29,519
     - Fast Conversion (AUC)
     - 0.561
     - 0.561
     - +0.0%
     - -

Overall, incorporating preference graph features provides a modest but consistent lift over strong baseline models (such as traditional RFM features).

Performance
-----------

PrefGraph achieves throughputs up to 49k+ users/sec using a Rayon/Rust backend. See the :doc:`Performance <performance>` page for detailed scaling metrics.

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
   api
   references
