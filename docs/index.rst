Preference Graphs
==================

.. code-block:: bash

   pip install prefgraph

.. raw:: html

   <div class="feature-grid">
     <div class="feature-card">
       <h3>1. Build Graph</h3>
       <p>Map budget and menu choices to directed graphs and test for cyclic violations.</p>
     </div>
     <div class="feature-card">
       <h3>2. Score Users</h3>
       <p>Compute Consistency, Rationality and Exploitability Scores like CCEI and Houtman-Maks.</p>
     </div>
     <div class="feature-card">
       <h3>3. Scale to Millions</h3>
       <p>Process up to 49k+ users per second with a Rust backend.</p>
     </div>
   </div>

In a preference graph, a cycle (A > B > C > A) represents a logical contradiction. Using fast algorithms like Tarjan's SCC (see :doc:`algorithms`), PrefGraph detects these cycles to quantify consistency and evaluate choice quality directly from the data.

Analyse Budgets & Menus
------------------------

.. raw:: html

   <div style="display: flex; gap: 20px; justify-content: space-between; flex-wrap: wrap;">
     <div style="width: 48%;">
       <img src="_static/budget_hero.gif" style="width: 100%; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);" alt="Budget Choices">
       <p class="gif-caption"><strong>Budget choices.</strong> CCEI measures how much budgets must shrink to remove contradictions.</p>
     </div>
     <div style="width: 48%;">
       <img src="_static/menu_hero.gif" style="width: 100%; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);" alt="Menu Choices">
       <p class="gif-caption"><strong>Menu choices.</strong> HM counts how many choices to discard to restore consistency.</p>
     </div>
   </div>

PrefGraph handles two types of choice data: :doc:`Budgets <budget/index>` (prices x quantities, e.g. retail shopping) and :doc:`Menus <menu/index>` (discrete selections, e.g. search clicks or LLM prompting). Menus support three data subtypes: deterministic (``MenuChoiceLog``), stochastic (``StochasticChoiceLog``), and risk/lotteries (``RiskChoiceLog``).

Load data from Polars, Pandas, Parquet, or raw NumPy arrays. See :doc:`quickstart` for all ingestion methods.

**Budget example** --- score 5,000 synthetic consumers in under 5 seconds:

.. code-block:: python

   from prefgraph.datasets import load_demo
   from prefgraph.engine import Engine, results_to_dataframe

   users = load_demo(n_users=5000)
   engine = Engine(metrics=["garp", "ccei", "mpi", "hm"])
   results = engine.analyze_arrays(users)
   df = results_to_dataframe(results)
   print(df[["is_garp", "n_violations", "ccei", "mpi", "hm_consistent", "hm_total"]].head())

.. code-block:: text

   Scored 5,000 users in 4.7s

      is_garp  n_violations   ccei    mpi  hm_consistent  hm_total
   0     True             0  1.000  0.000             15        15
   1     True             0  1.000  0.000             15        15
   2    False             2  0.969  0.134             14        15
   3    False             2  0.931  0.067             14        15
   4    False             2  0.998  0.022             14        15

**Menu example** --- check consistency of 3 users picking from small menus:

.. code-block:: python

   from prefgraph.engine import Engine, results_to_dataframe

   menus_data = [
       ([[0,1], [1,2], [0,2], [0,1,2]], [0, 1, 0, 0], 3),  # consistent
       ([[0,1], [1,2], [0,2], [0,1,2]], [0, 1, 2, 1], 3),  # has violations
       ([[0,1,2], [1,2,3], [0,2,3], [0,1,3], [0,1,2,3]], [0, 1, 2, 3, 0], 4),
   ]
   engine = Engine(metrics=["hm"])
   results = engine.analyze_menus(menus_data)
   df = results_to_dataframe(results)
   print(df[["is_sarp", "n_sarp_violations", "hm_consistent", "hm_total"]])

.. code-block:: text

      is_sarp  n_sarp_violations  hm_consistent  hm_total
   0     True                  0              3         3
   1    False                  3              2         3
   2    False                  6              2         4

Detecting Inconsistency in LLMs
---------------------------------

Do LLMs have stable action rankings? We build preference graphs from gpt-4o-mini decisions across 5 enterprise scenarios and test for cycles. Full results: :doc:`budget/app_llm_benchmark`.

.. list-table::
   :header-rows: 1
   :widths: 30 30 30

   * - Operational Scenario
     - Perfect Consistency (%)
     - Probabilistic Consistency (%)
   * - **Support**
     - 88
     - 54
   * - **Alert**
     - 92
     - 74
   * - **Content Moderation Task**
     - 82
     - 60
   * - **Jobs Task**
     - 74
     - 62
   * - **Procurement**
     - 84
     - 61

Predicting LTV
--------------

Do RP features improve predictive models? We benchmark GARP, CCEI, MPI, HM, and VEI features against spend/engagement baselines. Full results: :doc:`benchmarks_ecommerce`.

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
     - Churn
     - 0.711
     - 0.724
     - +1.8%
     - 0.622
   * - Amazon
     - 4,668
     - Spend Drop
     - 0.784
     - 0.798
     - +1.8%
     - 0.684
   * - Dunnhumby
     - 2,222
     - Future LTV (R²)
     - 0.577
     - 0.589
     - +0.012
     - 0.246
   * - Dunnhumby
     - 2,222
     - High Spender
     - 0.962
     - 0.965
     - +0.3%
     - 0.937

Blazingly Fast
--------------

Parallel Rust/Rayon backend with streaming for flat memory. Menus and budgets both scale linearly. Full benchmarks: :doc:`performance`.

.. list-table:: Throughput by Metric Configuration (T=20-100, K=5)
   :header-rows: 1
   :widths: 40 20 20

   * - Metrics
     - Throughput (Agents/sec)
     - Latency (per Agent)
   * - **GARP Only** (O(T²))
     - ~49,000
     - 20 μs
   * - **GARP + CCEI**
     - ~2,400
     - 420 μs
   * - **Comprehensive Metrics** (GARP, CCEI, MPI, HARP)
     - ~2,000
     - 500 μs

.. list-table:: Budget — Large-Scale
   :header-rows: 1
   :widths: 25 15 15 15

   * - Configuration
     - 10,000 Agents
     - 100,000 Agents
     - 1,000,000 Agents
   * - GARP (O(T²))
     - 0.1s
     - 2.0s
     - ~20s
   * - GARP + CCEI
     - 4.2s
     - 39.5s
     - ~6.6 min
   * - Comprehensive Suite
     - 6.8s
     - 67.1s
     - ~11 min

.. list-table:: Menu — Large-Scale
   :header-rows: 1
   :widths: 25 15 15 15

   * - Metric Configuration
     - 10,000 Agents
     - 100,000 Agents
     - 1,000,000 Agents
   * - SARP + WARP + HM
     - 0.3s
     - 5.2s
     - **85.6s**

Explore the :doc:`API Reference <api>` and :doc:`References <papers>` for more.

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
   papers
