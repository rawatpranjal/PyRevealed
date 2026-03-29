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

When users make choices, we can represent their decisions as a preference graph. If someone chooses A over B, B over C, and then C over A, they have formed a logic cycle. These cycles represent a logical contradiction in their decision making. PrefGraph runs high-speed graph algorithms (like Tarjan's SCC) to instantly detect these cycles. By identifying and counting these contradictions, we can strictly score a user's consistency and evaluate the factual quality of their behavior.

.. raw:: html

   <div style="margin: 2em 0; text-align: center;">
     <img src="_static/intro_graphs.png" style="width: 100%; max-width: 900px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);" alt="Example of Preference Graphs">
     <p class="gif-caption" style="margin-top: 10px; font-size: 0.9em; color: #555;"><strong>Consistency in action:</strong> User A makes straightforward, logical choices without any contradictions. Conversely, User B creates a logical cycle by choosing Item C over Item A. PrefGraph catches this graph reversal and immediately penalizes User B's score.</p>
   </div>

Analyse Budgets & Menus
------------------------

PrefGraph supports two separate domains of choice data. You can evaluate **Budgets** (purchased quantities constrained by given prices, like retail shopping behavior) and **Menus** (discrete selections chosen from a set of available items, like search engine clicks or AI agent prompting). Furthermore, you can map out three specific behavioral patterns inside menus: strict deterministic limits (``MenuChoiceLog``), probabilistic stochastic distributions (``StochasticChoiceLog``), and risk-based lotteries (``RiskChoiceLog``).

You can easily feed your data into PrefGraph using Polars DataFrames, Pandas, Parquet files, or raw NumPy arrays. See the :doc:`Loading Data <quickstart>` guide for straightforward code examples spanning each loading technique.

**Budget-choice example**

.. code-block:: python

   from prefgraph.datasets import load_demo
   from prefgraph.engine import Engine, results_to_dataframe

   users = load_demo(n_users=100_000)
   engine = Engine(metrics=["garp", "ccei"])
   results = engine.analyze_arrays(users)
   df = results_to_dataframe(results)
   print(df[["is_garp", "n_violations", "ccei"]].head())

.. code-block:: text

   Scored 100,000 users in 3.8s (26,165 users/sec)

      is_garp  n_violations      ccei
   0     True             0  1.000000
   1     True             0  1.000000
   2     True             0  1.000000
   3    False             4  0.972536
   4    False             2  0.978055

**Menu-choice example** --- score 100,000 random menu users:

.. code-block:: python

   import numpy as np
   from prefgraph.engine import Engine, results_to_dataframe

   np.random.seed(42)
   menus_data = []
   for _ in range(100_000):
       menus, choices = [], []
       for __ in range(10):
           menu = sorted(np.random.choice(5, np.random.randint(2, 6), replace=False).tolist())
           menus.append(menu)
           choices.append(menu[np.random.randint(len(menu))])
       menus_data.append((menus, choices, 5))

   engine = Engine(metrics=["hm"])
   results = engine.analyze_menus(menus_data)
   df = results_to_dataframe(results)
   print(df[["is_sarp", "n_sarp_violations", "hm_consistent", "hm_total"]].head())

.. code-block:: text

   Scored 100,000 users in 1.6s (61,093 users/sec)

      is_sarp  n_sarp_violations  hm_consistent  hm_total
   0    False                 10              3         5
   1    False                  6              3         5
   2    False                 10              3         5
   3    False                 10              3         5
   4    False                 10              3         5

Case Study 1: Inconsistency in AI Agents
--------------------------------------------------

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

Case Study 2: Predicting Customer LTV
-------------------------------------------------

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
   * - H&M
     - 46,757
     - Spend Change (R²)
     - 0.290
     - 0.295
     - +0.005
     - \-
   * - Taobao
     - 29,519
     - High Entropy (AP)
     - 0.789
     - 0.790
     - +0.1%
     - \-
   * - Amazon
     - 4,668
     - Spend Drop
     - 0.784
     - 0.798
     - +1.8%
     - 0.684
   * - Dunnhumby
     - 2,222
     - Churn
     - 0.711
     - 0.724
     - +1.8%
     - 0.622

Blazingly Fast
--------------

PrefGraph processes choices using a parallel Rust and Rayon backend paired with smart memory streaming. Because it streams the data sequentially, the memory footprint remains entirely flat. Both menus and budgets scale linearly on standard hardware. In practice, you can load and score 100,000 users end-to-end across five different metrics from a 110 MB Parquet file in under two minutes natively. File I/O adds less than 70 milliseconds of total overhead. You can view our extensive format and size comparisons on the :doc:`Performance Benchmarks <performance>` page.

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
