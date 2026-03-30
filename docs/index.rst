Preference Graphs
==================

.. code-block:: bash

   pip install prefgraph

.. raw:: html

   <div style="display: flex; gap: 24px; align-items: center; flex-wrap: wrap; margin: 1.5em 0;">
     <div style="flex: 1; min-width: 280px;">
       <p style="font-size: 1.05em; line-height: 1.6; margin: 0;">
         When users make choices, we can represent their decisions as a <strong>preference graph</strong>.
         If someone chooses A over B, B over C, and then C over A, they have formed a cycle.
         These cycles could represent an inconsistency in their decision making.
         PrefGraph uses graph algorithms (like Tarjan's SCC) to detect these cycles.
         By identifying and counting these violations, we can score a user's consistency.
       </p>
     </div>
     <div style="flex: 1; min-width: 320px;">
       <img src="_static/consistency.gif" style="width: 100%; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);" alt="Animated preference graph showing consistency detection">
     </div>
   </div>

Budgets & Menu Choices
----------------------

PrefGraph supports two separate domains of choice data. You can evaluate **Budgets** (purchased quantities constrained by given prices, like retail shopping behavior) and **Menus** (discrete selections chosen from a set of available items, like search engine clicks or AI agent prompting). Furthermore, you can map out three specific behavioral patterns inside menus: strict deterministic limits (``MenuChoiceLog``), probabilistic stochastic distributions (``StochasticChoiceLog``), and risk-based lotteries (``RiskChoiceLog``).

You can easily feed your data into PrefGraph using Polars DataFrames, Pandas, Parquet files, or raw NumPy arrays. See the :doc:`Loading Data <quickstart>` guide for straightforward code examples spanning each loading technique.

**Budget-choice example**

.. code-block:: python

   from prefgraph.datasets import load_demo
   from prefgraph.engine import Engine, results_to_dataframe

   # load_demo returns list[tuple[prices, quantities]] — synthetic shoppers
   users = load_demo(n_users=100_000)

   # Engine scores every user in parallel via Rust/Rayon
   engine = Engine(metrics=["garp", "ccei"])  # GARP = acyclicity test, CCEI = efficiency score
   results = engine.analyze_arrays(users)

   # Flatten to a DataFrame for analysis
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

**Menu-choice example**

.. code-block:: python

   from prefgraph import generate_random_menus
   from prefgraph.engine import Engine, results_to_dataframe

   # Generate discrete-choice data: each user picks one item from a menu
   menus_data = generate_random_menus(
       n_users=100_000, n_obs=10, n_items=5,
       menu_size=(2, 5),        # menus contain 2–5 items each
       choice_model="logit",    # logit model with some noise
       rationality=0.7, seed=42
   )

   # HM = Houtman-Maks: counts how many choices to discard for consistency
   engine = Engine(metrics=["hm"])
   results = engine.analyze_menus(menus_data)
   df = results_to_dataframe(results)
   print(df[["is_sarp", "n_sarp_violations", "hm_consistent", "hm_total"]].head())

.. code-block:: text

   Generated + scored 100,000 users in 2.6s (38,895 users/sec)

      is_sarp  n_sarp_violations  hm_consistent  hm_total
   0    False                  6              3         5
   1    False                  3              4         5
   2    False                  6              3         5
   3    False                  3              4         5
   4    False                  6              3         5

Case Study 1: Inconsistency in AI Agents
--------------------------------------------------

Do LLMs have stable action rankings, or does the ranking change when different
alternatives are shown? We queried GPT-4o-mini across 5 enterprise scenarios
(support triage, alert routing, content moderation, hiring, procurement), each
with 10 vignettes, 5 prompt frameworks, and 15 menus per vignette. The
deterministic stage collected 3,750 calls at temperature 0; the stochastic stage
sampled each menu 20 times at temperature 0.7, adding 75,000 calls — roughly
78,750 API calls in total over 15 hours. We built preference graphs from these
responses and tested for logical cycles. Full results: :doc:`budget/app_llm_benchmark`.

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

Performance
-----------

PrefGraph processes choices using a parallel Rust and Rayon backend with memory streaming. Because it streams the data sequentially, the memory footprint remains entirely flat. Both menus and budgets scale linearly on standard hardware. In practice, you can load and score 100,000 users end-to-end across five different metrics from a 110 MB Parquet file in under two minutes natively. File I/O adds less than 70 milliseconds of total overhead. You can view our extensive format and size comparisons on the :doc:`Performance Benchmarks <performance>` page.

.. raw:: html

   <div style="max-width: 640px;">

.. list-table:: Throughput by Metric Configuration (T=20-100, K=5)
   :header-rows: 1
   :widths: 40 20 20

   * - Metrics
     - Throughput (users/sec)
     - Latency (per user)
   * - **GARP Only** (O(T²))
     - ~49,000
     - 20 μs
   * - **GARP + CCEI**
     - ~2,400
     - 420 μs
   * - **Comprehensive** (GARP, CCEI, MPI, HARP)
     - ~2,000
     - 500 μs

.. list-table:: Budget — Large-Scale
   :header-rows: 1
   :widths: 25 15 15 15

   * - Configuration
     - 10K users
     - 100K users
     - 1M users
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

   * - Configuration
     - 10K users
     - 100K users
     - 1M users
   * - SARP + WARP + HM
     - 0.3s
     - 5.2s
     - **85.6s**

.. raw:: html

   </div>

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
