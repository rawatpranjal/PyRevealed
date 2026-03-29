Case Studies
============

Real-world applications of PrefGraph.

`Detecting Inconsistency in AI Agents <budget/app_llm_benchmark.html>`_
   Do LLMs keep a stable action ranking across menus? We build preference graphs from model choices and test for cycles (SARP, IIA), then quantify minimal edits (HM) to restore consistency.

`Predicting Customer Spend & Engagement <benchmarks_ecommerce.html>`_
   Do RP features improve predictive models? We benchmark GARP, CCEI, MPI, HM, and VEI features against spend/engagement baselines on churn, high-spender, engagement, and LTV tasks across 10+ datasets.

`Performance Benchmarks <performance.html>`_
   Throughput and scaling metrics for the Rust engine across dataset sizes, user counts, and choice dimensions.

.. toctree::
   :hidden:
   :maxdepth: 1

   budget/app_llm_benchmark
   benchmarks_ecommerce
   performance
