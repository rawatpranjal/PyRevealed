Benchmarks
==========

Empirical evaluation of PrefGraph on real data.

`Consistency Benchmarks (LLMs) <budget/app_llm_benchmark.html>`_
   Do LLMs keep a stable action ranking across menus? We construct preference graphs from model choices and test for cycles (SARP, IIA), then quantify minimal edits (HM) to restore consistency.

`Predictive Benchmarks (E-commerce) <benchmarks_ecommerce.html>`_
   Do RP features improve predictive models?
   We benchmark GARP, CCEI, MPI, HM, and VEI features against spend/engagement baselines on churn, high-spender, novelty, and LTV tasks.

`Performance Benchmarks <performance.html>`_
   Throughput and scaling metrics for the Rust engine across dataset sizes, user counts, and choice dimensions.

.. toctree::
   :hidden:
   :maxdepth: 1

   budget/app_llm_benchmark
   benchmarks_ecommerce
   performance
   performance
