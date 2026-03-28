Benchmarks
==========

Empirical evaluation of PrefGraph on real data.

`LLM Consistency Benchmarks <budget/app_llm_benchmark.html>`_
   Do LLMs keep a stable action ranking across menus? We construct preference graphs from model choices and test for cycles (SARP, IIA), then quantify minimal edits (HM) to restore consistency.

`Predictive Benchmarks <benchmarks_ecommerce.html>`_
   Do revealed-preference (RP) features improve predictive models? We add GARP/CCEI/MPI/HM/VEI-based features to strong spend/engagement baselines and evaluate lift on churn, high-spender, novelty, and LTV across multiple public datasets.

`Performance Benchmarks <performance.html>`_
   Throughput and scaling metrics for the Rust engine across dataset sizes, user counts, and choice dimensions.

.. toctree::
   :hidden:
   :maxdepth: 1

   budget/app_llm_benchmark
   benchmarks_ecommerce
   performance
