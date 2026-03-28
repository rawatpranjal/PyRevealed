Benchmarks
==========

Empirical evaluation of PrefGraph on real data: LLM decision consistency and e‑commerce prediction.

`LLM Consistency Benchmark <budget/app_llm_benchmark>`_
-------------------------------------------------------

Do LLMs keep a stable action ranking across menus? We construct preference graphs from model choices and test for cycles (SARP, IIA), then quantify minimal edits (HM) to restore consistency. The benchmark spans support, alerting, content, jobs, and procurement scenarios.

.. image:: _static/app_llm_hero.jpg
   :alt: LLM prompt consistency — stock illustration
   :width: 720

`E‑commerce Benchmarks <benchmarks_ecommerce>`_
-----------------------------------------------

Do revealed‑preference (RP) features improve predictive models? We add GARP/CCEI/MPI/HM/VEI‑based features to strong spend/engagement baselines and evaluate lift on churn, high‑spender, novelty, and LTV across multiple public datasets.

.. image:: _static/app_recsys_hero.jpg
   :alt: E‑commerce recommendations — stock illustration
   :width: 720

.. toctree::
   :hidden:
   :maxdepth: 1

   budget/app_llm_benchmark
   benchmarks_ecommerce
