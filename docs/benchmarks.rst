Benchmarks
==========

Empirical evaluation of PrefGraph on real data: LLM decision consistency and e‑commerce prediction.

.. raw:: html

   <div style="display:flex; gap:16px; align-items:flex-start;">
     <img src="_static/app_llm_hero.jpg" alt="LLM prompt consistency — stock illustration" style="width:180px; border-radius:6px; box-shadow: 0 2px 4px rgba(0,0,0,0.08);" />
     <div>
       <h2 style="margin:0 0 6px 0; font-size:1.6rem;">
         <a href="budget/app_llm_benchmark.html" style="text-decoration:none;">LLM Consistency Benchmark</a>
       </h2>
       <p style="margin:0;">
         Do LLMs keep a stable action ranking across menus? We construct preference graphs from model choices and test for cycles (SARP, IIA), then quantify minimal edits (HM) to restore consistency. The benchmark spans support, alerting, content, jobs, and procurement scenarios.
       </p>
     </div>
   </div>

`E‑commerce Benchmarks <benchmarks_ecommerce>`_
-----------------------------------------------

.. raw:: html

   <div style="display:flex; gap:16px; align-items:flex-start;">
     <img src="_static/app_recsys_hero.jpg" alt="E‑commerce recommendations — stock illustration" style="width:180px; border-radius:6px; box-shadow: 0 2px 4px rgba(0,0,0,0.08);" />
     <div>
       <p style="margin-top:0;">
         Do revealed‑preference (RP) features improve predictive models? We add GARP/CCEI/MPI/HM/VEI‑based features to strong spend/engagement baselines and evaluate lift on churn, high‑spender, novelty, and LTV across multiple public datasets.
       </p>
     </div>
   </div>

.. toctree::
   :hidden:
   :maxdepth: 1

   budget/app_llm_benchmark
   benchmarks_ecommerce
