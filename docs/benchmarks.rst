Benchmarks
==========

Empirical evaluation of PrefGraph on real data.

.. raw:: html

   <div style="display:flex; gap:16px; align-items:center;">
     <img src="_static/app_llm_benchmark_summary.png" alt="LLM benchmark summary (real results)" style="width:180px; border-radius:6px; box-shadow: 0 2px 4px rgba(0,0,0,0.08);" />
     <div style="margin-top:6px;">
       <h2 style="margin:0 0 6px 0; font-size:1.6rem;">
        <a href="budget/app_llm_benchmark.html" style="text-decoration:none;">LLM Consistency Benchmarks</a>
       </h2>
       <p style="margin:0;">
         Do LLMs keep a stable action ranking across menus? We construct preference graphs from model choices and test for cycles (SARP, IIA), then quantify minimal edits (HM) to restore consistency. Real outputs shown at right; full details in the linked page.
       </p>
     </div>
   </div>

.. raw:: html

   <div style="display:flex; gap:16px; align-items:center;">
        <img src="_static/eco_auc_comparison.png" alt="E-commerce benchmarking pipeline results map" style="width:180px; border-radius:6px; box-shadow: 0 2px 4px rgba(0,0,0,0.08);" />
     <div style="margin-top:6px;">
       <h2 style="margin:0 0 6px 0; font-size:1.6rem;">
         <a href="benchmarks_ecommerce.html" style="text-decoration:none;">E‑commerce Benchmarks</a>
       </h2>
       <p style="margin:0;">
         Do revealed‑preference (RP) features improve predictive models? We add GARP/CCEI/MPI/HM/VEI‑based features to strong spend/engagement baselines and evaluate lift on churn, high‑spender, novelty, and LTV across multiple public datasets.
       </p>
     </div>
   </div>

.. toctree::
   :hidden:
   :maxdepth: 1

   budget/app_llm_benchmark
   benchmarks_ecommerce
