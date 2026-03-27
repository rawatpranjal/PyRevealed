:html_theme.sidebar_secondary.remove:

PyRevealed
==========

PyRevealed is a high-performance computational library designed for the axiomatic analysis of choice behavior. It provides a robust framework for quantifying the consistency of observed decisions with the hypothesis of utility maximization, supporting both budget-constrained and menu-based choice environments.

.. raw:: html

   <div class="feature-grid">
     <div class="feature-card">
       <h3>Axiomatic Scoring</h3>
       <p>Quantify consistency via GARP, SARP, and WARP. Compute efficiency indices including CCEI (0–1), Money Pump Index (MPI), and Houtman-Maks noise fractions.</p>
     </div>
     <div class="feature-card">
       <h3>Heterogeneous Data</h3>
       <p>Support for budget-constrained choices (price-quantity pairs) and discrete menu-based selections (item-set pairs) across large-scale longitudinal datasets.</p>
     </div>
     <div class="feature-card">
       <h3>Computational Engine</h3>
       <p>Optimized Rust backend utilizing graph-theoretic algorithms and HiGHS linear programming, parallelized via Rayon for high-throughput processing.</p>
     </div>
   </div>

.. code-block:: bash

   pip install pyrevealed

.. raw:: html

   <div style="display: flex; gap: 20px; justify-content: space-between; flex-wrap: wrap;">
     <img src="_static/budget_hero.gif" style="width: 48%; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);" alt="Budget Choices">
     <img src="_static/menu_hero.gif" style="width: 48%; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);" alt="Menu Choices">
   </div>

Analytical Overview
-------------------

The primary utility of PyRevealed lies in its ability to transform raw behavioral observations into standardized measures of economic rationality. By applying the principles of revealed preference theory, the library enables researchers and practitioners to:

1. **Verify Rationalizability:** Determine if an agent's choices are reconcilable with any stable utility function.
2. **Quantify Efficiency:** Measure the severity of behavioral inconsistencies (e.g., the fraction of wealth "wasted" through sub-optimal choices).
3. **Detect Anomalies:** Identify decision-makers whose behavior deviates significantly from established economic benchmarks.
4. **Model Latent Preferences:** Recover ordinal rankings and latent utility values from observed choice sequences.

Integrated Analytical Pipelines
------------------------------

PyRevealed implements two primary workflows tailored to the structure of the observed data:

.. code-block:: text

   Budget-Constrained Choice (e.g., Grocery)   Discrete Menu-Based Choice (e.g., RecSys)
   ─────────────────────────────────────────   ─────────────────────────────────────────
        BehaviorLog(prices, quantities)             MenuChoiceLog(menus, choices)
                    │                                           │
                    ▼                                           ▼
       validate_consistency() [GARP]               validate_menu_sarp() [SARP]
                    │                                           │
                    ▼                                           ▼
      compute_integrity_score() [CCEI]            compute_menu_efficiency() [HM]
                    │                                           │
                    ▼                                           ▼
          Segmentation / Diagnostics                  Segmentation / Diagnostics

Performance Benchmarks
----------------------

In large-scale empirical applications, PyRevealed demonstrates high computational efficiency and provides meaningful behavioral insights:

* **Scalability:** The Rust compute engine processes over **10,000 agents per second** on commodity hardware, maintaining a flat memory profile via bounded streaming.
* **Consistency Benchmarks:** 
    * In controlled laboratory experiments, agents typically exhibit a **Critical Cost Efficiency Index (CCEI) of ~0.88** (Choi et al., 2014).
    * In digital recommendation environments, the **Houtman-Maks Index** identifies the maximal consistent subset of choices, facilitating the detection of stochastic noise in high-frequency interaction data.
* **Complexity Analysis:** Leverages **O(T²)** algorithms for GARP verification, ensuring stability for long-duration longitudinal datasets where T denotes the number of observations.

Why Revealed Preference?
------------------------

Unlike traditional econometric models that assume specific functional forms for utility, revealed preference analysis is **non-parametric**. It evaluates the internal consistency of data without imposing structural assumptions on preferences, making it an ideal tool for auditing the behavior of both human agents and algorithmic decision-making systems.

----

PyRevealed's methodology is grounded in the formal frameworks established in `Chambers & Echenique (2016) <https://www.amazon.com/Revealed-Preference-Econometric-Society-Monographs/dp/1107087805>`_. See :doc:`theory_landscape` for a complete map of all methods organized by data type and output type.

.. toctree::
   :maxdepth: 2
   :hidden:

   budget/index
   menu/index
   algorithms
   performance
   applications
   theory_landscape
   production/index
   intertemporal/index
   api
   references
