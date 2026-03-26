Applications
============

PyRevealed supports a diverse range of empirical applications, each grounded in established revealed preference research. These implementations demonstrate the library's capacity to process large-scale longitudinal datasets and evaluate behavioral consistency across different choice environments.

.. list-table::
   :header-rows: 1
   :widths: 25 20 15 20

   * - Application Domain
     - Dataset Characteristics
     - Analytical Methodology
     - Primary Reference
   * - **Consumer Grocery**
     - Dunnhumby scanner data (2,222 households)
     - GARP / CCEI / MPI
     - Dean & Martin (2016, *AER*)
   * - **LLM Alignment**
     - Controlled API experiments (GPT-4o-mini)
     - GARP / CCEI
     - Chen et al. (2023, *PNAS*)
   * - **Recommendation Systems**
     - RetailRocket click-stream logs
     - SARP / Houtman-Maks
     - Kallus & Udell (2016, *EC*)

Implementation Framework
------------------------

The empirical pipeline for revealed preference analysis follows a standardized four-stage process:

1. **Data Ingestion:** Transformation of raw behavioral logs into canonical formats—either budget-constrained (price-quantity pairs) or discrete (menu-selection pairs).
2. **Axiomatic Verification:** Evaluation of consistency via the Generalized Axiom of Revealed Preference (GARP) for budget data or the Strong Axiom of Revealed Preference (SARP) for menu-based selections.
3. **Efficiency Quantification:** Computation of continuous indices—such as CCEI, MPI, or the Houtman-Maks Index—to measure the degree of departure from rationalizability.
4. **Behavioral Segmentation:** Analysis of agent heterogeneity, identification of anomalous decision-makers, and longitudinal tracking of consistency trends.

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

Reference Implementations
-------------------------

Standardized Python implementations for each application are available in the ``applications/`` directory:

.. code-block:: bash

   # Consumer Grocery: Analysis of household scanner data
   python applications/01_grocery_scanner.py --households 200

   # LLM Alignment: Controlled axiomatic testing of language models
   python applications/02_llm_alignment.py --trials 100

   # Recommendation Systems: Behavioral analysis of click-stream data
   python applications/03_recommendation_clicks.py --max-users 200

API Reference for Empirical Pipelines
-------------------------------------

.. code-block:: python

   # Budget-Constrained Analysis (GARP Pipeline)
   from pyrevealed import BehaviorLog, validate_consistency
   from pyrevealed import compute_integrity_score, compute_confusion_metric

   log = BehaviorLog(cost_vectors=prices, action_vectors=quantities)
   garp = validate_consistency(log)
   ccei = compute_integrity_score(log)
   mpi  = compute_confusion_metric(log)

.. code-block:: python

   # Discrete Choice Analysis (SARP Pipeline)
   from pyrevealed import MenuChoiceLog
   from pyrevealed.algorithms.abstract_choice import validate_menu_sarp
   from pyrevealed.algorithms.abstract_choice import compute_menu_efficiency

   log = MenuChoiceLog(menus=menus, choices=choices)
   sarp = validate_menu_sarp(log)
   hm   = compute_menu_efficiency(log)

.. code-block:: python

   # Large-Scale Batch Processing
   from pyrevealed.engine import Engine

   engine = Engine(metrics=["garp", "ccei", "mpi", "hm"])
   results = engine.analyze_arrays([(prices, quantities) for each agent])

.. toctree::
   :maxdepth: 2

   app_grocery
   app_llm_alignment
   app_recsys
