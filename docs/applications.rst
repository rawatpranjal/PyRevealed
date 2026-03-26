Applications
============

Three real-world applications of revealed preference theory, each backed
by published research and runnable end-to-end. Click through for the full
tutorial with formal definitions, pseudocode, EDA, and citations.

.. list-table::
   :header-rows: 1
   :widths: 25 20 15 20

   * - Application
     - Data
     - Method
     - Paper
   * - :doc:`app_grocery`
     - Dunnhumby scanner (2,222 households)
     - GARP / CCEI / MPI
     - Dean & Martin (2016 AER)
   * - :doc:`app_llm_alignment`
     - GPT-4o-mini API (controlled experiment)
     - GARP / CCEI
     - Chen et al. (2023 PNAS)
   * - :doc:`app_recsys`
     - RetailRocket click-stream
     - SARP / Houtman-Maks
     - Kallus & Udell (2016 EC)

The generic pipeline
--------------------

All three applications follow the same structure:

1. **Load data** --- prices/quantities (budget) or menus/clicks (discrete)
2. **Test consistency** --- GARP (budget) or SARP (menu)
3. **Score** --- CCEI, MPI, Houtman-Maks (0 to 1)
4. **Segment** --- rank users, detect anomalies, track trends

.. code-block:: text

   Budget choice (grocery, LLM):     Discrete choice (recommendation):
   ─────────────────────────────     ─────────────────────────────────
   BehaviorLog(prices, quantities)   MenuChoiceLog(menus, choices)
        │                                  │
        ▼                                  ▼
   validate_consistency() [GARP]     validate_menu_sarp() [SARP]
        │                                  │
        ▼                                  ▼
   compute_integrity_score() [CCEI]  compute_menu_efficiency() [HM]
        │                                  │
        ▼                                  ▼
   Segment / rank / alert            Segment / rank / alert

Companion scripts
-----------------

Each application has a runnable Python script in ``applications/``:

.. code-block:: bash

   # Grocery: Dunnhumby scanner data (2,222 households)
   python applications/01_grocery_scanner.py --households 200

   # LLM: controlled GPT experiment (requires OPENAI_API_KEY)
   python applications/02_llm_alignment.py --trials 100

   # Recommendation: RetailRocket click-stream (requires Kaggle download)
   python applications/03_recommendation_clicks.py --max-users 200

Quick reference
---------------

.. code-block:: python

   # Budget choice (GARP pipeline)
   from pyrevealed import BehaviorLog, validate_consistency
   from pyrevealed import compute_integrity_score, compute_confusion_metric

   log = BehaviorLog(cost_vectors=prices, action_vectors=quantities)
   garp = validate_consistency(log)
   ccei = compute_integrity_score(log)
   mpi  = compute_confusion_metric(log)

.. code-block:: python

   # Discrete choice (SARP pipeline)
   from pyrevealed import MenuChoiceLog
   from pyrevealed.algorithms.abstract_choice import validate_menu_sarp
   from pyrevealed.algorithms.abstract_choice import compute_menu_efficiency

   log = MenuChoiceLog(menus=menus, choices=choices)
   sarp = validate_menu_sarp(log)
   hm   = compute_menu_efficiency(log)

.. code-block:: python

   # Batch scoring (thousands of users)
   from pyrevealed.engine import Engine

   engine = Engine(metrics=["garp", "ccei", "mpi", "hm"])
   results = engine.analyze_arrays([(prices, quantities) for each user])

.. toctree::
   :maxdepth: 2

   app_grocery
   app_llm_alignment
   app_recsys
