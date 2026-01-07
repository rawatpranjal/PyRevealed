Examples
========

PyRevealed has been validated on real-world data from the **Dunnhumby "The Complete Journey"** dataset—2 years of grocery transactions from 2,222 households.

.. note::

   For a complete walkthrough with code examples, see the :doc:`tutorial`.

Key Findings
------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Category
     - Finding
   * - **Consistency**
     - 4.5% perfectly consistent, mean integrity = 0.839
   * - **Exploitability**
     - Mean confusion = 0.225 (inversely correlated with integrity)
   * - **Statistical Power**
     - Test power = 0.845 (87.5% statistically significant)
   * - **Preference Structure**
     - 3.2% proportional-scaling, 0% income-invariant
   * - **Mental Accounting**
     - Only Protein vs Staples shows separate budgets (62%)
   * - **Cross-Price**
     - Mostly complements (Milk+Bread, Soda+Pizza)
   * - **Lancaster Model**
     - 5.4% "rescued" in characteristics-space, +5.1% mean integrity
   * - **Smooth Preferences**
     - 1.6% differentiable (SARP + uniqueness)
   * - **Strict Consistency**
     - 5.0% Acyclical P pass, 0.5% "approximately rational"
   * - **Price Preferences**
     - 0% GAPP pass (all have price preference cycles)

Dataset Overview
----------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Metric
     - Value
   * - Households analyzed
     - 2,222
   * - Product categories
     - 10 (Soda, Milk, Bread, Cheese, Chips, Soup, Yogurt, Beef, Pizza, Lunchmeat)
   * - Time period
     - 104 weeks (2 years)
   * - Total transactions
     - 645,288
   * - Processing time
     - ~92 seconds

Running the Analysis
--------------------

.. code-block:: bash

   # Download the Kaggle dataset (requires kaggle CLI)
   cd dunnhumby && ./download_data.sh

   # Run the full integration test suite
   python3 dunnhumby/run_all.py

   # Run individual analyses
   python3 dunnhumby/extended_analysis.py       # Income, spending, time trends
   python3 dunnhumby/comprehensive_analysis.py  # Confusion, separability
   python3 dunnhumby/advanced_analysis.py       # Complementarity, mental accounting
   python3 dunnhumby/encoder_analysis.py        # Auto-discovery, outlier fraction
   python3 dunnhumby/predictive_analysis.py     # Split-sample LightGBM
   python3 dunnhumby/new_algorithms_analysis.py # Test power, 2024 survey algorithms
   python3 dunnhumby/lancaster_analysis.py      # Lancaster characteristics model

   # Quick test mode (100 households sample)
   python3 dunnhumby/run_all.py --quick

See Also
--------

- :doc:`tutorial` — Full teaching tutorial with step-by-step code
- :doc:`quickstart` — Getting started with your own data
- :doc:`api` — Complete API reference
