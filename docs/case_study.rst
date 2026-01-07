Case Study: Dunnhumby
=====================

See `DUNNHUMBY.md <https://github.com/rawatpranjal/PyRevealed/blob/main/DUNNHUMBY.md>`_ for a real-world validation on 2,222 households from the Dunnhumby grocery dataset.

Key Findings
------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Category
     - Finding
   * - **Consistency**
     - 4.5% consistent, mean integrity = 0.839
   * - **Exploitability**
     - Mean confusion = 0.225
   * - **Statistical Power**
     - Test power = 0.845 (87.5% significant)
   * - **Preference Structure**
     - 3.2% proportional-scaling, 0% income-invariant
   * - **Feature Independence**
     - Only Protein vs Staples shows separate budgets (62%)
   * - **Robustness**
     - 17% "almost consistent" (outlier fraction < 0.1)
   * - **Cross-Price**
     - Mostly complements (Milk+Bread, Soda+Pizza)
   * - **Lancaster Model**
     - 5.4% "rescued" in characteristics-space, +5.1% mean integrity

Running the Analysis
--------------------

.. code-block:: bash

   # Download the Kaggle dataset (requires kaggle CLI)
   cd dunnhumby && ./download_data.sh

   # Run the full integration test suite
   python3 dunnhumby/run_all.py

   # Run extended analysis
   python3 dunnhumby/extended_analysis.py

   # Quick test mode (100 households sample)
   python3 dunnhumby/run_all.py --quick
