Intertemporal
=============

Analysis of time preferences — testing whether choices over time-dated
payoffs are consistent with exponential discounting or exhibit present bias.
Based on Echenique, Imai & Saito (2020, *AEJ: Micro*) and Laibson (1997, *QJE*).

.. admonition:: What can you do?

   - **Test**: Exponential discounting, quasi-hyperbolic discounting, present bias
   - **Recover**: Discount factor δ

.. code-block:: python

   from pyrevealed.contrib.intertemporal import (
       test_exponential_discounting, recover_discount_factor
   )

Tutorials
---------

.. toctree::
   :maxdepth: 1

   tutorial_intertemporal
