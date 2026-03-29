:orphan:

Production
==========

**Input:** ``ProductionLog`` - input prices/quantities × output prices/quantities

Production theory applies revealed preference to firm behavior, testing
whether input-output observations are consistent with profit maximization
or cost minimization. Based on Varian (1984, *Econometrica*) and
Chambers & Echenique (2016) Chapter 15.

.. admonition:: What can you do?

   - **Test**: Profit maximization (production GARP), cost minimization
   - **Score**: Technical efficiency
   - **Recover**: Returns to scale

.. code-block:: python

   from prefgraph import ProductionLog
   from prefgraph.algorithms.production import test_profit_maximization

   log = ProductionLog(input_prices=ip, input_quantities=iq,
                       output_prices=op, output_quantities=oq)
   result = test_profit_maximization(log)

Theory
------

.. toctree::
   :maxdepth: 1

   theory_production

Examples
--------

.. toctree::
   :maxdepth: 1

   examples
