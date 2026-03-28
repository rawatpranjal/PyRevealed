Production Theory
=================

Apply revealed preference to firm behavior.
Based on Chambers & Echenique (2016) Chapter 15.

Profit Maximization
-------------------

**Function:** ``test_profit_maximization(log)``

Define input vector :math:`x^t`, output vector :math:`y^t`, input prices :math:`w^t`, and output prices :math:`p^t`.

**Profit GARP:** No profit arbitrage cycles exist:

.. math::

   \pi^s \geq \pi^t(x^s, y^s) \implies \pi^t \geq \pi^s(x^t, y^t)

where :math:`\pi^t = p^t \cdot y^t - w^t \cdot x^t` is observed profit.

**Weak Axiom of Profit Maximization:**

.. math::

   p^s \cdot y^s - w^s \cdot x^s \geq p^s \cdot y^t - w^s \cdot x^t \implies
   p^t \cdot y^t - w^t \cdot x^t \geq p^t \cdot y^s - w^t \cdot x^s

Cost Minimization
-----------------

**Function:** ``check_cost_minimization(log)``

For a given output level :math:`y`, cost minimization requires:

.. math::

   w^t \cdot x^t \leq w^t \cdot x^s \text{ whenever } f(x^t) = f(x^s) = y

**GARP for Cost:**

.. math::

   w^s \cdot x^s \leq w^s \cdot x^t \implies w^t \cdot x^t \leq w^t \cdot x^s

Returns to Scale
----------------

**Function:** ``estimate_returns_to_scale(log)``

Estimate the production technology's returns to scale:

.. math::

   \text{RTS} = \frac{\partial \log y}{\partial \log x}

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - RTS
     - Interpretation
   * - > 1
     - Increasing returns to scale (economies of scale)
   * - = 1
     - Constant returns to scale
   * - < 1
     - Decreasing returns to scale (diseconomies of scale)

**Reference:** Varian (1984), Chambers & Echenique (2016) Ch. 15
