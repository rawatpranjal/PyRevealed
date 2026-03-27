Limited Attention
=================

Model choice under consideration set constraints.
Based on Chambers & Echenique (2016) Chapter 14.

.. note::

   Limited attention models explain apparent irrationality by assuming the decision-maker
   doesn't see all available options. A choice is "attention-rational" if it's optimal among
   the items actually considered.

Attention Filter Framework
--------------------------

**Function:** ``test_attention_rationality(log)``

A decision-maker has **limited attention** if they maximize utility over a consideration set:

.. math::

   c(A) = \arg\max_{x \in \Gamma(A)} u(x)

where :math:`\Gamma(A) \subseteq A` is the **consideration set** (items actually considered).

An **attention filter** :math:`\Gamma` maps menus to consideration sets with a key property:
removing items outside the consideration set doesn't change what's considered.

.. admonition:: Definition (Attention Filter)
   :class: important

   A map :math:`\Gamma` is an attention filter if for all menus :math:`A` and items :math:`x`:

   .. math::

      x \notin \Gamma(A) \implies \Gamma(A \setminus \{x\}) = \Gamma(A)

   This captures the intuition that unnoticed items can be removed without affecting consideration.


WARP(LA): WARP with Limited Attention
-------------------------------------

**Function:** ``test_warp_la(log)``

The WARP(LA) axiom (Masatlioglu, Nakajima & Ozbay, 2012) characterizes **Choice with Limited Attention (CLA)**.
It defines a revealed preference relation :math:`P` as:

.. math::

   x \, P \, y \iff \exists \text{ menu } T \text{ such that } c(T) = x \text{ and } c(T \setminus \{y\}) \neq x

In words: :math:`x` is revealed preferred to :math:`y` if removing :math:`y` from some menu
changes the choice away from :math:`x`.

.. admonition:: Theorem (Masatlioglu et al., 2012)
   :class: important

   A choice function :math:`c` is rationalizable by an attention filter and preference ordering
   if and only if the revealed preference relation :math:`P` is acyclic.

**Interpretation:** WARP(LA) is weaker than standard WARP. Data that violates WARP may still
satisfy WARP(LA) if the violations can be explained by attention effects.

Recovering Attention Filters
-----------------------------

**Function:** ``recover_preference_with_attention(log)``

When WARP(LA) is satisfied, we can construct an attention filter that rationalizes the data:

.. math::

   \Gamma(S) = \{c(S)\} \cup \{x \in S : c(S) \succ x \text{ in revealed preference}\}

This is the minimal consideration set needed to rationalize each choice.

Consideration Set Estimation
----------------------------

**Function:** ``estimate_consideration_sets(log)``

Estimate the consideration sets :math:`\Gamma(A)` that rationalize observed choices:

.. math::

   \Gamma^*(A) = \min \{ \Gamma : c(A) = \arg\max_{x \in \Gamma} u(x) \text{ for some } u \}

Salience Weights
----------------

**Function:** ``compute_salience_weights(log)``

Estimate how likely each item is to enter the consideration set:

.. math::

   \sigma_i = \Pr[i \in \Gamma(A) | i \in A]

Higher salience items are more likely to be considered.


Random Attention Model (RAM)
----------------------------

**Function:** ``fit_random_attention_model(log)``

The Random Attention Model (Cattaneo et al., 2020) extends attention theory to **stochastic choice**.
Instead of deterministic consideration, each item has an attention probability :math:`\mu_i`.

**Model:**

1. Each item :math:`i` is considered with probability :math:`\mu_i` (independently)
2. Consumer chooses the most preferred item among those considered
3. Choice probability depends on both preference rank and attention

.. math::

   P(\text{choose } x | S) = \sum_{\Gamma \subseteq S : x \in \Gamma} P(x \text{ is maximal in } \Gamma) \cdot P(\Gamma \text{ is considered})

.. admonition:: RAM Constraints
   :class: important

   Under RAM with preference :math:`\succ`, choice probabilities must satisfy:

   1. **Regularity bounds**: :math:`P(x|S) \leq P(x|T)` when :math:`T \subseteq S` and :math:`x \in T`
   2. **Monotonicity** (optional): :math:`\mu_i \geq \mu_j` if :math:`i \succ j`

RAM Assumptions
---------------

**Function:** ``fit_random_attention_model(log, assumption="...")``

Different assumptions on attention probabilities:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Assumption
     - Description
   * - ``"monotonic"``
     - Higher-ranked items have higher attention probability
   * - ``"independent"``
     - Attention probabilities are item-specific (no ranking constraint)
   * - ``"general"``
     - Minimal restrictions on attention

Attention Bounds
----------------

**Function:** ``compute_attention_bounds(log, preference, item, menu)``

RAM provides identified bounds on attention probabilities from choice data:

.. math::

   \underline{\mu}_i \leq \mu_i \leq \overline{\mu}_i

These bounds can be computed using linear programming.

**Reference:** Masatlioglu, Nakajima & Ozbay (2012), Cattaneo et al. (2020), Chambers & Echenique (2016) Ch. 14
