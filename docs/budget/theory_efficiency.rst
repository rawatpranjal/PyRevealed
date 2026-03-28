Efficiency and Power Indices
============================

In empirical settings where axiomatic consistency (e.g., GARP) is violated, efficiency indices provide a continuous measure to quantify the severity of departures from rationality and the statistical power of the associated tests.

Critical Cost Efficiency Index (CCEI)
-------------------------------------

**Reference Implementation:** ``compute_integrity_score(log)``

The Critical Cost Efficiency Index (CCEI), often referred to as the Afriat Efficiency Index (AEI), quantifies the minimal adjustment required to the agent's budget sets to eliminate all axiomatic violations.

**Formal Definition:**

.. math::

   \text{CCEI} = \sup \left\{ e \in [0,1] : \text{GARP is satisfied with } e \cdot p^i \cdot x^i \geq p^i \cdot x^j \right\}

**Computational Methodology:**

The CCEI is determined via binary search over the efficiency parameter :math:`e`. For a given :math:`e`, the modified revealed preference relation :math:`R_e` is defined as:

.. math::

   x^i \, R_e \, x^j \iff e \cdot (p^i \cdot x^i) \geq p^i \cdot x^j

**Interpretation and Benchmarks:**

The CCEI represents the fraction of wealth the agent "wastes" through inconsistent choices.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - CCEI Value
     - Behavioral Interpretation
   * - 1.00
     - Perfect consistency; behavior is fully reconcilable with utility maximization.
   * - [0.95, 1.00)
     - Near-rational behavior; deviations are often attributed to minor optimization errors or measurement noise.
   * - [0.85, 0.95)
     - Moderate inconsistency; presence of significant preference cycles.
   * - < 0.85
     - Substantial departures from the hypothesis of utility maximization.

.. note::

   **Empirical Context:** Choi et al. (2014) report a mean CCEI of approximately 0.88 in large-scale laboratory experiments. In high-frequency e-commerce data, CCEI values may be lower due to unobserved factors and higher levels of stochastic noise.

**References:** Afriat (1972), Varian (1990, 1991).


Money Pump Index (MPI)
----------------------

**Reference Implementation:** ``compute_confusion_metric(log)``

The Money Pump Index (MPI) provides an alternative measure of inconsistency by quantifying the maximal economic loss an agent could incur by being subjected to a sequence of trades corresponding to their revealed preference cycles.

**Formal Definition:**

For a detected violation cycle :math:`k_1 \to k_2 \to \cdots \to k_m \to k_1`:

.. math::

   \text{MPI} = \frac{\sum_{i=1}^{m} \max(0, p^{k_i} \cdot x^{k_i} - p^{k_i} \cdot x^{k_{i+1}})}{\sum_{i=1}^{m} p^{k_i} \cdot x^{k_i}}

**Interpretation:**

The MPI measures the "exploitability" of the agent. A higher MPI indicates that the agent's inconsistencies could lead to substantial wealth extraction in a market environment.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - MPI Value
     - Interpretation
   * - 0.00
     - No preference cycles detected; zero exploitability.
   * - (0.00, 0.10]
     - Marginal exploitability; minor transitivity violations.
   * - (0.10, 0.30]
     - Significant exploitability; noticeable behavioral contradictions.
   * - > 0.30
     - Severe exploitability; fundamental failure of transitive preferences.

**Reference:** Echenique, Lee, & Shum (2011).


Houtman-Maks Index (HM)
-----------------------

**Reference Implementation:** ``compute_minimal_outlier_fraction(log)``

The Houtman-Maks Index identifies the maximal subset of observations that are mutually consistent with GARP. It is typically expressed as the fraction of observations that must be discarded to achieve rationalizability.

**Formal Definition:**

.. math::

   \text{HM} = \min \left\{ \frac{|S|}{T} : \text{The subset of observations } \{1, \ldots, T\} \setminus S \text{ satisfies GARP} \right\}

This index is particularly useful for identifying agents who are "mostly" rational but exhibit a small number of anomalous choices.

**Reference:** Houtman & Maks (1985).


Granular Efficiency (Varian's Index)
------------------------------------

**Reference Implementation:** ``compute_granular_integrity(log)``

Varian's Index provides observation-specific efficiency scores by solving a constrained optimization problem. It identifies the minimal perturbation to each individual budget set required to satisfy GARP.

**Optimization Problem:**

.. math::

   \min \sum_{i=1}^{T} (1 - e_i)

Subject to:

.. math::

   e_i \cdot (p^i \cdot x^i) \geq p^i \cdot x^j \quad \forall \, (i,j) : x^i \, R^* \, x^j

.. math::

   0 \leq e_i \leq 1

This granular approach allows analysts to pinpoint specific temporal periods or choice environments where the agent's behavior deviates from the model.

**Reference:** Varian (1990).


Statistical Test Power (Bronars' Index)
---------------------------------------

**Reference Implementation:** ``compute_test_power(log)``

The validity of any revealed preference test depends on its ability to reject the null hypothesis of random behavior. Bronars' Index quantifies the statistical power of the test given the observed budget sets.

**Methodology:**

Power is estimated by simulating a cohort of "synthetic agents" who make random choices (typically drawn from a uniform distribution on the budget hyperplane) and calculating the frequency with which these random choices violate GARP.

.. math::

   \text{Power} = \mathbb{P}(\text{Random Behavior Violates GARP} \mid \text{Observed Budget Sets})

**Interpretation:**

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Power Value
     - Statistical Interpretation
   * - > 0.90
     - High Power: The observed budget sets provide a rigorous test; passing GARP is highly informative.
   * - [0.70, 0.90]
     - Sufficient Power: The test is capable of discriminating between rational and random behavior.
   * - < 0.50
     - Low Power: The budget sets do not overlap sufficiently; passing GARP may be a trivial result of the data structure rather than a reflection of agent rationality.

.. warning::

   A high CCEI (or AEI) observed in a low-power environment must be interpreted with caution, as it may be a spurious artifact of non-overlapping budget sets.

**Reference:** Bronars (1987).


Observation Graph Network Features
-----------------------------------

**Requires:** ``Engine(metrics=[..., "network"])``

The **observation graph** :math:`G = (V, E)` has one node per observation and a directed edge :math:`i \to j` whenever bundle :math:`j` was within the budget set at observation :math:`i` but was not selected (:math:`x^i \, R \, x^j`). The following features characterize the topology and edge-weight distribution of this graph. They are empirically uncorrelated (max :math:`|r| < 0.3`) with CCEI, MPI, and VEI across multiple datasets, indicating that they capture distinct aspects of choice behavior.

**Graph Density** (``r_density``)

.. math::

   \rho = \frac{|\{(i,j) : x^i \, R \, x^j, \, i \neq j\}|}{T(T-1)}

The proportion of observation pairs for which a revealed preference relation exists. Higher density indicates greater overlap among budget sets, providing more pairwise comparisons for preference inference. Lower density reflects distinct price or income regimes across observations. This quantity is closely related to Bronars power: sparse graphs yield tests with limited discriminatory ability.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Value
     - Interpretation
   * - > 0.5
     - Most observation pairs are comparable; the data is informative for preference inference.
   * - [0.2, 0.5]
     - Moderate budget overlap. Typical of monthly aggregated transaction data.
   * - < 0.2
     - Limited budget overlap. Consistency tests have low statistical power.

**Out-Degree Dispersion** (``r_out_degree_std``)

Standard deviation of :math:`\text{deg}^+(i) = |\{j : x^i \, R \, x^j\}|` across observations. Quantifies heterogeneity in the number of alternatives each observation dominates. Elevated values indicate that certain observations were made under conditions (e.g., higher expenditure or stable prices) that rendered many alternative bundles affordable, while others occurred under more constrained conditions.

**Degree Gini** (``degree_gini``)

Gini coefficient of the total degree distribution :math:`\text{deg}(i) = \text{deg}^+(i) + \text{deg}^-(i)`. Quantifies the concentration of revealed preference information across observations. A high Gini coefficient indicates that a small number of observations account for a disproportionate share of the graph's connectivity.

**Edge-Weight Distribution** (``ew_mean``, ``ew_std``, ``ew_skew``)

Requires ``"harp"`` in metrics. For each edge :math:`(i,j)` in :math:`R`, the HARP log-ratio weight is:

.. math::

   w_{ij} = \ln \frac{p^i \cdot x^i}{p^i \cdot x^j}

This is the logarithm of actual expenditure at observation :math:`i` relative to the cost of bundle :math:`j` at observation :math:`i`'s prices. The distribution of these weights across all edges characterizes the agent's substitution and income effect patterns:

- ``ew_mean``: Mean edge weight. Negative values indicate a tendency to select bundles that are more expensive than available alternatives, consistent with quality differentiation or brand attachment. Values near zero indicate price-responsive substitution.
- ``ew_std``: Standard deviation of edge weights. Measures the heterogeneity of substitution patterns across observations. Empirically, this is the most orthogonal network feature relative to existing consistency scores. High values indicate variable price sensitivity; low values indicate uniform response to price changes.
- ``ew_skew``: Skewness of the edge-weight distribution. Positive skewness indicates a concentration of moderate weights with occasional large positive values (available but unchosen low-cost alternatives). Negative skewness indicates occasional observations with unusually high relative expenditure.

.. note::

   Edge-weight features are statistically independent of consistency scores. An agent may satisfy GARP (CCEI = 1.0) while exhibiting substantial variation in substitution patterns (high ``ew_std``), or conversely, display low ``ew_std`` despite GARP violations.
