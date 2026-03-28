LLM Consistency
===============

Do LLMs have stable action rankings, or does the ranking change when
different alternatives are shown? We build preference graphs from LLM
decisions and check for cycles.

.. code-block:: text

   ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
   │ Vignettes  │───▶│ Query LLM  │───▶│ Build pref │───▶│ Test SARP  │
   │ (10 per    │    │ (15 menus  │    │ graph per  │    │ + IIA per  │
   │ scenario)  │    │ per vig.)  │    │ vignette   │    │ vignette   │
   └────────────┘    └────────────┘    └────────────┘    └────────────┘

Setup
-----

5 enterprise scenarios, 5 actions each, 5 system prompts. gpt-4o-mini.
For each vignette: fix the input, present all C(5,2)=10 pairwise menus
+ 5 size-3 menus. Deterministic (temp=0) and stochastic (temp=0.7, K=20).

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Scenario
     - Actions
   * - Support
     - auto-reply KB, bug ticket, billing, account mgr, escalate VP
   * - Alert
     - auto-resolve, P3 ticket, page on-call, incident channel, runbook
   * - Content
     - approve, warning, hide, remove+strike, suspend+legal
   * - Jobs
     - reject, hold, phone screen, technical, fast-track
   * - Procurement
     - auto-approve, tag, request quotes, escalate, deny

Prompts: *minimal, decision-tree, conservative, aggressive, chain-of-thought.*

Deterministic Results (temp=0)
------------------------------

Each cell = % of 10 vignettes where the preference graph is acyclic.

.. list-table:: SARP pass rate by prompt
   :header-rows: 1
   :widths: 18 13 13 13 13 13 13

   * -
     - Min
     - DecTree
     - Conserv
     - Aggress
     - CoT
     - Mean
   * - Support
     - 90
     - 80
     - 100
     - 80
     - 90
     - 88
   * - Alert
     - 80
     - 100
     - 90
     - 100
     - 90
     - 92
   * - Content
     - 90
     - 80
     - 70
     - 90
     - 80
     - 82
   * - Jobs
     - 70
     - 60
     - 80
     - 80
     - 80
     - 74
   * - Procurement
     - 70
     - 100
     - 70
     - 90
     - 90
     - 84

.. list-table:: SARP pass rate by vignette difficulty
   :header-rows: 1
   :widths: 18 16 16 16 16 16

   * -
     - Clear
     - Binary
     - Ambig.
     - Advers.
     - Mean
   * - Support
     - 87
     - 93
     - 90
     - 80
     - 88
   * - Alert
     - 93
     - 100
     - 90
     - 80
     - 92
   * - Content
     - 60
     - 80
     - 100
     - 100
     - 82
   * - Jobs
     - 87
     - 67
     - 70
     - 70
     - 74
   * - Procurement
     - 93
     - 73
     - 90
     - 80
     - 84

.. list-table:: IIA violations
   :header-rows: 1
   :widths: 20 15 15 20

   * -
     - Deterministic
     - Stochastic
     - Agreement
   * - Support
     - 3
     - 3
     - 95.8%
   * - Alert
     - 2
     - 3
     - 96.6%
   * - Content
     - 9
     - 12
     - 95.5%
   * - Jobs
     - 15
     - 14
     - 97.6%
   * - Procurement
     - 8
     - --
     - --

*IIA violation = adding a third option flips the pairwise preference.
Stochastic = majority-vote from K=20 reps. Agreement = % of menus
where temp=0 and temp=0.7 majority match. -- = procurement in progress.*

Stochastic Results (temp=0.7, K=20)
------------------------------------

.. list-table:: SARP pass rate (majority-vote)
   :header-rows: 1
   :widths: 18 13 13 13 13 13 13

   * -
     - Min
     - DecTree
     - Conserv
     - Aggress
     - CoT
     - Mean
   * - Support
     - 90
     - 80
     - 90
     - 90
     - 100
     - 90
   * - Alert
     - 80
     - 90
     - 90
     - 100
     - 90
     - 90
   * - Content
     - 90
     - 80
     - 60
     - 80
     - 70
     - 76
   * - Jobs
     - 80
     - 60
     - 80
     - 80
     - 90
     - 78

.. list-table:: % of menus with mixed responses
   :header-rows: 1
   :widths: 18 13 13 13 13 13 13

   * -
     - Min
     - DecTree
     - Conserv
     - Aggress
     - CoT
     - All
   * - Support
     - 8
     - 20
     - 5
     - 11
     - 11
     - 11
   * - Alert
     - 8
     - 11
     - 7
     - 9
     - 7
     - 8
   * - Content
     - 14
     - 7
     - 16
     - 5
     - 17
     - 12
   * - Jobs
     - 9
     - 10
     - 9
     - 5
     - 9
     - 8

Findings
--------

- **Job screening is the least consistent scenario.** 74% deterministic
  SARP pass, 78% stochastic. 15 deterministic + 14 stochastic IIA violations.
  Adding a third candidate changes which of two is preferred.
- **Content moderation "clear" vignettes pass only 47% stochastically.**
  Even unambiguous posts produce menu-dependent severity judgments.
  Stochastic IIA violations jump from 8 (old) to 12 — majority-voting
  doesn't eliminate context effects.
- **Decision-tree prompts score 60% on jobs but 90% on alert.** Conservative
  scores 90% on support, 60% on content. Chain-of-thought is the only
  prompt that hits 100% on any scenario (support). No universal best prompt.
- **Alert triage is the most consistent** (90% stochastic SARP). Actions
  have a clear ordinal severity structure that resists menu effects.
- **Stochastic sampling barely changes rankings.** 92-98% of menus agree
  between temp=0 and temp=0.7 majority vote. Only 8-12% of menus produce
  mixed responses across 20 reps. Inconsistency is structural, not noise.

Reproduce
---------

.. code-block:: bash

   pip install prefgraph openai
   export OPENAI_API_KEY=your_key
   cd examples

   # Deterministic (3,750 calls)
   python -m applications.llm_benchmark.v2.generate_vignettes --all
   python -m applications.llm_benchmark.v2.run_benchmark --all --stage 1
   python -m applications.llm_benchmark.v2.analyze --all

   # Stochastic (75,000 calls)
   python -m applications.llm_benchmark.v2.run_benchmark --all --stage 2 --k 20
   python -m applications.llm_benchmark.v2.analyze --all --stage 2

Appendix
--------

Pipeline
~~~~~~~~

.. code-block:: text

   1. VIGNETTES: 10 per scenario, 4 tiers (clear, binary, ambiguous,
      adversarial). Generated by gpt-4o-mini.

   2. MENUS: C(5,2)=10 pairwise + 5 size-3 = 15 menus per vignette.
      Same input, different options shown.

   3. QUERY: gpt-4o-mini. temp=0 (1 response) or temp=0.7 (K=20).

   4. GRAPH: Each choice adds edges: chosen → each unchosen item.

   5. TEST: SARP (acyclic?). IIA (pairwise vs triple comparison).

Code
~~~~

.. code-block:: python

   from prefgraph import MenuChoiceLog
   from prefgraph.algorithms.abstract_choice import validate_menu_sarp

   log = MenuChoiceLog(
       menus=[frozenset(r["menu"]) for r in records],
       choices=[r["choice"] for r in records],
   )
   result = validate_menu_sarp(log)
   # result.is_consistent, result.violations

Metrics
~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Metric
     - Range
     - Meaning
   * - SARP pass rate
     - 0--100%
     - % of vignettes with acyclic preference graph
   * - HM efficiency
     - 0--1
     - Fraction of items in largest acyclic subgraph
   * - IIA violations
     - 0--n
     - Adding a third option flips a pairwise preference
   * - % mixed
     - 0--100%
     - % of menus with different choices across K reps

Limitations
~~~~~~~~~~~

No ground truth (consistency ≠ accuracy). Synthetic vignettes. Single
model family. Procurement stochastic data still collecting.
