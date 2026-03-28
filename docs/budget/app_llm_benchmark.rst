LLM Consistency
===============

Do LLMs have stable action rankings, or does the ranking change depending
on which alternatives are shown? We construct item graphs from LLM
decisions and test for cycles.

.. code-block:: text

   Pipeline:
   ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
   │ Vignettes  │───▶│ Query LLM  │───▶│ Build pref │───▶│ Test SARP  │
   │ (10 per    │    │ (15 menus  │    │ graph per  │    │ + IIA per  │
   │ scenario)  │    │ per vig.)  │    │ vignette   │    │ vignette   │
   └────────────┘    └────────────┘    └────────────┘    └────────────┘

   5 scenarios × 5 prompts × 10 vignettes × 15 menus = 3,750 decisions (v2 det.)
   + 20 reps at temp=0.7 = 75,000 decisions (v2 stochastic)

Setup
-----

5 enterprise LLM deployment scenarios, each with 5 actions and 5 system
prompt strategies. gpt-4o-mini at temp=0 (deterministic) and temp=0.7
(stochastic, K=20 reps).

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Scenario
     - 5 Actions
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

Results
-------

For each vignette, we fix the input and vary only the menu. Preference
graph cycles are genuine — the LLM's ranking depends on which alternatives
are shown.

.. list-table:: SARP pass rate by scenario × prompt (% of 10 vignettes, deterministic)
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

.. list-table:: SARP pass rate by vignette difficulty tier
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

.. list-table:: IIA violations and deterministic/stochastic agreement
   :header-rows: 1
   :widths: 20 15 15 20

   * -
     - Det. IIA
     - Stoch. IIA
     - Det/Stoch agree
   * - Support
     - 3
     - 3
     - 98.2%
   * - Alert
     - 2
     - 3
     - 98.3%
   * - Content
     - 9
     - 8
     - 97.4%
   * - Jobs
     - 15
     - --
     - --
   * - Procurement
     - 8
     - --
     - --

*Det. IIA = cycles from temp=0 choices. Stoch. IIA = cycles from majority-vote
of K=20 reps at temp=0.7. Agreement = % of menus where both conditions pick
the same action. -- = in progress.*

.. list-table:: Stochastic SARP pass rate (majority-vote from K=20, temp=0.7)
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
     - 80
     - 80
     - 100
     - 100
     - 100
     - 92
   * - Alert
     - 80
     - 100
     - 90
     - 100
     - 90
     - 92
   * - Content
     - 88
     - 80
     - 62
     - 86
     - 75
     - 78
   * - Jobs
     - --
     - --
     - --
     - --
     - --
     - --
   * - Procurement
     - --
     - --
     - --
     - --
     - --
     - --

.. list-table:: % of menus with mixed responses (temp=0.7, K=20)
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
     - 4
     - 11
     - 11
     - 11
   * - Alert
     - 7
     - 12
     - 8
     - 9
     - 7
     - 8
   * - Content
     - 12
     - 8
     - 17
     - 3
     - 18
     - 12

Do Item Graphs Add Value?
--------------------------

Yes. Three findings that accuracy benchmarks do not capture:

1. **Decoy effects exist in LLMs.** Introducing a third alternative changes the
   ranking between two others in 15 instances for job screening. The item
   graph detects this as a cycle. Accuracy testing cannot.

2. **Consistency varies by scenario.** Alert triage (92%) vs job screening
   (74%). This ordering follows from item graph structure (ordinal
   actions produce fewer cycles), not task complexity.

3. **Prompt effects are scenario-dependent.** Decision-tree prompts are
   the most consistent on alert triage (100%) and least consistent on job screening
   (60%). Conservative is most consistent on support (100%) but least on content
   review (70%). Only per-vignette SARP testing reveals this.

**Stochastic confirms deterministic.** At temp=0.7 with K=20 reps,
97-98% of menus produce the same majority-vote choice as temp=0.
Stochastic SARP pass rates are similar to deterministic (support 92%
vs 88%, content 78% vs 82%). The preference graph structure is robust
to sampling noise — cycles are structural, not stochastic artifacts.

V1 Experiment (10,000 decisions)
---------------------------------

An initial experiment (v1) queried gpt-4o-mini and o4-mini across the same 5
scenarios and 5 prompts, collecting 10,000 total decisions (200 trials per
scenario × prompt × model group).

**Design:** Each trial used a *different* input vignette. SARP was tested
across 200 pooled trials per group.

**Results:**

- All 50 groups (5 scenarios × 5 prompts × 2 models) failed SARP.
- Item-level HM efficiency = 0.60 across all groups (3 of 5 items form the
  largest consistent subset).
- Observation-level HM efficiency ≈ 0.95 (95% of individual decisions are
  locally rationalizable).
- Permutation test p = 1.0 everywhere: violations are far fewer than random
  choice, indicating structured preferences that fall short of full transitivity.
- Both models were statistically indistinguishable on consistency.
- No prompt strategy achieved a SARP pass in any group.

**Confound:** Because the input vignette changed across trials, a reversal
(choosing A over B on vignette-1, then B over A on vignette-2) may reflect
correct context-dependent classification rather than genuine intransitivity.
V1 conflates two sources of variation: input change and menu change.

**What v1 established despite the confound:**

1. LLMs exhibit structured preferences (far better than random) but not
   perfect transitivity, even at temperature 0.
2. Prompt strategy has no detectable effect on consistency at the pooled level.
3. The two models are indistinguishable on rationality scores.

V2 corrected the design by fixing the vignette and varying only the menu,
enabling the per-vignette results reported above.

Reproduce
---------

.. code-block:: bash

   pip install prefgraph openai
   export OPENAI_API_KEY=your_key
   cd examples

   # v1 pooled (10,000 calls, ~$5) — different vignette per trial
   python -m applications.llm_benchmark.run_benchmark --all
   python -m applications.llm_benchmark.analyze --all

   # v2 deterministic (3,750 calls, ~$2) — fixed vignette, varied menu
   python -m applications.llm_benchmark.v2.generate_vignettes --all
   python -m applications.llm_benchmark.v2.run_benchmark --all --stage 1
   python -m applications.llm_benchmark.v2.analyze --all

   # v2 stochastic (75,000 calls, ~$40)
   python -m applications.llm_benchmark.v2.run_benchmark --all --stage 2 --k 20

Appendix
--------

Pipeline detail
~~~~~~~~~~~~~~~

.. code-block:: text

   1. VIGNETTES: 10 per scenario, curated across 4 tiers
      (clear, binary, ambiguous, adversarial). Generated by gpt-4o-mini.

   2. MENUS: For each vignette, present ALL C(5,2)=10 pairwise menus
      + 5 size-3 menus = 15 menus. Same vignette, different options shown.

   3. QUERY: For each (vignette, menu, prompt), call gpt-4o-mini.
      Deterministic: temp=0, 1 response.
      Stochastic: temp=0.7, K=20 responses.

   4. BUILD GRAPH: Each choice adds directed edges from chosen item
      to all unchosen items in the menu. One graph per (vignette, prompt).

   5. TEST: SARP on the item graph (is it acyclic?).
      IIA: compare pairwise choice in {A,B} vs A-vs-B in {A,B,C}.

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
   # result.is_consistent → bool
   # result.violations → list of cycles in item graph

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
     - % of vignettes where item graph is acyclic
   * - HM efficiency
     - 0--1
     - Fraction of items in largest acyclic subgraph
   * - IIA violations
     - 0--n
     - Third option flips a pairwise edge direction
   * - % mixed (stoch.)
     - 0--100%
     - % of menus with different choices across K reps

Limitations
~~~~~~~~~~~

No ground truth (consistency ≠ accuracy). Synthetic vignettes. Single
model family. Stochastic data still collecting for 2 scenarios.
