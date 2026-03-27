LLM Enterprise Consistency Benchmark
=====================================

Demonstrated application of PyRevealed's discrete choice axioms (SARP,
Houtman-Maks, IIA) to audit LLM decision-making across 5 enterprise
deployment scenarios. 13,750 API decisions, two experiment iterations.

What PyRevealed Reveals That Other Tools Don't
-----------------------------------------------

Standard LLM evaluation measures **accuracy** (% correct) and **latency**
(ms per call). Neither detects a failure mode that matters in production:
**menu-dependent preferences**.

An LLM might correctly handle 90% of support tickets. But when presented
with {escalate, auto-reply}, it picks escalate — and when presented with
{escalate, auto-reply, bug-ticket}, it switches to auto-reply. The third
option changed the ranking. No accuracy benchmark catches this.

PyRevealed's ``validate_menu_sarp()`` tests exactly this: given a set of
choices from varying menus, does a consistent ranking exist? If not,
``compute_menu_efficiency()`` quantifies how much of the behavior is
rationalizable. These are the same tools economists use to test whether
human consumers are rational (Varian 1982), now applied to LLM deployment.

**What you get that you can't get elsewhere:**

- **SARP pass rate**: Does the LLM have a stable action ranking for this
  input? Binary, per-vignette.
- **Houtman-Maks efficiency**: What fraction of actions can be consistently
  ranked? Tells you which items participate in cycles.
- **IIA violation detection**: Does adding a third option flip the preference
  between two others? The "decoy effect" — classic in behavioral economics,
  unmeasured in LLM evaluation until now.
- **Preference graph structure**: Which action pairs create cycles? Which
  items are in the same strongly connected component? Directly actionable
  for guardrail design.

Scenarios and Prompts
---------------------

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - Scenario
     - Input
     - 5 Actions
   * - Support Router
     - Customer ticket
     - auto-reply KB, bug ticket, billing, account mgr, escalate VP
   * - Alert Triage
     - Monitoring alert
     - auto-resolve, P3 ticket, page on-call, incident channel, runbook
   * - Content Review
     - Flagged post
     - approve, warning, hide, remove+strike, suspend+legal
   * - Job Screen
     - Resume + JD
     - reject, hold, phone screen, technical, fast-track
   * - Procurement
     - Purchase request
     - auto-approve, tag, request quotes, escalate, deny

5 prompt strategies per scenario: *minimal*, *decision tree*, *conservative*,
*aggressive*, *chain-of-thought*. Full production prompts (100--300 words).

Experiment v1: Pooled SARP (10,000 decisions)
----------------------------------------------

**Design**: 5 scenarios x 5 prompts x 2 models x 200 vignettes x 1 menu each.
Different vignette per trial. SARP tested across all 200 trials per group.

**Result**: 50/50 groups fail SARP. HM=0.60 everywhere. Zero variation.

**What we learned:**

- At temp=0, both gpt-4o-mini and o4-mini fail SARP on every scenario-prompt
  combination with maximum violations (10/10 pairwise cycles)
- Observation-level bootstrap HM is 0.95 [0.93, 0.97] — 95% of individual
  decisions are locally rationalizable
- Permutation p=1.000 everywhere — LLMs are far more consistent than random
- gpt-4o-mini and o4-mini are indistinguishable on consistency
- Prompts shift choice distributions (KL divergence 0.02--0.84) but never
  eliminate preference cycles
- Conservative prompts flip 6/10 pairwise preferences vs aggressive, but both
  produce 10 SARP violations

**Design flaw identified:** Each trial uses a different vignette. SARP
violations reflect the LLM correctly adapting to different inputs, not
actual menu-dependent inconsistency. EDA confirmed: 50% of vignettes
produce identical choices across all 5 prompts.

Experiment v2: Per-Vignette SARP (3,750 decisions)
----------------------------------------------------

**Design fix**: Hold vignette constant, vary only the menu. For each of 10
curated vignettes per scenario, present all C(5,2)=10 pairwise menus + 5
size-3 menus = 15 menus. Test SARP within each vignette.

**Scale**: 5 scenarios x 5 prompts x 10 vignettes x 15 menus = 3,750 calls.
gpt-4o-mini only, temp=0.

v2 Results
~~~~~~~~~~

SARP pass rates by scenario and prompt:

.. list-table::
   :header-rows: 1
   :widths: 22 13 13 13 13 13

   * - Scenario
     - Minimal
     - Dec. Tree
     - Conserv.
     - Aggress.
     - CoT
   * - Support Router
     - 90%
     - 80%
     - 100%
     - 80%
     - 90%
   * - Alert Triage
     - 80%
     - 100%
     - 90%
     - 100%
     - 90%
   * - Content Review
     - 90%
     - 80%
     - 70%
     - 90%
     - 80%
   * - Job Screen
     - 70%
     - 60%
     - 80%
     - 80%
     - 80%
   * - Procurement
     - 70%
     - 100%
     - 70%
     - 90%
     - 90%

*Percentage of 10 vignettes where SARP is satisfied (deterministic, temp=0).
Higher = more consistent for that prompt-scenario combination.*

SARP pass rates by vignette difficulty tier:

.. list-table::
   :header-rows: 1
   :widths: 22 16 16 16 16

   * - Scenario
     - Clear
     - Binary
     - Ambiguous
     - Adversarial
   * - Support Router
     - 87%
     - 93%
     - 90%
     - 80%
   * - Alert Triage
     - 93%
     - 100%
     - 90%
     - 80%
   * - Content Review
     - 60%
     - 80%
     - 100%
     - 100%
   * - Job Screen
     - 87%
     - 67%
     - 70%
     - 70%
   * - Procurement
     - 93%
     - 73%
     - 90%
     - 80%

*Averaged across 5 prompts per tier. Clear = unambiguous input, Binary = 2
actions compete, Ambiguous = 3+ plausible, Adversarial = designed to trigger
menu effects.*

IIA violations (menu-dependence):

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Scenario
     - IIA violations
     - Interpretation
   * - Support Router
     - 3
     - Adding a third option rarely changes pairwise preference
   * - Alert Triage
     - 2
     - Nearly menu-independent decisions
   * - Content Review
     - 9
     - Moderate menu effects — adding options shifts moderation decisions
   * - Job Screen
     - 15
     - Strongest menu effects — candidate ranking changes with alternatives shown
   * - Procurement
     - 8
     - Moderate menu effects on spending decisions

*IIA = Independence of Irrelevant Alternatives. A violation means adding a
third option to a pairwise menu changed which of two options is preferred.*

Suggestive Takeaways
~~~~~~~~~~~~~~~~~~~~

These results are from synthetic vignettes and a single model family.
They are illustrative, not definitive. But they suggest patterns that
would be invisible without revealed preference tools:

- **LLMs exhibit decoy effects.** Adding a third option to a menu changes
  which of two actions the LLM prefers — 15 times in job screening alone.
  This is the Independence of Irrelevant Alternatives (IIA) violation that
  behavioral economists study in human choice. No standard LLM benchmark
  measures this. ``validate_menu_sarp()`` catches it automatically.

- **"Clear" inputs aren't.** Content moderation vignettes designed to be
  unambiguous still produce menu-dependent decisions 40% of the time. The
  LLM's severity judgment shifts when you change which alternative actions
  are shown. A deployment team would never discover this with accuracy
  testing alone.

- **Some prompt strategies create more decision boundaries to trip over.**
  Decision-tree prompts score 60% SARP-consistent on job screening (vs 80%
  for simpler prompts). The explicit rules create edge cases where menu
  composition determines which rule fires. ``compute_menu_efficiency()``
  identifies exactly which action pairs participate in the resulting cycles.

- **Scenario difficulty varies in non-obvious ways.** Alert triage (92%) is
  easier than support routing (88%) which is easier than job screening (74%).
  This ranking doesn't follow from the scenarios' apparent complexity — it
  follows from the preference graph structure, which only SARP reveals.

- **The experimental design itself matters.** v1 (different input per trial)
  showed 0% SARP pass rate; v2 (same input, varied menus) showed 60--100%.
  The difference is entirely methodological. Revealed preference theory
  provides the framework to design the right test — pooled vs per-vignette
  SARP are different questions with different answers.

What This Means for Practitioners
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you deploy an LLM as a decision system (triage, moderation, screening):

- **Audit before deployment.** Run the per-vignette SARP test on your
  specific inputs. A 70% pass rate means 30% of inputs produce decisions
  that depend on which options you show — not just on the input itself.

- **Identify fragile action pairs.** SARP violations tell you exactly which
  pairs of actions form preference cycles. Add guardrails (confidence
  thresholds, human review) specifically for those pairs.

- **Test prompt candidates on consistency, not just accuracy.** Two prompts
  can have identical accuracy but different SARP pass rates. The one with
  higher consistency produces more predictable behavior in production.

- **Measure IIA before designing menus.** If your system shows the LLM
  different action sets in different contexts (e.g., filtering options by
  eligibility), IIA violations mean the filtered set changes the outcome
  — not just the available options.

Reproduce
---------

.. code-block:: bash

   pip install pyrevealed openai
   export OPENAI_API_KEY=your_key
   cd examples

   # v1 (pooled SARP, 10K decisions)
   python -m applications.llm_benchmark.generate_vignettes --all --trials 200
   python -m applications.llm_benchmark.run_benchmark --all --trials 200
   python -m applications.llm_benchmark.analyze --all

   # v2 (per-vignette SARP, 3.75K decisions)
   python -m applications.llm_benchmark.v2.generate_vignettes --all
   python -m applications.llm_benchmark.v2.run_benchmark --all --stage 1
   python -m applications.llm_benchmark.v2.analyze --all

All stages are resumable. v1 data in ``llm_benchmark/data/``,
v2 data in ``llm_benchmark/v2/data/``.

Appendix: Full Pipeline Documentation
--------------------------------------

Data Generation
~~~~~~~~~~~~~~~

.. code-block:: text

   ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
   │ Vignette Gen      │     │ Menu Gen          │     │ LLM Querying     │
   │                   │     │                   │     │                  │
   │ v1: o4-mini, 200  │     │ v1: 1 random menu │     │ For each         │
   │ random per scen.  │────▶│ per vignette      │────▶│ (vig, menu,      │
   │                   │     │                   │     │  prompt, model): │
   │ v2: gpt-4o-mini,  │     │ v2: ALL 15 menus  │     │  → OpenAI API    │
   │ 10 curated tiers  │     │ per vignette      │     │  → parse choice  │
   │                   │     │ (10 pair + 5 tri) │     │  → append JSONL  │
   └──────────────────┘     └──────────────────┘     └──────────────────┘

Feature Extraction
~~~~~~~~~~~~~~~~~~

Each ``(vignette, prompt)`` group produces a ``MenuChoiceLog``:

.. code-block:: python

   from pyrevealed import MenuChoiceLog
   from pyrevealed.algorithms.abstract_choice import validate_menu_sarp

   log = MenuChoiceLog(
       menus=[frozenset(r["menu"]) for r in records],
       choices=[r["choice"] for r in records],
       item_labels=["auto_reply_kb", "create_bug_ticket", ...],
   )
   result = validate_menu_sarp(log)
   # result.is_consistent, result.violations, result.transitive_closure

Analysis Pipeline
~~~~~~~~~~~~~~~~~

.. code-block:: text

   MenuChoiceLog
       │
       ├─── validate_menu_sarp() ─── SARP pass/fail + cycle list
       ├─── compute_menu_efficiency() ─── HM efficiency (0-1)
       └─── IIA detection ─── compare pairwise vs triple menu choices

Metrics Reference
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Metric
     - Range
     - Interpretation
   * - SARP pass rate
     - [0%, 100%]
     - Fraction of vignettes with transitive action ranking (v2)
   * - HM efficiency
     - [0, 1]
     - Fraction of items in largest consistent subset
   * - IIA violations
     - [0, n_triples]
     - Cases where adding a third option flips a pairwise preference
   * - Max SCC
     - [1, n]
     - Largest strongly connected component. 1=acyclic, n=all items in one cycle

Limitations
~~~~~~~~~~~

1. **No ground truth**: We measure consistency, not accuracy. A perfectly
   consistent but wrong system would score well.

2. **5 synthetic scenarios**: Results may not generalize to all LLM deployments.

3. **Deterministic only (v2 Stage 1)**: SARP assumes deterministic choice.
   Stochastic testing (RUM, K=20 repetitions at temp=0.7) planned for Stage 2.

4. **Single model**: v2 tests gpt-4o-mini only. v1 showed o4-mini is
   indistinguishable, but other model families (Claude, Gemini) untested.
