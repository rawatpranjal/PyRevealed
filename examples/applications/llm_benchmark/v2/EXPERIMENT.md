# LLM Benchmark v2 — Experiment Log

## Design

**Fix from v1**: Hold vignette constant, vary only menu. Per-vignette SARP.

- **Scenarios**: 5 (same as v1: support_ticket, alert_triage, content_review, job_screen, procurement)
- **Items**: 5 per scenario (same as v1)
- **Prompts**: 5 per scenario (same as v1: minimal, decision_tree, conservative, aggressive, chain_of_thought)
- **Model**: gpt-4o-mini only (v1 showed models indistinguishable on consistency)
- **Vignettes**: 10 per scenario, curated across 4 tiers (clear=3, binary=3, ambiguous=2, adversarial=2)
- **Menus per vignette**: 15 (C(5,2)=10 pairwise + 5 size-3 triples)

## Stages

### Stage 1: Deterministic Screen
- Temperature: 0.0
- Repetitions: 1 per (vignette, menu, prompt)
- Total calls: 5 × 5 × 10 × 15 = 3,750
- Tests: Per-vignette SARP, IIA violations
- Output: `data/responses/{scenario}__stage1.jsonl`

### Stage 2: Stochastic Deep Dive
- Temperature: 0.7
- Repetitions: K=20 per (vignette, menu, prompt)
- Scope: Only (vignette, prompt) combos that FAIL Stage 1 SARP
- Tests: RUM consistency, regularity, choice probability estimation
- Output: `data/responses/{scenario}__stage2.jsonl`

## Data Layout
```
v2/
  EXPERIMENT.md          ← this file
  data/
    vignettes/           ← 10 curated vignettes per scenario
      support_ticket.jsonl
      alert_triage.jsonl
      content_review.jsonl
      job_screen.jsonl
      procurement.jsonl
    responses/
      {scenario}__stage1.jsonl   ← deterministic responses
      {scenario}__stage2.jsonl   ← stochastic responses
    results/
      summary_v2.json            ← analysis output
```

## v1 Data (preserved, not overwritten)
All v1 data remains in `../data/` (the parent `llm_benchmark/data/` directory).
v1 experiment: 10,000 decisions, 5 scenarios × 5 prompts × 2 models × 200 trials.
