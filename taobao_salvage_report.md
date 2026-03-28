# Taobao Salvage Audit Report

**Script**: `examples/eda/taobao_salvage_audit.py`
**Sample**: First 20M of 100M rows (20%)
**Date**: 2026-03-28

---

## Q2. Timestamp corruption - manageable

| Metric | Value |
|--------|-------|
| Total events | 20,000,000 |
| Bad timestamps (outside Nov–Dec 2017 window) | 435 (0.002%) |
| Users affected | 77 of 198,002 (0.04%) |
| p100 gap before filtering | 41,042 days |
| p100 gap after filtering | 29.0 days |

Corruption is concentrated in 435 events across 77 users. Dropping corrupted events (not entire users) removes the problem. After filtering, the worst gap is 29 days - unusual but not structurally broken. Sessionization is sound after this filter.

**Decision**: Drop 435 bad events. Continue.

---

## Q1. Clean pre-buy choice occasions - collapse is severe

### Session funnel (20M row sample, post timestamp filter)

| Stage | Sessions | % of total |
|-------|---------|-----------|
| Total sessions (30-min gap) | 3,209,439 | 100% |
| After ≥1 buy | 310,515 | 9.7% |
| After ≥1 pre-buy pv | 188,911 | 5.9% |
| After buy item was viewed before buy | 89,868 | 2.8% |
| After menu size [2, 20] | **83,120** | **2.6%** |

### User survival

| Threshold | Users | % of 198K sample | vs loose loader (7,513) |
|-----------|-------|-----------------|------------------------|
| ≥1 clean session | 61,214 | 30.9% | - |
| ≥3 clean sessions | 4,095 | 2.1% | 54.5% |
| ≥5 clean sessions | **376** | **0.2%** | **5.0%** |

Strict rules reduce the benchmark population from ~7,513 (loose, 20M sample) to **376 users** with ≥5 sessions. At 20% sampling rate, the full dataset would yield approximately **1,880 strict-clean users** vs the 4,239 reported by the current loose loader.

### Why the "viewed-and-bought" rule is so costly

The single most destructive rule is requiring the bought item to have been viewed before purchase:
- Eliminates 99,043 sessions (3.1% of total, 52% of sessions with a buy AND pre-buy views)
- This reflects the reality documented in D5: 39.3% of purchase sessions have phantom buy insertion, and 26.9% have no views at all

The "pre-buy only" rule (cutting post-purchase views) is cheap by comparison - it mainly removes sessions where the menu inflated but the choice itself was observed.

---

## Q3. Viability of strict-clean sample - too thin for RP

### Session depth per user

| Percentile | Sessions per user |
|------------|-----------------|
| p10 | 1 |
| p25 | 1 |
| p50 | **1** |
| p75 | 2 |
| p90 | 2 |
| p95 | 3 |
| p99 | 4 |

Median sessions per user = 1. This is fatal for preference graph construction.

- A user with 1 session has 0 pairwise comparisons.
- A user with 5 sessions has C(5,2)=10 pairwise comparisons - the minimum for sparse RP testing.
- Mean pairwise comparisons per user in the clean sample: **0.5** (i.e., most users have 0).

### Qualification breakdown

| Threshold | Users | % of strict-clean users |
|-----------|-------|------------------------|
| ≥1 session | 61,214 | 100% |
| ≥2 sessions | 15,881 | 25.9% |
| ≥3 sessions | 4,095 | 6.7% |
| ≥5 sessions | 376 | **0.6%** |
| ≥10 sessions | 21 | 0.03% |

Only 0.6% of users with any clean session reach the 5-session threshold. The cleaned distribution is far more skewed than the loose-loader distribution (where p95 = 5 sessions per user).

### Menu size (strict)

Median: 5 items. p75: 8 items. Menu size is actually healthy - the filtering has selected sessions with genuine pre-buy consideration. The problem is not menu size, it is session depth.

---

## Decision

### What the three questions reveal together

| Question | Answer | Implication |
|----------|--------|-------------|
| Q2: Timestamps | 0.002% corrupted; 77 users; clean after filter | Non-issue |
| Q1: Clean sessions | 83K sessions, but only **376 users with ≥5** | Sample collapses |
| Q3: RP viability | Median sessions per user = 1; mean pairwise = 0.5 | No RP graph |

### Verdict: DROP from main benchmark

The strict-clean construction leaves too little per-user data for revealed preference testing:
- The RP preference graph for a user requires repeated choices from overlapping menus. With median 1 session per user, there are no overlapping menus.
- Relaxing to ≥3 sessions gives 4,095 users, but at 3 sessions the graphs are trivially sparse (3 pairwise comparisons max).
- The current reported benchmark results (4,239 users, AUC 0.913/0.915/0.925) were produced with the loose loader, where 66% of "choices" are phantom insertions or direct purchases with no observed menu.

### Possible salvage paths

1. **Lower min_sessions to 3 and accept thin graphs.** This gives ~4,095 users in the 20M sample (~20K in full dataset). RP features will be computed on 1–3 session graphs. The structure may still carry signal (the current AUC suggests it does), but the theoretical justification is weaker.

2. **Document phantom insertion explicitly and keep the loose loader.** The benchmark results are empirically reproducible. The caveat is that the "choice" in 66% of sessions is a direct purchase or unobserved consideration, and the RP features (SARP violations, HM score) measure structure in the augmented menus, not pure revealed preference. This is a different but not necessarily invalid estimand.

3. **Drop Taobao and replace with a dataset where session boundaries are server-defined and purchase follows from a genuinely observed menu.** REES46 is an example where session IDs come from the server and menus are the clicked items.

The current benchmark entry should either be removed or labeled with explicit caveats about the phantom insertion rate and the non-strict menu construction.
