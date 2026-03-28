# Tenrec Salvage Experiment: Results Report

**Date:** 2026-03-28
**Dataset:** QB-video.csv sample (500K rows, 3,120 users)
**Constructions tested:** 3 (1 category-run, 2 fixed-K variants)
**Acceptance threshold:** 4+ out of 5 criteria pass

---

## Constructions Tested

| Construction | Rule | Parameters |
|---|---|---|
| **A: Category-run micro-sessions** | Contiguous rows in same category; reset on category break or positive action | Max length: capped by action boundary |
| **B1: K=5 window before like** | Last 5 exposures before each like event | Fixed K=5 |
| **B2: K=10 window before like** | Last 10 exposures before each like event | Fixed K=10 |

---

## Results Table

| Construction | N Sessions | Median Size | p95 | p99 | Dup% | Same-Cat% | Chosen-Last% |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cat-run micro-sessions | 211,684 | 1.0 | 6.0 | 12.0 | 0.0% | 99.8% | 100.0% |
| K=5 before like | 8,285 | 5.0 | 5.0 | 5.0 | 0.0% | 22.1% | 100.0% |
| K=10 before like | 8,285 | 10.0 | 10.0 | 10.0 | 0.0% | 8.6% | 100.0% |

---

## Acceptance Criteria Evaluation

**Threshold:** Pass 4 out of 5 criteria.

### Criterion 1: Median unique items ≥ 4

| Construction | Value | Pass? |
|---|---:|---|
| Cat-run | 1.0 | ✗ FAIL |
| K=5 | 5.0 | ✓ PASS |
| K=10 | 10.0 | ✓ PASS |

**Finding:** Fixed-K constructions meet size threshold. Category-run is too granular (median 1).

### Criterion 2: Duplicate share < 15%

| Construction | Value | Pass? |
|---|---:|---|
| Cat-run | 0.0% | ✓ PASS |
| K=5 | 0.0% | ✓ PASS |
| K=10 | 0.0% | ✓ PASS |

**Finding:** All constructions have zero duplicate items within sessions. Clean.

### Criterion 3: Top-category share > 70% (same-category coherence)

| Construction | Value | Pass? |
|---|---:|---|
| Cat-run | 99.8% | ✓ PASS |
| K=5 | 22.1% | ✗ FAIL |
| K=10 | 8.6% | ✗ FAIL |

**Finding:** Category-run is highly coherent. Fixed-K constructions destroy category alignment (only 8-22% same-category).

### Criterion 4: Chosen-last share < 100%

| Construction | Value | Pass? |
|---|---:|---|
| Cat-run | 100.0% | ✗ FAIL |
| K=5 | 100.0% | ✗ FAIL |
| K=10 | 100.0% | ✗ FAIL |

**Finding:** **ALL THREE CONSTRUCTIONS HAVE CHOSEN-LAST = 100%.** The liked item is always the terminal event in the window.

**Why:** By construction definition, windows are built ending at a positive action (like). That action is mechanically last. No amount of window redefinition will change this.

### Criterion 5: N sessions > 100

| Construction | Value | Pass? |
|---|---:|---|
| Cat-run | 211,684 | ✓ PASS |
| K=5 | 8,285 | ✓ PASS |
| K=10 | 8,285 | ✓ PASS |

**Finding:** All constructions have sufficient volume.

---

## Scores

| Construction | Criteria Passing | Pass Rate | Threshold | Status |
|---|---:|---:|---|---|
| Cat-run micro-sessions | 3/5 | 60% | 4/5 (80%) | ✗ FAIL |
| K=5 before like | 3/5 | 60% | 4/5 (80%) | ✗ FAIL |
| K=10 before like | 3/5 | 60% | 4/5 (80%) | ✗ FAIL |

---

## Key Tensions

### Trade-off 1: Size vs. Category Coherence

**Cat-run:** Preserves category coherence (99.8%) but produces tiny median size (1.0)
- Most "sessions" are single-item exposures
- Economically indefensible as a choice set

**K-windows:** Achieve reasonable size (5 or 10) but destroy category coherence (8-22%)
- Mixing items across multiple categories
- Not a coherent consideration set

**Trade-off:** Bounded size and category coherence are in **direct conflict** under the sequential feed structure.

### Trade-off 2: Chosen-Item Position

**Finding:** 100% of all windows have the chosen (liked) item in the last position.

**Why:** Windows are defined ending at the positive-action event. There is no way to change this without abandoning the "like" as the choice signal itself.

**Economic interpretation:** This is **not** "the user chose from among alternatives." It is "the user stopped scrolling at this item." It measures stopping behavior, not preference.

---

## Why Salvage Failed

### Structural Issue: Sequential Feed, Terminal Choice

Tenrec data are inherently sequential:
1. User scrolls through items
2. User occasionally clicks (to see more detail)
3. User occasionally likes (positive feedback)
4. *The like is the terminal event in the sequence.*

When you define a "choice set" as items leading up to that like, the like is mechanically last. There is no way to construct a window where:
- The liked item is not last, AND
- The window is still anchored on the like event

You can trade off other properties (size, category coherence) but you cannot fix the chosen-last problem without:
- **Abandoning like/share/follow as the choice signal**, or
- **Defining sessions without anchoring on positive actions**

### Why Category-Run Doesn't Work

Category-run micro-sessions do achieve high same-category coherence (99.8%), but:

1. **Median session size = 1.0.** Most sessions have only 1 item. A one-item "choice set" is not a choice set; it's a trivial decision.

2. **Sessions are too small to sustain RP analysis.** You need 3+ observations per choice set, and sets need non-trivial size. Here, 50%+ of sessions are single-item.

### Why Fixed-K Doesn't Work

Fixed-K windows (K=5, K=10) do achieve reasonable size, but:

1. **Category coherence collapses (8-22%).** When you take 5 or 10 consecutive items from a sequential feed, they span multiple categories. This is the opposite of a curated menu of substitutes.

2. **Chosen-last = 100%.** The liked item is still always last. No preference signal.

---

## Conclusion

**All three salvage constructions fail to achieve acceptable local consideration sets.**

### The Core Problem

The like (positive feedback) is the only credible "choice signal" in Tenrec. But:
- Like events are terminal in the sequence
- Like events are not uniformly distributed (sparse: 1.1% of rows)
- When you build windows ending at a like, the like is mechanically last
- This confounds choice with sequential stopping

This is not an artifact of window definition. **It is structural to the data.**

### The Unresolvable Trade-off

You cannot simultaneously achieve:
1. Reasonable session size (≥4 items)
2. High category coherence (>70% same-category)
3. Non-terminal choice position (<100% last)

Category-run gets (2) but not (1) or (3).
Fixed-K gets (1) but not (2) or (3).
Nothing gets (3) — chosen-last is always 100%.

---

## Recommendation

**Stop attempting to salvage Tenrec as a menu-choice benchmark.**

### Path Forward: Two Options

#### Option A: Move Tenrec to Appendix

Reframe as:
- **"Sequential engagement consistency"** (not static revealed preference)
- Measure behavioral stability within category-coherent feed episodes
- Use simple features: like-rate by category, recency bias, repetition patterns
- Explicitly acknowledge: "This is not menu-choice RP; it's feed engagement."

#### Option B: Drop Tenrec Entirely

Rationale:
- No salvageable classical RP signal
- Appendix claim is weak without a strong alternative framing
- Your main benchmark (Instacart, Dunnhumby, REES46, H&M) is sufficient
- Avoid diluting the main narrative with a dataset that doesn't fit the framework

**Recommendation:** **Option A (appendix) is defensible, Option B (drop) is also reasonable.** Choose based on whether a sequential-engagement paper claim interests you.

---

## Data Artifact or Real Property?

**Could this be an artifact of the sample or the data release?**

No. The full file (2.44M rows) would have the same structure:
- No timestamps, so no true sessioning possible
- Feed is sequential by design
- Like/follow/share remain terminal actions
- Category coherence is an inherent property of the Tencent recommendation system

This is not a data-cleaning problem. It's structural to what Tenrec is: **a recommender-system engagement log, not a revealed-preference choice dataset.**

---

## Acceptance Criteria: Final Tally

| Criterion | Cat-run | K=5 | K=10 |
|---|---|---|---|
| Median size ≥ 4 | ✗ | ✓ | ✓ |
| Duplicates < 15% | ✓ | ✓ | ✓ |
| Same-category > 70% | ✓ | ✗ | ✗ |
| Chosen-last < 100% | ✗ | ✗ | ✗ |
| N sessions > 100 | ✓ | ✓ | ✓ |
| **Pass Rate** | 3/5 (60%) | 3/5 (60%) | 3/5 (60%) |
| **Threshold** | 4/5 (80%) | 4/5 (80%) | 4/5 (80%) |
| **Status** | ✗ FAIL | ✗ FAIL | ✗ FAIL |

**All three constructions fail.** No path to a credible classical RP interpretation.
