# Tenrec Decision: Salvage Attempt Failed — Proceed to Path B or Drop

**Status:** Decision point reached. Salvage attempt 3 constructions. All failed acceptance threshold.

**Recommendation:** Move to Appendix (Path B) OR drop entirely (Path A).

---

## Summary of Evidence

### EDA Results (tenrec_eda_facts.md)

**Sample:** 500K rows, 3,120 users, 81K items

**Diagnostic 2 (Window-Size Distribution):**
| Rule | Median | p90 | p99 | Size-1% | Chosen-Last% |
|---|---:|---:|---:|---:|---:|
| Exposures since last like | 2.0 | 47.0 | 372.3 | 37.9% | 100.0% |
| Clicks since last like | 2.0 | 26.0 | 198.7 | 45.6% | 100.0% |
| Pos. action | 3.0 | 57.0 | 360.2 | 35.7% | 100.0% |
| Last 5 before like | 5.0 | 5.0 | 5.0 | 0.4% | 100.0% |
| Last 10 before like | 10.0 | 10.0 | 10.0 | 0.4% | 100.0% |

**Diagnostic 3 (Category Coherence):**
- Median unique categories per window: 1.0
- Top-category share: 85.9%
- Windows are single-category or dual-category

**Diagnostic 4 (Position of Chosen Item):**
- % chosen-item always last: 100.0%
- Normalized position distribution: 100% in [0.8–1.0] bucket

---

### Salvage Experiment Results (tenrec_salvage_facts.md)

**Three constructions tested:**

| Construction | N Sessions | Median Size | Same-Cat% | Chosen-Last% |
|---|---:|---:|---:|---:|
| Cat-run micro-sessions | 211,684 | 1.0 | 99.8% | 100.0% |
| K=5 before like | 8,285 | 5.0 | 22.1% | 100.0% |
| K=10 before like | 8,285 | 10.0 | 8.6% | 100.0% |

**Acceptance Criteria (4+ of 5 must pass):**

| Criterion | Cat-run | K=5 | K=10 | Threshold |
|---|---|---|---|---|
| Median size ≥ 4 | ✗ | ✓ | ✓ | Pass |
| Duplicates < 15% | ✓ | ✓ | ✓ | Pass |
| Same-category > 70% | ✓ | ✗ | ✗ | Fail |
| Chosen-last < 100% | ✗ | ✗ | ✗ | **Fail (All)** |
| N sessions > 100 | ✓ | ✓ | ✓ | Pass |
| **Score** | 3/5 | 3/5 | 3/5 | **4/5 required** |

**Verdict:** All three constructions score 3/5 (60%), below 4/5 (80%) threshold.

---

## Why Salvage Failed

### Problem 1: Chosen-Last = 100% (Structural, Not Fixable)

**Fact:** Every construction has chosen-item-last = 100%.

**Why:** Windows are defined ending at a like (the only usable choice signal). By definition, the like is terminal. You cannot build a window ending at an event and have that event be non-terminal.

**Economic interpretation:** Chosen-last = 100% means "the liked item is always the final exposure the user saw." This is sequential stopping behavior, not preference from a set.

**Fixability:** Cannot be fixed without abandoning "like" as the choice signal. Abandoning "like" removes the only credible choice indicator in the data.

### Problem 2: Size vs. Category Coherence Trade-off

**Fact 1:** Category-run achieves 99.8% same-category but median size 1.0 (too small).
**Fact 2:** K=5 achieves size 5.0 but only 22.1% same-category.
**Fact 3:** K=10 achieves size 10.0 but only 8.6% same-category.

**Why:** Tenrec is a sequential feed. Items naturally cluster by category (user scrolls through Comedy, then Food, then Travel). But:
- To preserve categories, you must stop at category boundaries → tiny sessions (median 1)
- To get reasonable size, you must cross categories → lose category coherence

**Trade-off is mathematically inevitable** under the sequential feed structure.

### Problem 3: Insufficient Clean Windows

- Cat-run: 211K sessions, but 50%+ are size-1 (trivial)
- K-windows: 8.3K windows (sparse after user filtering)

Only the K-window constructions produce enough non-degenerate sessions, but they destroy category coherence in the process.

---

## Why This Cannot Be Salvaged as Classical RP

**Fact:** In revealed preference analysis, a "choice" must:
1. Be from a **non-trivial set** (at least 2–3 alternatives)
2. Be **cross-sectional** (simultaneous decision)
3. Have **variation in the chosen item's position** (not always last)

**Tenrec fails all three:**
1. ✗ Most windows are size 1 (trivial) or cross-category (not coherent)
2. ✗ Data are sequential (user scrolls), not simultaneous (user chooses from a display)
3. ✗ Chosen item is always last (100%), never first or middle

---

## Decision Tree

**Question 1:** Can we build windows with median size ≥ 4?
- **Answer:** Yes, but only by mixing categories (K-windows: 22% same-cat, down to 8% at K=10)

**Question 2:** Can we preserve category coherence (>70%)?
- **Answer:** Yes, but only with tiny windows (cat-run: median size 1.0)

**Question 3:** Can we fix chosen-item-last = 100%?
- **Answer:** No. This is structural to the data. Like events are terminal.

**Conclusion:** No construction passes all three requirements. Salvage fails.

---

## Path Forward: Two Options

### Option A: Move to Appendix (Path B — Exploratory)

**Claim:** Sequential exposure consistency, not static RP.

**Implementation:**
1. Use K=5 or cat-run construction (pick one)
2. Extract simple behavioral features:
   - Like-rate by category context
   - Recency bias (does user like newer items?)
   - Repetition vs novelty
   - Category transition stability
   - Engagement entropy
3. Test if features predict engagement (not using SARP/WARP)
4. Frame as: "Can revealed preference features predict future engagement in sequential recommendation contexts?"

**Tradeoff:** Weaker claim, but honest. Avoids forcing RP axioms onto feed data.

**Effort:** Low. Reuse existing feature extractors, different target variable.

### Option B: Drop Entirely (Path A — Conservative)

**Rationale:**
- Tenrec is fundamentally a recommender-system engagement log
- No credible way to extract menu-choice observations
- Main benchmark (Instacart, Dunnhumby, REES46, H&M) is sufficient for the paper's core claims
- Appendix claim is weak without strong alternative framing

**Tradeoff:** Simplify the narrative. Focus on where you have strong data.

**Effort:** Minimal. Remove tenrec_bench.py from runner.

---

## Recommendation

**Based on the salvage failure and structural analysis:**

### Primary: **Option B — Drop Tenrec**

**Why:**
1. All salvage constructions fail acceptance threshold
2. Chosen-last = 100% is structural, not fixable
3. Core issue is not data quality, but fundamental mismatch between feed structure and menu-choice model
4. You have strong benchmarks already (Instacart aisle-level single-reorder, H&M, Dunnhumby, REES46)
5. Better to have 4 strong benchmarks than 5 mediocre ones

### Secondary: **Option A — If you want a sequential-engagement claim**

Use this only if:
- You are interested in proving RP features help in sequential contexts
- You can reframe to "engagement consistency" not "menu choice"
- You are willing to add a full appendix section on the limitations

---

## Files Produced

**Diagnostic reports:**
- `tenrec_eda_facts.md` — All 6 diagnostics, raw numbers, code queries
- `tenrec_eda_summary.md` — Diagnostic interpretation
- `tenrec_salvage_facts.md` — Salvage experiment queries and numbers
- `tenrec_salvage_report.md` — Salvage analysis and decision framework

**Code:**
- `examples/eda/tenrec_eda.py` — Full 6-diagnostic runner
- `examples/eda/tenrec_salvage.py` — Salvage experiment runner

---

## Decision Summary Table

| Aspect | Finding | Status |
|---|---|---|
| **EDA Complete** | All 6 diagnostics run | ✓ Done |
| **Salvage Attempted** | 3 constructions tested | ✓ Done |
| **Acceptance Threshold** | 4/5 criteria required | ✗ All failed (3/5) |
| **Chosen-last Issue** | 100% across all constructions | ✗ Unfixable |
| **Size-Category Trade-off** | Unresolvable | ✗ Confirmed |
| **Recommendation** | Drop or reframe as appendix | ✓ Clear |

---

## Next Action

**If choosing Option B (drop):**
1. Remove `case_studies/benchmarks/datasets/tenrec_bench.py`
2. Remove tenrec from `case_studies/benchmarks/runner.py` AVAILABLE_DATASETS
3. Update documentation

**If choosing Option A (appendix):**
1. Keep tenrec_bench.py
2. Reframe feature extraction and target
3. Add appendix narrative explaining limitations

**Recommendation: Proceed with Option B.** Tenrec cannot credibly be used for revealed preference benchmarking under any construction.
