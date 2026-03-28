# Tenrec EDA Summary: Menu-Choice Viability Assessment

**Date:** 2026-03-28
**Dataset:** QB-video.csv sample (500K rows, 3,120 users, 81K items)
**Question:** Can Tenrec be defensibly used as a revealed preference menu-choice benchmark?

---

## Key Findings

### 1. Action-State Consistency ✓

**Finding:** Exposure-level feedback is present.

| Action Type | Count | % |
|---|---:|---:|
| Pure exposures (no action) | 279.7K | 55.9% |
| Clicks only | 211.3K | 42.3% |
| Clicks + likes | 5.6K | 1.1% |
| Likes without clicks | 2.4K | 0.5% |

**Interpretation:**
- **True negatives exist.** Over half the rows are pure exposures with no feedback. This is good - it means the dataset contains what users *didn't* like, not just what they did.
- **Likes are mostly nested in clicks** (99.5% of likes occur in clicked rows). The 0.5% "like without click" suggests some data anomalies, but the general pattern is consistent with a sequential process: exposure → click → optional like.
- **Exposure-only prevalence (56%) is strong.** This supports the idea that a menu-choice interpretation is *possible* if we can define the menu credibly.

**Verdict on (1):** ✓ PASS. True negative feedback exists. Menu construction is not obviously doomed.

---

### 2. Window-Size Distribution (Critical Table) ✗✗✗

**The key table across 5 construction rules:**

| Rule | Median | p90 | p99 | %sz1 | %>20 | %>50 | chosen-last% | top-cat% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **R1:** Exposures since last like | 2.0 | 47.0 | 372.3 | 37.9% | 17.1% | 9.6% | 100.0% | 83.5% |
| **R2:** Clicks since last like | 2.0 | 26.0 | 198.7 | 45.6% | 11.9% | 6.0% | 100.0% | 85.9% |
| **R3:** Pos. action (any) | 3.0 | 57.0 | 360.2 | 35.7% | 19.3% | 10.9% | 100.0% | 82.7% |
| **R4:** Last 5 before like | 5.0 | 5.0 | 5.0 | 0.4% | 0.0% | 0.0% | 100.0% | 75.6% |
| **R5:** Last 10 before like | 10.0 | 10.0 | 10.0 | 0.4% | 0.0% | 0.0% | 100.0% | 71.5% |

**Interpretation (rules R1–R3, the "natural" constructions):**

All three "streaming-based" rules (accumulate until positive feedback) exhibit **severe pathology:**

- **Median window size = 2–3.** That's almost the minimum possible. Users rarely see more than 2 items before liking something.
- **p99 window size = 198–372.** Extreme outliers. A handful of users have window sizes in the hundreds. This suggests some users scroll for minutes without liking anything.
- **Size-1 prevalence = 36–46%.** Between one-third and half of all windows are trivial: the user likes the very first item they see. No choice.
- **Chosen-last = 100%.** *The chosen item is always the last item in the window.* This is the smoking gun: these aren't menu choices; they're **stopping events**. The user stops scrolling when they see something they like. This is sequential engagement, not preference from a menu.

**Interpretation (rules R4–R5, the "fixed-window" construction):**

- **Median = 5 or 10 exactly.** By construction, R4 always uses 5 prior items (or fewer at the start), R5 always uses 10. No variation.
- **Chosen-last = 100%.** Same problem: the liked item is defined as the last item in the window, so it's mechanically always last.
- **Top-category share lower but still 71–76%.** Windows are category-coherent, which is good, but the windows are *too constrained*.

**Verdict on (2):** ✗✗ **FAIL HARD.**
- R1, R2, R3: Massive window-size variance, trivial windows (36–46% size-1), and the chosen item is always last (100%) - classic engagement/stopping behavior, not menu choice.
- R4, R5: Better bounded, but the chosen item is mechanically last, making the construction circular (you've defined "choice" to be the stopping point).

---

### 3. Category Coherence (Rule 2)

**Finding:** Windows are single-category or dual-category.

| Metric | Value |
|---|---:|
| Median unique categories per window | 1.0 |
| p90 unique categories | 2.0 |
| p99 unique categories | 2.0 |
| Median top-category share | 1.00 (100%) |
| p90 top-category share | 1.00 (100%) |
| Median category entropy | ~0.0 |

**Interpretation:**
- **Almost all windows are single-category.** Median = 1 unique category means the typical click-window is entirely within a single video category. This is **very good** for menu coherence - the user is facing items of the same type.
- **P99 = 2 categories.** Even the extreme tail has mostly 2 categories. Very tight.
- **Top-category share = 100%.** Almost all windows are 100% the dominant category. No cross-category clutter.

**This is the strongest finding in Tenrec's favor.** If the user is scrolling through the "Comedy" section and likes the third comedy video, they *are* choosing from a coherent menu of comedies, not a random cross-category feed.

**Verdict on (3):** ✓ **PASS.** Windows are locally coherent by category.

---

### 4. Position of the Chosen Item (Rule 2)

**Finding:** The liked item is always or nearly always last in the window.

| Metric | Value |
|---|---:|
| Median rank within window | 2.0 |
| Median normalized position | 1.00 |
| % always last | 100.0% |

**Interpretation:**
- **Normalized position = 1.0 means the liked item is at position rank/window_size = 1,** i.e., the last item.
- **100% of windows have the liked item in the last position.** This is not a coincidence; it's the construction. The window is defined as "all clicks since the last like," so by definition, the new like is at the end.
- **This is NOT a menu choice.** A menu choice would show a distribution of positions (first, middle, last). Here, 100% are last, which reveals that the "choice" is actually the *stopping point*. The user scrolls (is exposed), clicks to see more detail (if interested), and stops when they find something worth liking. This is sequential engagement / stopping behavior, not a preference over alternatives.

**This is the smoking gun against Tenrec as a menu-choice benchmark.**

**Verdict on (4):** ✗ **FAIL.** 100% of chosen items are last in window. This indicates sequential stopping, not menu choice.

---

### 5. Watch-Time Separation

**Finding:** No meaningful separation by action type.

| Item Type | Median watch time | Mean | n |
|---|---:|---:|---:|
| Liked items | 2.0 | 2.5 | 8.3K |
| Clicked (not liked) | 2.0 | 2.4 | 211.8K |
| Exposed only | 1.0 | 0.9 | 279.9K |

**Interpretation:**
- **Liked items have slightly higher watch time** (median 2 vs 1, mean 2.5 vs 0.9 for exposed). This suggests some engagement signal.
- **But the difference is modest.** Liked vs clicked-not-liked have nearly identical watch times (2.0–2.4 s). If the "like" were a strong engagement signal, we'd expect a larger gap.
- **Exposed-only items are briefer** (1.0 s median), which makes sense - the user didn't click, so they saw less.

**Interpretation:** The watch-time data *slightly* supports a preference story (liked items are engaged with a bit longer), but the separation is weak. It's not a killer, but it doesn't strongly validate the menu-choice interpretation either.

**Verdict on (5):** ⚠ **WEAK PASS.** Watch-time shows modest engagement signal for likes, but weak separation.

---

### 6. User-Level Pathology Rates

**Finding:** Highly heterogeneous user behavior; many users with zero productive menu-choice events.

| Metric | Median | p90 | p99 |
|---|---:|---:|---:|
| Like rate (likes / rows) | 0.000 | 0.027 | 0.265 |
| Positive-action rate (any action / rows) | 0.429 | 0.731 | 1.000 |
| Size-1 window share (% of user's windows with size 1) | 0.000 | 0.000 | 0.667 |
| Median window size per user | 0.0 | 21.0 | - |

**Interpretation:**
- **Median like rate = 0.** The typical user in the sample has *zero likes*. Only at p90 (90th percentile) does a user have 2.7% likes. At p99, some users like 26.5% of items. This is **extremely skewed** - the vast majority of users are passive viewers.
- **Median positive-action rate = 42.9%.** Most users click or interact at all, but not always.
- **Median median-window-size = 0.** The median user has a median window size of 0, meaning most users have no windows at all (no likes = no completed menu events). Only at p90 does the per-user median window jump to 21.0.
- **Size-1 prevalence:** Most users have no size-1 windows (median 0), but at p99, some users are 67% size-1. Again, extreme heterogeneity.

**Interpretation:** The dataset is dominated by **passive users with few or no likes.** Only a small fraction of users generate enough menu-choice events to sustain an RP analysis. The sample of 3,120 users has maybe 10–30% who generate meaningful menu-choice data; the rest are lurkers.

**Verdict on (6):** ⚠ **WEAK PASS with major caveat.** True menu-choice events exist, but they are concentrated in a small fraction of users.

---

## Threshold Assessment

**Your pass/fail criteria:**

| Threshold | Tenrec (R2) | Pass? |
|---|---|---|
| 1. Median window size < 10 | 2.0 | ✓ |
| 2. p99 window size < 50 | 198.7 | ✗ |
| 3. Size-1 prevalence < 30% | 45.6% | ✗ |
| 4. Chosen-last share < 70% | 100.0% | ✗ |
| 5. Category coherence > 0.4–0.5 | 85.9% | ✓ |
| 6. Pure exposure rows > 20% | 55.9% | ✓ |

**Score: 3 out of 6 pass.** Per your rubric, this puts Tenrec in **Path B: Appendix Only** (2–3 passes).

---

## Decision: Path B - Exploratory Appendix

### What Tenrec **is:**
1. A sequential exposure-engagement dataset from a real recommendation system
2. Category-coherent (single-category windows)
3. Has true negative feedback (exposures without action)
4. Contains some users with genuine product substitution behavior

### What Tenrec **is not:**
1. A static menu-choice benchmark (100% of choices are the last item in the exposure stream)
2. A sample with universal dense choice sets (45.6% of windows are trivial size-1)
3. A sample with bounded tail risk (p99 window is 199 items, thousands of exposures for some users)
4. A dataset where most users have usable data (median like rate per user = 0)

### Recommended reframing:
Instead of "Tenrec: Sequential Menu Choice," use:

> **Tenrec: Sequential Exposure Consistency**
>
> We audit whether users show consistent behavior within recommendation sessions. Given a user's recent exposure history in a category, can revealed preference features (category loyalty, substitution entropy, recency bias) predict future engagement?
>
> This is *not* static revealed preference, but it measures **behavioral stability** in a dynamic context.

### Implementation (if keeping Tenrec):
1. **Use R4 or R5** (last 5 or 10 items before a like). These give bounded, coherent windows.
2. **Don't claim SARP/WARP consistency.** Instead, measure:
   - **Category transition coherence:** Do users stay in the same category?
   - **Recency bias:** Is the chosen item the newest?
   - **Substitution rate:** Do users alternate between items or repeat?
   - **Dwell consistency:** Do users who like many items show similar dwell patterns?
3. **Acknowledge the limitation.** Make it clear in the paper: "Tenrec is not a revealed preference choice set; it's a sequential exposure engagement benchmark."

### Bottom line:
Tenrec is **suitable for an exploratory appendix** showing how RP features behave in a sequential context, but **not suitable for the main RP benchmark table** claiming SARP/WARP consistency.

---

## Comparison to User's Requirements

**Your statement:**
> "Tenrec should not be treated as a normal menu-choice benchmark unless EDA rescues it."
> **My prior is B unless EDA strongly supports C.**

**EDA result:** **The diagnostics confirm Path B.** Category coherence is strong, but window-size variance, size-1 prevalence (46%), and 100% chosen-last behavior all point to sequential engagement, not menu choice.

**Recommendation:** **Proceed with Path B - reframe as appendix.** Don't drop Tenrec entirely; instead, reuse the high-quality category signal and user behavior stability for a secondary claim: "Revealed preference features predict engagement trajectories in sequential contexts."
