# Taobao EDA: Menu-Choice Construction Assumption Audit

**Sample**: First 20M of 100M rows (20% of dataset)
**Script**: `examples/eda/taobao_eda.py`
**Date**: 2026-03-28

---

## Raw Data Summary

| Metric | Value |
|--------|-------|
| Rows sampled | 20,000,000 |
| Unique users | 198,002 |
| Unique items | 2,209,565 |
| Unique categories | 8,565 |
| Timestamp range | 1905-07-13 to 2037-04-09 (!) |

**Behavior mix:**
- pv (pageview): 89.6%
- cart: 5.5%
- fav: 2.9%
- buy: 2.0%

---

## Assumption Audit Results

### A1 - Session gap = 30 minutes

Gap distribution for within-user consecutive events:

| Percentile | Gap |
|------------|-----|
| p50 | 1.3 min |
| p75 | 5.2 min |
| p90 | 3.3 hr |
| p95 | 12.5 hr |
| p99 | 1.4 days |
| p100 | **41,042 days** (corrupted timestamp) |

Fraction of gaps below threshold:

| Threshold | Fraction below (stays in session) |
|-----------|----------------------------------|
| 1 min | 44.3% |
| 5 min | 74.6% |
| 15 min | 82.0% |
| **30 min** | **84.2%** |
| 1 hr | 86.2% |

**Gap histogram is unimodal** (heavy mass below 5 min, smooth tail). No bimodal structure straddling 30 min - the threshold is not knife-edge. Changing to 15 min would add +14% more session breaks; changing to 60 min would reduce by 12%.

**Issue identified**: p100 gap = 41,042 days = 3.55 billion seconds. The dataset timestamp range is 1905 to 2037 despite being a 2017 dataset. Corrupted timestamps exist. These will create spurious session breaks at the boundary between a normal event and a corrupted-timestamp event. Impact is likely small (extreme outliers in a 100M row dataset) but means session assignment is noisy for affected users.

**Verdict**: 30-min threshold is reasonable and not knife-edge. ✓ with caveat about corrupted timestamps.

---

### A2 - Menu = viewed items (pv events)

Sessions created by 30-min gap: **3,316,666 total**

Sessions with at least 1 pv event: 3,125,120 (94.2%)

**Issue identified**: The menu is constructed from ALL pv events in a session, regardless of whether they occurred before or after the purchase. D6 shows that 61.4% of sessions have pv events occurring AFTER the buy event. These post-purchase pageviews are included in the menu even though they were not part of the consideration set at decision time.

**Verdict**: Menu includes post-purchase views. The effective menu (viewed items before buying) is smaller than the recorded menu. ⚠

---

### A3/A4 - Exactly 1 unique purchased item per session

| Unique buys per session | Count | Fraction |
|-------------------------|-------|----------|
| 0 (browsing only) | 3,006,349 | **90.6%** |
| 1 | 255,104 | **7.7%** - KEPT |
| 2 | 38,603 | 1.2% |
| ≥3 | ~16,000 | 0.5% |

Only 7.7% of sessions contain a purchase. 90.6% are pure browsing and produce no observations. This is a fundamental sparsity of the purchase signal - 12× more browsing sessions than purchase sessions.

**Verdict**: Harsh but clean filter. The 1.7% dropped for multiple purchases is small. ✓

---

### A5 - Purchased item inserted into menu if not viewed

For the 255,104 valid (exactly 1 buy) sessions:

| Status | Count | Fraction |
|--------|-------|----------|
| Purchase was viewed (pv) before buying | 86,144 | 33.8% |
| Purchase NOT viewed - phantom insertion | 100,237 | **39.3%** |
| Session had zero pv events | 68,723 | **26.9%** |

**Critical finding**: In 66.2% of valid sessions (39.3% + 26.9%), the purchased item was never observed as a pageview in the same session. The loader inserts it into the menu via `menu | {choice}`.

This means:
- **26.9%**: "Direct purchase" sessions with no observed consideration set at all. The menu is a singleton {bought_item}, then bumped to size 2 only if paired with any view. These are not menu-choice observations in any meaningful sense.
- **39.3%**: User viewed other items but bought something they did not view (perhaps from a recommendation widget, search, or external link). The menu is contaminated with a phantom item.
- **33.8%**: Clean observations where the purchased item appears in the viewed set. Only these are credible simultaneous-menu-choice observations.

**Verdict**: Only ~34% of valid sessions satisfy the core assumption that the chosen item was part of the observed menu. ✗

---

### A6 - Menu size in [2, 50]

Of 255,104 valid sessions:

| Menu size (unique viewed items) | Count | Fraction |
|---------------------------------|-------|----------|
| 0 (no pv) | 68,723 | 26.9% - DROPPED |
| 1 | 27,608 | 10.8% - DROPPED (or bumped to 2 by A5 insert) |
| 2–50 | 157,691 | **61.8%** - KEPT |
| >50 | 1,082 | 0.4% - DROPPED |

Menu size percentiles (unique items, for sessions with pv):
p25 = 2, p50 = 5, p75 = 9, p90 = 17, p99 = 43

41.9% of sessions have repeat views (same item viewed multiple times). Mean 3.5 re-views per such session. The menu is a set so these are deduplicated - raw pv event count is larger than menu size.

**Verdict**: Filter removes 38.2% of valid sessions. The kept sessions have reasonable menu sizes (median 5). ✓

---

### D6 - Purchase temporal position (Tenrec test)

**This is not a Tenrec-like issue.**

- Bought item = last viewed item before purchase: **0.9%** (Tenrec was 100%)
- Bought item = first viewed item in session: 11.2%
- Mean fraction of pv events before the buy: 60.9%

The purchase timestamp splits the session: 60.9% of views come before, 39.1% after. Users browse, buy, then continue browsing. The bought item is NOT systematically terminal.

However, the implication is that the menu as constructed contains ~39% post-purchase views on average, i.e., items the user viewed **after** completing their transaction. These items could not have influenced the purchase decision.

**Verdict**: No stopping-rule bias (unlike Tenrec). ✓
But post-purchase views contaminate the menu. ⚠

---

### D8 - Category coherence

Of 186,381 sessions with pv data:

| Categories in menu | Count | Fraction |
|--------------------|-------|----------|
| 1 (coherent) | 58,542 | **31.4%** |
| 2 | 40,781 | 21.9% |
| 3 | 25,870 | 13.9% |
| 4 | 17,576 | 9.4% |
| 5 | 11,820 | 6.3% |
| ≥10 | 9,685 | 5.2% |

Median: 2 categories. p75: 4 categories. p90: 7 categories.

Only 31.4% of sessions are single-category. 68.6% span multiple unrelated product categories (e.g., user browses shoes, then electronics, then clothing, then buys one item from one of those categories).

For discrete-choice RP analysis, cross-category menus are not inherently invalid - the user is choosing among items from their session. But the theoretical cleanness is lower: RP results on a shoe-vs-laptop menu are harder to interpret economically than results on a shoes-only menu.

**Verdict**: Significant cross-category mixing. Economic interpretation of RP violations is less clean. ⚠

---

### D9 - User qualification (min_sessions = 5)

Users with ≥1 valid session (in 20M row sample): 124,333

| min_sessions | Qualifying users | Fraction |
|-------------|-----------------|----------|
| 1 | 124,333 | 100.0% |
| 3 | 32,375 | 26.0% |
| **5 (default)** | **7,513** | **6.0%** |
| 8 | 1,157 | 0.9% |
| 10 | 464 | 0.4% |

The min_sessions=5 filter retains only 6% of eligible users. This is a severe selection effect: the benchmark population is the top 6% most-active buyers, not a representative sample of Taobao users.

p50 valid sessions per user = 2. p75 = 3. p95 = 5.

The full 100M row dataset will yield more sessions per user and thus higher qualification rates. The benchmark reports 4,239 qualifying users from 50K max → suggesting the full dataset roughly 4-6× the 20M sample rate.

**Verdict**: Strong selection bias toward power users. Results describe the top 5–10% of buyers by activity. ⚠

---

## Summary Scorecard

| Assumption | Finding | Verdict |
|-----------|---------|---------|
| A1: 30-min session gap | 84.2% of gaps < 30 min; unimodal gap distribution; but corrupted timestamps (p100 = 41,042 days) | ✓ with caveat |
| A2: Menu = pv events | 61.4% of sessions have post-purchase pv events in the menu | ⚠ |
| A3/A4: Exactly 1 buy/session | 7.7% sessions have 1 buy; filter is clean | ✓ |
| A5: Buy inserted if not viewed | **39.3% phantom insertions; 26.9% direct-purchase sessions** | ✗ |
| A6: Menu size [2, 50] | 61.8% of valid sessions pass; median size = 5 | ✓ |
| D6: Purchase not terminal | 0.9% chosen-last (vs 100% in Tenrec) | ✓ |
| D8: Category coherence | 31.4% single-category; median 2 categories | ⚠ |
| A7: min_sessions = 5 | 6% of users qualify; strong power-user selection | ⚠ |

---

## Key Issues for Benchmark Validity

**Issue 1 (Critical): Phantom menu construction**
Only 33.8% of valid sessions have the purchased item in the viewed set. The remaining 66.2% have either no views (26.9%) or a phantom-inserted purchase item (39.3%). The RP analysis runs on menus where most choices were not observed as simultaneous alternatives. This is the most serious validity concern.

**Issue 2 (Significant): Post-purchase views in menu**
The menu includes items viewed after the purchase in 61.4% of sessions. The consideration set at decision time is a subset of the recorded menu. This overstates menu size and may include irrelevant items.

**Issue 3 (Moderate): Cross-category menus**
68.6% of menus span multiple product categories. RP violations may reflect budget allocation across categories rather than preference inconsistency within a category.

**Issue 4 (Moderate): Power-user selection**
min_sessions=5 keeps only 6% of eligible users. Benchmark results describe highly active repeat purchasers, not typical users.

**Comparison to Tenrec**: Taobao does NOT have the Tenrec structural flaw (100% chosen-last). The purchase can appear at any point in the session timeline. However, the phantom insertion rate (39.3% + 26.9%) is a different but comparably serious validity concern.
