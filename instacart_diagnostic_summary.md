# Instacart Aisle-Level Single-Reorder Diagnostic Report

## Executive Summary

The Instacart dataset, when reconstructed at **aisle granularity with single-reorder filtering**, provides **11.6M clean choice events** across **2.76M user-aisle pairs**. However, five critical diagnostics reveal that the dataset has **fundamental limitations** for revealed preference analysis:

- **75% of events have no real choice menu** (0-1 prior alternatives)
- **58.6% of repeated user-aisles show pure habit**, not switching
- **Data is sparse**: median 2 events per user-aisle pair
- **Only 19% of aisles are high-quality** (remaining are commodities/niche)

**Verdict**: Instacart is **borderline publishable** as a sensitivity benchmark if heavily filtered, but is **not strong enough** to be a flagship benchmark. Consider dropping entirely if data quality is a hard constraint.

---

## Dataset Context (from Kaggle)

| Property | Value |
|----------|-------|
| Users | 206,209 |
| Orders | 3.2M (prior) + 131K (train) + 75K (test) |
| Products | 50,000+ |
| Aisles | 134 |
| Departments | 21 |
| Source | [Kaggle Market Basket Analysis](https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis) |

---

## Construction Tested

| Element | Specification |
|---------|--------------|
| Observation unit | `user × order × aisle` |
| Choice signal | Exactly 1 reordered SKU per (order, aisle) event |
| Total events generated | 11.6M clean single-choice occasions |
| Total user-aisle pairs | 2.76M |

---

## Five Diagnostic Findings

### 1. Menu Non-Triviality: **FAIL** ✗

**Question**: After aisle + single reorder, what fraction have prior known alternatives (M >= 2 or >= 3)?

**Answer**:
- Events with ≥2 prior items: **25.9%** (3.0M) ⚠
- Events with ≥3 prior items: **10.2%** (1.2M) ✗

**Detailed Distribution**:
```
0 prior items:  2.76M ( 23.8%)  ← No choice at all
1 prior item:   5.84M ( 50.3%)  ← Trivial choice (1 option)
2 prior items:  1.83M ( 15.8%)
3 prior items:  0.66M (  5.7%)
4+ prior items: 0.54M (  4.4%)
```

**Interpretation**:
- **76% of events are NOT meaningful choices** - they either have no menu or only one known alternative
- These are repeat purchases with empty/trivial menus, not discrete selections
- The "choice" is binary (buy/don't buy), not among substitutes
- Menu inflation: formally non-empty but behaviorally trivial

**Verdict**: 🚨 **RED FLAG**. Most observations have no real substitution set.

---

### 2. Repeated User-Aisle Pairs: **WEAK** ⚠

**Question**: How many repeated events exist per user-aisle pair?

**Answer**:
- Total user-aisle pairs: **2.76M**
- Median events per pair: **2**
- Mean events per pair: **4.2**

**Distribution**:
```
1 event:   967K (35.0%)  ← One-shot, no RP content
2 events:  513K (18.6%)
3 events:  314K (11.4%)
4 events:  211K (7.6%)
5 events:  150K (5.4%)
6+ events: 606K (22.0%)
```

**Interpretation**:
- Median 2 means **half of pairs have ≤2 occurrences**
- RP analysis needs N ≥ 3–5 per pair for meaningful cycle/violation detection
- 35% of pairs are **singletons** (one purchase, zero RP content)
- Only ~30–40% have enough repetition (5+ events)
- Graph is **very sparse** compared to dense preference networks

**Verdict**: 🟡 **BORDERLINE**. Enough pairs for basic analysis, but sparsity is concerning.

---

### 3. Product Switching: **WEAK** ✗

**Question**: How often does the chosen product switch across events within the same user-aisle pair?

**Answer**:
- User-aisle pairs with ≥2 events: **1.79M**
- Always same product (loyalty): **58.6%** (1.05M)
- Switch products (substitution): **41.4%** (0.74M)
- Mean switching rate (among switchers): **23.6%** per transition
- Median switching rate: **0%** (many pairs never switch)

**Interpretation**:
- **58.6% of repeating user-aisles are pure habit** - user buys same SKU every time
- No preference variation = no choice signal
- Only 41.4% show *any* substitution behavior
- Even within switching pairs, changes are infrequent (median 0%)
- Suggests user-aisle behavior is dominated by routine/replenishment, not deliberation

**Verdict**: 🚨 **WEAK**. More than half the data is habitual loyalty with zero choice content.

---

### 4. Menu Definition (Trailing vs All-Time): **RECOMMEND TRAILING** →

**Question**: Should menu be all-time history or trailing window (3–5 orders)?

**Example**: User 33779, Aisle 115 (28 orders over time):
```
Order  2: chose product 19660, all-time menu=0, trailing-3=0
Order  5: chose product 49520, all-time menu=1, trailing-3=1
Order  6: chose product 19660, all-time menu=1, trailing-3=2
...
Order 57: chose product 19660, all-time menu=1, trailing-3=1
```

**Key observations**:
- All-time menu grows as user accumulates history (inflates choice set)
- Trailing-3 menu stays small and bounded (more realistic)
- User likely only recalls recent aisles, not 2-year history

**Recommendation**: **Use trailing-3 or trailing-5 orders, NOT all-time**

**Impact**:
- Trailing windows shrink menus further
- Reduces non-trivial events even more (already at 25.9%)
- But behaviorally more credible

---

### 5. Usable Aisles: **LIMITED** ⚠

**Question**: Which aisles represent real substitution, not commodity replenishment?

**High-Quality Aisles** (≥5 unique products, ≥1.5 events/user, ≥100 total events):
```
Aisle ID | Events  | Users  | Unique Prod | Events/User | Type
───────────────────────────────────────────────────────────────
   24    | 745K    | 135K   |    346      |    5.50     | ✓ Fresh Produce
  123    | 592K    | 109K   |    590      |    5.39     | ✓ Dairy
   84    | 555K    |  77K   |    232      |    7.21     | ✓ Packaged Foods
   83    | 524K    | 111K   |    518      |    4.70     | ✓ Beverages
  120    | 423K    |  80K   |    968      |    5.25     | ✓ Frozen Foods
   21    | 373K    |  79K   |    852      |    4.69     | ✓ Snacks
  115    | 371K    |  65K   |    327      |    5.71     | ✓ Breakfast
```

**Low-Quality Aisles** (commodity, thin):
```
Aisle ID | Events | Users  | Unique Prod | Events/User | Type
──────────────────────────────────────────────────────────────
   80    |  3.4K  | 1.9K   |    277      |    1.76     | ✗ Niche/Specialty
  103    |  2.9K  | 1.3K   |     73      |    2.16     | ✗ Bulk/Limited
   55    |  2.5K  | 1.3K   |    168      |    1.98     | ✗ Thin/Seasonal
```

**Summary**:
- **25 out of 134 aisles are high-quality** → 19% usable
- 109 aisles are thin, commodity, or niche
- Top aisles (fresh, dairy, beverages, frozen) have real product variety
- Bottom aisles (specialty, bulk, niche) have no substitution

**Recommendation**: **Filter to top ~25 high-quality aisles only**

**Verdict**: 🟡 **LIMITED**. Most aisles are unusable; only 19% remain.

---

## Overall Assessment

### Summary Table

| Diagnostic | Result | Threshold | Status |
|-----------|--------|-----------|--------|
| Menu Non-Triviality | 25.9% ≥2 items | >50% | ❌ FAIL |
| Repeated Events | Median 2/pair | ≥3 median | ⚠ WEAK |
| Switching Behavior | 41.4% switchers | >60% | ❌ WEAK |
| Menu Definition | Use trailing | behavioral fit | → RECOMMEND |
| Usable Aisles | 25/134 (19%) | >40% | ⚠ LIMITED |

### Crux Issue

> **Are aisle-level events economically meaningful choices among alternatives, or just repeated habit purchases with a formally non-empty menu?**

**Evidence**:
- 76% of events have trivial menus (0–1 priors)
- 59% of repeated user-aisles are pure habit (no switching)
- Median 2 events per user-aisle pair (sparse)
- Only 19% of aisles are high-quality

**Conclusion**: **Mostly habit, not choice.** The data inflates event count but sacrifices signal quality.

---

## Recommendations

### Path A: Build Instacart V2 (If Proceeding)

Use **strict filters**:

1. ✓ Aisle-level granularity (not department)
2. ✓ Single-reorder events only
3. ✓ Trailing-3 or trailing-5 history for menu construction
4. ✓ **Require menu size ≥ 2** (25.9% of events)
5. ✓ **Require ≥3 events per user-aisle pair** (further filters)
6. ✓ **Use only high-quality aisles** (top ~25 by volume)
7. ✓ Do NOT use `add_to_cart_order` (noise)

**Result**: ~500K–1M clean events, ~10–15K users
**Status**: Defensible as a **sensitivity benchmark**, not flagship

### Path B: Drop Instacart (Recommended)

Focus on **confirmed strong datasets**:
- **REES46**: Gold standard (platform sessions, real comparison shopping)
- **H&M**: Validated (real article counts, moderate quality)
- **Dunnhumby**: Proven (production benchmark)

**Rationale**: A smaller, clean benchmark outweighs a larger, weak one.

---

## Sources

- [Kaggle: Instacart Market Basket Analysis](https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis)
