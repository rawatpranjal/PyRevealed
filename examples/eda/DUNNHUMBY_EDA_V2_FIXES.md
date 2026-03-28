# Dunnhumby EDA v2: Fixes & Conservative Conclusions

## Status

✅ **Three critical issues fixed**
⚠️ **Four overclaimed conclusions softened**
🔴 **Do NOT sign off yet** - requires sensitivity testing before publication

---

## Critical Fixes (v1 → v2)

### 1. Block 4: T Unit Bug

**v1 (broken):**
```
T = len(hh_data)  # Transaction count
  Median: 206
  Max: 1,477
Edge density: 0.017 (1.7%)
```

**v2 (fixed):**
```
T = len(weeks_observed)  # Household-weeks
  Median: 39
  Max: 97
Edge density: 0.504 (50.4%)
```

**Root cause:** Line 470 used `len(hh_data)` (filtered transaction rows) instead of `len(hh_pivot.index)` (household-weeks after weekly pivoting).

**Impact:**
- Old edge density (1.7%) was meaningless-suggested sparse, weak RP support
- New edge density (50.4%) is healthy-half of all pairs have direct RP edges

### 2. Sample Accounting

**v1:** Claimed "2,496 qualifying households (≥10 active weeks)"
**v2:** Correctly states "2,222 qualifying households"

**Reality:**
- Total households: 2,496
- Households with ≥ 10 active weeks: 2,222
- Filtered out: 274 households with sparse activity

### 3. Data Provenance Clarity

**v1 issues:**
- Block 3 says "0.0% on promo" because aggregated dataset lacks RETAIL_DISC
- Block 5 presents "promo week event study" without clarifying data source

**v2:**
- Block 3 now computes oracle error BY CATEGORY (MAE ranges $0.38–$2.51)
- Block 5 explicitly states: uses RAW transaction data, not aggregated

---

## Softened Conclusions

### Block 1: Zero-Fill Rejection

**v1 claim:** "Zero-weeks would create pathological GARP violations because shopping weeks dominate zero-weeks"

**v2 reframe:** Zero weeks likely represent shopping OUTSIDE tracked categories, not zero demand. Including them would confound two different choice problems.

**Why this matters:** The v1 argument about GARP pathology was technically correct but missed the real point.

---

### Block 3: Price Oracle Accuracy

**v1:** "Price oracle is acceptable baseline"
**v2:** "Price oracle is noisy but usable approximation"

**Evidence:**
| Commodity | MAE | % of Median Price |
|-----------|-----|------------------|
| Milk | $0.38 | 20% |
| Bread | $0.60 | 32% |
| Soup | $0.62 | 43% |
| Cheese | $0.81 | 34% |
| Beef | $2.51 | 53% |
| Soda | $1.44 | 71% |

**Implication:** Beef and Soda have massive measurement error (50%+). Any GARP violations concentrated in these categories are suspect.

---

### Block 4: RP Identification

**v1:** "Low crossing rate (0.3%) indicates strong identification"
**v2:** "Low crossing rate is ambiguous-could mean budget variety (good) OR insufficient overlap (bad)"

**Key point:** 50% edge density is healthy, but we can't conclude strong identification without:
- Checking violation patterns (are they in high-noise categories?)
- Comparing with null models (randomized quantities, permuted weeks)
- Verifying price variation drives edges (not just quantity variation)

---

### Block 5: IID-Across-Weeks

**v1:** "Stockpiling effects <2%, so IID assumption is defensible"
**v2:** "We do NOT detect large week-level spikes, but this does NOT prove IID holds"

**Why:**
- Weekly aggregation masks intra-week dynamics (daily buying patterns)
- Category aggregation masks sub-category effects (e.g., promo on Diet Coke but not Coke)
- Stockpiling may happen over longer windows (bi-weekly planning)

**Honest verdict:** IID remains an unvalidated assumption.

---

## Before Publication: Required Checks

### 1. Category-Level RP Metrics

Compute GARP consistency, CCEI, MPI separately by commodity category.

**Question:** Are violations concentrated in high-error categories (Beef, Soda)?

**If yes:** Price mismeasurement is likely the culprit, not preference inconsistency.

### 2. Store-Week vs Chain-Week Price Sensitivity

- Run RP metrics under chain-week oracle (current)
- Run RP metrics under store-week oracle (if feasible)
- Compare household GARP rates, CCEI distributions

**Question:** Do RP scores stabilize or diverge significantly?

**Why:** Chain-week error (MAE $0.38–$2.51) could be creating artificial violations.

### 3. Violation Concentration Check

For households with GARP violations:
- Which commodities dominate the violation cycles?
- Are they high-error categories (Beef, Soda) or low-error (Milk, Bread)?

**Question:** Are "violations" real preference inconsistency or measurement artifacts?

### 4. Budget Variety Validation

- Plot (price, quantity) scatterplots for top 5 households
- Check if budgets actually cross in 2D price-quantity space

**Question:** Does 50% edge density reflect real budget variety or geometric artifact?

---

## Summary Table

| Check | v1 Status | v2 Status | Ready for Paper? |
|-------|-----------|-----------|------------------|
| Sample accounting | ❌ 2,496 vs 2,222 | ✅ Fixed (2,222) | Yes |
| Block 4 T unit | ❌ T = transactions | ✅ T = weeks | Yes |
| Data provenance | ⚠️ Unclear | ✅ Explicit | Yes |
| Zero-fill argument | ⚠️ Overclaimed | ✅ Reframed | Yes |
| Price oracle claim | ⚠️ "Acceptable" | ✅ "Noisy" | Yes, with caveat |
| ID strength claim | ❌ "Strong ID" | ✅ "Ambiguous" | Yes, conditional |
| IID assumption | ⚠️ "Defensible" | ✅ "Unvalidated" | Yes, but document |
| Category-level errors | ❌ Not reported | ✅ Reported | Needs sensitivity |
| Store-week comparison | ❌ Not done | ❌ Not done | ⚠️ **Required** |
| Violation attribution | ❌ Not done | ❌ Not done | ⚠️ **Required** |

---

## Recommended Next Steps

### Tier 1: Must Do Before Submission

1. **Store-week price sensitivity check**
   - Run RP analysis on 100-household subsample under both oracles
   - Compute correlation of GARP rates, CCEI, MPI
   - Report "RP results are [stable / divergent] across price specifications"

2. **Category-level violation audit**
   - For top 50 violating households, compute violation cycles
   - Which commodities dominate? (beef, soda, or distributed?)
   - Cross-tabulate with oracle error category

### Tier 2: Nice to Have

3. **Null model check**
   - Randomly permute quantities within household
   - Recompute GARP rates
   - Report: "Real RP violations are [X%], null model predicts [Y%]"

4. **Budget scatterplots**
   - Pick 5 active households
   - Plot 2D (price[commodity1], quantity[commodity1]) with budget ellipses
   - Visually confirm budget crossings exist

---

## Dunnhumby Verdict (Current)

✅ **Modeling story is sound:** Conditional sub-basket repeated demand (not full budgeting)

✅ **Data structure is defensible:** Active-week, 2,222 HH, 39 median weeks

⚠️ **Price mismeasurement is substantial:** MAE $0.38–$2.51 by category

⚠️ **RP identification is ambiguous:** Edge density 50%, but requires validation

❌ **Not ready for publication** without Tier 1 sensitivity tests

---

## Files

- `dunnhumby_stress_eda_v2_conservative.py` - Fixed script with conservative conclusions (449 lines)
- `DUNNHUMBY_STRESS_TEST_REPORT.md` - v1 report (archive; do not use for paper)
