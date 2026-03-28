# Dunnhumby Validation Results: Three Red Flags

## Executive Summary

Three critical validation checks (Tier 1) reveal that **Dunnhumby is severely compromised by price measurement error**:

1. **RP metrics shift meaningfully under store-week prices** (-2.4%, p<0.0001)
2. **73% of violations cluster in high-error categories** (soda dominates at 65%)
3. **Observed violations are statistically indistinguishable from randomized quantities** (p=0.47)

**Recommendation:** Dunnhumby should **NOT** be used as a headline benchmark without major price quality improvements.

---

## Check 1: Store-Week vs Chain-Week Price Sensitivity

### Results

```
Chain-week oracle (current):
  Mean consistency rate: 96.2%
  Median:               97.6%
  Std:                  4.5%

Store-week oracle (alternative):
  Mean consistency rate: 93.9%
  Median:               95.3%
  Std:                  5.2%

Difference:
  Mean shift:           -2.4% (SIGNIFICANT)
  Correlation:          0.928
  Paired t-test:        p < 0.0001
```

### Interpretation

- RP consistency **drops 2.4%** when switching from chain-week to store-week prices
- The shift is **statistically significant** (p<0.0001)
- However, **correlation is high** (0.928), meaning relative rankings of households are preserved

### What This Means

**Chain-week oracle is inadequate as the true price specification.** The fact that results shift meaningfully under store-week prices tells us:

- Chain-week prices introduce **systematic measurement bias**
- Any GARP violations could be partially or entirely explained by price mismeasurement
- The 96.2% consistency rate on chain-week prices is **optimistic**

---

## Check 2: Violation Concentration by Commodity

### Results

```
Commodity                 Violations  % of Total  Oracle MAE
------------------------------------------------------
Soda                      581         64.8%       $1.44 ← HIGH ERROR
Pizza                     66          7.4%        $1.41 ← HIGH ERROR
Cheese                    77          8.6%        $0.81
Milk                      70          7.8%        $0.38
Chips                     30          3.3%        $0.80
Soup                      27          3.0%        $0.62
Bread                     26          2.9%        $0.60
Beef                      12          1.3%        $2.51 ← HIGH ERROR
Lunch                     8           0.9%        $0.81
Yogurt                    0           0.0%        $0.64

High-error categories (MAE > $1.0):
  Soda + Pizza + Beef = 65% + 7% + 1% = 73% of all violations
```

### Interpretation

**Nearly 3 out of 4 violations are in high-error price categories.**

- **Soda dominates at 65%** of all violations, despite being only 1 of 10 categories
- Soda's oracle error is **$1.44 (71% of median price)**-this is enormous measurement noise
- Pizza adds another **7%** of violations with **$1.41 error (58% of median)**
- Low-error categories (milk, bread, soup) contribute only **11%** of violations combined

### What This Means

**The "violations" are primarily oracle artifacts, not revealed preference inconsistencies.**

This is the smoking gun. If Soda alone accounts for 65% of GARP violations, and Soda prices are mismeasured by 71%, then:

- Most observed violations are **not real preference inconsistencies**
- They are **price mismeasurement creating false cycles** in the preference graph
- The household isn't irrational; the data is noisy

---

## Check 3: Null Model (Permutation Test)

### Results

```
Observed violations (real quantities):
  Mean: 34.20
  Median: 24.0
  Std: 33.34

Null model violations (permuted quantities):
  Mean: 33.02
  Median: 22.0
  Std: 34.95

Comparison:
  Observed - Null: +1.18 violations
  Effect size: 0.03 standard deviations
  Paired t-test: p = 0.4665 (NOT SIGNIFICANT)
```

### Interpretation

**There is no statistically significant difference between real and randomized quantities.**

This is the most damning result:

- If you randomly permute a household's weekly quantities (keeping weeks × prices fixed)
- You get **almost exactly the same number of GARP violations**
- The difference is **indistinguishable from noise** (p = 0.47)

### What This Means

**Dunnhumby has essentially NO genuine revealed preference signal.**

The RP violations are:
1. Not driven by actual preference inconsistencies (they don't exceed randomized baseline)
2. Concentrated in price-noisy categories (Check 2)
3. Sensitive to price specification (Check 1)

This suggests violations are **stochastic artifacts of measurement error**, not economic meaning.

---

## Combined Interpretation

The three checks form a coherent story:

| Check | Finding | Implication |
|-------|---------|-------------|
| **Check 1** | Metrics shift with price spec | Price mismeasurement is significant |
| **Check 2** | 73% violations in high-error cats | Violations are oracle artifacts |
| **Check 3** | No difference from random quantities | No genuine RP signal detected |

**Conclusion:** Dunnhumby's apparent GARP violations are **overwhelmingly explained by price measurement error**, not actual preference inconsistency.

---

## What Went Wrong

### Root Cause: Chain-Week Oracle

The chain-week price oracle (current approach) uses a **single median price per commodity per week** across all 477 stores. This creates two problems:

1. **Store heterogeneity:** A household shopping at a low-price Aldi pays different prices than a household shopping at premium Whole Foods. Using a chain median erases this variation.

2. **Composition bias:** High-discount weeks have more sales, so the median price is pulled down by discounted transactions. This creates false affordability.

### Example: Soda

- Median soda price: $2.02
- Cross-store IQR: $1.97 (range: $0.37 to $3.94)
- Relative IQR: 97.5% of median
- Measurement error (MAE): $1.44

A household that buys soda at a discount store pays $0.50, but the oracle says $2.02. This creates **artificial budget constraints** that generate false GARP violations.

---

## Recommended Path Forward

### Option A: Improve Price Oracle (Recommended)

**Store-week prices** would reduce measurement error substantially:
- Compute per-store-week median (477 stores × 104 weeks × 10 commodities)
- Use sparse matrix + fallback to chain-week for missing cells
- Re-run RP analysis and recompute validation checks

**Estimated improvement:** Check 3 should show p < 0.05 (significant RP signal above null model).

### Option B: Filter Noisy Categories

If store-week prices are infeasible:
- **Drop Soda (65% of violations, 71% price error)**
- **Drop Pizza (7% of violations, 58% price error)**
- Recompute RP metrics on remaining 8 categories

**Risk:** Shrinks choice problem substantially (removes high-variance categories).

### Option C: Acknowledge Limitations and Use Cautiously

- Use Dunnhumby only as a **secondary / robustness benchmark**
- Frame results as: "Households show [X]% GARP consistency on a sub-basket, but violations are sensitive to price specification"
- Do NOT claim this as headline evidence of household irrationality

### Option D: Move Dunnhumby to Secondary Suite

Use Dunnhumby only for:
- Testing feature extraction on real data
- Benchmarking computational performance
- Sensitivity analysis

Do NOT use for:
- Empirical claims about household rationality
- Comparison of behavioral metrics across populations

---

## Revised Recommendation

### Can Dunnhumby Stay in the Benchmark Suite?

**Yes, but only as a secondary robustness check.**

- Dunnhumby is real data with real challenges ✅
- The modeling story (conditional sub-basket demand) is sound ✅
- But the price mismeasurement is too severe for headline use ❌

### Should Dunnhumby Be a Headline Benchmark?

**No. Not without Option A (store-week prices) first.**

The validation checks show that 70%+ of observed violations are oracle artifacts. Publishing this as evidence of household irrationality would be misleading.

### What to Do Before Using in a Paper

1. **Implement store-week prices** (1-2 hours of coding + 5 min runtime)
2. **Re-run validation checks** on improved oracle
3. **If Check 3 p < 0.05:** Dunnhumby is ready for headline use
4. **If Check 3 p ≥ 0.05:** Use Option B (filter Soda) or Option C (secondary only)

---

## Files

- `dunnhumby_eda_validation_checks.py` - Tier 1 validation code (445 lines)
- `DUNNHUMBY_VALIDATION_RESULTS.md` - This document

---

## Conclusion

**The good news:** Dunnhumby's observational unit and modeling story are sound.

**The bad news:** Price mismeasurement is so severe that observed violations are indistinguishable from noise.

**The path forward:** Implement store-week prices and revalidate. If that fixes the null model problem (p < 0.05), Dunnhumby is a keeper. If not, relegate to secondary robustness suite.

---

## Appendix: Statistical Details

### Check 1: Paired t-test

```
H0: Store-week and chain-week consistency rates are equal
Test statistic: t = 4.82
Degrees of freedom: 99
p-value: p < 0.0001
Result: REJECT H0 (significant difference)
```

### Check 3: Permutation t-test

```
H0: Observed violations = Null model violations
Test statistic: t = 0.72
Degrees of freedom: 49
p-value: p = 0.4665
Result: FAIL TO REJECT H0 (no difference)
```

This means: **We have no evidence that observed RP violations exceed what random quantities would produce.**

---

## Impact on Publication

**Before validation checks:**
- Could claim Dunnhumby shows 96% GARP consistency
- Could use as headline dataset

**After validation checks:**
- Must acknowledge violations are oracle artifacts
- Should either improve oracle OR acknowledge limitations
- Recommend: improve oracle and revalidate before publication
