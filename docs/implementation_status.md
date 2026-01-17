# PyRevealed Implementation Status vs. Literature Survey

## Summary

Based on the comprehensive survey of 65+ revealed preference methods from the academic literature, here's what PyRevealed currently implements and what's missing.

**Overall Coverage: ~60% of methods from the survey are implemented**

---

## Implementation Status by Section

### 1. Consistency and Rationality Scores [MOSTLY COMPLETE]

| Method | Status | Function | Notes |
|--------|--------|----------|-------|
| CCEI (Afriat) | ✅ | `compute_aei()`, `compute_integrity_score()` | Binary search, O(T² log(1/ε)) |
| Min/Max MPI | ✅ | `compute_mpi()`, `compute_confusion_metric()` | Polynomial time |
| Houtman-Maks | ✅ | `compute_houtman_maks_index()`, `compute_menu_efficiency()` | MILP-based |
| Swaps Index | ✅ | `compute_swaps_index()` | Apesteguia & Ballester 2015 |
| Varian per-obs efficiency | ✅ | `compute_vei()`, `compute_granular_integrity()` | Per-observation scores |
| **Minimum Cost Index** | ❌ | - | Dean & Martin 2016 - NOT implemented |

### 2. Graph-Based Methods [COMPLETE]

| Method | Status | Function |
|--------|--------|----------|
| Warshall-Floyd transitive closure | ✅ | `floyd_warshall_transitive_closure()` |
| GARP/SARP cycle detection | ✅ | `check_garp()`, via DFS |
| ViolationGraph export | ✅ | `to_networkx()`, `to_dict()`, `find_shortest_cycles()` |
| Centrality computation | ✅ | `compute_centrality()` |

### 3. Consideration Set Models [MOSTLY COMPLETE]

| Method | Status | Function | Paper |
|--------|--------|----------|-------|
| WARP(LA) | ✅ | `test_warp_la()` | Masatlioglu et al. 2012 |
| Random Attention Model | ✅ | `fit_random_attention_model()` | Cattaneo et al. 2020 |
| Attention overload | ✅ | `test_attention_overload()` | Lleras et al. 2017 |
| **i-Asymmetry/i-Independence** | ❌ | - | Manzini & Mariotti 2014 - NOT implemented |

### 4. Menu and Context Effects [MOSTLY MISSING]

| Method | Status | Notes |
|--------|--------|-------|
| **Decoy/Attraction Effect** | ❌ | Huber et al. 1982 - NOT implemented |
| **Compromise Effect** | ❌ | Simonson 1989 - NOT implemented |
| **Context-dependent utility** | ❌ | Tversky & Simonson 1993 - NOT implemented |
| Regularity violations (decoy detection) | ✅ | `test_regularity()` detects when P(x|A) < P(x|B) for A⊂B |

### 5. Stochastic Choice Tests [MOSTLY COMPLETE]

| Method | Status | Function |
|--------|--------|----------|
| Block-Marschak polynomials | ✅ | Via `test_rum_consistency()` |
| RUM consistency | ✅ | `test_rum_consistency()` |
| Regularity test | ✅ | `test_regularity()` |
| IIA test | ✅ | `check_iia()` |
| Stochastic transitivity (WST/MST/SST) | ⚠️ | Partial - only via McFadden axioms |
| **Additive Perturbed Utility axioms** | ❌ | Fudenberg et al. 2015 - NOT explicitly implemented |

### 6. Preference Recovery/Bounds [PARTIALLY COMPLETE]

| Method | Status | Function |
|--------|--------|----------|
| Afriat utility bounds | ✅ | `recover_utility()`, `fit_latent_values()` |
| **E-bounds** | ❌ | Blundell et al. 2008 - NOT implemented |
| **i-bounds** | ❌ | Blundell et al. 2015 - NOT implemented |
| **Stochastic rationality bounds** | ❌ | Hoderlein & Stoye 2014 - NOT implemented |

### 7. Welfare Bounds [PARTIALLY COMPLETE]

| Method | Status | Function |
|--------|--------|----------|
| CV/EV | ✅ | `compute_cv()`, `compute_ev()` |
| Cost function recovery | ✅ | `recover_cost_function()` |
| **GAPP axiom** | ⚠️ | `check_gapp()` exists but limited |
| **Population welfare bounds** | ❌ | Deb et al. 2023 **HIGH PRIORITY** - NOT fully implemented |
| **Partial demand welfare** | ❌ | NOT implemented |

### 8. Temporal Methods [MOSTLY MISSING]

| Method | Status | Notes |
|--------|--------|-------|
| **Habit formation tests** | ❌ | Dynan 2000 - NOT implemented |
| **ATCA/RTCA axioms** | ❌ | Demuynck 2009 - NOT implemented |
| **Preference drift detection** | ❌ | NOT implemented |
| **Discount rate recovery** | ❌ | Adams et al. 2014 - NOT implemented |

### 9. Pairwise and Ranking Methods [MOSTLY MISSING]

| Method | Status | Notes |
|--------|--------|-------|
| **Bradley-Terry MLE** | ❌ | NOT directly implemented |
| **Heterogeneous annotators** | ❌ | Jin et al. 2020 - NOT implemented |
| **Kendall tau** | ❌ | NOT implemented |
| **Rank-Biased Overlap** | ❌ | NOT implemented |
| Condorcet cycles | ✅ | Via graph cycle detection |

### 10. Power Analysis [COMPLETE]

| Method | Status | Function |
|--------|--------|----------|
| Bronars power | ✅ | `compute_bronars_power()`, `compute_test_power()` |
| Fast power | ✅ | `compute_test_power_fast()` |
| **Bootstrap methods** | ❌ | Andreoni et al. 2013 - NOT fully implemented |

---

## Current Coverage Summary

| Category | Implemented | Missing | Coverage |
|----------|-------------|---------|----------|
| Consistency Scores | 5/6 | 1 | 83% |
| Graph Methods | 6/6 | 0 | 100% |
| Consideration Sets | 4/5 | 1 | 80% |
| Context Effects | 1/4 | 3 | 25% |
| Stochastic Choice | 5/7 | 2 | 71% |
| Preference Bounds | 2/5 | 3 | 40% |
| Welfare Analysis | 3/5 | 2 | 60% |
| Temporal Methods | 0/4 | 4 | 0% |
| Pairwise/Ranking | 1/5 | 4 | 20% |
| Power Analysis | 2/3 | 1 | 67% |

---

## Gap Analysis: High-Priority Missing Methods

### Tier 1: High Impact, Moderate Effort

| Method | Paper | Why Important | Effort |
|--------|-------|---------------|--------|
| **Context Effects (Decoy/Attraction)** | Huber et al. 1982 | Detect manipulation opportunities in menus | Medium |
| **Bradley-Terry Scores** | Bradley-Terry 1952 | Core method for RLHF/ranking from pairwise data | Low |
| **Stochastic Transitivity (explicit)** | Luce 1959 | WST/MST/SST as separate tests | Low |

### Tier 2: High Impact, High Effort

| Method | Paper | Why Important | Effort |
|--------|-------|---------------|--------|
| **GAPP + Population Welfare Bounds** | Deb et al. 2023 | "Highest-priority paper for ML welfare analysis" | High |
| **E-bounds/i-bounds** | Blundell et al. 2008/2015 | Tightest demand response bounds | High |
| **Temporal/Habit Methods** | Dynan 2000, Demuynck 2009 | Preference drift detection | High |

### Tier 3: Specialized Applications

| Method | Paper | Use Case |
|--------|-------|----------|
| Kendall tau / RBO | - | Ranking comparison metrics |
| Heterogeneous annotator model | Jin et al. 2020 | Crowdsourced preference data |
| Minimum Cost Index | Dean & Martin 2016 | Alternative violation severity |
| i-Asymmetry/i-Independence | Manzini & Mariotti 2014 | Item-level attention scores |

---

## Recommended Phase 3 Priorities

Based on production ML value and effort:

1. **Bradley-Terry MLE** - Core for RLHF, low effort
2. **Explicit WST/MST/SST tests** - Low effort, high interpretability
3. **Decoy/Attraction Effect Detection** - High value for recommendation systems
4. **Temporal preference tests** - Critical for A/B testing and drift detection
5. **Kendall tau and ranking metrics** - Standard benchmarking tools

---

## Key References

### Foundational Papers
- Afriat (1967, 1973) - Utility recovery, CCEI
- Varian (1982, 1990) - GARP formalization, per-observation efficiency
- Chambers & Echenique (2016) - Definitive textbook

### Consistency Indices
- Echenique, Lee & Shum (2011) - Money Pump Index
- Apesteguia & Ballester (2015) - Swaps Index
- Dean & Martin (2016) - Minimum Cost Index
- Smeulders et al. (2019) - NP-hardness results

### Attention Models
- Masatlioglu, Nakajima & Ozbay (2012) - WARP(LA)
- Manzini & Mariotti (2014) - Random consideration
- Cattaneo et al. (2020) - Random Attention Model
- Lleras et al. (2017) - Attention overload

### Stochastic Choice
- Block & Marschak (1960) - Block-Marschak polynomials
- Falmagne (1978) - RUM characterization
- Kitamura & Stoye (2018) - Statistical inference
- Smeulders et al. (2021) - Computational methods

### Welfare Analysis
- Blundell et al. (2008, 2015) - E-bounds, i-bounds
- Deb et al. (2023) - Population welfare bounds

### Power Analysis
- Bronars (1987) - Power against random behavior
- Andreoni et al. (2013) - Bootstrap methods
