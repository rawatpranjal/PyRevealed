# PyRevealed User Workflow Testing Summary

**Date**: 2026-01-09
**Tester**: Claude (acting as new user)
**Package Version**: 0.3.0

## Overview

Tested the PyRevealed package by creating and running 3 tutorial notebooks that simulate new user workflows.

## Notebooks Created

1. `01_basic_workflow.ipynb` - Core API smoke tests
2. `02_real_data_analysis.ipynb` - Real data from Prest project
3. `03_preference_structure.ipynb` - Advanced structure tests

## Issues Found

### Bug: scipy OptimizeWarning

**Location**: `src/pyrevealed/algorithms/utility.py:106`
**Severity**: Low (cosmetic)
**Description**: When calling `PreferenceEncoder.fit()`, scipy emits a warning:
```
OptimizeWarning: Unrecognized options detected: {'tol': 1e-08}
```
**Root cause**: The `tol` parameter is being passed to `linprog()` but the HiGHS solver doesn't recognize it.
**Suggested fix**: Either remove the `tol` option or use the correct HiGHS tolerance parameter.

### API Issue: Inconsistent Parameter Names

**Location**: `src/pyrevealed/encoder.py` - `PreferenceEncoder.predict_choice()`
**Severity**: Medium (UX inconsistency)
**Description**: The method uses `cost_vector` and `resource_limit` parameters, which don't match the tech-friendly naming convention used elsewhere (`prices`/`budget` would be more intuitive).
**Current signature**: `predict_choice(cost_vector, resource_limit)`
**Expected by users**: `predict_choice(prices, budget)`

### Environment Issue: SSL Certificates

**Location**: External (Python installation)
**Severity**: N/A (not a package bug)
**Description**: On some macOS Python installations, loading remote data fails with SSL certificate errors. This is a system issue, not a PyRevealed bug.

### Feature Request: Batch Processing API

**Location**: `BehavioralAuditor`
**Severity**: Enhancement
**Description**: When analyzing many users/subjects, the user must loop manually. A batch API like `auditor.audit_batch([log1, log2, ...])` would be convenient.

## What Works Well

### Core Functionality (All Passing)
- [x] Package imports correctly
- [x] `BehaviorLog` creation with `cost_vectors`/`action_vectors`
- [x] `validate_consistency()` - GARP checking
- [x] `compute_integrity_score()` - AEI calculation
- [x] `compute_confusion_metric()` - MPI calculation
- [x] `BehavioralAuditor.full_audit()` - High-level API
- [x] `PreferenceEncoder.fit()` - ML feature extraction
- [x] `PreferenceEncoder.extract_latent_values()`
- [x] `PreferenceEncoder.extract_marginal_weights()`

### Structure Tests (All Passing)
- [x] `test_feature_independence()` with `group_a`/`group_b` params
- [x] `test_cross_price_effect()` with `good_g`/`good_h` params

### Error Handling (Excellent)
- [x] Clear error messages with hints
- [x] `ValueRangeError` for negative prices
- [x] `DimensionError` for shape mismatches
- [x] All error messages include actionable suggestions

### Edge Cases (All Passing)
- [x] Single observation (trivially consistent)
- [x] Known WARP violations detected correctly
- [x] Violations list correctly populated

## Recommendations

1. **High Priority**: Fix scipy warning by updating `linprog()` call
2. **Medium Priority**: Rename `predict_choice` params to `prices`/`budget`
3. **Low Priority**: Add batch processing API to `BehavioralAuditor`

## Test Commands Used

```bash
# Install package
pip install .

# Run quick smoke test
python3 -c "from pyrevealed import BehaviorLog, validate_consistency; print('OK')"

# Run full test suite
pytest tests/
```

## Conclusion

The PyRevealed package is functional and well-designed. The tech-friendly API works as documented for the most part. Error handling is excellent with clear, actionable messages. The main issues are cosmetic (scipy warning) and a minor API naming inconsistency.
