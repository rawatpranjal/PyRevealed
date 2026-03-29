"""ML Benchmark: Per-user temporal split + user holdout + bootstrap CI + grouped importance.

Protocol:
  1. Per-user event-time split (first 70% → features, last 30% → targets)
  2. 80/20 random user holdout (stratified for classification)
  3. CatBoost defaults (random_seed=42, verbose=0)
  4. Bootstrap CI on test lift (1000 iterations)
  5. Grouped permutation importance (RP block vs baseline block)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, asdict, field
from typing import Literal

import numpy as np
import pandas as pd

from case_studies.benchmarks.config import SEED


MODEL_PARAMS = {
    "random_seed": SEED,
    "verbose": 0,
}


@dataclass
class BenchmarkResult:
    dataset: str
    target: str
    task_type: str
    n_users: int
    n_train: int
    n_test: int
    positive_rate: float = 0.0

    # Test set metrics
    auc_rp: float = 0.0
    auc_base: float = 0.0
    auc_combined: float = 0.0
    ap_rp: float = 0.0
    ap_base: float = 0.0
    ap_combined: float = 0.0
    r2_rp: float = 0.0
    r2_base: float = 0.0
    r2_combined: float = 0.0

    # Bootstrap CI on lift
    lift_pct: float = 0.0
    lift_ci_lower: float = 0.0
    lift_ci_upper: float = 0.0
    lift_p_value: float = 1.0

    # Grouped permutation importance
    group_importance: dict = field(default_factory=dict)

    top_features: list | None = None
    wall_time_s: float = 0.0

    # Performance metrics (disk-to-scores pipeline)
    load_time_s: float = 0.0       # Time to load raw data from disk
    engine_time_s: float = 0.0     # Time for Engine scoring (Rust backend)
    feature_time_s: float = 0.0    # Time for full feature extraction
    peak_memory_mb: float = 0.0    # Peak memory during scoring

    def to_dict(self) -> dict:
        return asdict(self)


def _bootstrap_lift(y_test, pred_base, pred_combined, metric_fn, n_boot=1000):
    """Bootstrap CI on lift percentage."""
    rng = np.random.default_rng(SEED)
    lifts = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_test), size=len(y_test), replace=True)
        try:
            b = metric_fn(y_test[idx], pred_base[idx])
            c = metric_fn(y_test[idx], pred_combined[idx])
            if b > 0:
                lifts.append((c - b) / b * 100)
        except ValueError:
            pass
    if not lifts:
        return 0.0, 0.0, 0.0, 1.0
    ci_lower, ci_upper = np.percentile(lifts, [2.5, 97.5])
    p_value = float(np.mean(np.array(lifts) <= 0))
    return float(np.mean(lifts)), ci_lower, ci_upper, p_value


def _grouped_permutation_importance(model, X_test, y_test, feature_names, rp_cols, base_cols, metric_fn, task_type):
    """Shuffle RP block and baseline block separately to measure group contribution."""
    if task_type == "classification":
        base_score = metric_fn(y_test, model.predict_proba(X_test)[:, 1])
    else:
        base_score = metric_fn(y_test, model.predict(X_test))

    results = {}
    rng = np.random.default_rng(SEED)

    for group_name, cols in [("RP_features", rp_cols), ("Baseline_features", base_cols)]:
        col_indices = [i for i, name in enumerate(feature_names) if name in cols]
        if not col_indices:
            results[group_name] = 0.0
            continue

        drops = []
        for _ in range(5):  # 5 repeats
            X_shuf = X_test.copy()
            perm = rng.permutation(len(X_test))
            for ci in col_indices:
                X_shuf[:, ci] = X_shuf[perm, ci]
            if task_type == "classification":
                shuf_score = metric_fn(y_test, model.predict_proba(X_shuf)[:, 1])
            else:
                shuf_score = metric_fn(y_test, model.predict(X_shuf))
            drops.append(base_score - shuf_score)
        results[group_name] = float(np.mean(drops))

    return results


def compute_lift_pct(combined: float, base: float) -> float:
    if base > 0.5:
        return (combined - base) / base * 100
    return 0.0


def run_three_way(
    X_rp: pd.DataFrame,
    X_base: pd.DataFrame,
    y: np.ndarray,
    dataset: str,
    target: str,
    task_type: Literal["classification", "regression"] = "classification",
    y_continuous: np.ndarray | None = None,
    threshold_pctl: float | None = None,
) -> BenchmarkResult:
    """Run the full protocol: train, evaluate, bootstrap CI, grouped importance.

    For classification with threshold_pctl: pass raw continuous values in
    y_continuous and the percentile (e.g. 66.67). The threshold is computed
    on TRAIN users only, then applied to TEST users. Zero leakage.
    """
    import time as _time
    _t0 = _time.time()

    from catboost import CatBoostClassifier, CatBoostRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, average_precision_score, r2_score

    # Align
    common_idx = X_rp.index.intersection(X_base.index)
    X_rp = X_rp.loc[common_idx]
    X_base = X_base.loc[common_idx]

    X_combined = pd.concat([X_base, X_rp], axis=1)
    X_combined = X_combined.loc[:, ~X_combined.columns.duplicated()]

    rp_cols = set(X_rp.columns)
    base_cols = set(X_base.columns)
    combined_names = list(X_combined.columns)

    feature_sets = {
        "rp": X_rp.values,
        "base": X_base.values,
        "combined": X_combined.values,
    }

    n_users = len(y)
    pos_rate = float(np.mean(y)) if task_type == "classification" else 0.0

    # 80/20 user holdout
    stratify = y if task_type == "classification" else None
    idx = np.arange(n_users)
    tr, te = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=stratify)

    # Fix threshold leakage: re-binarize using TRAIN-only threshold
    if y_continuous is not None and threshold_pctl is not None:
        train_threshold = np.percentile(y_continuous[tr], threshold_pctl)
        y = (y_continuous > train_threshold).astype(int)
        pos_rate = float(np.mean(y))

    y_train, y_test = y[tr], y[te]

    # Train 3 models, collect test predictions
    metrics = {}
    test_preds = {}
    combined_model = None

    for name, X_arr in feature_sets.items():
        # Impute with train medians only
        train_df = pd.DataFrame(X_arr[tr])
        med = train_df.median()
        X_tr = train_df.fillna(med).replace([np.inf, -np.inf], 0).values
        X_te = pd.DataFrame(X_arr[te]).fillna(med).replace([np.inf, -np.inf], 0).values

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if task_type == "classification":
                model = CatBoostClassifier(**MODEL_PARAMS)
                model.fit(X_tr, y_train)
                y_prob = model.predict_proba(X_te)[:, 1]
                try:
                    auc = roc_auc_score(y_test, y_prob)
                except ValueError:
                    auc = 0.5
                try:
                    ap = average_precision_score(y_test, y_prob)
                except ValueError:
                    ap = 0.0
                metrics[name] = {"auc": auc, "ap": ap}
                test_preds[name] = y_prob
            else:
                model = CatBoostRegressor(**MODEL_PARAMS)
                model.fit(X_tr, y_train)
                y_pred = model.predict(X_te)
                metrics[name] = {"r2": float(r2_score(y_test, y_pred))}
                test_preds[name] = y_pred

            if name == "combined":
                combined_model = model
                X_te_combined = X_te

    # Bootstrap CI on lift (always on AUC-PR as primary metric)
    if task_type == "classification":
        lift_mean, ci_lo, ci_hi, p_val = _bootstrap_lift(
            y_test, test_preds["base"], test_preds["combined"], average_precision_score
        )
    else:
        lift_mean, ci_lo, ci_hi, p_val = _bootstrap_lift(
            y_test, test_preds["base"], test_preds["combined"], r2_score
        )

    # Grouped permutation importance on test set
    metric_fn_perm = roc_auc_score if task_type == "classification" else r2_score
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        group_imp = _grouped_permutation_importance(
            combined_model, X_te_combined, y_test,
            combined_names, rp_cols, base_cols, metric_fn_perm, task_type
        )

    result = BenchmarkResult(
        dataset=dataset, target=target, task_type=task_type,
        n_users=n_users, n_train=len(tr), n_test=len(te),
        positive_rate=pos_rate,
        lift_pct=lift_mean, lift_ci_lower=ci_lo, lift_ci_upper=ci_hi,
        lift_p_value=p_val,
        group_importance=group_imp,
    )

    if task_type == "classification":
        result.auc_rp = metrics["rp"]["auc"]
        result.auc_base = metrics["base"]["auc"]
        result.auc_combined = metrics["combined"]["auc"]
        result.ap_rp = metrics["rp"]["ap"]
        result.ap_base = metrics["base"]["ap"]
        result.ap_combined = metrics["combined"]["ap"]
    else:
        result.r2_rp = metrics["rp"]["r2"]
        result.r2_base = metrics["base"]["r2"]
        result.r2_combined = metrics["combined"]["r2"]

    result.wall_time_s = _time.time() - _t0
    return result
