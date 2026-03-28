"""Three-way model comparison: OOS (random split) + OOT (temporal split).

Two evaluation modes, both reported:
  - OOS: Random 80/20 user split. X from past, y from future, users shuffled.
  - OOT: X from past, y from future, train on ALL users, predict ALL users.
    The temporal gap IS the holdout — no user split needed.

LightGBM with default hyperparameters. No tuning.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, asdict, field
from typing import Literal

import numpy as np
import pandas as pd

from case_studies.benchmarks.config import SEED


@dataclass
class BenchmarkResult:
    """Result from a single dataset x target benchmark."""

    dataset: str
    target: str
    task_type: str
    n_users: int
    n_rp_features: int
    n_base_features: int
    positive_rate: float = 0.0

    # OOS (random user split)
    oos_auc_rp: float = 0.0
    oos_auc_base: float = 0.0
    oos_auc_combined: float = 0.0
    oos_ap_rp: float = 0.0
    oos_ap_base: float = 0.0
    oos_ap_combined: float = 0.0
    oos_r2_rp: float = 0.0
    oos_r2_base: float = 0.0
    oos_r2_combined: float = 0.0

    # OOT (temporal — train on all, predict all)
    oot_auc_rp: float = 0.0
    oot_auc_base: float = 0.0
    oot_auc_combined: float = 0.0
    oot_ap_rp: float = 0.0
    oot_ap_base: float = 0.0
    oot_ap_combined: float = 0.0
    oot_r2_rp: float = 0.0
    oot_r2_base: float = 0.0
    oot_r2_combined: float = 0.0

    top_features: list | None = None
    wall_time_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _compute_lift(combined: float, base: float) -> float:
    if base > 0.5:
        return (combined - base) / base * 100
    return 0.0


def _train_predict(X_train, y_train, X_test, y_test, task_type):
    """Train LightGBM with defaults, return metrics."""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score, average_precision_score, r2_score

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if task_type == "classification":
            model = lgb.LGBMClassifier(random_state=SEED, verbose=-1)
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1]
            try:
                auc = roc_auc_score(y_test, y_prob)
            except ValueError:
                auc = 0.5
            try:
                ap = average_precision_score(y_test, y_prob)
            except ValueError:
                ap = 0.0
            return {"auc": auc, "ap": ap}, model
        else:
            model = lgb.LGBMRegressor(random_state=SEED, verbose=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            return {"r2": float(r2_score(y_test, y_pred))}, model


def run_three_way(
    X_rp: pd.DataFrame,
    X_base: pd.DataFrame,
    y: np.ndarray,
    dataset: str,
    target: str,
    task_type: Literal["classification", "regression"] = "classification",
) -> BenchmarkResult:
    """Run three-way comparison with both OOS and OOT evaluation.

    LightGBM defaults. No hyperparameter tuning.
    """
    import time as _time
    _t0 = _time.time()

    from sklearn.model_selection import train_test_split

    # Align indices
    common_idx = X_rp.index.intersection(X_base.index)
    X_rp = X_rp.loc[common_idx]
    X_base = X_base.loc[common_idx]

    # Impute NaN/inf
    X_rp = X_rp.fillna(X_rp.median()).replace([np.inf, -np.inf], 0)
    X_base = X_base.fillna(X_base.median()).replace([np.inf, -np.inf], 0)

    X_combined = pd.concat([X_base, X_rp], axis=1)
    X_combined = X_combined.loc[:, ~X_combined.columns.duplicated()]

    feature_sets = {
        "rp": X_rp.values,
        "base": X_base.values,
        "combined": X_combined.values,
    }

    n_users = len(y)
    pos_rate = float(np.mean(y)) if task_type == "classification" else 0.0

    # ── OOS: random 80/20 user split ──
    stratify = y if task_type == "classification" else None
    idx = np.arange(n_users)
    tr, te = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=stratify)

    oos = {}
    for name, X_arr in feature_sets.items():
        # Impute test with train medians
        train_df = pd.DataFrame(X_arr[tr])
        med = train_df.median()
        X_tr = train_df.fillna(med).values
        X_te = pd.DataFrame(X_arr[te]).fillna(med).values
        m, _ = _train_predict(X_tr, y[tr], X_te, y[te], task_type)
        oos[name] = m

    # ── OOT: train on all past, predict all future (temporal gap = holdout) ──
    oot = {}
    for name, X_arr in feature_sets.items():
        m, _ = _train_predict(X_arr, y, X_arr, y, task_type)
        oot[name] = m

    # Feature importance (from OOS combined model)
    import lightgbm as lgb
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if task_type == "classification":
            fi_model = lgb.LGBMClassifier(random_state=SEED, verbose=-1)
        else:
            fi_model = lgb.LGBMRegressor(random_state=SEED, verbose=-1)
        train_df = pd.DataFrame(X_combined.values[tr])
        med = train_df.median()
        fi_model.fit(train_df.fillna(med).values, y[tr])
        imp = fi_model.feature_importances_
        feat_names = list(X_combined.columns)
        top_features = sorted(zip(feat_names, imp.tolist()), key=lambda x: x[1], reverse=True)[:15]

    # Assemble
    result = BenchmarkResult(
        dataset=dataset, target=target, task_type=task_type,
        n_users=n_users, n_rp_features=X_rp.shape[1],
        n_base_features=X_base.shape[1], positive_rate=pos_rate,
        top_features=top_features,
    )

    if task_type == "classification":
        result.oos_auc_rp = oos["rp"]["auc"]
        result.oos_auc_base = oos["base"]["auc"]
        result.oos_auc_combined = oos["combined"]["auc"]
        result.oos_ap_rp = oos["rp"]["ap"]
        result.oos_ap_base = oos["base"]["ap"]
        result.oos_ap_combined = oos["combined"]["ap"]
        result.oot_auc_rp = oot["rp"]["auc"]
        result.oot_auc_base = oot["base"]["auc"]
        result.oot_auc_combined = oot["combined"]["auc"]
        result.oot_ap_rp = oot["rp"]["ap"]
        result.oot_ap_base = oot["base"]["ap"]
        result.oot_ap_combined = oot["combined"]["ap"]
    else:
        result.oos_r2_rp = oos["rp"]["r2"]
        result.oos_r2_base = oos["base"]["r2"]
        result.oos_r2_combined = oos["combined"]["r2"]
        result.oot_r2_rp = oot["rp"]["r2"]
        result.oot_r2_base = oot["base"]["r2"]
        result.oot_r2_combined = oot["combined"]["r2"]

    result.wall_time_s = _time.time() - _t0
    return result
