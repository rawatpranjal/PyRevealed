"""Three-way model comparison with rigorous ML evaluation.

Evaluation modes:
  - Out-of-sample: 5-fold stratified CV (splits users, not time)
  - In-sample: Train on all data, predict on all data (overfitting check)

Data leakage prevention:
  Each dataset module must implement a GLOBAL CALENDAR CUTOFF:
  - Features extracted from observations BEFORE the cutoff date
  - Targets computed from observations AFTER the cutoff date
  - This ensures no temporal overlap across users

Metrics (classification):
  - AUC-ROC: Discrimination ability across all thresholds
  - AUC-PR: Average precision (better for imbalanced targets)
  - Log Loss: Calibration quality
  - F1: Balanced precision-recall at default threshold

Metrics (regression):
  - R²: Explained variance
  - RMSE: Root mean squared error
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, asdict
from typing import Literal

import numpy as np
import pandas as pd

from case_studies.benchmarks.config import (
    LGBM_CLASSIFIER_PARAMS,
    LGBM_REGRESSOR_PARAMS,
    N_FOLDS,
    SEED,
)


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

    # --- Out-of-sample classification ---
    auc_rp: float = 0.0
    auc_base: float = 0.0
    auc_combined: float = 0.0
    auc_rp_std: float = 0.0
    auc_base_std: float = 0.0
    auc_combined_std: float = 0.0

    # AUC-PR (average precision)
    ap_rp: float = 0.0
    ap_base: float = 0.0
    ap_combined: float = 0.0

    # Log loss
    logloss_rp: float = 0.0
    logloss_base: float = 0.0
    logloss_combined: float = 0.0

    f1_rp: float = 0.0
    f1_base: float = 0.0
    f1_combined: float = 0.0

    # --- In-sample classification ---
    auc_rp_train: float = 0.0
    auc_base_train: float = 0.0
    auc_combined_train: float = 0.0

    # --- Out-of-sample regression ---
    rmse_rp: float = 0.0
    rmse_base: float = 0.0
    rmse_combined: float = 0.0
    r2_rp: float = 0.0
    r2_base: float = 0.0
    r2_combined: float = 0.0

    # --- In-sample regression ---
    r2_rp_train: float = 0.0
    r2_base_train: float = 0.0
    r2_combined_train: float = 0.0

    top_features: list | None = None
    auc_lift: float = 0.0
    auc_lift_pct: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def run_three_way(
    X_rp: pd.DataFrame,
    X_base: pd.DataFrame,
    y: np.ndarray,
    dataset: str,
    target: str,
    task_type: Literal["classification", "regression"] = "classification",
) -> BenchmarkResult:
    """Run three-way model comparison with cross-validation.

    Three models: (a) RP only, (b) Baseline only, (c) RP + Baseline.
    """
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold, KFold
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        log_loss,
        f1_score,
        mean_squared_error,
        r2_score,
    )

    # Align indices
    common_idx = X_rp.index.intersection(X_base.index)
    X_rp = X_rp.loc[common_idx]
    X_base = X_base.loc[common_idx]

    # Handle NaN/inf
    X_rp = X_rp.fillna(X_rp.median()).replace([np.inf, -np.inf], 0)
    X_base = X_base.fillna(X_base.median()).replace([np.inf, -np.inf], 0)

    X_combined = pd.concat([X_base, X_rp], axis=1)
    X_combined = X_combined.loc[:, ~X_combined.columns.duplicated()]

    feature_sets = {
        "rp": X_rp.values,
        "base": X_base.values,
        "combined": X_combined.values,
    }
    feature_names = {
        "rp": list(X_rp.columns),
        "base": list(X_base.columns),
        "combined": list(X_combined.columns),
    }

    n_users = len(y)
    pos_rate = float(np.mean(y)) if task_type == "classification" else 0.0

    if task_type == "classification":
        kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        splits = list(kf.split(np.zeros(n_users), y))
        params = LGBM_CLASSIFIER_PARAMS.copy()
    else:
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        splits = list(kf.split(np.zeros(n_users)))
        params = LGBM_REGRESSOR_PARAMS.copy()

    # --- Out-of-sample: K-fold CV ---
    fold_metrics = {
        name: {"auc": [], "ap": [], "logloss": [], "f1": [], "rmse": [], "r2": []}
        for name in feature_sets
    }

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        y_train, y_test = y[train_idx], y[test_idx]

        for name, X_arr in feature_sets.items():
            X_train, X_test = X_arr[train_idx], X_arr[test_idx]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if task_type == "classification":
                    model = lgb.LGBMClassifier(**params)
                    model.fit(X_train, y_train)
                    y_prob = model.predict_proba(X_test)[:, 1]
                    y_pred = model.predict(X_test)

                    try:
                        fold_metrics[name]["auc"].append(roc_auc_score(y_test, y_prob))
                    except ValueError:
                        fold_metrics[name]["auc"].append(0.5)
                    try:
                        fold_metrics[name]["ap"].append(average_precision_score(y_test, y_prob))
                    except ValueError:
                        fold_metrics[name]["ap"].append(0.0)
                    try:
                        fold_metrics[name]["logloss"].append(log_loss(y_test, y_prob))
                    except ValueError:
                        fold_metrics[name]["logloss"].append(1.0)
                    fold_metrics[name]["f1"].append(f1_score(y_test, y_pred, zero_division=0))
                else:
                    model = lgb.LGBMRegressor(**params)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    fold_metrics[name]["rmse"].append(np.sqrt(mean_squared_error(y_test, y_pred)))
                    fold_metrics[name]["r2"].append(r2_score(y_test, y_pred))

    # --- In-sample ---
    in_sample = {}
    for name, X_arr in feature_sets.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if task_type == "classification":
                model = lgb.LGBMClassifier(**params)
                model.fit(X_arr, y)
                y_prob_train = model.predict_proba(X_arr)[:, 1]
                try:
                    in_sample[name] = roc_auc_score(y, y_prob_train)
                except ValueError:
                    in_sample[name] = 0.5
            else:
                model = lgb.LGBMRegressor(**params)
                model.fit(X_arr, y)
                in_sample[name] = r2_score(y, model.predict(X_arr))

    # --- Feature importance ---
    top_features = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if task_type == "classification":
            final_model = lgb.LGBMClassifier(**params)
        else:
            final_model = lgb.LGBMRegressor(**params)
        final_model.fit(X_combined.values, y)
        importances = final_model.feature_importances_
        feat_imp = sorted(
            zip(feature_names["combined"], importances.tolist()),
            key=lambda x: x[1], reverse=True,
        )
        top_features = feat_imp[:15]

    # --- Assemble ---
    result = BenchmarkResult(
        dataset=dataset, target=target, task_type=task_type,
        n_users=n_users, n_rp_features=X_rp.shape[1],
        n_base_features=X_base.shape[1], positive_rate=pos_rate,
        top_features=top_features,
    )

    if task_type == "classification":
        result.auc_rp = float(np.mean(fold_metrics["rp"]["auc"]))
        result.auc_base = float(np.mean(fold_metrics["base"]["auc"]))
        result.auc_combined = float(np.mean(fold_metrics["combined"]["auc"]))
        result.auc_rp_std = float(np.std(fold_metrics["rp"]["auc"]))
        result.auc_base_std = float(np.std(fold_metrics["base"]["auc"]))
        result.auc_combined_std = float(np.std(fold_metrics["combined"]["auc"]))
        result.ap_rp = float(np.mean(fold_metrics["rp"]["ap"]))
        result.ap_base = float(np.mean(fold_metrics["base"]["ap"]))
        result.ap_combined = float(np.mean(fold_metrics["combined"]["ap"]))
        result.logloss_rp = float(np.mean(fold_metrics["rp"]["logloss"]))
        result.logloss_base = float(np.mean(fold_metrics["base"]["logloss"]))
        result.logloss_combined = float(np.mean(fold_metrics["combined"]["logloss"]))
        result.f1_rp = float(np.mean(fold_metrics["rp"]["f1"]))
        result.f1_base = float(np.mean(fold_metrics["base"]["f1"]))
        result.f1_combined = float(np.mean(fold_metrics["combined"]["f1"]))
        result.auc_lift = result.auc_combined - result.auc_base
        if result.auc_base > 0.5:
            result.auc_lift_pct = (result.auc_combined - result.auc_base) / result.auc_base * 100
        result.auc_rp_train = in_sample["rp"]
        result.auc_base_train = in_sample["base"]
        result.auc_combined_train = in_sample["combined"]
    else:
        result.rmse_rp = float(np.mean(fold_metrics["rp"]["rmse"]))
        result.rmse_base = float(np.mean(fold_metrics["base"]["rmse"]))
        result.rmse_combined = float(np.mean(fold_metrics["combined"]["rmse"]))
        result.r2_rp = float(np.mean(fold_metrics["rp"]["r2"]))
        result.r2_base = float(np.mean(fold_metrics["base"]["r2"]))
        result.r2_combined = float(np.mean(fold_metrics["combined"]["r2"]))
        result.r2_rp_train = in_sample["rp"]
        result.r2_base_train = in_sample["base"]
        result.r2_combined_train = in_sample["combined"]

    return result
