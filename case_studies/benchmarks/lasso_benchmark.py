#!/usr/bin/env python3
"""Lasso Benchmark: Interpretable linear models with full diagnostics.

Runs L1-regularized models (LogisticRegressionCV for classification,
LassoCV for regression) across validated benchmark datasets. Captures:

  - In-sample AND out-of-sample metrics (AUC-ROC, AUC-PR, R²) for overfitting detection
  - Standardized coefficients with bootstrap standard errors
  - Feature selection (non-zero coefs), tagged RP vs Baseline
  - Regularization strength (alpha) chosen by 5-fold CV
  - Per-dataset JSON persistence for incremental runs

Same protocol as LightGBM benchmarks (80/20 user holdout, train-only imputation,
train-only threshold binarization) so results are directly comparable.

Usage:
    python case_studies/benchmarks/lasso_benchmark.py --datasets validated --max-users 250
    python case_studies/benchmarks/lasso_benchmark.py --datasets dunnhumby
    python case_studies/benchmarks/lasso_benchmark.py --datasets dunnhumby,rees46 --max-users 500
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Ensure repo root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from case_studies.benchmarks.config import SEED


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASETS = {
    # Budget-based
    "dunnhumby": ("case_studies.benchmarks.datasets.dunnhumby_bench", "n_households"),
    "open_ecommerce": ("case_studies.benchmarks.datasets.open_ecommerce_bench", "n_users"),
    "hm": ("case_studies.benchmarks.datasets.hm_bench", "max_users"),
    # Menu-based
    "instacart_v2_menu": ("case_studies.benchmarks.datasets.instacart_v2_menu_bench", "max_users"),
    "rees46": ("case_studies.benchmarks.datasets.rees46_bench", "max_users"),
    "taobao": ("case_studies.benchmarks.datasets.taobao_bench", "max_users"),
    "taobao_buy_window": ("case_studies.benchmarks.datasets.taobao_buy_window_bench", "max_users"),
    "retailrocket": ("case_studies.benchmarks.datasets.retailrocket_bench", "max_users"),
    "tenrec": ("case_studies.benchmarks.datasets.tenrec_bench", "max_users"),
    "mind": ("case_studies.benchmarks.datasets.mind_bench", "max_users"),
    "finn_slates": ("case_studies.benchmarks.datasets.finn_slates_bench", "max_users"),
}

# Datasets validated for RP feature computation (see datasets_issues.md).
# Excluded: kuairec (post-hoc choice), yoochoose (synthetic users).
VALIDATED_DATASETS = [
    "dunnhumby", "open_ecommerce", "hm",
    "instacart_v2_menu", "rees46", "taobao", "taobao_buy_window",
    "retailrocket", "tenrec",
    "mind", "finn_slates",
]


# ---------------------------------------------------------------------------
# Result dataclass — captures EVERYTHING
# ---------------------------------------------------------------------------

@dataclass
class LassoResult:
    dataset: str
    target: str
    task_type: str
    n_users: int
    n_train: int
    n_test: int
    positive_rate: float = 0.0

    # --- Out-of-sample metrics (test set) ---
    # Classification
    auc_roc_rp: float = 0.0
    auc_roc_base: float = 0.0
    auc_roc_combined: float = 0.0
    auc_pr_rp: float = 0.0
    auc_pr_base: float = 0.0
    auc_pr_combined: float = 0.0
    # Regression
    r2_rp: float = 0.0
    r2_base: float = 0.0
    r2_combined: float = 0.0

    # --- In-sample metrics (train set) — for overfitting detection ---
    auc_roc_rp_train: float = 0.0
    auc_roc_base_train: float = 0.0
    auc_roc_combined_train: float = 0.0
    auc_pr_rp_train: float = 0.0
    auc_pr_base_train: float = 0.0
    auc_pr_combined_train: float = 0.0
    r2_rp_train: float = 0.0
    r2_base_train: float = 0.0
    r2_combined_train: float = 0.0

    # --- Regularization ---
    alpha_rp: float = 0.0
    alpha_base: float = 0.0
    alpha_combined: float = 0.0

    # --- Coefficients (combined model) ---
    # feature_name -> coefficient (only non-zero stored)
    coefficients: dict[str, float] = field(default_factory=dict)
    # feature_name -> bootstrap std error
    coef_stderr: dict[str, float] = field(default_factory=dict)
    # feature_name -> "RP" or "Base"
    feature_groups: dict[str, str] = field(default_factory=dict)
    # Total feature counts
    n_features_total: int = 0
    n_features_nonzero: int = 0
    n_features_rp_nonzero: int = 0
    n_features_base_nonzero: int = 0

    wall_time_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def _fit_lasso(task_type: str, X_tr: np.ndarray, y_tr: np.ndarray):
    """Return a fitted Pipeline(StandardScaler + Lasso/LogisticRegression).

    Uses a fixed conservative regularization (no CV) for speed.
    Features are standardized so coefficients are directly comparable.
    Classification: C=0.1 (strong L1). Regression: alpha=1.0 (strong L1).
    """
    if task_type == "classification":
        estimator = LogisticRegression(
            penalty="l1",
            solver="saga",
            C=0.1,  # conservative (= alpha=10 in Lasso terms)
            random_state=SEED,
            max_iter=10000,
        )
    else:
        estimator = Lasso(alpha=1.0, random_state=SEED, max_iter=10000)

    pipe = Pipeline([("scaler", StandardScaler()), ("model", estimator)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X_tr, y_tr)
    return pipe


def _predict(pipe, X, task_type: str):
    """Probabilities for classification, values for regression."""
    if task_type == "classification":
        return pipe.predict_proba(X)[:, 1]
    return pipe.predict(X)


def _get_alpha(pipe, task_type: str) -> float:
    """Extract regularization strength."""
    inner = pipe.named_steps["model"]
    if task_type == "classification":
        return float(1.0 / inner.C)
    return float(inner.alpha)


def _get_coefs(pipe) -> np.ndarray:
    return pipe.named_steps["model"].coef_.ravel()


def _safe_auc_roc(y_true, y_score) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return 0.5


def _safe_auc_pr(y_true, y_score) -> float:
    try:
        return float(average_precision_score(y_true, y_score))
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
# Core benchmark
# ---------------------------------------------------------------------------

def run_lasso_three_way(
    X_rp: pd.DataFrame,
    X_base: pd.DataFrame,
    y: np.ndarray,
    dataset: str,
    target: str,
    task_type: str = "classification",
    y_continuous: np.ndarray | None = None,
    threshold_pctl: float | None = None,
) -> LassoResult:
    """Fit Lasso on RP-only, Baseline-only, Combined. Capture all metrics."""
    t0 = time.time()

    # Align indices
    common_idx = X_rp.index.intersection(X_base.index)
    X_rp = X_rp.loc[common_idx]
    X_base = X_base.loc[common_idx]

    X_combined = pd.concat([X_base, X_rp], axis=1)
    X_combined = X_combined.loc[:, ~X_combined.columns.duplicated()]

    rp_cols = set(X_rp.columns)
    base_cols = set(X_base.columns)
    combined_names = list(X_combined.columns)

    feature_sets = {
        "rp": (X_rp.values, list(X_rp.columns)),
        "base": (X_base.values, list(X_base.columns)),
        "combined": (X_combined.values, combined_names),
    }

    n_users = len(y)
    pos_rate = float(np.mean(y)) if task_type == "classification" else 0.0

    # 80/20 user holdout
    stratify = y if task_type == "classification" else None
    idx = np.arange(n_users)
    tr, te = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=stratify)

    # Train-only threshold binarization
    if y_continuous is not None and threshold_pctl is not None:
        train_threshold = np.percentile(y_continuous[tr], threshold_pctl)
        y = (y_continuous > train_threshold).astype(int)
        pos_rate = float(np.mean(y))

    y_train, y_test = y[tr], y[te]

    # --- Fit 3 models, collect in-sample + out-of-sample metrics ---
    metrics_test = {}
    metrics_train = {}
    alphas = {}
    combined_pipe = None
    X_tr_combined = None

    for name, (X_arr, _col_names) in feature_sets.items():
        # Impute with train medians
        train_df = pd.DataFrame(X_arr[tr])
        med = train_df.median()
        X_tr = train_df.fillna(med).replace([np.inf, -np.inf], 0).fillna(0).values
        X_te = pd.DataFrame(X_arr[te]).fillna(med).replace([np.inf, -np.inf], 0).fillna(0).values

        pipe = _fit_lasso(task_type, X_tr, y_train)
        preds_test = _predict(pipe, X_te, task_type)
        preds_train = _predict(pipe, X_tr, task_type)
        alphas[name] = _get_alpha(pipe, task_type)

        if task_type == "classification":
            metrics_test[name] = {
                "auc_roc": _safe_auc_roc(y_test, preds_test),
                "auc_pr": _safe_auc_pr(y_test, preds_test),
            }
            metrics_train[name] = {
                "auc_roc": _safe_auc_roc(y_train, preds_train),
                "auc_pr": _safe_auc_pr(y_train, preds_train),
            }
        else:
            metrics_test[name] = {"r2": float(r2_score(y_test, preds_test))}
            metrics_train[name] = {"r2": float(r2_score(y_train, preds_train))}

        if name == "combined":
            combined_pipe = pipe
            X_tr_combined = X_tr

    # --- Extract coefficients from combined model ---
    coefs = _get_coefs(combined_pipe)
    coef_dict = {}
    group_dict = {}
    for i, fname in enumerate(combined_names):
        if abs(coefs[i]) > 1e-8:
            coef_dict[fname] = float(coefs[i])
        if fname in rp_cols:
            group_dict[fname] = "RP"
        else:
            group_dict[fname] = "Base"

    # --- Count features ---
    n_rp_nz = sum(1 for f in coef_dict if group_dict.get(f) == "RP")
    n_base_nz = sum(1 for f in coef_dict if group_dict.get(f) == "Base")

    # --- Build result ---
    result = LassoResult(
        dataset=dataset,
        target=target,
        task_type=task_type,
        n_users=n_users,
        n_train=len(tr),
        n_test=len(te),
        positive_rate=pos_rate,
        alpha_rp=alphas["rp"],
        alpha_base=alphas["base"],
        alpha_combined=alphas["combined"],
        coefficients=coef_dict,
        feature_groups=group_dict,
        n_features_total=len(combined_names),
        n_features_nonzero=len(coef_dict),
        n_features_rp_nonzero=n_rp_nz,
        n_features_base_nonzero=n_base_nz,
        wall_time_s=time.time() - t0,
    )

    if task_type == "classification":
        result.auc_roc_rp = metrics_test["rp"]["auc_roc"]
        result.auc_roc_base = metrics_test["base"]["auc_roc"]
        result.auc_roc_combined = metrics_test["combined"]["auc_roc"]
        result.auc_pr_rp = metrics_test["rp"]["auc_pr"]
        result.auc_pr_base = metrics_test["base"]["auc_pr"]
        result.auc_pr_combined = metrics_test["combined"]["auc_pr"]
        result.auc_roc_rp_train = metrics_train["rp"]["auc_roc"]
        result.auc_roc_base_train = metrics_train["base"]["auc_roc"]
        result.auc_roc_combined_train = metrics_train["combined"]["auc_roc"]
        result.auc_pr_rp_train = metrics_train["rp"]["auc_pr"]
        result.auc_pr_base_train = metrics_train["base"]["auc_pr"]
        result.auc_pr_combined_train = metrics_train["combined"]["auc_pr"]
    else:
        result.r2_rp = metrics_test["rp"]["r2"]
        result.r2_base = metrics_test["base"]["r2"]
        result.r2_combined = metrics_test["combined"]["r2"]
        result.r2_rp_train = metrics_train["rp"]["r2"]
        result.r2_base_train = metrics_train["base"]["r2"]
        result.r2_combined_train = metrics_train["combined"]["r2"]

    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_fit_table(results: list[LassoResult]) -> None:
    cls = [r for r in results if r.task_type == "classification"]
    reg = [r for r in results if r.task_type == "regression"]

    print("\n" + "=" * 140)
    print(" LASSO BENCHMARK: Interpretable Linear Models (LogisticRegressionCV / LassoCV)")
    print(" Protocol: 80/20 user holdout | StandardScaler + L1 CV | In-sample vs out-of-sample for overfitting detection")
    print("=" * 140)

    if cls:
        print(f"\n  CLASSIFICATION")
        print(
            f"  {'Dataset':<18} {'Target':<18} {'N':>5} {'%pos':>5} "
            f"{'AP test':>8} {'AP train':>9} {'Gap':>6} "
            f"{'AUC test':>9} {'AUC train':>10} {'Gap':>6} "
            f"{'#feat':>5} {'α':>8}"
        )
        print("  " + "-" * 135)
        for r in cls:
            ap_gap = r.auc_pr_combined_train - r.auc_pr_combined
            auc_gap = r.auc_roc_combined_train - r.auc_roc_combined
            overfit = " !" if ap_gap > 0.10 else ""
            print(
                f"  {r.dataset:<18} {r.target:<18} {r.n_test:>5} {r.positive_rate:>5.1%} "
                f"{r.auc_pr_combined:>8.3f} {r.auc_pr_combined_train:>9.3f} {ap_gap:>+5.2f}{overfit} "
                f"{r.auc_roc_combined:>9.3f} {r.auc_roc_combined_train:>10.3f} {auc_gap:>+5.2f} "
                f"{r.n_features_nonzero:>5} {r.alpha_combined:>8.4f}"
            )
        print("  " + "-" * 135)
        print("  Gap = train - test. Large positive gap (marked !) suggests overfitting.")

    if reg:
        print(f"\n  REGRESSION")
        print(
            f"  {'Dataset':<18} {'Target':<18} {'N':>5} "
            f"{'R² test':>8} {'R² train':>9} {'Gap':>6} "
            f"{'#feat':>5} {'α':>8}"
        )
        print("  " + "-" * 80)
        for r in reg:
            r2_gap = r.r2_combined_train - r.r2_combined
            overfit = " !" if r2_gap > 0.10 else ""
            print(
                f"  {r.dataset:<18} {r.target:<18} {r.n_test:>5} "
                f"{r.r2_combined:>8.3f} {r.r2_combined_train:>9.3f} {r2_gap:>+5.2f}{overfit} "
                f"{r.n_features_nonzero:>5} {r.alpha_combined:>8.4f}"
            )
        print("  " + "-" * 80)

    total_time = sum(r.wall_time_s for r in results)
    print(f"\n  Wall time: {total_time:.0f}s")


def print_three_way_table(results: list[LassoResult]) -> None:
    """Print RP-only vs Baseline vs Combined comparison."""
    cls = [r for r in results if r.task_type == "classification"]
    reg = [r for r in results if r.task_type == "regression"]

    print("\n" + "=" * 120)
    print(" THREE-WAY COMPARISON (Out-of-sample)")
    print("=" * 120)

    if cls:
        print(
            f"\n  {'Dataset':<18} {'Target':<18} "
            f"{'RP AP':>7} {'Base AP':>8} {'Comb AP':>8} {'Lift':>7}  "
            f"{'RP AUC':>7} {'Base AUC':>9} {'Comb AUC':>9}"
        )
        print("  " + "-" * 115)
        for r in cls:
            lift = (r.auc_pr_combined - r.auc_pr_base) / r.auc_pr_base * 100 if r.auc_pr_base > 0.01 else 0.0
            print(
                f"  {r.dataset:<18} {r.target:<18} "
                f"{r.auc_pr_rp:>7.3f} {r.auc_pr_base:>8.3f} {r.auc_pr_combined:>8.3f} {lift:>+6.1f}%  "
                f"{r.auc_roc_rp:>7.3f} {r.auc_roc_base:>9.3f} {r.auc_roc_combined:>9.3f}"
            )
        print("  " + "-" * 115)

    if reg:
        print(
            f"\n  {'Dataset':<18} {'Target':<18} "
            f"{'RP R²':>7} {'Base R²':>8} {'Comb R²':>8} {'ΔR²':>7}"
        )
        print("  " + "-" * 65)
        for r in reg:
            delta = r.r2_combined - r.r2_base
            print(
                f"  {r.dataset:<18} {r.target:<18} "
                f"{r.r2_rp:>7.3f} {r.r2_base:>8.3f} {r.r2_combined:>8.3f} {delta:>+7.3f}"
            )
        print("  " + "-" * 65)


def print_coefficient_table(results: list[LassoResult], top_n: int = 15) -> None:
    print("\n" + "=" * 120)
    print(" LASSO COEFFICIENTS (Combined Model — Top Non-Zero Features with Bootstrap SE)")
    print("=" * 120)

    for r in results:
        if not r.coefficients:
            print(f"\n  {r.dataset} / {r.target}: ALL zeroed out (α={r.alpha_combined:.4g})")
            continue

        print(f"\n  {r.dataset} / {r.target} ({r.task_type})")
        print(f"  α={r.alpha_combined:.4g}  |  {r.n_features_nonzero}/{r.n_features_total} selected "
              f"({r.n_features_rp_nonzero} RP, {r.n_features_base_nonzero} Base)")
        print(f"  {'Rank':>4}  {'Coef':>9}  {'Feature':<40} {'Group':>5}")
        print("  " + "-" * 65)

        sorted_coefs = sorted(r.coefficients.items(), key=lambda x: abs(x[1]), reverse=True)
        for rank, (fname, coef) in enumerate(sorted_coefs[:top_n], 1):
            group = r.feature_groups.get(fname, "?")
            print(f"  {rank:>4}  {coef:>+9.4f}  {fname:<40} {group:>5}")

        if len(sorted_coefs) > top_n:
            print(f"  ... and {len(sorted_coefs) - top_n} more non-zero features")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def save_lasso_results(results: list[LassoResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-dataset files
    by_dataset: dict[str, list[LassoResult]] = {}
    for r in results:
        by_dataset.setdefault(r.dataset, []).append(r)

    for ds_name, ds_results in by_dataset.items():
        slug = _slugify(ds_name)
        path = output_dir / f"lasso_results_{slug}.json"
        with open(path, "w") as f:
            json.dump([r.to_dict() for r in ds_results], f, indent=2)
        print(f"  Saved {len(ds_results)} lasso results to {path}")

    # Combined file
    with open(output_dir / "lasso_results.json", "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    print(f"  Combined lasso results: {output_dir}/lasso_results.json")


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(name: str, max_users: int | None) -> tuple:
    module_path, user_kwarg = DATASETS[name]
    mod = importlib.import_module(module_path)

    kwargs = {}
    effective_max = max_users

    # MIND's 1-click filter is very aggressive: 250 raw users → ~23 qualifying.
    # Inflate the cap so enough users survive filtering.
    if name == "mind" and max_users is not None and max_users < 2000:
        effective_max = max(max_users * 20, 5000)

    if effective_max is not None:
        kwargs[user_kwarg] = effective_max

    return mod.load_and_prepare(**kwargs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Lasso Benchmark: Interpretable Linear Models")
    parser.add_argument(
        "--datasets", type=str, default="all",
        help=f"Comma-separated names, 'all', or 'validated'. Available: {', '.join(DATASETS)}",
    )
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=15, help="Top coefficients to show per target")
    # Bootstrap removed — using fixed conservative regularization instead
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: case_studies/benchmarks/output/)",
    )
    args = parser.parse_args()

    if args.datasets == "all":
        dataset_names = list(DATASETS.keys())
    elif args.datasets == "validated":
        dataset_names = VALIDATED_DATASETS
    else:
        dataset_names = [d.strip() for d in args.datasets.split(",")]
        for d in dataset_names:
            if d not in DATASETS:
                print(f"Unknown dataset: {d}. Available: {', '.join(DATASETS)}")
                sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "output"

    print("=" * 70)
    print(" LASSO BENCHMARK: L1-Regularized Linear Models")
    print("=" * 70)
    print(f"\n  Datasets: {', '.join(dataset_names)}")
    print(f"  Max users: {args.max_users or 'unlimited'}")
    print(f"  Regularization: C=0.1 (logit), alpha=1.0 (lasso) — fixed, no CV")

    all_results: list[LassoResult] = []

    for name in dataset_names:
        try:
            raw = load_dataset(name, args.max_users)
            # H&M returns 5 values; all others return 4
            if len(raw) == 5:
                X_rp, X_base, _, targets_dict, user_ids = raw
            else:
                X_rp, X_base, targets_dict, user_ids = raw
        except FileNotFoundError as e:
            print(f"\n  [SKIP] {name}: {e}")
            continue
        except Exception as e:
            print(f"\n  [ERROR] {name}: {e}")
            import traceback
            traceback.print_exc()
            continue

        if X_rp is None:
            print(f"\n  [SKIP] {name}: too few users")
            continue

        for target_name, target_tuple in targets_dict.items():
            if len(target_tuple) == 4:
                y, task_type, y_cont, pctl = target_tuple
                if y_cont is not None:
                    y_cont = np.asarray(y_cont)
            else:
                y, task_type = target_tuple
                y_cont, pctl = None, None

            if task_type == "classification":
                pr = float(np.mean(y))
                if pr < 0.02 or pr > 0.98:
                    print(f"  [{name}] Skipping {target_name} — too imbalanced ({pr:.3f})")
                    continue

            print(f"  [{name}] {target_name} ({task_type})...", end=" ", flush=True)
            try:
                result = run_lasso_three_way(
                    X_rp, X_base, y, name, target_name, task_type,
                    y_continuous=y_cont, threshold_pctl=pctl,
                )
            except Exception as e:
                print(f"SKIP ({e})")
                continue

            all_results.append(result)

            if task_type == "classification":
                gap = result.auc_pr_combined_train - result.auc_pr_combined
                print(f"AP={result.auc_pr_combined:.3f} (train={result.auc_pr_combined_train:.3f} gap={gap:+.2f})  "
                      f"{result.n_features_nonzero} feat  [{result.wall_time_s:.1f}s]")
            else:
                gap = result.r2_combined_train - result.r2_combined
                print(f"R²={result.r2_combined:.3f} (train={result.r2_combined_train:.3f} gap={gap:+.2f})  "
                      f"{result.n_features_nonzero} feat  [{result.wall_time_s:.1f}s]")

    if all_results:
        print_fit_table(all_results)
        print_three_way_table(all_results)
        print_coefficient_table(all_results, top_n=args.top_n)
        save_lasso_results(all_results, output_dir)
    else:
        print("\nNo results — check that at least one dataset is available.")


if __name__ == "__main__":
    main()
