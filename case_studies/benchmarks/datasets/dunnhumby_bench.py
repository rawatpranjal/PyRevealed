"""Dunnhumby grocery benchmark: global calendar cutoff + user holdout.

Protocol:
  1. Read raw transaction DAY column → find 70th percentile day
  2. Features from transactions BEFORE cutoff day
  3. Targets from transactions AFTER cutoff day
  4. Users with zero post-cutoff spend → churned (test_spend = 0)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from case_studies.benchmarks.config import MIN_OBS_BUDGET
from case_studies.benchmarks.core.features import extract_budget_baseline, extract_budget_rp
from case_studies.benchmarks.core.evaluation import run_three_way, BenchmarkResult


DATASET_NAME = "Dunnhumby"
DEFAULT_DATA_DIR = str(Path(__file__).resolve().parents[2] / "dunnhumby" / "data")
TOP_COMMODITIES = [
    "FLUID MILK WHITE ONLY", "SOFT DRINKS", "COLD CEREAL",
    "CHEESE", "YOGURT", "ICE CREAM/SHERBET",
    "SHREDDED CHEESE", "LUNCH MEAT", "BREAD",
    "BUTTER/MARGARINE",
]


def load_and_prepare(data_dir=None, n_households=None):
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    data_path = Path(data_dir)

    print(f"\n[{DATASET_NAME}] Loading dataset...")

    # Read raw transactions with DAY column for global cutoff
    trans = pd.read_csv(data_path / "transaction_data.csv",
                        usecols=["household_key", "DAY", "PRODUCT_ID", "QUANTITY",
                                 "SALES_VALUE", "RETAIL_DISC", "COUPON_DISC", "WEEK_NO"])
    products = pd.read_csv(data_path / "product.csv",
                           usecols=["PRODUCT_ID", "COMMODITY_DESC"])

    trans = trans.merge(products, on="PRODUCT_ID")
    trans = trans[trans["COMMODITY_DESC"].isin(TOP_COMMODITIES)]

    # Compute unit price
    trans["unit_price"] = (trans["SALES_VALUE"] - trans["RETAIL_DISC"] - trans["COUPON_DISC"]) / trans["QUANTITY"]
    trans = trans[(trans["unit_price"] > 0.01) & (trans["unit_price"] < 50)]

    # Global calendar cutoff: 70th percentile of DAY
    cutoff_day = int(np.percentile(trans["DAY"].unique(), 70))
    print(f"  Global cutoff: DAY {cutoff_day} (70th pctl of {trans['DAY'].nunique()} unique days)")

    # Split transactions
    trans_train = trans[trans["DAY"] <= cutoff_day]
    trans_test = trans[trans["DAY"] > cutoff_day]

    # All users who have ANY train data
    train_users = set(trans_train["household_key"].unique())
    if n_households:
        train_users = set(list(train_users)[:n_households])

    # Price oracle from TRAIN data only
    price_oracle = (trans_train.groupby(["WEEK_NO", "COMMODITY_DESC"])["unit_price"]
                    .median().unstack().reindex(columns=TOP_COMMODITIES).ffill().bfill())
    price_grid = price_oracle.values  # (n_weeks, n_commodities)

    # Build per-user (prices, quantities) for TRAIN window
    user_ids = []
    train_tuples = []
    train_mean_spends = []
    test_mean_spends = []
    test_total_spends = []

    for hh in train_users:
        hh_train = trans_train[trans_train["household_key"] == hh]
        weeks_train = sorted(hh_train["WEEK_NO"].unique())

        if len(weeks_train) < MIN_OBS_BUDGET:
            continue

        # Build train arrays
        T = len(weeks_train)
        K = len(TOP_COMMODITIES)
        qty_matrix = np.zeros((T, K))
        week_to_idx = {w: i for i, w in enumerate(weeks_train)}

        for _, row in hh_train.iterrows():
            t = week_to_idx.get(row["WEEK_NO"])
            if t is None:
                continue
            c = TOP_COMMODITIES.index(row["COMMODITY_DESC"]) if row["COMMODITY_DESC"] in TOP_COMMODITIES else -1
            if c >= 0:
                qty_matrix[t, c] += row["QUANTITY"]

        # Price matrix from oracle
        price_indices = [min(w - 1, len(price_grid) - 1) for w in weeks_train]
        price_matrix = price_grid[price_indices]
        price_matrix = np.nan_to_num(price_matrix, nan=1.0)

        train_tuples.append((price_matrix.astype(np.float64), qty_matrix.astype(np.float64)))
        user_ids.append(f"household_{hh}")

        train_spend = np.sum(price_matrix * qty_matrix, axis=1)
        train_mean_spends.append(float(np.mean(train_spend)))

        # Test window: same user, post-cutoff transactions
        hh_test = trans_test[trans_test["household_key"] == hh]
        if len(hh_test) > 0:
            test_spend_total = float((hh_test["SALES_VALUE"] - hh_test["RETAIL_DISC"] - hh_test["COUPON_DISC"]).sum())
            n_test_weeks = hh_test["WEEK_NO"].nunique()
            test_mean_spends.append(test_spend_total / max(n_test_weeks, 1))
            test_total_spends.append(test_spend_total)
        else:
            # No post-cutoff data → churned
            test_mean_spends.append(0.0)
            test_total_spends.append(0.0)

    train_mean_spends = np.array(train_mean_spends)
    test_mean_spends = np.array(test_mean_spends)
    test_total_spends = np.array(test_total_spends)

    n_churned = int(np.sum(test_total_spends == 0))
    print(f"  Users: {len(user_ids)} ({n_churned} churned — zero post-cutoff spend)")

    # Features from PAST
    print(f"  Extracting baseline features...")
    X_base = extract_budget_baseline(train_tuples, user_ids)
    print(f"  Extracting RP features via Engine...")
    X_rp = extract_budget_rp(train_tuples, user_ids)

    # Targets from FUTURE
    spend_ratio = test_mean_spends / np.maximum(train_mean_spends, 1e-6)
    churn = (spend_ratio < 0.5).astype(int)

    active_spends = test_total_spends[test_total_spends > 0]
    threshold = np.percentile(active_spends, 66.67) if len(active_spends) > 0 else 0
    high_spender = (test_total_spends > threshold).astype(int)

    spend_change = test_mean_spends - train_mean_spends

    targets_dict = {
        "Churn": (churn, "classification"),
        "High Spender": (high_spender, "classification"),
        "Spend Change": (spend_change, "regression"),
    }

    return X_rp, X_base, targets_dict, user_ids


def run_benchmark(data_dir=None, n_households=None) -> list[BenchmarkResult]:
    X_rp, X_base, targets_dict, user_ids = load_and_prepare(data_dir, n_households)
    results = []
    for target_name, (y, task_type) in targets_dict.items():
        print(f"  [{DATASET_NAME}] Target: {target_name} ({task_type})")
        if task_type == "classification":
            pos_rate = np.mean(y)
            if pos_rate < 0.02 or pos_rate > 0.98:
                print(f"    Skipping — too imbalanced (pos_rate={pos_rate:.3f})")
                continue
        result = run_three_way(X_rp, X_base, y, DATASET_NAME, target_name, task_type)
        results.append(result)
        if task_type == "classification":
            print(f"    AUC: RP={result.auc_rp:.3f}  Base={result.auc_base:.3f}  "
                  f"Combined={result.auc_combined:.3f}  Lift={result.auc_combined - result.auc_base:+.3f}")
        else:
            print(f"    R2: RP={result.r2_rp:.3f}  Base={result.r2_base:.3f}  "
                  f"Combined={result.r2_combined:.3f}")
    return results
