#!/usr/bin/env python3
"""Predictive Validation Study: Can PyRevealed features predict future behavior?

Split-sample study:
- First half of each household's data → Extract features
- Second half → Compute targets
- Use LightGBM with CV to predict second-half outcomes from first-half features
"""

from __future__ import annotations

import sys
import pickle
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pyrevealed import (
    BehaviorLog,
    BehavioralAuditor,
    PreferenceEncoder,
    compute_aei,
)

from config import OUTPUT_DIR, TOP_COMMODITIES, COMMODITY_SHORT_NAMES
from session_builder import HouseholdData


def load_sessions() -> Dict[int, HouseholdData]:
    """Load pre-built sessions."""
    cache_file = OUTPUT_DIR.parent / "cache" / "sessions.pkl"
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    raise FileNotFoundError("Run run_all.py first to build session cache")


def split_behavior_log(log: BehaviorLog) -> Tuple[BehaviorLog, BehaviorLog]:
    """Split a BehaviorLog temporally at midpoint."""
    T = log.num_records
    mid = T // 2

    first_half = BehaviorLog(
        cost_vectors=log.cost_vectors[:mid],
        action_vectors=log.action_vectors[:mid],
        user_id=f"{log.user_id}_first" if log.user_id else None,
    )

    second_half = BehaviorLog(
        cost_vectors=log.cost_vectors[mid:],
        action_vectors=log.action_vectors[mid:],
        user_id=f"{log.user_id}_second" if log.user_id else None,
    )

    return first_half, second_half


def extract_features(log: BehaviorLog, auditor: BehavioralAuditor) -> dict:
    """Extract PyRevealed features from a BehaviorLog."""
    features = {}

    # Basic stats
    features['n_observations'] = log.num_records
    features['total_spend'] = float(np.sum(log.total_spend))
    features['mean_spend_per_obs'] = float(np.mean(log.total_spend))
    features['std_spend'] = float(np.std(log.total_spend))

    # BehavioralAuditor features
    try:
        report = auditor.full_audit(log)
        features['integrity_score'] = report.integrity_score
        features['confusion_score'] = report.confusion_score
        features['bot_risk'] = report.bot_risk
        features['shared_account_risk'] = report.shared_account_risk
        features['ux_confusion_risk'] = report.ux_confusion_risk
        features['is_consistent'] = 1.0 if report.is_consistent else 0.0
    except Exception:
        features['integrity_score'] = np.nan
        features['confusion_score'] = np.nan
        features['bot_risk'] = np.nan
        features['shared_account_risk'] = np.nan
        features['ux_confusion_risk'] = np.nan
        features['is_consistent'] = np.nan

    # PreferenceEncoder features (if fittable)
    try:
        encoder = PreferenceEncoder()
        encoder.fit(log)
        if encoder.is_fitted:
            latent_vals = encoder.extract_latent_values()
            marginal_wts = encoder.extract_marginal_weights()
            features['mean_latent'] = float(np.mean(latent_vals))
            features['std_latent'] = float(np.std(latent_vals))
            features['mean_marginal'] = float(encoder.mean_marginal_weight or 0)
            features['encoder_fitted'] = 1.0
        else:
            features['mean_latent'] = 0.0
            features['std_latent'] = 0.0
            features['mean_marginal'] = 0.0
            features['encoder_fitted'] = 0.0
    except Exception:
        features['mean_latent'] = 0.0
        features['std_latent'] = 0.0
        features['mean_marginal'] = 0.0
        features['encoder_fitted'] = 0.0

    # Category spending shares (10 categories)
    total_qty = np.sum(log.action_vectors)
    if total_qty > 0:
        category_shares = np.sum(log.action_vectors, axis=0) / total_qty
    else:
        category_shares = np.zeros(log.num_features)

    for i, name in enumerate(COMMODITY_SHORT_NAMES.values()):
        if i < len(category_shares):
            features[f'share_{name}'] = float(category_shares[i])

    return features


def compute_targets(log: BehaviorLog) -> dict:
    """Compute prediction targets from a BehaviorLog."""
    targets = {}

    # Target 1: Integrity score (AEI)
    try:
        aei_result = compute_aei(log, tolerance=1e-4)
        targets['target_integrity'] = aei_result.efficiency_index
    except Exception:
        targets['target_integrity'] = np.nan

    # Target 2: Total spending
    targets['target_total_spend'] = float(np.sum(log.total_spend))

    # Target 3: Category shares
    total_qty = np.sum(log.action_vectors)
    if total_qty > 0:
        category_shares = np.sum(log.action_vectors, axis=0) / total_qty
    else:
        category_shares = np.zeros(log.num_features)

    for i, name in enumerate(COMMODITY_SHORT_NAMES.values()):
        if i < len(category_shares):
            targets[f'target_share_{name}'] = float(category_shares[i])

    return targets


def build_dataset(sessions: Dict[int, HouseholdData], min_obs: int = 20) -> pd.DataFrame:
    """Build feature/target dataset from sessions."""
    print("\n[1/4] Building dataset...")

    auditor = BehavioralAuditor()
    records = []

    eligible = {k: v for k, v in sessions.items() if v.num_observations >= min_obs}
    print(f"  Eligible households (≥{min_obs} obs): {len(eligible)}")

    for i, (key, hh_data) in enumerate(eligible.items()):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(eligible)}...")

        try:
            log = hh_data.behavior_log
            first_half, second_half = split_behavior_log(log)

            # Extract features from first half
            features = extract_features(first_half, auditor)
            features['household_key'] = key

            # Compute targets from second half
            targets = compute_targets(second_half)

            # Combine
            record = {**features, **targets}
            records.append(record)

        except Exception as e:
            continue

    df = pd.DataFrame(records)
    print(f"  Dataset size: {len(df)} households")

    return df


def get_feature_groups(feature_cols: list) -> dict:
    """Categorize features into groups for ablation study."""
    basic_stats = ['n_observations', 'total_spend', 'mean_spend_per_obs', 'std_spend']
    category_shares = [c for c in feature_cols if c.startswith('share_')]

    pyrevealed_auditor = ['integrity_score', 'confusion_score', 'bot_risk',
                          'shared_account_risk', 'ux_confusion_risk', 'is_consistent']
    pyrevealed_encoder = ['mean_latent', 'std_latent', 'mean_marginal', 'encoder_fitted']

    return {
        'basic': basic_stats + category_shares,
        'pyrevealed': pyrevealed_auditor + pyrevealed_encoder,
        'auditor': pyrevealed_auditor,
        'encoder': pyrevealed_encoder,
    }


def train_and_evaluate(df: pd.DataFrame) -> dict:
    """Train LightGBM models and evaluate against baselines."""
    import lightgbm as lgb
    from sklearn.model_selection import cross_val_predict, KFold
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

    print("\n[2/4] Training models...")

    # Feature columns (exclude targets and household_key)
    feature_cols = [c for c in df.columns
                    if not c.startswith('target_') and c != 'household_key']

    # Drop rows with NaN in features or main targets
    main_targets = ['target_integrity', 'target_total_spend']
    df_clean = df.dropna(subset=feature_cols + main_targets)
    print(f"  Clean samples: {len(df_clean)}")

    X = df_clean[feature_cols].values
    feature_names = feature_cols

    results = {}
    all_importances = {}

    # Define targets to predict
    targets = {
        'Integrity (AEI)': 'target_integrity',
        'Total Spending': 'target_total_spend',
    }

    # Add category share targets
    share_cols = [c for c in df.columns if c.startswith('target_share_')]
    for col in share_cols[:5]:  # First 5 categories
        name = col.replace('target_share_', '')
        targets[f'Share: {name}'] = col

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for target_name, target_col in targets.items():
        if target_col not in df_clean.columns:
            continue

        y = df_clean[target_col].values

        # Skip if too few valid values
        if np.sum(~np.isnan(y)) < 50:
            continue

        # Baseline 1: Mean predictor
        mean_pred = np.full_like(y, np.nanmean(y))
        baseline_rmse = np.sqrt(mean_squared_error(y, mean_pred))

        # Baseline 2: Persistence (use first-half value as prediction)
        # For integrity: use first-half integrity
        # For spending: use first-half spending
        # For shares: use first-half shares
        if target_col == 'target_integrity':
            persist_col = 'integrity_score'
        elif target_col == 'target_total_spend':
            persist_col = 'total_spend'
        else:
            persist_col = target_col.replace('target_', '')

        if persist_col in df_clean.columns:
            persist_pred = df_clean[persist_col].values
            persist_rmse = np.sqrt(mean_squared_error(y, persist_pred))
        else:
            persist_rmse = baseline_rmse

        # LightGBM with CV
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'n_estimators': 100,
        }

        model = lgb.LGBMRegressor(**lgb_params)

        # Cross-validated predictions
        y_pred = cross_val_predict(model, X, y, cv=kf)

        lgb_rmse = np.sqrt(mean_squared_error(y, y_pred))
        lgb_r2 = r2_score(y, y_pred)
        lgb_mae = mean_absolute_error(y, y_pred)

        # Train final model for feature importance
        model.fit(X, y)
        importances = model.feature_importances_

        results[target_name] = {
            'baseline_rmse': baseline_rmse,
            'persist_rmse': persist_rmse,
            'lgb_rmse': lgb_rmse,
            'lgb_r2': lgb_r2,
            'lgb_mae': lgb_mae,
            'y_true': y,
            'y_pred': y_pred,
            'improvement_vs_baseline': (baseline_rmse - lgb_rmse) / baseline_rmse * 100,
            'improvement_vs_persist': (persist_rmse - lgb_rmse) / persist_rmse * 100,
        }

        all_importances[target_name] = dict(zip(feature_names, importances))

        print(f"  {target_name}: RMSE={lgb_rmse:.4f}, R²={lgb_r2:.4f}")

    # Ablation study: Basic vs Basic+PyRevealed
    print("\n  Running ablation study (Basic vs Basic+PyRevealed)...")
    ablation_results = {}

    feature_groups = get_feature_groups(feature_cols)
    basic_cols = [c for c in feature_groups['basic'] if c in feature_cols]
    pyrevealed_cols = [c for c in feature_groups['pyrevealed'] if c in feature_cols]

    for target_name in ['Integrity (AEI)', 'Total Spending']:
        if target_name not in results:
            continue

        target_col = targets[target_name]
        y = df_clean[target_col].values

        # Model with basic features only
        X_basic = df_clean[basic_cols].values
        model_basic = lgb.LGBMRegressor(**lgb_params)
        y_pred_basic = cross_val_predict(model_basic, X_basic, y, cv=kf)
        rmse_basic = np.sqrt(mean_squared_error(y, y_pred_basic))
        r2_basic = r2_score(y, y_pred_basic)

        # Full model (basic + pyrevealed)
        rmse_full = results[target_name]['lgb_rmse']
        r2_full = results[target_name]['lgb_r2']

        # Incremental value of PyRevealed
        rmse_reduction = (rmse_basic - rmse_full) / rmse_basic * 100
        r2_lift = r2_full - r2_basic

        ablation_results[target_name] = {
            'rmse_basic': rmse_basic,
            'r2_basic': r2_basic,
            'rmse_full': rmse_full,
            'r2_full': r2_full,
            'rmse_reduction_pct': rmse_reduction,
            'r2_lift': r2_lift,
        }

        print(f"    {target_name}: Basic R²={r2_basic:.3f} → Full R²={r2_full:.3f} "
              f"(+{r2_lift:.3f} from PyRevealed)")

    return {
        'results': results,
        'feature_importances': all_importances,
        'feature_names': feature_names,
        'ablation': ablation_results,
        'feature_groups': feature_groups,
    }


def generate_results_table(results: dict) -> pd.DataFrame:
    """Generate formatted results table."""
    rows = []
    for target_name, metrics in results['results'].items():
        rows.append({
            'Target': target_name,
            'Mean Baseline RMSE': f"{metrics['baseline_rmse']:.4f}",
            'Persistence RMSE': f"{metrics['persist_rmse']:.4f}",
            'LightGBM RMSE': f"{metrics['lgb_rmse']:.4f}",
            'R²': f"{metrics['lgb_r2']:.4f}",
            'MAE': f"{metrics['lgb_mae']:.4f}",
            '% Improve (vs Mean)': f"{metrics['improvement_vs_baseline']:.1f}%",
            '% Improve (vs Persist)': f"{metrics['improvement_vs_persist']:.1f}%",
        })

    return pd.DataFrame(rows)


def plot_results(results: dict, output_dir: Path):
    """Generate visualization plots."""
    import matplotlib.pyplot as plt

    print("\n[3/4] Generating visualizations...")

    # Plot 1: Actual vs Predicted for main targets
    main_targets = ['Integrity (AEI)', 'Total Spending']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, target_name in zip(axes, main_targets):
        if target_name not in results['results']:
            continue

        metrics = results['results'][target_name]
        y_true = metrics['y_true']
        y_pred = metrics['y_pred']

        ax.scatter(y_true, y_pred, alpha=0.4, s=20)
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
                'r--', linewidth=2, label='Perfect prediction')

        ax.set_xlabel(f'Actual {target_name}')
        ax.set_ylabel(f'Predicted {target_name}')
        ax.set_title(f'{target_name}\nR²={metrics["lgb_r2"]:.3f}, RMSE={metrics["lgb_rmse"]:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'predictive_actual_vs_predicted.png', dpi=150)
    plt.close()

    # Plot 2: Feature Importance (for Integrity prediction)
    if 'Integrity (AEI)' in results['feature_importances']:
        importances = results['feature_importances']['Integrity (AEI)']

        # Sort by importance
        sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:15]
        names = [x[0] for x in sorted_imp]
        values = [x[1] for x in sorted_imp]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(names)), values, color='steelblue')
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel('Feature Importance')
        ax.set_title('Top 15 Features for Predicting Second-Half Integrity (AEI)')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig(output_dir / 'predictive_feature_importance.png', dpi=150)
        plt.close()

    # Plot 3: Model comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    target_names = list(results['results'].keys())[:5]  # First 5 targets
    x = np.arange(len(target_names))
    width = 0.25

    baseline_rmse = [results['results'][t]['baseline_rmse'] for t in target_names]
    persist_rmse = [results['results'][t]['persist_rmse'] for t in target_names]
    lgb_rmse = [results['results'][t]['lgb_rmse'] for t in target_names]

    ax.bar(x - width, baseline_rmse, width, label='Mean Baseline', color='lightgray')
    ax.bar(x, persist_rmse, width, label='Persistence', color='coral')
    ax.bar(x + width, lgb_rmse, width, label='LightGBM', color='steelblue')

    ax.set_xlabel('Target')
    ax.set_ylabel('RMSE')
    ax.set_title('Model Comparison: RMSE by Target')
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace('Share: ', '') for t in target_names], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'predictive_model_comparison.png', dpi=150)
    plt.close()


def run_predictive_study() -> dict:
    """Run the full predictive validation study."""
    print("=" * 70)
    print(" PREDICTIVE VALIDATION STUDY")
    print(" Can first-half PyRevealed features predict second-half behavior?")
    print("=" * 70)

    # Load sessions
    sessions = load_sessions()
    print(f"  Loaded {len(sessions)} sessions")

    # Build dataset
    df = build_dataset(sessions, min_obs=20)

    # Train and evaluate
    results = train_and_evaluate(df)

    # Generate results table
    table = generate_results_table(results)

    # Plot results
    plot_results(results, OUTPUT_DIR)

    print("\n[4/4] Results complete!")

    return {
        'table': table,
        'results': results,
        'n_samples': len(df),
    }


if __name__ == "__main__":
    study = run_predictive_study()

    print("\n" + "=" * 70)
    print(" PREDICTIVE PERFORMANCE BENCHMARK")
    print("=" * 70)

    print("\n" + study['table'].to_string(index=False))

    print("\n\nKey Findings:")
    results = study['results']['results']

    if 'Integrity (AEI)' in results:
        r = results['Integrity (AEI)']
        print(f"  - Integrity prediction: R²={r['lgb_r2']:.3f}, "
              f"{r['improvement_vs_persist']:.1f}% better than persistence")

    if 'Total Spending' in results:
        r = results['Total Spending']
        print(f"  - Spending prediction: R²={r['lgb_r2']:.3f}, "
              f"{r['improvement_vs_persist']:.1f}% better than persistence")

    # Ablation study results
    if 'ablation' in study['results'] and study['results']['ablation']:
        print("\n" + "=" * 70)
        print(" ABLATION STUDY: Incremental Value of PyRevealed Features")
        print("=" * 70)

        print("\n  Feature Sets:")
        groups = study['results']['feature_groups']
        print(f"    Basic: {len(groups['basic'])} features (spend stats + category shares)")
        print(f"    PyRevealed: {len(groups['pyrevealed'])} features (auditor + encoder)")

        print("\n  Results:")
        print(f"  {'Target':<20} {'Basic R²':>10} {'Full R²':>10} {'R² Lift':>10} {'RMSE Δ':>10}")
        print("  " + "-" * 62)

        for target_name, abl in study['results']['ablation'].items():
            print(f"  {target_name:<20} {abl['r2_basic']:>10.3f} {abl['r2_full']:>10.3f} "
                  f"{abl['r2_lift']:>+10.3f} {abl['rmse_reduction_pct']:>+9.1f}%")

    # Top features for integrity (grouped)
    if 'Integrity (AEI)' in study['results']['feature_importances']:
        print("\n" + "=" * 70)
        print(" FEATURE IMPORTANCE: Predicting Second-Half Integrity")
        print("=" * 70)

        imp = study['results']['feature_importances']['Integrity (AEI)']
        groups = study['results']['feature_groups']

        # Sum importance by group
        group_importance = {}
        for group_name, group_cols in groups.items():
            total = sum(imp.get(c, 0) for c in group_cols)
            group_importance[group_name] = total

        print("\n  Importance by Feature Group:")
        total_imp = sum(group_importance.values())
        for group_name in ['basic', 'pyrevealed']:
            if group_name in group_importance:
                pct = group_importance[group_name] / total_imp * 100 if total_imp > 0 else 0
                print(f"    {group_name.capitalize():<12}: {pct:5.1f}%")

        # Top individual features
        top_features = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:10]
        print("\n  Top 10 Individual Features:")
        for i, (name, val) in enumerate(top_features, 1):
            # Mark PyRevealed features
            marker = " [PyRevealed]" if name in groups['pyrevealed'] else ""
            print(f"    {i:2}. {name:<25} {val:6.1f}{marker}")

    print(f"\nVisualizations saved to: {OUTPUT_DIR}")
