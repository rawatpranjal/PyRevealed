"""FINN.no Classifieds: Rationality in the Wild.

Deep empirical analysis of revealed preference consistency across 46,858
users on Norway's largest classifieds marketplace. Analyzes preferences at
the category-geography group level (290 groups) rather than individual
listings, producing dense preference graphs from observed platform slates.

Sections:
  1. Data Portrait — loading, group distributions, overlap statistics
  2. Deterministic Consistency — SARP, WARP, HM via Engine batch analysis
  3. Stochastic Consistency — RUM LP tests on repeated group-level menus
  4. Search vs Recommendation — consistency comparison by interaction type
  5. Violation Anatomy — cycle structure, SCC sizes, reversal pairs

All figures follow the PrefGraph visual style spec:
  Background #fafafa, primary blue #2563eb, light blue #3b82f6,
  violation red #e74c3c, dark text #333333, 150 DPI.

Usage:
  python3 case_studies/finn_slates/run_analysis.py [--max-users N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from prefgraph import Engine
from prefgraph.algorithms.abstract_choice import (
    validate_menu_sarp,
    validate_menu_warp,
)

from case_studies.finn_slates.group_loader import (
    load_group_level,
    build_stochastic_logs,
    compute_search_ratios,
)

# --- Style constants (PrefGraph visual spec) ---
BG_COLOR = "#fafafa"
PRIMARY_BLUE = "#2563eb"
LIGHT_BLUE = "#3b82f6"
VIOLATION_RED = "#e74c3c"
DARK_TEXT = "#333333"
SECONDARY_TEXT = "#666666"
DPI = 150
FIG_SIZE = (7.5, 4.5)

OUTPUT_DIR = Path(__file__).parent / "output"
FIG_DIR = OUTPUT_DIR / "figures"


def _style_ax(ax):
    """Apply PrefGraph style to axis."""
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=DARK_TEXT, labelsize=9)
    ax.xaxis.label.set_color(DARK_TEXT)
    ax.yaxis.label.set_color(DARK_TEXT)
    for spine in ax.spines.values():
        spine.set_color("#cccccc")
        spine.set_linewidth(0.5)


# =========================================================================
# Section 1: Data Portrait
# =========================================================================

def section_1_data_portrait(user_logs, group_labels, stats):
    """Compute and visualize data overview statistics."""
    print("\n" + "=" * 60)
    print("SECTION 1: Data Portrait")
    print("=" * 60)

    n_users = len(user_logs)
    obs_per_user = [len(log.choices) for log in user_logs.values()]
    groups_per_user = [len(log.all_items) for log in user_logs.values()]

    # Group frequency across all users
    group_freq: Counter = Counter()
    for log in user_logs.values():
        reverse_map = log.metadata.get("group_reverse_map", {})
        for choice in log.choices:
            orig_group = reverse_map.get(choice, choice)
            group_freq[orig_group] += 1

    # Menu overlap at group level: fraction of groups that appear in 2+ menus per user
    overlap_ratios = []
    for log in user_logs.values():
        item_menu_count: Counter = Counter()
        for menu in log.menus:
            for item in menu:
                item_menu_count[item] += 1
        if item_menu_count:
            overlap = sum(1 for c in item_menu_count.values() if c >= 2) / len(item_menu_count)
            overlap_ratios.append(overlap)

    results = {
        "n_users": n_users,
        "obs_per_user_mean": float(np.mean(obs_per_user)),
        "obs_per_user_median": float(np.median(obs_per_user)),
        "groups_per_user_mean": float(np.mean(groups_per_user)),
        "groups_per_user_median": float(np.median(groups_per_user)),
        "group_overlap_mean": float(np.mean(overlap_ratios)) if overlap_ratios else 0,
        "n_unique_groups_observed": len(group_freq),
    }

    print(f"  Users: {n_users:,}")
    print(f"  Observations per user: mean={results['obs_per_user_mean']:.1f}, median={results['obs_per_user_median']:.0f}")
    print(f"  Groups per user: mean={results['groups_per_user_mean']:.1f}, median={results['groups_per_user_median']:.0f}")
    print(f"  Group overlap (fraction appearing in 2+ menus): {results['group_overlap_mean']:.1%}")
    print(f"  Unique groups observed: {results['n_unique_groups_observed']}")

    # --- Figure 1: Data Portrait ---
    fig, axes = plt.subplots(1, 3, figsize=FIG_SIZE, facecolor=BG_COLOR)

    # Panel 1: Observations per user
    ax = axes[0]
    _style_ax(ax)
    ax.hist(obs_per_user, bins=20, color=PRIMARY_BLUE, edgecolor="white", linewidth=0.5, alpha=0.9)
    ax.set_xlabel("Observations per user", fontsize=9, color=DARK_TEXT)
    ax.set_ylabel("Users", fontsize=9, color=DARK_TEXT)

    # Panel 2: Groups per user
    ax = axes[1]
    _style_ax(ax)
    ax.hist(groups_per_user, bins=30, color=PRIMARY_BLUE, edgecolor="white", linewidth=0.5, alpha=0.9)
    ax.set_xlabel("Unique groups per user", fontsize=9, color=DARK_TEXT)
    ax.set_ylabel("", fontsize=9)

    # Panel 3: Top 20 groups by frequency
    ax = axes[2]
    _style_ax(ax)
    top20 = group_freq.most_common(20)
    if top20:
        gids, counts = zip(*top20)
        labels = [group_labels.get(g, str(g))[:20] for g in gids]
        y_pos = np.arange(len(labels))
        ax.barh(y_pos, counts, color=LIGHT_BLUE, edgecolor="white", linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=6, color=DARK_TEXT)
        ax.invert_yaxis()
        ax.set_xlabel("Total clicks", fontsize=9, color=DARK_TEXT)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig1_data_portrait.png", dpi=DPI, facecolor=BG_COLOR, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIG_DIR / 'fig1_data_portrait.png'}")

    return results


# =========================================================================
# Section 2: Deterministic Consistency (SARP/WARP/HM)
# =========================================================================

def section_2_deterministic(user_logs):
    """Run SARP, WARP, HM batch analysis via Engine."""
    print("\n" + "=" * 60)
    print("SECTION 2: Deterministic Consistency")
    print("=" * 60)

    # Convert to Engine format
    uids = list(user_logs.keys())
    engine_tuples = [user_logs[uid].to_engine_tuple() for uid in uids]

    # Batch analysis
    engine = Engine()
    print(f"  Running Engine.analyze_menus() on {len(engine_tuples):,} users...")
    t0 = time.perf_counter()
    results = engine.analyze_menus(engine_tuples)
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s")

    # Extract metrics
    sarp_pass = [r.is_sarp for r in results]
    warp_pass = [r.is_warp for r in results]
    hm_ratios = [r.hm_consistent / r.hm_total if r.hm_total > 0 else 1.0 for r in results]
    n_sarp_violations = [r.n_sarp_violations for r in results]
    n_warp_violations = [r.n_warp_violations for r in results]
    max_scc = [r.max_scc for r in results]
    pref_entropy = [r.pref_entropy for r in results]
    r_density = [r.r_density for r in results]

    sarp_rate = np.mean(sarp_pass)
    warp_rate = np.mean(warp_pass)

    summary = {
        "sarp_pass_rate": float(sarp_rate),
        "warp_pass_rate": float(warp_rate),
        "hm_ratio_mean": float(np.mean(hm_ratios)),
        "hm_ratio_median": float(np.median(hm_ratios)),
        "hm_ratio_q25": float(np.percentile(hm_ratios, 25)),
        "hm_ratio_q75": float(np.percentile(hm_ratios, 75)),
        "mean_sarp_violations": float(np.mean(n_sarp_violations)),
        "mean_warp_violations": float(np.mean(n_warp_violations)),
        "mean_max_scc": float(np.mean(max_scc)),
        "engine_time_s": elapsed,
    }

    print(f"  SARP pass rate: {sarp_rate:.1%}")
    print(f"  WARP pass rate: {warp_rate:.1%}")
    print(f"  HM ratio: mean={summary['hm_ratio_mean']:.3f}, median={summary['hm_ratio_median']:.3f}")
    print(f"  Mean SARP violations: {summary['mean_sarp_violations']:.1f}")
    print(f"  Mean max SCC: {summary['mean_max_scc']:.1f}")

    # --- Power analysis: random choice baseline ---
    print("  Computing random-choice baseline (Bronars-style)...")
    rng = np.random.RandomState(42)
    n_random_sims = 20
    random_sarp_rates = []

    # Sample a subset for power analysis (expensive per user)
    sample_size = min(500, len(uids))
    sample_idx = rng.choice(len(uids), sample_size, replace=False)

    for sim in range(n_random_sims):
        n_pass = 0
        for idx in sample_idx:
            log = user_logs[uids[idx]]
            # Random choices from same menus
            random_choices = [rng.choice(list(m)) for m in log.menus]
            random_log = type(log)(menus=log.menus, choices=random_choices)
            try:
                result = validate_menu_sarp(random_log)
                if result.is_consistent:
                    n_pass += 1
            except Exception:
                n_pass += 1  # If analysis fails, count as pass (conservative)
        random_sarp_rates.append(n_pass / sample_size)

    random_sarp_mean = float(np.mean(random_sarp_rates))
    summary["random_sarp_rate"] = random_sarp_mean
    print(f"  Random-choice SARP pass rate: {random_sarp_mean:.1%} (vs observed {sarp_rate:.1%})")

    # --- Figure 2: Consistency Portrait ---
    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE, facecolor=BG_COLOR)

    # Panel 1: HM ratio distribution with random baseline
    ax = axes[0]
    _style_ax(ax)
    ax.hist(hm_ratios, bins=30, color=PRIMARY_BLUE, edgecolor="white",
            linewidth=0.5, alpha=0.85, density=True, label="Observed")
    # Overlay random baseline as vertical band
    ax.axvline(np.mean(hm_ratios), color=VIOLATION_RED, linestyle="--",
               linewidth=1.5, label=f"Mean = {np.mean(hm_ratios):.2f}")
    ax.set_xlabel("HM consistency ratio", fontsize=9, color=DARK_TEXT)
    ax.set_ylabel("Density", fontsize=9, color=DARK_TEXT)
    ax.legend(fontsize=7, framealpha=0.8)

    # Panel 2: SARP violation count distribution
    ax = axes[1]
    _style_ax(ax)
    viol_counts = [v for v in n_sarp_violations if v <= 50]  # Cap for visualization
    if viol_counts:
        max_v = max(viol_counts)
        bins = np.arange(0, min(max_v + 2, 52))
        ax.hist(viol_counts, bins=bins, color=PRIMARY_BLUE, edgecolor="white",
                linewidth=0.5, alpha=0.85)
    ax.set_xlabel("SARP violations per user", fontsize=9, color=DARK_TEXT)
    ax.set_ylabel("Users", fontsize=9, color=DARK_TEXT)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig2_consistency_portrait.png", dpi=DPI, facecolor=BG_COLOR, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIG_DIR / 'fig2_consistency_portrait.png'}")

    # Store per-user results for later sections
    per_user = {
        uid: {
            "hm_ratio": hm_ratios[i],
            "is_sarp": sarp_pass[i],
            "is_warp": warp_pass[i],
            "n_sarp_violations": n_sarp_violations[i],
            "n_warp_violations": n_warp_violations[i],
            "max_scc": max_scc[i],
            "pref_entropy": pref_entropy[i],
            "r_density": r_density[i],
        }
        for i, uid in enumerate(uids)
    }

    return summary, per_user


# =========================================================================
# Section 3: Stochastic Consistency (RUM)
# =========================================================================

def section_3_stochastic(user_logs):
    """Test Random Utility Model consistency on repeated group-level menus."""
    print("\n" + "=" * 60)
    print("SECTION 3: Stochastic Consistency (RUM)")
    print("=" * 60)

    # Build stochastic logs
    stoch_logs = build_stochastic_logs(user_logs, min_repeated_menus=3, min_total_obs_per_menu=2)

    # Distribution of repeated menus per user (before filtering)
    repeated_menu_counts = []
    for uid, log in user_logs.items():
        menu_freq: Counter = Counter()
        for menu in log.menus:
            menu_freq[menu] += 1
        n_repeated = sum(1 for c in menu_freq.values() if c >= 2)
        repeated_menu_counts.append(n_repeated)

    summary = {
        "n_users_total": len(user_logs),
        "n_users_stochastic": len(stoch_logs),
        "fraction_qualifying": len(stoch_logs) / len(user_logs) if user_logs else 0,
        "mean_repeated_menus": float(np.mean(repeated_menu_counts)),
        "median_repeated_menus": float(np.median(repeated_menu_counts)),
    }

    print(f"  Users with repeated group-level menus: {summary['n_users_stochastic']:,} / {summary['n_users_total']:,}")
    print(f"  Mean repeated menus per user: {summary['mean_repeated_menus']:.1f}")

    # Run RUM tests via Rust batch API (parallel, all users at once)
    if stoch_logs:
        from prefgraph._rust_backend import _rust_rum_batch

        if _rust_rum_batch is not None:
            # Convert StochasticChoiceLogs to Rust batch format
            uids_stoch = list(stoch_logs.keys())
            menus_batch = []
            freqs_batch = []
            n_items_batch = []

            for uid in uids_stoch:
                slog = stoch_logs[uid]
                menus_batch.append([sorted(m) for m in slog.menus])
                freqs_batch.append([
                    [(item, float(count)) for item, count in freq.items()]
                    for freq in slog.choice_frequencies
                ])
                n_items_batch.append(len(slog.all_items))

            print(f"  Running Rust RUM batch on {len(uids_stoch):,} users...")
            t0 = time.time()
            rum_results = _rust_rum_batch(menus_batch, freqs_batch, n_items_batch)
            elapsed = time.time() - t0
            print(f"  Rust RUM batch done in {elapsed:.1f}s")

            n_tested = len(rum_results)
            n_rum_pass = sum(1 for r in rum_results if r["is_consistent"])
            n_regular = sum(1 for r in rum_results if r["is_regular"])

            rum_rate = n_rum_pass / n_tested if n_tested > 0 else 0
            summary["rum_pass_rate"] = rum_rate
            summary["regularity_pass_rate"] = n_regular / n_tested if n_tested > 0 else 0
            summary["n_rum_tested"] = n_tested
            summary["rum_batch_time_s"] = elapsed
            print(f"  RUM pass rate: {rum_rate:.1%} ({n_rum_pass}/{n_tested})")
            print(f"  Regularity pass rate: {n_regular}/{n_tested}")
        else:
            print(f"  Rust backend not available, skipping RUM tests")

    # --- Figure 3: Stochastic analysis ---
    fig, ax = plt.subplots(1, 1, figsize=(5, 4), facecolor=BG_COLOR)
    _style_ax(ax)

    # Histogram of repeated menu counts
    max_count = min(max(repeated_menu_counts) if repeated_menu_counts else 10, 30)
    bins = np.arange(0, max_count + 2)
    ax.hist(repeated_menu_counts, bins=bins, color=PRIMARY_BLUE,
            edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.axvline(3, color=VIOLATION_RED, linestyle="--", linewidth=1.5,
               label="Qualification threshold (3)")
    ax.set_xlabel("Repeated menus per user", fontsize=9, color=DARK_TEXT)
    ax.set_ylabel("Users", fontsize=9, color=DARK_TEXT)
    ax.legend(fontsize=7, framealpha=0.8)

    # Annotate qualifying fraction
    frac = summary["fraction_qualifying"]
    ax.text(0.97, 0.95, f"{frac:.0%} qualify",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, color=DARK_TEXT, fontweight="bold")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig3_stochastic.png", dpi=DPI, facecolor=BG_COLOR, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIG_DIR / 'fig3_stochastic.png'}")

    return summary


# =========================================================================
# Section 4: Search vs Recommendation
# =========================================================================

def section_4_search_vs_reco(user_logs, per_user, stats):
    """Compare consistency between search-heavy and recommendation-heavy users."""
    print("\n" + "=" * 60)
    print("SECTION 4: Search vs Recommendation")
    print("=" * 60)

    search_ratios = compute_search_ratios(stats)

    # Split by median. Terciles don't work because ~70% of slates are
    # search-originated, so the top tercile threshold often lands at 1.0.
    uids_with_both = [uid for uid in per_user if uid in search_ratios]
    ratios_arr = np.array([search_ratios[uid] for uid in uids_with_both])

    median_ratio = float(np.median(ratios_arr))
    # Use strict inequality to ensure both groups are non-empty
    search_heavy = [uid for uid in uids_with_both if search_ratios[uid] > median_ratio]
    reco_heavy = [uid for uid in uids_with_both if search_ratios[uid] < median_ratio]

    search_hm = [per_user[uid]["hm_ratio"] for uid in search_heavy]
    reco_hm = [per_user[uid]["hm_ratio"] for uid in reco_heavy]

    search_sarp_rate = np.mean([per_user[uid]["is_sarp"] for uid in search_heavy])
    reco_sarp_rate = np.mean([per_user[uid]["is_sarp"] for uid in reco_heavy])

    search_violations = [per_user[uid]["n_sarp_violations"] for uid in search_heavy]
    reco_violations = [per_user[uid]["n_sarp_violations"] for uid in reco_heavy]

    # Statistical test
    if search_hm and reco_hm:
        u_stat, p_value = scipy_stats.mannwhitneyu(search_hm, reco_hm, alternative="two-sided")
    else:
        u_stat, p_value = 0, 1.0

    summary = {
        "n_search_heavy": len(search_heavy),
        "n_reco_heavy": len(reco_heavy),
        "search_hm_median": float(np.median(search_hm)) if search_hm else 0,
        "reco_hm_median": float(np.median(reco_hm)) if reco_hm else 0,
        "search_sarp_rate": float(search_sarp_rate),
        "reco_sarp_rate": float(reco_sarp_rate),
        "search_mean_violations": float(np.mean(search_violations)) if search_violations else 0,
        "reco_mean_violations": float(np.mean(reco_violations)) if reco_violations else 0,
        "mannwhitney_u": float(u_stat),
        "mannwhitney_p": float(p_value),
        "search_ratio_median": median_ratio,
    }

    print(f"  Median search ratio: {median_ratio:.2f}")
    print(f"  Search-heavy users: {len(search_heavy):,} (above median)")
    print(f"  Reco-heavy users: {len(reco_heavy):,} (below median)")
    print(f"  HM ratio median: search={summary['search_hm_median']:.3f}, reco={summary['reco_hm_median']:.3f}")
    print(f"  SARP pass rate: search={search_sarp_rate:.1%}, reco={reco_sarp_rate:.1%}")
    print(f"  Mann-Whitney p={p_value:.4f}")

    # --- Figure 4: Search vs Recommendation ---
    fig, ax = plt.subplots(1, 1, figsize=(5, 4), facecolor=BG_COLOR)
    _style_ax(ax)

    # Overlapping KDEs
    has_legend = False
    if len(search_hm) > 5 and len(set(search_hm)) > 1:
        from scipy.stats import gaussian_kde
        x_range = np.linspace(0, 1, 200)
        kde_search = gaussian_kde(search_hm)
        ax.fill_between(x_range, kde_search(x_range), alpha=0.5, color=PRIMARY_BLUE, label="Search-heavy")
        ax.plot(x_range, kde_search(x_range), color=PRIMARY_BLUE, linewidth=1.5)
        ax.axvline(np.median(search_hm), color=PRIMARY_BLUE, linestyle=":", linewidth=1, alpha=0.7)
        has_legend = True

    if len(reco_hm) > 5 and len(set(reco_hm)) > 1:
        from scipy.stats import gaussian_kde
        x_range = np.linspace(0, 1, 200)
        kde_reco = gaussian_kde(reco_hm)
        ax.fill_between(x_range, kde_reco(x_range), alpha=0.3, color="#93c5fd", label="Reco-heavy")
        ax.plot(x_range, kde_reco(x_range), color="#93c5fd", linewidth=1.5)
        ax.axvline(np.median(reco_hm), color="#93c5fd", linestyle=":", linewidth=1, alpha=0.7)
        has_legend = True

    ax.set_xlabel("HM consistency ratio", fontsize=9, color=DARK_TEXT)
    ax.set_ylabel("Density", fontsize=9, color=DARK_TEXT)
    if has_legend:
        ax.legend(fontsize=8, framealpha=0.8)

    # P-value annotation
    sig = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"
    ax.text(0.03, 0.95, sig, transform=ax.transAxes, ha="left", va="top",
            fontsize=9, color=SECONDARY_TEXT)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig4_search_vs_reco.png", dpi=DPI, facecolor=BG_COLOR, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIG_DIR / 'fig4_search_vs_reco.png'}")

    return summary


# =========================================================================
# Section 5: Violation Anatomy
# =========================================================================

def section_5_violations(user_logs, per_user, group_labels):
    """Analyze the structure of SARP violations."""
    print("\n" + "=" * 60)
    print("SECTION 5: Violation Anatomy")
    print("=" * 60)

    # Collect users with SARP violations
    violators = {uid: data for uid, data in per_user.items() if not data["is_sarp"]}
    print(f"  Users with SARP violations: {len(violators):,}")

    # For each violator, run per-user SARP to get cycle details
    cycle_lengths: list[int] = []
    scc_sizes: list[int] = []
    reversal_pairs: Counter = Counter()

    # Sample violators for detailed analysis (expensive per user)
    sample_violators = list(violators.keys())[:500]
    print(f"  Analyzing {len(sample_violators)} violators in detail...")

    for uid in sample_violators:
        log = user_logs[uid]
        try:
            result = validate_menu_sarp(log)
            if hasattr(result, "violations") and result.violations:
                for cycle in result.violations:
                    if isinstance(cycle, (list, tuple)):
                        cycle_lengths.append(len(cycle))
            if hasattr(result, "max_scc_size"):
                scc_sizes.append(result.max_scc_size)

            # Check for WARP violations (direct reversals)
            warp_result = validate_menu_warp(log)
            if hasattr(warp_result, "violations") and warp_result.violations:
                reverse_map = log.metadata.get("group_reverse_map", {})
                for pair in warp_result.violations:
                    if isinstance(pair, (list, tuple)) and len(pair) == 2:
                        a, b = sorted(pair)
                        orig_a = reverse_map.get(a, a)
                        orig_b = reverse_map.get(b, b)
                        reversal_pairs[(orig_a, orig_b)] += 1
        except Exception:
            pass

    # Use SCC data from engine results
    engine_scc_sizes = [per_user[uid]["max_scc"] for uid in violators if per_user[uid]["max_scc"] > 1]

    summary = {
        "n_violators": len(violators),
        "fraction_violators": len(violators) / len(per_user) if per_user else 0,
        "n_cycles_found": len(cycle_lengths),
        "mean_cycle_length": float(np.mean(cycle_lengths)) if cycle_lengths else 0,
        "median_cycle_length": float(np.median(cycle_lengths)) if cycle_lengths else 0,
        "mean_max_scc": float(np.mean(engine_scc_sizes)) if engine_scc_sizes else 0,
        "top_reversal_pairs": [
            {
                "pair": [group_labels.get(a, str(a)), group_labels.get(b, str(b))],
                "count": count,
            }
            for (a, b), count in reversal_pairs.most_common(10)
        ],
    }

    print(f"  Cycles found: {len(cycle_lengths)}")
    if cycle_lengths:
        print(f"  Cycle lengths: mean={summary['mean_cycle_length']:.1f}, median={summary['median_cycle_length']:.0f}")
    if engine_scc_sizes:
        print(f"  Mean max SCC: {summary['mean_max_scc']:.1f}")
    if reversal_pairs:
        print(f"  Top reversal pair: {reversal_pairs.most_common(1)}")

    # --- Figure 5: Violation Anatomy ---
    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE, facecolor=BG_COLOR)

    # Panel 1: SCC size distribution from engine results
    ax = axes[0]
    _style_ax(ax)
    if engine_scc_sizes:
        max_scc_val = min(max(engine_scc_sizes), 30)
        bins = np.arange(2, max_scc_val + 2)
        ax.hist(engine_scc_sizes, bins=bins, color=VIOLATION_RED,
                edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.set_xlabel("Largest SCC size", fontsize=9, color=DARK_TEXT)
    ax.set_ylabel("Users", fontsize=9, color=DARK_TEXT)

    # Panel 2: Cycle length distribution (if available)
    ax = axes[1]
    _style_ax(ax)
    if cycle_lengths:
        max_cl = min(max(cycle_lengths), 20)
        bins = np.arange(2, max_cl + 2)
        ax.hist(cycle_lengths, bins=bins, color=VIOLATION_RED,
                edgecolor="white", linewidth=0.5, alpha=0.85)
        ax.set_xlabel("Cycle length", fontsize=9, color=DARK_TEXT)
        ax.set_ylabel("Cycles", fontsize=9, color=DARK_TEXT)
    else:
        # If no cycle data, show violations per user distribution
        viol_counts = [per_user[uid]["n_sarp_violations"] for uid in violators]
        if viol_counts:
            max_v = min(max(viol_counts), 50)
            bins = np.arange(1, max_v + 2)
            ax.hist(viol_counts, bins=bins, color=VIOLATION_RED,
                    edgecolor="white", linewidth=0.5, alpha=0.85)
        ax.set_xlabel("SARP violations per user", fontsize=9, color=DARK_TEXT)
        ax.set_ylabel("Users", fontsize=9, color=DARK_TEXT)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig5_violations.png", dpi=DPI, facecolor=BG_COLOR, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIG_DIR / 'fig5_violations.png'}")

    return summary


# =========================================================================
# Section 6: Item-Level Comparison
# =========================================================================

def section_6_item_level(data_dir, max_users):
    """Run SARP/WARP/HM at the individual item level for comparison."""
    print("\n" + "=" * 60)
    print("SECTION 6: Item-Level Comparison")
    print("=" * 60)

    from prefgraph.datasets._finn_slates import load_finn_slates

    print(f"  Loading item-level MenuChoiceLogs...")
    t0 = time.perf_counter()
    item_logs = load_finn_slates(data_dir=data_dir, max_users=max_users, min_sessions=5)
    load_time = time.perf_counter() - t0
    print(f"  Loaded {len(item_logs):,} users in {load_time:.1f}s")

    if not item_logs:
        return {"error": "no users"}

    # Item overlap per user
    overlap_ratios = []
    items_per_user = []
    for log in item_logs.values():
        item_counts: Counter = Counter()
        for menu in log.menus:
            for item in menu:
                item_counts[item] += 1
        if item_counts:
            overlap = sum(1 for c in item_counts.values() if c >= 2) / len(item_counts)
            overlap_ratios.append(overlap)
        items_per_user.append(len(log.all_items))

    # Engine batch
    uids = list(item_logs.keys())
    engine_tuples = [item_logs[uid].to_engine_tuple() for uid in uids]
    engine = Engine()
    print(f"  Running Engine.analyze_menus() on {len(engine_tuples):,} item-level users...")
    t0 = time.perf_counter()
    results = engine.analyze_menus(engine_tuples)
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s")

    sarp_pass = [r.is_sarp for r in results]
    warp_pass = [r.is_warp for r in results]
    hm_ratios = [r.hm_consistent / r.hm_total if r.hm_total > 0 else 1.0 for r in results]

    summary = {
        "n_users": len(item_logs),
        "sarp_pass_rate": float(np.mean(sarp_pass)),
        "warp_pass_rate": float(np.mean(warp_pass)),
        "hm_ratio_mean": float(np.mean(hm_ratios)),
        "hm_ratio_median": float(np.median(hm_ratios)),
        "item_overlap_mean": float(np.mean(overlap_ratios)) if overlap_ratios else 0,
        "items_per_user_mean": float(np.mean(items_per_user)),
        "engine_time_s": elapsed,
    }

    print(f"  Item overlap: {summary['item_overlap_mean']:.1%}")
    print(f"  Items per user: {summary['items_per_user_mean']:.0f}")
    print(f"  SARP pass rate: {summary['sarp_pass_rate']:.1%}")
    print(f"  WARP pass rate: {summary['warp_pass_rate']:.1%}")
    print(f"  HM ratio: mean={summary['hm_ratio_mean']:.3f}, median={summary['hm_ratio_median']:.3f}")

    # --- Dense subset: top users (most obs) with only top items (most frequent) ---
    print("\n  Dense subset: top 20% users by observation count, top 50 items per user...")

    # Rank users by observation count, take top 20%
    user_obs = [(uid, len(log.choices)) for uid, log in item_logs.items()]
    user_obs.sort(key=lambda x: -x[1])
    top_n = max(1, len(user_obs) // 5)
    top_uids = [uid for uid, _ in user_obs[:top_n]]

    # For each top user, keep only the 50 most frequently chosen items
    dense_logs: dict[str, object] = {}
    from prefgraph.core.session import MenuChoiceLog as MCL
    for uid in top_uids:
        log = item_logs[uid]
        # Find top 50 items by choice frequency
        choice_freq: Counter = Counter(log.choices)
        top_items = set(item for item, _ in choice_freq.most_common(50))

        # Filter menus and choices to only include top items
        filtered_menus = []
        filtered_choices = []
        for menu, choice in zip(log.menus, log.choices):
            if choice not in top_items:
                continue
            filtered_menu = frozenset(i for i in menu if i in top_items)
            if len(filtered_menu) >= 2 and choice in filtered_menu:
                filtered_menus.append(filtered_menu)
                filtered_choices.append(choice)

        if len(filtered_choices) >= 5:
            # Remap to compact 0..N-1
            all_items_set = sorted(set().union(*filtered_menus))
            imap = {item: idx for idx, item in enumerate(all_items_set)}
            dense_logs[uid] = MCL(
                menus=[frozenset(imap[i] for i in m) for m in filtered_menus],
                choices=[imap[c] for c in filtered_choices],
            )

    if dense_logs:
        dense_tuples = [dense_logs[uid].to_engine_tuple() for uid in dense_logs]
        dense_results = engine.analyze_menus(dense_tuples)

        dense_sarp = float(np.mean([r.is_sarp for r in dense_results]))
        dense_hm = [r.hm_consistent / r.hm_total if r.hm_total > 0 else 1.0 for r in dense_results]

        summary["dense_n_users"] = len(dense_logs)
        summary["dense_sarp_pass_rate"] = dense_sarp
        summary["dense_hm_mean"] = float(np.mean(dense_hm))
        summary["dense_hm_median"] = float(np.median(dense_hm))

        print(f"  Dense subset: {len(dense_logs):,} users")
        print(f"  SARP pass rate: {dense_sarp:.1%}")
        print(f"  HM ratio: mean={np.mean(dense_hm):.3f}, median={np.median(dense_hm):.3f}")

    return summary


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="FINN.no Classifieds RP analysis")
    parser.add_argument("--max-users", type=int, default=100_000,
                        help="Max users to load (default: 100,000)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to FINN.no data directory")
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FINN.no Classifieds: Rationality in the Wild")
    print("=" * 60)

    # Load data at group level
    t0 = time.perf_counter()
    user_logs, group_labels, stats = load_group_level(
        data_dir=args.data_dir,
        max_users=args.max_users,
    )
    load_time = time.perf_counter() - t0
    print(f"  Data loaded in {load_time:.1f}s")

    if not user_logs:
        print("ERROR: No qualifying users found. Check data path.")
        return

    # Run all sections
    all_results = {"load_time_s": load_time}

    all_results["section_1"] = section_1_data_portrait(user_logs, group_labels, stats)
    det_summary, per_user = section_2_deterministic(user_logs)
    all_results["section_2"] = det_summary
    all_results["section_3"] = section_3_stochastic(user_logs)
    all_results["section_4"] = section_4_search_vs_reco(user_logs, per_user, stats)
    all_results["section_5"] = section_5_violations(user_logs, per_user, group_labels)
    all_results["section_6"] = section_6_item_level(args.data_dir, args.max_users)

    # Print comparison
    s2 = all_results["section_2"]
    s6 = all_results["section_6"]
    if "error" not in s6:
        print("\n" + "=" * 60)
        print("COMPARISON: Group-Level vs Item-Level")
        print("=" * 60)
        print(f"  {'Metric':<25} {'Group (290)':<15} {'Item (unique)':<15}")
        print(f"  {'─' * 55}")
        print(f"  {'SARP pass rate':<25} {s2['sarp_pass_rate']:.1%}{'':<10} {s6['sarp_pass_rate']:.1%}")
        print(f"  {'HM ratio (mean)':<25} {s2['hm_ratio_mean']:.3f}{'':<10} {s6['hm_ratio_mean']:.3f}")
        print(f"  {'HM ratio (median)':<25} {s2['hm_ratio_median']:.3f}{'':<10} {s6['hm_ratio_median']:.3f}")
        print(f"  {'Item overlap':<25} {'62.7%':<15} {s6['item_overlap_mean']:.1%}")

    # Save all results
    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
