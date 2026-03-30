use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use std::time::Instant;

use rpt_core::ccei::ccei_search;
use rpt_core::garp::{garp_check, garp_check_with_closure};
use rpt_core::graph::PreferenceGraph;
use rpt_core::harp::harp_check;
use rpt_core::houtman_maks::{houtman_maks, houtman_maks_exact};
// 2026-03-29 performance audit: mpi_karp_v2 is the default (sparse predecessor
// lists, 4.3x faster at T=500). mpi_karp (dense inner loop) kept as _legacy
// for A/B benchmarking. Both produce identical results to 1e-12 tolerance.
// If you suspect an MPI discrepancy, run bench_champion_vs_challenger.py to
// compare both variants on the same data.
use rpt_core::mpi::mpi_karp_v2 as mpi_karp;
use rpt_core::utility::recover_utility;
use rpt_core::vei::{compute_vei as run_vei, compute_vei_exact as run_vei_exact};

use crate::convert::extract_user_data;

/// Linear interpolation percentile (matches numpy default method='linear').
fn percentile(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 { return 0.0; }
    if n == 1 { return sorted[0]; }
    let idx = p * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = idx - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Compute (std, q25, q75) from a VEI efficiency vector.
fn compute_vei_stats(efficiencies: &[f64]) -> (f64, f64, f64) {
    let n = efficiencies.len();
    if n == 0 {
        return (0.0, 1.0, 1.0);
    }
    let mean = efficiencies.iter().sum::<f64>() / n as f64;
    let variance = efficiencies.iter().map(|&e| (e - mean).powi(2)).sum::<f64>() / n as f64;
    let std = variance.sqrt();

    let mut sorted = efficiencies.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q25 = percentile(&sorted, 0.25);
    let q75 = percentile(&sorted, 0.75);

    (std, q25, q75)
}

/// Compute graph network features from the R matrix and edge weights.
/// Returns (r_density, r_out_degree_std, degree_gini, ew_mean, ew_std, ew_skew).
/// Edge-weight fields are 0.0 if weights have not been computed.
fn compute_graph_stats(graph: &PreferenceGraph) -> (f64, f64, f64, f64, f64, f64) {
    let t = graph.t;
    if t < 2 {
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }
    let n_possible = (t * (t - 1)) as f64;

    // --- R-based features ---
    let mut out_deg = vec![0u32; t];
    let mut in_deg = vec![0u32; t];
    let mut r_edges: u64 = 0;
    for i in 0..t {
        for j in 0..t {
            if i != j && graph.r[i * t + j] {
                r_edges += 1;
                out_deg[i] += 1;
                in_deg[j] += 1;
            }
        }
    }
    let r_density = r_edges as f64 / n_possible;

    // Out-degree std
    let out_mean = r_edges as f64 / t as f64;
    let out_var = out_deg.iter().map(|&d| {
        let diff = d as f64 - out_mean;
        diff * diff
    }).sum::<f64>() / t as f64;
    let r_out_degree_std = out_var.sqrt();

    // Degree Gini: Gini of total_degree = out_deg + in_deg
    let mut total_deg: Vec<f64> = (0..t).map(|i| (out_deg[i] + in_deg[i]) as f64).collect();
    let deg_sum: f64 = total_deg.iter().sum();
    let degree_gini = if deg_sum > 0.0 {
        total_deg.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = t as f64;
        let weighted_sum: f64 = total_deg.iter().enumerate()
            .map(|(i, &d)| (i as f64 + 1.0) * d)
            .sum();
        (2.0 * weighted_sum - (n + 1.0) * deg_sum) / (n * deg_sum)
    } else {
        0.0
    };

    // --- Edge-weight features (only if HARP has been computed) ---
    let (ew_mean, ew_std, ew_skew) = if graph.has_weights && r_edges > 0 {
        let mut weights = Vec::with_capacity(r_edges as usize);
        for i in 0..t {
            for j in 0..t {
                if i != j && graph.r[i * t + j] {
                    let w = graph.edge_weights[i * t + j];
                    if w.is_finite() {
                        weights.push(w);
                    }
                }
            }
        }
        if weights.is_empty() {
            (0.0, 0.0, 0.0)
        } else {
            let n = weights.len() as f64;
            let mean = weights.iter().sum::<f64>() / n;
            let var = weights.iter().map(|&w| (w - mean).powi(2)).sum::<f64>() / n;
            let std = var.sqrt();
            let skew = if std > 1e-12 {
                weights.iter().map(|&w| ((w - mean) / std).powi(3)).sum::<f64>() / n
            } else {
                0.0
            };
            (mean, std, skew)
        }
    } else {
        (0.0, 0.0, 0.0)
    };

    (r_density, r_out_degree_std, degree_gini, ew_mean, ew_std, ew_skew)
}

/// Compute menu-specific network features from the R matrix and choices.
/// Returns (r_density, pref_entropy, choice_diversity).
fn compute_menu_graph_stats(graph: &PreferenceGraph, choices: &[usize]) -> (f64, f64, f64) {
    let t = graph.t;
    if t < 2 {
        return (0.0, 0.0, 0.0);
    }
    let n_possible = (t * (t - 1)) as f64;

    // R density
    let mut r_edges: u64 = 0;
    let mut out_deg = vec![0u32; t];
    for i in 0..t {
        for j in 0..t {
            if i != j && graph.r[i * t + j] {
                r_edges += 1;
                out_deg[i] += 1;
            }
        }
    }
    let r_density = if n_possible > 0.0 { r_edges as f64 / n_possible } else { 0.0 };

    // Preference entropy: Shannon entropy of out-degree distribution
    let deg_sum: f64 = out_deg.iter().map(|&d| d as f64).sum();
    let pref_entropy = if deg_sum > 0.0 {
        out_deg.iter()
            .filter(|&&d| d > 0)
            .map(|&d| {
                let p = d as f64 / deg_sum;
                -p * p.ln() / std::f64::consts::LN_2
            })
            .sum()
    } else {
        0.0
    };

    // Choice diversity: unique choices / total choices
    let n_choices = choices.len();
    let choice_diversity = if n_choices > 0 {
        let mut sorted = choices.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        sorted.len() as f64 / n_choices as f64
    } else {
        0.0
    };

    (r_density, pref_entropy, choice_diversity)
}

/// Metric flags controlling which analyses to run.
#[derive(Clone, Copy)]
struct MetricFlags {
    ccei: bool,
    mpi: bool,
    harp: bool,
    hm: bool,
    utility: bool,
    vei: bool,
    vei_exact: bool,
    network: bool,
}

/// Output from processing one user.
struct UserOut {
    is_garp: bool,
    n_violations: u32,
    ccei: f64,
    mpi: f64,
    is_harp: bool,
    hm_consistent: u32,
    hm_total: u32,
    utility_success: bool,
    vei_mean: f64,
    vei_min: f64,
    vei_exact_mean: f64,
    vei_exact_min: f64,
    max_scc: u32,
    time_us: u64,
    // Extended fields: expose intermediate values
    vei_std: f64,
    vei_q25: f64,
    vei_q75: f64,
    vei_exact_std: f64,
    vei_exact_q25: f64,
    vei_exact_q75: f64,
    n_scc: u32,
    harp_severity: f64,
    scc_mean_size: f64,
    // Network/graph features
    r_density: f64,
    r_out_degree_std: f64,
    degree_gini: f64,
    ew_mean: f64,
    ew_std: f64,
    ew_skew: f64,
}

/// Process a batch of users in parallel using Rayon. Shared by both
/// `analyze_batch` (numpy input) and `analyze_parquet_file` (Parquet input).
fn process_users_parallel(
    users: &[(Vec<f64>, Vec<f64>, usize, usize)],
    flags: MetricFlags,
    tolerance: f64,
) -> Vec<UserOut> {
    let max_t = users.iter().map(|(_, _, t, _)| *t).max().unwrap_or(0);

    users
        .par_iter()
        .map_init(
            || PreferenceGraph::new(max_t),
            |graph, (p_flat, q_flat, t, k)| {
                let start = Instant::now();
                let t = *t;
                let k = *k;

                graph.reset();
                graph.parse_budget(p_flat, q_flat, t, k, tolerance);

                let needs_closure = flags.mpi || flags.vei;
                let garp = if needs_closure {
                    garp_check_with_closure(graph)
                } else {
                    garp_check(graph)
                };

                let mpi = if flags.mpi && !garp.is_consistent {
                    mpi_karp(graph)
                } else {
                    0.0
                };

                // HARP: binary test for homothetic preferences.
                // Varian (1983), C&E (2016) Thm 4.2: pass/fail only.
                // harp_severity is always 1.0 - no severity metric in the literature.
                let (is_harp, harp_severity) = if flags.harp {
                    let harp = harp_check(graph, tolerance);
                    (harp.is_consistent, harp.max_cycle_product)
                } else {
                    (false, 1.0)
                };

                // Houtman-Maks: use exact ILP (Demuynck & Rehbeck 2023) for
                // T ≤ 200 where MILP is tractable; greedy FVS for larger T
                // where the NP-hard problem is approximated.
                // Reference: Smeulders et al. (2014) prove HM is NP-hard.
                let (hm_c, hm_t) = if flags.hm {
                    if graph.t <= 200 {
                        houtman_maks_exact(graph)
                    } else {
                        houtman_maks(graph)
                    }
                } else {
                    (0, 0)
                };

                let utility_success = if flags.utility {
                    recover_utility(graph).success
                } else {
                    false
                };

                let (vei_mean, vei_min, vei_std, vei_q25, vei_q75) = if flags.vei {
                    let vei = run_vei(graph);
                    let (std, q25, q75) = compute_vei_stats(&vei.efficiency_vector);
                    (vei.mean_efficiency, vei.min_efficiency, std, q25, q75)
                } else {
                    (1.0, 1.0, 0.0, 1.0, 1.0)
                };

                let (vei_exact_mean, vei_exact_min, vei_exact_std, vei_exact_q25, vei_exact_q75) = if flags.vei_exact {
                    let vei = run_vei_exact(graph);
                    let (std, q25, q75) = compute_vei_stats(&vei.efficiency_vector);
                    (vei.mean_efficiency, vei.min_efficiency, std, q25, q75)
                } else {
                    (1.0, 1.0, 0.0, 1.0, 1.0)
                };

                // Graph network features - compute before CCEI which may modify state
                let (r_density, r_out_degree_std, degree_gini, ew_mean, ew_std, ew_skew) =
                    if flags.network { compute_graph_stats(graph) } else { (0.0, 0.0, 0.0, 0.0, 0.0, 0.0) };

                let ccei = if flags.ccei && !garp.is_consistent {
                    ccei_search(graph, tolerance).ccei
                } else if garp.is_consistent {
                    1.0
                } else {
                    -1.0
                };

                let n_scc = garp.n_components;
                let scc_mean_size = if garp.n_components > 0 {
                    t as f64 / garp.n_components as f64
                } else {
                    t as f64
                };

                let time_us = start.elapsed().as_micros() as u64;

                UserOut {
                    is_garp: garp.is_consistent,
                    n_violations: garp.n_violations,
                    ccei,
                    mpi,
                    is_harp,
                    hm_consistent: hm_c as u32,
                    hm_total: hm_t as u32,
                    utility_success,
                    vei_mean,
                    vei_min,
                    vei_exact_mean,
                    vei_exact_min,
                    max_scc: garp.max_scc_size,
                    time_us,
                    vei_std,
                    vei_q25,
                    vei_q75,
                    vei_exact_std,
                    vei_exact_q25,
                    vei_exact_q75,
                    n_scc,
                    harp_severity,
                    scc_mean_size,
                    r_density,
                    r_out_degree_std,
                    degree_gini,
                    ew_mean,
                    ew_std,
                    ew_skew,
                }
            },
        )
        .collect()
}

/// Convert a Vec<UserOut> to Python dicts.
fn results_to_pydicts<'py>(py: Python<'py>, results: Vec<UserOut>) -> Vec<Bound<'py, PyDict>> {
    results
        .into_iter()
        .map(|r| {
            let dict = PyDict::new_bound(py);
            dict.set_item("is_garp", r.is_garp).unwrap();
            dict.set_item("n_violations", r.n_violations).unwrap();
            dict.set_item("ccei", r.ccei).unwrap();
            dict.set_item("mpi", r.mpi).unwrap();
            dict.set_item("is_harp", r.is_harp).unwrap();
            dict.set_item("hm_consistent", r.hm_consistent).unwrap();
            dict.set_item("hm_total", r.hm_total).unwrap();
            dict.set_item("utility_success", r.utility_success).unwrap();
            dict.set_item("vei_mean", r.vei_mean).unwrap();
            dict.set_item("vei_min", r.vei_min).unwrap();
            dict.set_item("vei_exact_mean", r.vei_exact_mean).unwrap();
            dict.set_item("vei_exact_min", r.vei_exact_min).unwrap();
            dict.set_item("max_scc", r.max_scc).unwrap();
            dict.set_item("compute_time_us", r.time_us).unwrap();
            dict.set_item("vei_std", r.vei_std).unwrap();
            dict.set_item("vei_q25", r.vei_q25).unwrap();
            dict.set_item("vei_q75", r.vei_q75).unwrap();
            dict.set_item("vei_exact_std", r.vei_exact_std).unwrap();
            dict.set_item("vei_exact_q25", r.vei_exact_q25).unwrap();
            dict.set_item("vei_exact_q75", r.vei_exact_q75).unwrap();
            dict.set_item("n_scc", r.n_scc).unwrap();
            dict.set_item("harp_severity", r.harp_severity).unwrap();
            dict.set_item("scc_mean_size", r.scc_mean_size).unwrap();
            dict.set_item("r_density", r.r_density).unwrap();
            dict.set_item("r_out_degree_std", r.r_out_degree_std).unwrap();
            dict.set_item("degree_gini", r.degree_gini).unwrap();
            dict.set_item("ew_mean", r.ew_mean).unwrap();
            dict.set_item("ew_std", r.ew_std).unwrap();
            dict.set_item("ew_skew", r.ew_skew).unwrap();
            dict
        })
        .collect()
}

/// Analyze a batch of users in parallel using Rayon.
///
/// Uses PreferenceGraph with lazy computation - expenditure matrix built once,
/// R/P/closure reused across metrics. MPI uses Karp's max-mean-weight cycle
/// (theory-correct). CCEI runs last since it may modify graph state.
#[pyfunction]
#[pyo3(signature = (prices_list, quantities_list, compute_ccei=true, compute_mpi=false, compute_harp=false, compute_hm=false, compute_utility=false, compute_vei=false, compute_vei_exact=false, compute_network=false, tolerance=1e-10))]
pub fn analyze_batch<'py>(
    py: Python<'py>,
    prices_list: Vec<PyReadonlyArray2<'py, f64>>,
    quantities_list: Vec<PyReadonlyArray2<'py, f64>>,
    compute_ccei: bool,
    compute_mpi: bool,
    compute_harp: bool,
    compute_hm: bool,
    compute_utility: bool,
    compute_vei: bool,
    compute_vei_exact: bool,
    compute_network: bool,
    tolerance: f64,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let n_users = prices_list.len();

    let users: Vec<(Vec<f64>, Vec<f64>, usize, usize)> = (0..n_users)
        .map(|i| extract_user_data(&prices_list[i], &quantities_list[i]))
        .collect();

    let flags = MetricFlags {
        ccei: compute_ccei,
        mpi: compute_mpi,
        harp: compute_harp,
        hm: compute_hm,
        utility: compute_utility,
        vei: compute_vei,
        vei_exact: compute_vei_exact,
        network: compute_network,
    };

    let results = process_users_parallel(&users, flags, tolerance);
    Ok(results_to_pydicts(py, results))
}

/// Analyze a batch of users' MENU choice data in parallel.
///
/// Each user has: menus (list of lists of item indices) + choices (list of chosen item).
/// Computes SARP, WARP, Houtman-Maks, and optionally WARP-LA per user.
///
/// This is the "rec/search click" path - no prices, just menus and picks.
#[pyfunction]
#[pyo3(signature = (menus_list, choices_list, n_items_list, compute_warp_la=false, compute_network=false))]
pub fn analyze_menu_batch<'py>(
    py: Python<'py>,
    menus_list: Vec<Vec<Vec<usize>>>,   // menus_list[user][obs] = vec of item indices
    choices_list: Vec<Vec<usize>>,       // choices_list[user][obs] = chosen item index
    n_items_list: Vec<usize>,            // n_items per user (max item index + 1)
    compute_warp_la: bool,
    compute_network: bool,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    use rpt_core::menu::{menu_sarp_check, menu_warp_check, menu_houtman_maks};
    use rpt_core::attention::warp_la_check;

    let n_users = menus_list.len();

    // Find max n_items for graph sizing
    let max_items = n_items_list.iter().cloned().max().unwrap_or(0);

    struct MenuOut {
        is_sarp: bool,
        is_warp: bool,
        is_warp_la: bool,
        n_sarp_violations: u32,
        n_warp_violations: u32,
        hm_consistent: u32,
        hm_total: u32,
        max_scc: u32,
        n_scc: u32,
        r_density: f64,
        pref_entropy: f64,
        choice_diversity: f64,
        time_us: u64,
    }

    // Pack user data for Send across threads
    let users: Vec<(&Vec<Vec<usize>>, &Vec<usize>, usize)> = (0..n_users)
        .map(|i| (&menus_list[i], &choices_list[i], n_items_list[i]))
        .collect();

    let results: Vec<MenuOut> = users
        .par_iter()
        .map_init(
            || PreferenceGraph::new(max_items),
            |graph, (menus, choices, n_items)| {
                let start = Instant::now();

                graph.reset();
                graph.parse_menu(menus, choices, *n_items);

                let sarp = menu_sarp_check(graph);
                let warp = menu_warp_check(graph);
                let (hm_c, hm_t) = menu_houtman_maks(graph);

                let is_warp_la = if compute_warp_la {
                    warp_la_check(graph).is_rationalizable
                } else {
                    false
                };

                let (r_density, pref_entropy, choice_diversity) =
                    if compute_network { compute_menu_graph_stats(graph, choices) } else { (0.0, 0.0, 0.0) };

                let time_us = start.elapsed().as_micros() as u64;

                MenuOut {
                    is_sarp: sarp.is_consistent,
                    is_warp: warp.is_consistent,
                    is_warp_la,
                    n_sarp_violations: sarp.n_violations,
                    n_warp_violations: warp.n_violations,
                    hm_consistent: hm_c as u32,
                    hm_total: hm_t as u32,
                    max_scc: sarp.max_scc_size,
                    n_scc: sarp.n_components,
                    r_density,
                    pref_entropy,
                    choice_diversity,
                    time_us,
                }
            },
        )
        .collect();

    let py_results: Vec<Bound<'py, PyDict>> = results
        .into_iter()
        .map(|r| {
            let dict = PyDict::new_bound(py);
            dict.set_item("is_sarp", r.is_sarp).unwrap();
            dict.set_item("is_warp", r.is_warp).unwrap();
            dict.set_item("is_warp_la", r.is_warp_la).unwrap();
            dict.set_item("n_sarp_violations", r.n_sarp_violations).unwrap();
            dict.set_item("n_warp_violations", r.n_warp_violations).unwrap();
            dict.set_item("hm_consistent", r.hm_consistent).unwrap();
            dict.set_item("hm_total", r.hm_total).unwrap();
            dict.set_item("max_scc", r.max_scc).unwrap();
            dict.set_item("n_scc", r.n_scc).unwrap();
            dict.set_item("r_density", r.r_density).unwrap();
            dict.set_item("pref_entropy", r.pref_entropy).unwrap();
            dict.set_item("choice_diversity", r.choice_diversity).unwrap();
            dict.set_item("compute_time_us", r.time_us).unwrap();
            dict
        })
        .collect();

    Ok(py_results)
}

/// Build an observation graph from budget data and return it as numpy arrays.
///
/// Tier 2 entry point: Python modules can consume the Rust-computed graph.
#[pyfunction]
#[pyo3(signature = (prices, quantities, tolerance=1e-10))]
pub fn build_preference_graph<'py>(
    py: Python<'py>,
    prices: PyReadonlyArray2<'py, f64>,
    quantities: PyReadonlyArray2<'py, f64>,
    tolerance: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let p = prices.as_array();
    let q = quantities.as_array();
    let t = p.nrows();
    let k = p.ncols();
    let p_flat: Vec<f64> = p.iter().cloned().collect();
    let q_flat: Vec<f64> = q.iter().cloned().collect();

    let mut graph = PreferenceGraph::new(t);
    graph.parse_budget(&p_flat, &q_flat, t, k, tolerance);
    // Must use garp_check_with_closure (Varian 1982, O(T³) Floyd-Warshall)
    // rather than garp_check (Talla Nobibon et al. 2015 JOTA, O(T²) SCC-only)
    // because this function returns r_star to Python for violation cycle
    // extraction. The O(T²) algorithm correctly determines is_consistent
    // but does not populate r_star, leaving it all-zeros.
    let garp = garp_check_with_closure(&mut graph);
    graph.ensure_weights();

    let dict = PyDict::new_bound(py);

    let r_u8: Vec<u8> = graph.r[..t * t].iter().map(|&b| b as u8).collect();
    let p_u8: Vec<u8> = graph.p[..t * t].iter().map(|&b| b as u8).collect();
    let rs_u8: Vec<u8> = graph.r_star[..t * t].iter().map(|&b| b as u8).collect();

    dict.set_item("r", PyArray2::from_owned_array_bound(py,
        ndarray::Array2::from_shape_vec((t, t), r_u8).unwrap()))?;
    dict.set_item("p", PyArray2::from_owned_array_bound(py,
        ndarray::Array2::from_shape_vec((t, t), p_u8).unwrap()))?;
    dict.set_item("r_star", PyArray2::from_owned_array_bound(py,
        ndarray::Array2::from_shape_vec((t, t), rs_u8).unwrap()))?;
    dict.set_item("expenditure", PyArray2::from_owned_array_bound(py,
        ndarray::Array2::from_shape_vec((t, t), graph.e[..t * t].to_vec()).unwrap()))?;
    dict.set_item("edge_weights", PyArray2::from_owned_array_bound(py,
        ndarray::Array2::from_shape_vec((t, t), graph.edge_weights[..t * t].to_vec()).unwrap()))?;

    dict.set_item("own_expenditure", PyArray1::from_vec_bound(py, graph.own_exp[..t].to_vec()))?;
    dict.set_item("scc_labels", PyArray1::from_vec_bound(py, graph.scc_labels[..t].to_vec()))?;

    dict.set_item("n_violations", garp.n_violations)?;
    dict.set_item("max_scc", garp.max_scc_size)?;
    dict.set_item("n_components", garp.n_components)?;
    dict.set_item("is_garp", garp.is_consistent)?;
    dict.set_item("t", t)?;

    Ok(dict)
}

/// Analyze a Parquet file directly in Rust - no Python data prep overhead.
///
/// Reads Parquet row groups, groups by user_col, extracts cost/action columns,
/// and feeds users to the Rayon-parallel analysis pipeline. Results returned
/// as list of (user_id, result_dict) tuples.
#[cfg(feature = "parquet")]
#[pyfunction]
#[pyo3(signature = (path, user_col, cost_cols, action_cols, compute_ccei=true, compute_mpi=false, compute_harp=false, compute_hm=false, compute_utility=false, compute_vei=false, compute_vei_exact=false, compute_network=false, tolerance=1e-10, chunk_size=50000))]
pub fn analyze_parquet_file<'py>(
    py: Python<'py>,
    path: &str,
    user_col: &str,
    cost_cols: Vec<String>,
    action_cols: Vec<String>,
    compute_ccei: bool,
    compute_mpi: bool,
    compute_harp: bool,
    compute_hm: bool,
    compute_utility: bool,
    compute_vei: bool,
    compute_vei_exact: bool,
    compute_network: bool,
    tolerance: f64,
    chunk_size: usize,
) -> PyResult<Vec<(String, Bound<'py, PyDict>)>> {
    use crate::parquet_reader::read_parquet_users_wide;

    let flags = MetricFlags {
        ccei: compute_ccei,
        mpi: compute_mpi,
        harp: compute_harp,
        hm: compute_hm,
        utility: compute_utility,
        vei: compute_vei,
        vei_exact: compute_vei_exact,
        network: compute_network,
    };

    // Read and chunk users from Parquet (GIL released during I/O + Rayon)
    let chunks = py.allow_threads(|| {
        read_parquet_users_wide(path, user_col, &cost_cols, &action_cols, chunk_size)
    }).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

    let mut all_results: Vec<(String, Bound<'py, PyDict>)> = Vec::new();

    for chunk in chunks {
        let user_ids: Vec<String> = chunk.iter().map(|(uid, _, _, _, _)| uid.clone()).collect();
        let user_data: Vec<(Vec<f64>, Vec<f64>, usize, usize)> = chunk
            .into_iter()
            .map(|(_, p, q, t, k)| (p, q, t, k))
            .collect();

        // Run Rayon-parallel analysis (GIL released)
        let results = py.allow_threads(|| {
            process_users_parallel(&user_data, flags, tolerance)
        });

        let dicts = results_to_pydicts(py, results);
        for (uid, dict) in user_ids.into_iter().zip(dicts) {
            all_results.push((uid, dict));
        }
    }

    Ok(all_results)
}

// ============================================================================
// Champion vs Challenger benchmark comparison functions
// ============================================================================

use rpt_core::mpi::mpi_karp_v2;
use rpt_core::closure::scc_transitive_closure_v2;

/// Benchmark MPI: champion (dense inner loop) vs challenger (sparse predecessors).
///
/// Runs both on the same user data and returns:
/// (mpi_champion, mpi_challenger, microseconds_champion, microseconds_challenger)
#[pyfunction]
#[pyo3(signature = (prices, quantities, tolerance=1e-10))]
pub fn benchmark_mpi<'py>(
    _py: Python<'py>,
    prices: PyReadonlyArray2<'py, f64>,
    quantities: PyReadonlyArray2<'py, f64>,
    tolerance: f64,
) -> PyResult<(f64, f64, u64, u64)> {
    let (p_flat, q_flat, t, k) = extract_user_data(&prices, &quantities);

    // Champion: mpi_karp (dense)
    let mut graph = PreferenceGraph::new(t);
    graph.parse_budget(&p_flat, &q_flat, t, k, tolerance);
    let g = garp_check(&mut graph);
    if g.is_consistent {
        return Ok((0.0, 0.0, 0, 0));
    }
    let start = Instant::now();
    let mpi_champ = rpt_core::mpi::mpi_karp(&graph);
    let us_champ = start.elapsed().as_micros() as u64;

    // Challenger: mpi_karp_v2 (sparse)
    let start = Instant::now();
    let mpi_chall = mpi_karp_v2(&graph);
    let us_chall = start.elapsed().as_micros() as u64;

    Ok((mpi_champ, mpi_chall, us_champ, us_chall))
}

/// Benchmark HM greedy: champion (per-SCC copies) vs challenger (direct scoring).
///
/// Uses T > 200 to trigger greedy path (not ILP).
/// Returns: (hm_champion, hm_challenger, microseconds_champion, microseconds_challenger)
/// where hm = n_consistent (integer).
#[pyfunction]
#[pyo3(signature = (prices, quantities, tolerance=1e-10))]
pub fn benchmark_hm<'py>(
    _py: Python<'py>,
    prices: PyReadonlyArray2<'py, f64>,
    quantities: PyReadonlyArray2<'py, f64>,
    tolerance: f64,
) -> PyResult<(u32, u32, u32, u64, u64)> {
    use rpt_core::houtman_maks::{houtman_maks, houtman_maks_greedy_v2_pub};
    let (p_flat, q_flat, t, k) = extract_user_data(&prices, &quantities);

    // Champion: houtman_maks (greedy with per-SCC copies)
    let mut graph = PreferenceGraph::new(t);
    graph.parse_budget(&p_flat, &q_flat, t, k, tolerance);
    let start = Instant::now();
    let (c_champ, t_champ) = houtman_maks(&mut graph);
    let us_champ = start.elapsed().as_micros() as u64;

    // Challenger: greedy_v2 (direct scoring, no per-SCC copies)
    graph.reset();
    graph.parse_budget(&p_flat, &q_flat, t, k, tolerance);
    graph.ensure_closure();
    let start = Instant::now();
    let (c_chall, _) = houtman_maks_greedy_v2_pub(&graph);
    let us_chall = start.elapsed().as_micros() as u64;

    Ok((c_champ as u32, c_chall as u32, t_champ as u32, us_champ, us_chall))
}

/// Benchmark closure: champion (Vec<bool> reachability) vs challenger (u64 bitsets).
///
/// Runs both on the same preference graph and compares resulting closure matrices.
/// Returns: (closures_match, microseconds_champion, microseconds_challenger)
#[pyfunction]
#[pyo3(signature = (prices, quantities, tolerance=1e-10))]
pub fn benchmark_closure<'py>(
    _py: Python<'py>,
    prices: PyReadonlyArray2<'py, f64>,
    quantities: PyReadonlyArray2<'py, f64>,
    tolerance: f64,
) -> PyResult<(bool, u64, u64)> {
    let (p_flat, q_flat, t, k) = extract_user_data(&prices, &quantities);

    // Build R matrix
    let mut graph = PreferenceGraph::new(t);
    graph.parse_budget(&p_flat, &q_flat, t, k, tolerance);
    graph.ensure_r(tolerance);

    let r_copy = graph.r[..t * t].to_vec();

    // Champion: scc_transitive_closure (Vec<bool> reach)
    let mut c1 = vec![false; t * t];
    let mut l1 = vec![0u32; t];
    let start = Instant::now();
    rpt_core::closure::scc_transitive_closure(&r_copy, t, &mut c1, &mut l1);
    let us_champ = start.elapsed().as_micros() as u64;

    // Challenger: scc_transitive_closure_v2 (u64 bitsets + FW pivot skip)
    let mut c2 = vec![false; t * t];
    let mut l2 = vec![0u32; t];
    let start = Instant::now();
    scc_transitive_closure_v2(&r_copy, t, &mut c2, &mut l2);
    let us_chall = start.elapsed().as_micros() as u64;

    // Compare closures
    let match_ = c1 == c2;

    Ok((match_, us_champ, us_chall))
}
