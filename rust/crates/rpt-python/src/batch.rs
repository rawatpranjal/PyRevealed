use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use std::time::Instant;

use rpt_core::ccei::ccei_search;
use rpt_core::garp::{garp_check, garp_check_with_closure};
use rpt_core::graph::PreferenceGraph;
use rpt_core::harp::harp_check;
use rpt_core::houtman_maks::houtman_maks;
use rpt_core::mpi::{mpi_karp, mpi_fast};
use rpt_core::utility::recover_utility;
use rpt_core::vei::{compute_vei as run_vei, compute_vei_exact as run_vei_exact};

use crate::convert::extract_user_data;

/// Analyze a batch of users in parallel using Rayon.
///
/// Uses PreferenceGraph with lazy computation — expenditure matrix built once,
/// R/P/closure reused across metrics. MPI uses Karp's max-mean-weight cycle
/// (theory-correct). CCEI runs last since it may modify graph state.
#[pyfunction]
#[pyo3(signature = (prices_list, quantities_list, compute_ccei=true, compute_mpi=false, compute_harp=false, compute_hm=false, compute_utility=false, compute_vei=false, compute_vei_exact=false, tolerance=1e-10))]
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
    tolerance: f64,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let n_users = prices_list.len();

    let users: Vec<(Vec<f64>, Vec<f64>, usize, usize)> = (0..n_users)
        .map(|i| extract_user_data(&prices_list[i], &quantities_list[i]))
        .collect();

    let max_t = users.iter().map(|(_, _, t, _)| *t).max().unwrap_or(0);

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
    }

    let results: Vec<UserOut> = users
        .par_iter()
        .map_init(
            || PreferenceGraph::new(max_t),
            |graph, (p_flat, q_flat, t, k)| {
                let start = Instant::now();
                let t = *t;
                let k = *k;

                graph.reset();
                graph.parse_budget(p_flat, q_flat, t, k, tolerance);

                // GARP: use O(T²) when only bool needed, O(T³) when MPI/VEI need closure
                let needs_closure = compute_mpi || compute_vei;
                let garp = if needs_closure {
                    garp_check_with_closure(graph)
                } else {
                    garp_check(graph)
                };

                // MPI via Karp's algorithm (theory-correct, before CCEI)
                let mpi = if compute_mpi && !garp.is_consistent {
                    mpi_karp(graph)
                } else {
                    0.0
                };

                // HARP
                let is_harp = if compute_harp {
                    harp_check(graph, tolerance).is_consistent
                } else {
                    false
                };

                // Houtman-Maks
                let (hm_c, hm_t) = if compute_hm {
                    houtman_maks(graph)
                } else {
                    (0, 0)
                };

                // Utility recovery (Afriat LP)
                let utility_success = if compute_utility {
                    recover_utility(graph).success
                } else {
                    false
                };

                // VEI (per-observation efficiency, LP relaxation)
                let (vei_mean, vei_min) = if compute_vei {
                    let vei = run_vei(graph);
                    (vei.mean_efficiency, vei.min_efficiency)
                } else {
                    (1.0, 1.0)
                };

                // VEI exact (Mononen binary LP + row generation)
                let (vei_exact_mean, vei_exact_min) = if compute_vei_exact {
                    let vei = run_vei_exact(graph);
                    (vei.mean_efficiency, vei.min_efficiency)
                } else {
                    (1.0, 1.0)
                };

                // CCEI (last — may overwrite R/P/closure)
                let ccei = if compute_ccei && !garp.is_consistent {
                    ccei_search(graph, tolerance).ccei
                } else if garp.is_consistent {
                    1.0
                } else {
                    -1.0
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
                }
            },
        )
        .collect();

    let py_results: Vec<Bound<'py, PyDict>> = results
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
            dict
        })
        .collect();

    Ok(py_results)
}

/// Analyze a batch of users' MENU choice data in parallel.
///
/// Each user has: menus (list of lists of item indices) + choices (list of chosen item).
/// Computes SARP, WARP, Houtman-Maks, and optionally WARP-LA per user.
///
/// This is the "rec/search click" path — no prices, just menus and picks.
#[pyfunction]
#[pyo3(signature = (menus_list, choices_list, n_items_list, compute_warp_la=false))]
pub fn analyze_menu_batch<'py>(
    py: Python<'py>,
    menus_list: Vec<Vec<Vec<usize>>>,   // menus_list[user][obs] = vec of item indices
    choices_list: Vec<Vec<usize>>,       // choices_list[user][obs] = chosen item index
    n_items_list: Vec<usize>,            // n_items per user (max item index + 1)
    compute_warp_la: bool,
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
            dict.set_item("compute_time_us", r.time_us).unwrap();
            dict
        })
        .collect();

    Ok(py_results)
}

/// Build a preference graph from budget data and return it as numpy arrays.
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
    let garp = garp_check(&mut graph);
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
