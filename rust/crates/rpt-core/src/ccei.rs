use crate::graph::PreferenceGraph;
use crate::types::CceiResult;

/// Compute CCEI (Critical Cost Efficiency Index) via discrete binary search.
///
/// Requires: graph has expenditure built (call parse_budget or ensure_expenditure first).
/// Reuses the expenditure matrix — only rebuilds R/P at each efficiency level.
pub fn ccei_search(graph: &mut PreferenceGraph, tolerance: f64) -> CceiResult {
    let t = graph.t;

    // Quick check: is data already GARP-consistent at e=1?
    graph.ensure_r(tolerance);
    if garp_consistent(graph) {
        return CceiResult {
            ccei: 1.0,
            iterations: 0,
            is_perfectly_consistent: true,
        };
    }

    // Collect all T^2 efficiency ratios in (0, 1) from cached expenditure
    let mut candidates: Vec<f64> = Vec::with_capacity(t * t);
    for i in 0..t {
        if graph.own_exp[i] <= 0.0 {
            continue;
        }
        for j in 0..t {
            let ratio = graph.e[i * t + j] / graph.own_exp[i];
            if ratio > 0.0 && ratio < 1.0 {
                candidates.push(ratio);
            }
        }
    }

    candidates.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());
    candidates.dedup_by(|a, b| (*a - *b).abs() < 1e-15);

    if candidates.is_empty() {
        return CceiResult {
            ccei: 0.0,
            iterations: 0,
            is_perfectly_consistent: false,
        };
    }

    // Binary search over candidates
    let mut lo = 0usize;
    let mut hi = candidates.len() - 1;
    let mut best_e = 0.0f64;
    let mut iterations = 0u32;

    while lo <= hi {
        let mid = (lo + hi) / 2;
        let e = candidates[mid];
        iterations += 1;

        // Rebuild R/P at this efficiency level (reuses cached E and own_exp)
        graph.build_r_at_efficiency(e, tolerance);
        graph.ensure_closure();

        if garp_consistent(graph) {
            best_e = e;
            if mid == 0 {
                break;
            }
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
    }

    // Restore R/P at e=1.0 so downstream algorithms get the right state
    graph.build_r_at_efficiency(1.0, tolerance);
    graph.ensure_closure();

    CceiResult {
        ccei: best_e,
        iterations,
        is_perfectly_consistent: false,
    }
}

/// Check GARP consistency on current graph state (no allocation).
fn garp_consistent(graph: &mut PreferenceGraph) -> bool {
    graph.ensure_closure();
    let t = graph.t;
    for i in 0..t {
        for j in 0..t {
            if graph.r_star[i * t + j] && graph.p[j * t + i] {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consistent_ccei_is_one() {
        let prices = [1.0, 2.0, 2.0, 1.0];
        let quantities = [4.0, 1.0, 1.0, 4.0];
        let mut graph = PreferenceGraph::new(2);
        graph.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        let result = ccei_search(&mut graph, 1e-10);
        assert_eq!(result.ccei, 1.0);
        assert!(result.is_perfectly_consistent);
    }

    #[test]
    fn test_violation_ccei_below_one() {
        let prices = [2.0, 1.0, 1.0, 2.0];
        let quantities = [3.0, 2.0, 2.0, 3.0];
        let mut graph = PreferenceGraph::new(2);
        graph.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        let result = ccei_search(&mut graph, 1e-10);
        assert!(result.ccei < 1.0);
        assert!(result.ccei > 0.0);
        assert!((result.ccei - 0.875).abs() < 0.001);
    }

    #[test]
    fn test_ccei_restores_state() {
        // After CCEI, graph should have R/P at e=1.0
        let prices = [2.0, 1.0, 1.0, 2.0];
        let quantities = [3.0, 2.0, 2.0, 3.0];
        let mut graph = PreferenceGraph::new(2);
        graph.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        let _ = ccei_search(&mut graph, 1e-10);
        // Closure should be at e=1.0 — GARP violations should exist
        assert!(graph.has_closure);
        let t = graph.t;
        let mut has_violation = false;
        for i in 0..t {
            for j in 0..t {
                if graph.r_star[i * t + j] && graph.p[j * t + i] {
                    has_violation = true;
                }
            }
        }
        assert!(has_violation);
    }
}
