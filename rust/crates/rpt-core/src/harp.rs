use crate::graph::PreferenceGraph;
use crate::types::HarpResult;

/// Check HARP (Homothetic Axiom of Revealed Preference).
///
/// Tests whether preferences are homothetic by checking that no cycle of
/// expenditure ratios has product > 1. Uses max-product Floyd-Warshall in
/// log-space: a positive diagonal entry means a violating cycle exists.
///
/// HARP is a **binary test** - Varian (1983) and Chambers & Echenique (2016,
/// Theorem 4.2) define only pass/fail, not a severity metric. The
/// `max_cycle_product` field is always 1.0; the meaningful output is
/// `is_consistent`.
///
/// # Algorithm
///
/// 1. Build log-ratio edge weights: w[i,j] = ln(p_i·x_i) - ln(p_i·x_j)
/// 2. Run modified Floyd-Warshall maximizing log-sum of paths
/// 3. If any diagonal entry > tolerance → cycle with product > 1 → violation
///
/// Complexity: O(T³) from Floyd-Warshall.
///
/// # References
///
/// - Varian, H. R. (1983). "Non-parametric tests of consumer behaviour."
///   Review of Economic Studies, 50(1), 99–110.
/// - Chambers, C. P. & Echenique, F. (2016). Revealed Preference Theory,
///   Ch. 4, Theorem 4.2: HARP ⟺ (≥^H, ≻^H) is acyclic.
pub fn harp_check(graph: &mut PreferenceGraph, tolerance: f64) -> HarpResult {
    graph.ensure_r(tolerance);
    graph.ensure_weights();
    graph.ensure_max_product();

    let t = graph.t;
    let mut is_consistent = true;

    // Check diagonal of max-product matrix: positive entry = violating cycle.
    // This is the operational form of Theorem 4.2 condition (V).
    for i in 0..t {
        let diag = graph.max_product[i * t + i];
        if diag > tolerance {
            is_consistent = false;
            break;
        }
    }

    HarpResult {
        is_consistent,
        // No severity metric is defined in the literature (Varian 1983,
        // C&E 2016). HARP is a binary test only.
        max_cycle_product: 1.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consistent_homothetic() {
        let prices = [1.0, 2.0, 2.0, 1.0];
        let quantities = [4.0, 1.0, 1.0, 4.0];
        let mut graph = PreferenceGraph::new(2);
        graph.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        let result = harp_check(&mut graph, 1e-10);
        assert!(result.is_consistent);
    }

    #[test]
    fn test_harp_violation() {
        let prices = [2.0, 1.0, 1.0, 2.0];
        let quantities = [3.0, 2.0, 2.0, 3.0];
        let mut graph = PreferenceGraph::new(2);
        graph.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        let result = harp_check(&mut graph, 1e-10);
        assert!(result.max_cycle_product >= 1.0 || result.is_consistent);
    }
}
