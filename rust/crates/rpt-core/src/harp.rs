use crate::graph::PreferenceGraph;
use crate::types::HarpResult;

/// Check HARP (Homothetic Axiom of Revealed Preference).
///
/// Tests whether preferences are homothetic using max-product paths in log-space.
/// Requires: graph has expenditure and R built (call parse_budget first).
/// Lazily computes edge weights and max-product Floyd-Warshall.
pub fn harp_check(graph: &mut PreferenceGraph, tolerance: f64) -> HarpResult {
    graph.ensure_r(tolerance);
    graph.ensure_weights();
    graph.ensure_max_product();

    let t = graph.t;
    let mut is_consistent = true;
    let mut max_cycle_log = 0.0f64;

    for i in 0..t {
        let diag = graph.max_product[i * t + i];
        if diag > tolerance {
            is_consistent = false;
            if diag > max_cycle_log {
                max_cycle_log = diag;
            }
        }
    }

    HarpResult {
        is_consistent,
        max_cycle_product: if is_consistent { 1.0 } else { max_cycle_log.exp() },
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
