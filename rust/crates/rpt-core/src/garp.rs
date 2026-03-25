use crate::graph::PreferenceGraph;
use crate::types::GarpResult;

/// Check GARP consistency using SCC-optimized transitive closure.
///
/// Requires: graph has expenditure and R/P built (call parse_budget first).
/// Computes transitive closure lazily if not already done.
pub fn garp_check(graph: &mut PreferenceGraph) -> GarpResult {
    graph.ensure_closure();
    let t = graph.t;

    // GARP violation: R*[i,j] AND P[j,i]
    let mut n_violations = 0u32;
    for i in 0..t {
        for j in 0..t {
            if graph.r_star[i * t + j] && graph.p[j * t + i] {
                n_violations += 1;
            }
        }
    }

    GarpResult {
        is_consistent: n_violations == 0,
        n_violations,
        max_scc_size: graph.max_scc_size as u32,
        n_components: graph.n_components as u32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consistent_data() {
        let prices = [1.0, 2.0, 2.0, 1.0];
        let quantities = [4.0, 1.0, 1.0, 4.0];
        let mut graph = PreferenceGraph::new(2);
        graph.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        let result = garp_check(&mut graph);
        assert!(result.is_consistent);
        assert_eq!(result.n_violations, 0);
    }

    #[test]
    fn test_warp_violation() {
        let prices = [2.0, 1.0, 1.0, 2.0];
        let quantities = [3.0, 2.0, 2.0, 3.0];
        let mut graph = PreferenceGraph::new(2);
        graph.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        let result = garp_check(&mut graph);
        assert!(!result.is_consistent);
        assert!(result.n_violations > 0);
    }

    #[test]
    fn test_graph_reuse() {
        let mut graph = PreferenceGraph::new(10);

        // First user: consistent
        let p1 = [1.0, 2.0, 2.0, 1.0];
        let q1 = [4.0, 1.0, 1.0, 4.0];
        graph.parse_budget(&p1, &q1, 2, 2, 1e-10);
        let r1 = garp_check(&mut graph);
        assert!(r1.is_consistent);

        // Second user: violation (same graph, reset)
        graph.reset();
        let p2 = [2.0, 1.0, 1.0, 2.0];
        let q2 = [3.0, 2.0, 2.0, 3.0];
        graph.parse_budget(&p2, &q2, 2, 2, 1e-10);
        let r2 = garp_check(&mut graph);
        assert!(!r2.is_consistent);
    }
}
