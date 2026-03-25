use crate::lp::solve_afriat_lp;
use crate::graph::PreferenceGraph;

/// Result of utility recovery.
#[derive(Debug, Clone)]
pub struct UtilityResult {
    pub success: bool,
    pub utility_values: Vec<f64>,
    pub lagrange_multipliers: Vec<f64>,
}

/// Recover utility values from a preference graph using Afriat's LP.
///
/// Requires: graph has expenditure computed (call parse_budget first).
/// Uses the HiGHS LP solver in Rust (no scipy dependency).
pub fn recover_utility(graph: &PreferenceGraph) -> UtilityResult {
    let t = graph.t;

    match solve_afriat_lp(&graph.e[..t * t], &graph.own_exp[..t], t, 1e-10) {
        Some((u, lambdas)) => UtilityResult {
            success: true,
            utility_values: u,
            lagrange_multipliers: lambdas,
        },
        None => UtilityResult {
            success: false,
            utility_values: vec![0.0; t],
            lagrange_multipliers: vec![0.0; t],
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_utility_recovery_consistent() {
        let mut graph = PreferenceGraph::new(2);
        let prices = [1.0, 2.0, 2.0, 1.0];
        let quantities = [4.0, 1.0, 1.0, 4.0];
        graph.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        let result = recover_utility(&graph);
        assert!(result.success);
        assert_eq!(result.utility_values.len(), 2);
        assert!(result.lagrange_multipliers.iter().all(|&l| l > 0.0));
    }

    #[test]
    fn test_utility_recovery_violation() {
        let mut graph = PreferenceGraph::new(2);
        let prices = [2.0, 1.0, 1.0, 2.0];
        let quantities = [3.0, 2.0, 2.0, 3.0];
        graph.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        let result = recover_utility(&graph);
        assert!(!result.success);
    }
}
