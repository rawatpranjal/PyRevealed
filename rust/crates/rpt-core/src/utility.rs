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

/// Recover utility values via Bellman-Ford shortest paths (Shiozawa, 2016).
///
/// Uses the SPP connection: GARP ⟺ no negative-weight cycles in the preference
/// graph with edge weights w(i→j) = E[i,j] - own_exp[i] (≤ 0 for direct preferences).
/// Utility values are shortest-path distances from a virtual source, which satisfy
/// Afriat's inequalities with λ_i = 1.
///
/// Faster than the LP for large T (graph-only, no solver call).
/// For integer data, the returned utility values are integer distances.
///
/// Returns success=false if GARP is violated (negative cycle detected).
///
/// Reference: Shiozawa (2016, JME 67), "Revealed Preference Test and Shortest Path Problem."
pub fn recover_utility_bellman_ford(graph: &mut PreferenceGraph) -> UtilityResult {
    graph.ensure_r(graph.tolerance);
    let t = graph.t;

    if t == 0 {
        return UtilityResult {
            success: true,
            utility_values: vec![],
            lagrange_multipliers: vec![],
        };
    }

    // Build edges: for each direct preference (i,j) where R[i][j],
    // weight w(i→j) = E[i,j] - own_exp[i] (≤ 0 for actual revealed preferences)
    let mut edges: Vec<(usize, usize, f64)> = Vec::new();
    for i in 0..t {
        for j in 0..t {
            if i != j && graph.r[i * t + j] {
                let w = graph.e[i * t + j] - graph.own_exp[i];
                edges.push((i, j, w));
            }
        }
    }

    // Bellman-Ford from virtual source (all nodes at distance 0 initially)
    // This is equivalent to adding a virtual source s with 0-weight edges to all nodes.
    let mut dist = vec![0.0f64; t];

    // Relax T-1 times
    for _ in 0..t {
        let mut changed = false;
        for &(u, v, w) in &edges {
            if dist[u] + w < dist[v] - 1e-12 {
                dist[v] = dist[u] + w;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    // Check for negative cycles (= GARP violation)
    for &(u, v, w) in &edges {
        if dist[u] + w < dist[v] - 1e-10 {
            return UtilityResult {
                success: false,
                utility_values: vec![0.0; t],
                lagrange_multipliers: vec![0.0; t],
            };
        }
    }

    // Utility values are the (negated) shortest-path distances.
    // Negate so higher utility = "better" (shorter distance = more constrained = lower utility).
    // Actually: the distances are non-positive (weights ≤ 0, initial dist = 0).
    // Shift so min utility is 0: U_i = dist[i] - min(dist)
    let min_dist = dist.iter().cloned().fold(f64::INFINITY, f64::min);
    let utility_values: Vec<f64> = dist.iter().map(|&d| d - min_dist).collect();

    UtilityResult {
        success: true,
        utility_values,
        lagrange_multipliers: vec![1.0; t], // λ_i = 1 (fixed normalization)
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

    // --- Bellman-Ford utility recovery tests ---

    #[test]
    fn test_bf_utility_consistent() {
        let mut graph = PreferenceGraph::new(2);
        let prices = [1.0, 2.0, 2.0, 1.0];
        let quantities = [4.0, 1.0, 1.0, 4.0];
        graph.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        let result = recover_utility_bellman_ford(&mut graph);
        assert!(result.success);
        assert_eq!(result.utility_values.len(), 2);
        // λ_i = 1 (fixed normalization)
        assert!(result.lagrange_multipliers.iter().all(|&l| l == 1.0));
    }

    #[test]
    fn test_bf_utility_violation() {
        let mut graph = PreferenceGraph::new(2);
        let prices = [2.0, 1.0, 1.0, 2.0];
        let quantities = [3.0, 2.0, 2.0, 3.0];
        graph.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        let result = recover_utility_bellman_ford(&mut graph);
        // GARP violation → negative cycle → success=false
        assert!(!result.success);
    }

    #[test]
    fn test_bf_vs_lp_consistent() {
        // Both methods should succeed on consistent data
        let mut graph = PreferenceGraph::new(2);
        let prices = [1.0, 2.0, 2.0, 1.0];
        let quantities = [4.0, 1.0, 1.0, 4.0];
        graph.parse_budget(&prices, &quantities, 2, 2, 1e-10);

        let lp_result = recover_utility(&graph);
        let bf_result = recover_utility_bellman_ford(&mut graph);

        assert!(lp_result.success);
        assert!(bf_result.success);

        // Both should produce valid utility values (may differ, both are valid)
        // Check Afriat inequalities: U_j ≤ U_i + λ_i·(E[i,j] - own_exp[i]) for R-edges
        let t = graph.t;
        for i in 0..t {
            for j in 0..t {
                if i != j && graph.r[i * t + j] {
                    let rhs = bf_result.utility_values[i]
                        + 1.0 * (graph.e[i * t + j] - graph.own_exp[i]);
                    assert!(
                        bf_result.utility_values[j] <= rhs + 1e-8,
                        "Afriat violated: U[{j}]={} > U[{i}]+w={}",
                        bf_result.utility_values[j],
                        rhs,
                    );
                }
            }
        }
    }
}
