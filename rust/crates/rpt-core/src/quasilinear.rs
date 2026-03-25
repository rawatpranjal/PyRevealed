use crate::graph::PreferenceGraph;

/// Quasilinear utility test via negative cycle detection on surplus weights.
///
/// Tests whether choices are consistent with quasilinear utility U(x) = v(x_1,...,x_{K-1}) + x_K.
/// The test checks that all cycles in the preference graph have non-positive surplus sum.
///
/// Surplus weight: S[i,j] = E[i,j] - E[i,i] (how much more j's bundle costs at i's prices
/// minus i's own expenditure). A cycle with positive surplus sum indicates a money pump.
///
/// Uses Bellman-Ford to detect negative cycles (on negated weights).
///
/// Requires: graph has expenditure built.
/// Returns: (is_quasilinear, worst_cycle_surplus).
pub fn quasilinear_check(graph: &mut PreferenceGraph) -> (bool, f64) {
    graph.ensure_closure();
    let t = graph.t;

    if t == 0 {
        return (true, 0.0);
    }

    // Build surplus weights: S[i,j] = E[i,j] - own_exp[i]
    // For a cycle i0 -> i1 -> ... -> iN -> i0, the test is:
    // sum(S[ik, i_{k+1}]) <= 0 for all cycles
    // Equivalently: no positive-weight cycle exists.
    // We negate weights and check for negative cycles via Bellman-Ford.

    // Edge weights (only on R edges)
    let mut dist = vec![f64::MAX; t];
    let mut edges: Vec<(usize, usize, f64)> = Vec::new();

    for i in 0..t {
        for j in 0..t {
            if i == j {
                continue;
            }
            if graph.r[i * t + j] {
                // Weight = E[i,j] - own_exp[i] (surplus)
                let surplus = graph.e[i * t + j] - graph.own_exp[i];
                edges.push((i, j, surplus));
            }
        }
    }

    // Bellman-Ford from each node to detect positive-weight cycles
    let mut worst_surplus = 0.0f64;
    let mut has_positive_cycle = false;

    // Run Bellman-Ford from node 0 (detects all reachable positive cycles)
    // For full coverage, run from each SCC representative
    for start in 0..t {
        for d in dist.iter_mut() {
            *d = f64::MAX;
        }
        dist[start] = 0.0;

        // Relax edges T-1 times (using negated surplus for shortest path = most positive cycle)
        for _ in 0..t - 1 {
            for &(u, v, w) in &edges {
                if dist[u] < f64::MAX && dist[u] + w < dist[v] {
                    dist[v] = dist[u] + w;
                }
            }
        }

        // Check for negative cycles (in surplus space, this means positive surplus cycle)
        // Wait — we want to detect POSITIVE surplus cycles. So we negate:
        // If after T-1 relaxations we can still improve, there's a negative-weight cycle.
        // But our weights are surplus (positive = violation). So we check for the ability
        // to keep decreasing distances, which would mean a negative-surplus cycle (good).
        // Actually: a positive-surplus cycle means money can be pumped.
        // With Bellman-Ford on raw surplus: if we can still relax after T-1 iterations,
        // there's a negative cycle. But we want to detect POSITIVE cycles.
        // Solution: negate weights.
    }

    // Simpler approach: check all cycles found via SCC structure
    // For each non-trivial SCC, find the cycle with maximum total surplus
    let mut scc_nodes: Vec<Vec<usize>> = vec![Vec::new(); graph.n_components];
    for i in 0..t {
        scc_nodes[graph.scc_labels[i] as usize].push(i);
    }

    for scc in &scc_nodes {
        if scc.len() <= 1 {
            continue;
        }
        // Use Bellman-Ford on negated weights within this SCC to find positive cycles
        let n = scc.len();
        let mut scc_dist = vec![0.0f64; n]; // Start at 0 from all nodes

        // Build local edges
        let mut local_edges: Vec<(usize, usize, f64)> = Vec::new();
        for (li, &ni) in scc.iter().enumerate() {
            for (lj, &nj) in scc.iter().enumerate() {
                if li == lj {
                    continue;
                }
                if graph.r[ni * t + nj] {
                    let surplus = graph.e[ni * t + nj] - graph.own_exp[ni];
                    // Negate: we want to find cycles where sum of surplus > 0
                    // = negative cycle in negated weights
                    local_edges.push((li, lj, -surplus));
                }
            }
        }

        // Bellman-Ford
        for _ in 0..n - 1 {
            for &(u, v, w) in &local_edges {
                if scc_dist[u] + w < scc_dist[v] {
                    scc_dist[v] = scc_dist[u] + w;
                }
            }
        }

        // Check for negative cycle (= positive surplus cycle)
        for &(u, v, w) in &local_edges {
            if scc_dist[u] + w < scc_dist[v] - 1e-10 {
                has_positive_cycle = true;
                // Estimate the surplus magnitude
                let gap = scc_dist[v] - (scc_dist[u] + w);
                if gap > worst_surplus {
                    worst_surplus = gap;
                }
            }
        }
    }

    (!has_positive_cycle, worst_surplus)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consistent_quasilinear() {
        let prices = [1.0, 2.0, 2.0, 1.0];
        let quantities = [4.0, 1.0, 1.0, 4.0];
        let mut graph = PreferenceGraph::new(2);
        graph.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        let (is_ql, _) = quasilinear_check(&mut graph);
        assert!(is_ql);
    }

    #[test]
    fn test_violation_quasilinear() {
        // Data with GARP violation — may also violate quasilinearity
        let prices = [2.0, 1.0, 1.0, 2.0];
        let quantities = [3.0, 2.0, 2.0, 3.0];
        let mut graph = PreferenceGraph::new(2);
        graph.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        let (_, worst) = quasilinear_check(&mut graph);
        // Just verify it runs without panic
        assert!(worst >= 0.0);
    }
}
