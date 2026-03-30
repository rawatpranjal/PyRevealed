use crate::graph::PreferenceGraph;
use crate::scc::tarjan_scc;
use crate::types::GarpResult;

/// Check GARP consistency in O(T²) using the SCC-based algorithm.
///
/// Talla Nobibon, Smeulders & Spieksma (2015), ""; testing GARP in
/// O(n²)", JOTA 166(3), Corollary 4.3:
///
///   "GARP can be tested in O(n²) time."
///
/// The key insight (Lemma 4.1) is that GARP is a special case of
/// arc coloring: GARP is violated iff any SCC of the weak preference
/// graph R contains a strict preference arc P. This avoids the O(T³)
/// transitive closure from the classical Varian (1982) algorithm.
///
/// Algorithm:
///   1. Build R in O(T²)
///   2. Tarjan's SCC on R in O(T + |A|) ≤ O(T²)
///   3. Check intra-SCC arcs for strict preference in O(T²)
///
/// This function only determines the boolean is_consistent and does NOT
/// populate graph.r_star. Use garp_check_with_closure() when the caller
/// needs r_star (e.g., for violation cycle extraction, MPI, or VEI).
pub fn garp_check(graph: &mut PreferenceGraph) -> GarpResult {
    graph.ensure_r(graph.tolerance);
    let t = graph.t;

    // Step 1: Tarjan's SCC on the weak preference graph R₀
    let n_comp = tarjan_scc(&graph.r[..t * t], t, &mut graph.scc_labels[..t]);
    graph.n_components = n_comp;

    // Compute max SCC size
    let mut scc_sizes = vec![0u32; n_comp];
    for i in 0..t {
        scc_sizes[graph.scc_labels[i] as usize] += 1;
    }
    graph.max_scc_size = *scc_sizes.iter().max().unwrap_or(&0) as usize;

    // Step 2: Check each intra-SCC arc for strict preference
    // GARP violated iff exists (i,j) in same SCC where P[i,j] = true
    let mut n_violations = 0u32;
    for i in 0..t {
        for j in 0..t {
            if i != j
                && graph.scc_labels[i] == graph.scc_labels[j]
                && graph.p[i * t + j]
            {
                n_violations += 1;
            }
        }
    }

    GarpResult {
        is_consistent: n_violations == 0,
        n_violations,
        max_scc_size: graph.max_scc_size as u32,
        n_components: n_comp as u32,
    }
}

/// Check GARP with full transitive closure via Floyd-Warshall (O(T³)).
///
/// Varian (1982), "The Nonparametric Approach to Demand Analysis",
/// Econometrica 50(4), pp. 945-973:
///
///   GARP violated iff exists (i,j) where R*[i,j] = 1 and P[j,i] = 1,
///   where R* is the transitive closure of the weak preference graph R.
///
/// This populates graph.r_star which is needed for violation cycle
/// extraction, MPI (Echenique et al. 2011), and VEI (Varian 1990).
/// Prefer garp_check() when only the boolean result is needed.
pub fn garp_check_with_closure(graph: &mut PreferenceGraph) -> GarpResult {
    graph.ensure_closure();
    let t = graph.t;

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
