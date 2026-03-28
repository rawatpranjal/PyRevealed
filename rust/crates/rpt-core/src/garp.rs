use crate::graph::PreferenceGraph;
use crate::scc::tarjan_scc;
use crate::types::GarpResult;

/// Check GARP consistency in O(T²) - the provably optimal algorithm.
///
/// Talla Nobibon, Smeulders & Spieksma (2015, JOTA 166(3)):
/// GARP is violated iff any SCC of the weak preference graph R₀ contains
/// a strict preference arc P₀. No transitive closure needed.
///
/// 1. Build R₀ in O(T²)
/// 2. Tarjan's SCC in O(T + |A|) ≤ O(T²)
/// 3. Check intra-SCC arcs for strict preference in O(T²)
///
/// This is the default GARP check. Use garp_check_with_closure() if you
/// also need the full transitive closure matrix (e.g., for MPI or VEI).
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

/// Check GARP with full transitive closure (O(T³) but provides R* matrix).
///
/// Use this when downstream algorithms need the closure (MPI, VEI, etc.).
/// The O(T²) garp_check() is preferred when only the bool result is needed.
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
