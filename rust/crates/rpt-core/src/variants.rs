use crate::closure::scc_transitive_closure;
use crate::graph::PreferenceGraph;
use crate::types::GarpResult;

/// Check SARP (Strong Axiom of Revealed Preference).
///
/// Stricter than GARP: violation if R*[i,j] AND R*[j,i] for any i!=j.
/// Requires: graph has expenditure and R built (call parse_budget first).
pub fn sarp_check(graph: &mut PreferenceGraph) -> GarpResult {
    graph.ensure_closure();
    let t = graph.t;

    // SARP violation: R*[i,j] AND R*[j,i] with i!=j (any cycle)
    let mut n_violations = 0u32;
    for i in 0..t {
        for j in (i + 1)..t {
            if graph.r_star[i * t + j] && graph.r_star[j * t + i] {
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

/// Check Acyclical-P (strict preference acyclicity).
///
/// Builds transitive closure of P (strict) instead of R (weak).
/// Violation if any non-trivial SCC exists in the strict preference graph.
///
/// NOTE: This temporarily overwrites graph.r with P values and recomputes closure.
/// Call this AFTER garp_check/mpi if you need e=1.0 R closure for those.
pub fn acyclical_p_check(graph: &mut PreferenceGraph, tolerance: f64) -> GarpResult {
    let t = graph.t;

    // Build P (strict) into a temporary buffer, then run closure on it
    // We use graph.r as scratch since we're done with weak R
    for i in 0..t {
        for j in 0..t {
            let idx = i * t + j;
            graph.r[idx] = graph.own_exp[i] > graph.e[idx] + tolerance;
        }
        graph.r[i * t + i] = false;
    }
    graph.has_r = true;
    graph.has_closure = false;

    let (n_comp, max_scc) = scc_transitive_closure(
        &graph.r[..t * t],
        t,
        &mut graph.r_star[..t * t],
        &mut graph.scc_labels[..t],
    );
    graph.n_components = n_comp;
    graph.max_scc_size = max_scc;
    graph.has_closure = true;

    // Violation: any SCC of size > 1 in strict preference graph
    let mut scc_sizes = vec![0u32; n_comp];
    for i in 0..t {
        scc_sizes[graph.scc_labels[i] as usize] += 1;
    }
    let mut n_violations = 0u32;
    for &s in &scc_sizes {
        if s > 1 {
            n_violations += s * (s - 1);
        }
    }

    GarpResult {
        is_consistent: n_violations == 0,
        n_violations,
        max_scc_size: max_scc as u32,
        n_components: n_comp as u32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sarp_consistent() {
        let prices = [1.0, 2.0, 2.0, 1.0];
        let quantities = [4.0, 1.0, 1.0, 4.0];
        let mut graph = PreferenceGraph::new(2);
        graph.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        let result = sarp_check(&mut graph);
        assert!(result.is_consistent);
    }

    #[test]
    fn test_sarp_violation() {
        let prices = [2.0, 1.0, 1.0, 2.0];
        let quantities = [3.0, 2.0, 2.0, 3.0];
        let mut graph = PreferenceGraph::new(2);
        graph.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        let result = sarp_check(&mut graph);
        assert!(!result.is_consistent);
    }

    #[test]
    fn test_acyclical_p_consistent() {
        let prices = [1.0, 2.0, 2.0, 1.0];
        let quantities = [4.0, 1.0, 1.0, 4.0];
        let mut graph = PreferenceGraph::new(2);
        graph.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        let result = acyclical_p_check(&mut graph, 1e-10);
        assert!(result.is_consistent);
    }
}
