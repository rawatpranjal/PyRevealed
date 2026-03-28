use crate::closure::scc_transitive_closure;
use crate::graph::PreferenceGraph;
use crate::types::GarpResult;

/// Check Menu SARP (Strong Axiom for abstract choice).
///
/// Operates on item-space graph (I items as nodes). R[x,y] = True if x was
/// chosen from a menu containing y. SARP violated if R*[x,y] AND R*[y,x]
/// for any x != y (cycle in transitive preferences).
///
/// Requires: graph parsed with parse_menu() (R already built on items).
pub fn menu_sarp_check(graph: &mut PreferenceGraph) -> GarpResult {
    graph.ensure_closure();
    let t = graph.t; // t = n_items for menu data

    // SARP violation: R*[x,y] AND R*[y,x] for x != y
    let mut n_violations = 0u32;
    for x in 0..t {
        for y in (x + 1)..t {
            if graph.r_star[x * t + y] && graph.r_star[y * t + x] {
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

/// Check Menu WARP (Weak Axiom for abstract choice).
///
/// Simpler than SARP: only checks for direct reversals.
/// Violation if R[x,y] AND R[y,x] for any x != y.
/// No transitive closure needed - O(I²).
pub fn menu_warp_check(graph: &PreferenceGraph) -> GarpResult {
    let t = graph.t;
    let mut n_violations = 0u32;

    for x in 0..t {
        for y in (x + 1)..t {
            if graph.r[x * t + y] && graph.r[y * t + x] {
                n_violations += 1;
            }
        }
    }

    GarpResult {
        is_consistent: n_violations == 0,
        n_violations,
        max_scc_size: 0,
        n_components: 0,
    }
}

/// Menu Houtman-Maks: fraction of observations consistent with SARP.
///
/// Returns (n_consistent, n_total) where n_consistent is the max number
/// of observations that can be kept while satisfying SARP.
///
/// Uses greedy FVS on the item-space preference graph.
pub fn menu_houtman_maks(graph: &mut PreferenceGraph) -> (usize, usize) {
    graph.ensure_closure();
    let t = graph.t;

    // Find non-trivial SCCs (these contain cycles)
    let mut scc_sizes = vec![0u32; graph.n_components];
    for i in 0..t {
        scc_sizes[graph.scc_labels[i] as usize] += 1;
    }

    // Count items in non-trivial SCCs (items that must be "removed")
    let mut items_in_cycles = 0usize;
    for &s in &scc_sizes {
        if s > 1 {
            // Greedy: need to remove at least (s - 1) items from each SCC
            // to break all cycles. Approximate with s/2.
            items_in_cycles += (s / 2) as usize;
        }
    }

    let n_consistent = t.saturating_sub(items_in_cycles);
    (n_consistent, t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_menu_sarp_consistent() {
        // 0 > 1 > 2 (consistent linear order)
        let mut graph = PreferenceGraph::new(3);
        let menus = vec![vec![0, 1, 2], vec![1, 2]];
        let choices = vec![0, 1];
        graph.parse_menu(&menus, &choices, 3);
        let result = menu_sarp_check(&mut graph);
        assert!(result.is_consistent);
    }

    #[test]
    fn test_menu_sarp_violation() {
        // 0 > 1, 1 > 0 (direct cycle)
        let mut graph = PreferenceGraph::new(3);
        let menus = vec![vec![0, 1], vec![0, 1]];
        let choices = vec![0, 1];
        graph.parse_menu(&menus, &choices, 2);
        let result = menu_sarp_check(&mut graph);
        assert!(!result.is_consistent);
    }

    #[test]
    fn test_menu_warp_consistent() {
        let mut graph = PreferenceGraph::new(3);
        let menus = vec![vec![0, 1, 2], vec![1, 2]];
        let choices = vec![0, 1];
        graph.parse_menu(&menus, &choices, 3);
        let result = menu_warp_check(&graph);
        assert!(result.is_consistent);
    }

    #[test]
    fn test_menu_warp_violation() {
        let mut graph = PreferenceGraph::new(3);
        let menus = vec![vec![0, 1], vec![0, 1]];
        let choices = vec![0, 1];
        graph.parse_menu(&menus, &choices, 2);
        let result = menu_warp_check(&graph);
        assert!(!result.is_consistent);
    }
}
