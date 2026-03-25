use crate::graph::PreferenceGraph;
use crate::lp::solve_hm_ilp;
use crate::scc::tarjan_scc;

/// Houtman-Maks index: maximum subset of observations consistent with GARP.
///
/// Uses greedy FVS with SCC-recomputation after each removal (matches the
/// standard textbook heuristic). After removing a node, SCCs are recomputed
/// so only still-cyclic nodes are candidates for further removal.
///
/// Returns (n_consistent, n_total).
///
/// Reference: Houtman & Maks (1985). "Determining all maximal data subsets
/// consistent with revealed preference."
pub fn houtman_maks(graph: &mut PreferenceGraph) -> (usize, usize) {
    graph.ensure_closure();
    let t = graph.t;

    if t == 0 {
        return (0, 0);
    }

    // Check if there are any GARP violations
    let mut has_violation = false;
    for i in 0..t {
        for j in 0..t {
            if graph.r_star[i * t + j] && graph.p[j * t + i] {
                has_violation = true;
                break;
            }
        }
        if has_violation {
            break;
        }
    }

    if !has_violation {
        return (t, t);
    }

    // Greedy FVS with SCC recomputation
    houtman_maks_greedy(graph)
}

/// Greedy FVS: repeatedly remove the highest-degree node in the largest
/// non-trivial SCC, recomputing SCCs after each removal.
///
/// This matches Python's `greedy_feedback_vertex_set` which recomputes
/// SCCs after each removal step.
fn houtman_maks_greedy(graph: &PreferenceGraph) -> (usize, usize) {
    let t = graph.t;

    // Work on a mutable copy of the R adjacency
    let mut adj = vec![false; t * t];
    adj[..t * t].copy_from_slice(&graph.r[..t * t]);

    let mut active = vec![true; t];
    let mut removed_count = 0usize;
    let mut scc_labels = vec![0u32; t];

    loop {
        // Collect active node indices
        let active_nodes: Vec<usize> = (0..t).filter(|&i| active[i]).collect();
        let n_active = active_nodes.len();
        if n_active < 2 {
            break;
        }

        // Build sub-adjacency for active nodes
        let mut sub_adj = vec![false; n_active * n_active];
        for (li, &ni) in active_nodes.iter().enumerate() {
            for (lj, &nj) in active_nodes.iter().enumerate() {
                sub_adj[li * n_active + lj] = adj[ni * t + nj];
            }
        }

        // Find SCCs on the active subgraph
        let mut sub_labels = vec![0u32; n_active];
        let n_comp = tarjan_scc(&sub_adj, n_active, &mut sub_labels);

        // Find non-trivial SCCs (size > 1)
        let mut scc_sizes = vec![0usize; n_comp];
        for &l in &sub_labels {
            scc_sizes[l as usize] += 1;
        }

        let has_nontrivial = scc_sizes.iter().any(|&s| s > 1);
        if !has_nontrivial {
            break;
        }

        // Find the node with highest (in_deg + out_deg) within its non-trivial SCC
        let mut best_local = usize::MAX;
        let mut best_score = 0usize;

        for comp in 0..n_comp {
            if scc_sizes[comp] <= 1 {
                continue;
            }

            // Get nodes in this SCC
            let scc_local: Vec<usize> = (0..n_active)
                .filter(|&i| sub_labels[i] == comp as u32)
                .collect();

            // Build SCC sub-adjacency
            let scc_n = scc_local.len();
            let mut scc_adj = vec![false; scc_n * scc_n];
            for (si, &li) in scc_local.iter().enumerate() {
                for (sj, &lj) in scc_local.iter().enumerate() {
                    scc_adj[si * scc_n + sj] = sub_adj[li * n_active + lj];
                }
            }

            // Score each node: out_degree + in_degree within this SCC
            for si in 0..scc_n {
                let out_deg: usize = (0..scc_n).filter(|&sj| scc_adj[si * scc_n + sj]).count();
                let in_deg: usize = (0..scc_n).filter(|&sj| scc_adj[sj * scc_n + si]).count();
                let score = out_deg + in_deg;
                if score > best_score {
                    best_score = score;
                    best_local = active_nodes[scc_local[si]]; // map back to global index
                }
            }
        }

        if best_local == usize::MAX {
            break;
        }

        // Remove the best node
        active[best_local] = false;
        // Zero out edges in adjacency
        for j in 0..t {
            adj[best_local * t + j] = false;
            adj[j * t + best_local] = false;
        }
        removed_count += 1;
    }

    (t - removed_count, t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consistent_data_full_hm() {
        let prices = [1.0, 2.0, 2.0, 1.0];
        let quantities = [4.0, 1.0, 1.0, 4.0];
        let mut graph = PreferenceGraph::new(2);
        graph.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        let (consistent, total) = houtman_maks(&mut graph);
        assert_eq!(consistent, 2);
        assert_eq!(total, 2);
    }

    #[test]
    fn test_violation_removes_one() {
        let prices = [2.0, 1.0, 1.0, 2.0];
        let quantities = [3.0, 2.0, 2.0, 3.0];
        let mut graph = PreferenceGraph::new(2);
        graph.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        let (consistent, total) = houtman_maks(&mut graph);
        assert_eq!(consistent, 1);
        assert_eq!(total, 2);
    }

    #[test]
    fn test_menu_houtman_maks() {
        let menus = vec![vec![0, 1], vec![1, 2], vec![0, 2]];
        let choices = [0, 1, 2];
        let mut graph = PreferenceGraph::new(3);
        graph.parse_menu(&menus, &choices, 3);
        let (consistent, total) = houtman_maks(&mut graph);
        assert!(consistent < total);
        assert_eq!(total, 3);
    }
}
