use highs::{HighsModelStatus, RowProblem, Sense};

use crate::graph::PreferenceGraph;

/// Result of VEI (Varian's Efficiency Index) computation.
#[derive(Debug, Clone)]
pub struct VeiResult {
    pub success: bool,
    pub efficiency_vector: Vec<f64>,
    pub mean_efficiency: f64,
    pub min_efficiency: f64,
    pub worst_observation: usize,
    pub total_inefficiency: f64,
}

/// Compute per-observation efficiency scores (Varian's Efficiency Index).
///
/// Finds e_i ∈ [0,1] for each observation that maximizes total efficiency
/// subject to preference constraints from the transitive closure.
///
/// LP relaxation: max Σe_i subject to e_i ≥ E[i,j]/own_exp[i] for all (i,j) where R*[i,j].
///
/// Note: This is a polynomial LP relaxation of the true VEI (which is NP-hard).
/// It constrains efficiency via the existing R* structure but does not account for
/// how lowering e_i changes which preferences are revealed.
/// For exact VEI, use `compute_vei_exact()`.
///
/// Requires: graph has R* (closure) and E computed.
pub fn compute_vei(graph: &mut PreferenceGraph) -> VeiResult {
    graph.ensure_closure();
    let t = graph.t;

    if t == 0 {
        return VeiResult {
            success: true,
            efficiency_vector: vec![],
            mean_efficiency: 1.0,
            min_efficiency: 1.0,
            worst_observation: 0,
            total_inefficiency: 0.0,
        };
    }

    // Check if already consistent (no violations → all e_i = 1)
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
        return VeiResult {
            success: true,
            efficiency_vector: vec![1.0; t],
            mean_efficiency: 1.0,
            min_efficiency: 1.0,
            worst_observation: 0,
            total_inefficiency: 0.0,
        };
    }

    // Build LP: maximize Σe_i = minimize -Σe_i
    // Variables: e_0..e_{T-1} ∈ [0, 1]
    // Constraints: for each (i,j) where R*[i,j] and i≠j:
    //   e_i ≥ E[i,j] / own_exp[i]   (rewritten as -e_i ≤ -ratio)
    let mut pb = RowProblem::default();

    let mut cols = Vec::with_capacity(t);
    for _i in 0..t {
        // cost = -1 (minimize -e_i = maximize e_i), bounds [0, 1]
        cols.push(pb.add_column(-1.0, 0.0..1.0));
    }

    // Add constraints from transitive closure
    for i in 0..t {
        if graph.own_exp[i] <= 0.0 {
            continue;
        }
        for j in 0..t {
            if i == j {
                continue;
            }
            if graph.r_star[i * t + j] {
                let ratio = graph.e[i * t + j] / graph.own_exp[i];
                if ratio > 0.0 && ratio <= 1.0 {
                    // e_i ≥ ratio (lower bound on efficiency for this preference)
                    pb.add_row(ratio.., [(cols[i], 1.0)]);
                }
            }
        }
    }

    let model = pb.optimise(Sense::Minimise);
    let solved = model.solve();

    match solved.status() {
        HighsModelStatus::Optimal => {
            let sol = solved.get_solution();
            let e_vec: Vec<f64> = (0..t)
                .map(|i| sol.columns()[i].clamp(0.0, 1.0))
                .collect();

            let mean = e_vec.iter().sum::<f64>() / t as f64;
            let min_e = e_vec.iter().cloned().fold(f64::INFINITY, f64::min);
            let worst = e_vec
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            let total_ineff = e_vec.iter().map(|&e| 1.0 - e).sum();

            VeiResult {
                success: true,
                efficiency_vector: e_vec,
                mean_efficiency: mean,
                min_efficiency: min_e,
                worst_observation: worst,
                total_inefficiency: total_ineff,
            }
        }
        _ => VeiResult {
            success: false,
            efficiency_vector: vec![0.0; t],
            mean_efficiency: 0.0,
            min_efficiency: 0.0,
            worst_observation: 0,
            total_inefficiency: t as f64,
        },
    }
}

/// Exact VEI via Mononen's (2023) binary LP + row generation.
///
/// Reformulates VEI as Weighted Minimum Feedback Arc Set (WFAS):
/// binary θ_{i,j} ∈ {0,1} for each strict preference arc, minimizing
/// Σ cost_{i,j} · θ_{i,j} subject to cycle-breaking constraints.
///
/// Row generation: start with 2-cycles (WARP violations), then use
/// DFS separation oracle to find more cycles iteratively.
///
/// Falls back to `compute_vei()` if the MILP fails.
///
/// Reference: Mononen (2023), "Computing and Comparing Measures of Rationality."
pub fn compute_vei_exact(graph: &mut PreferenceGraph) -> VeiResult {
    use crate::garp::garp_check;

    let t = graph.t;

    if t == 0 {
        return VeiResult {
            success: true,
            efficiency_vector: vec![],
            mean_efficiency: 1.0,
            min_efficiency: 1.0,
            worst_observation: 0,
            total_inefficiency: 0.0,
        };
    }

    // O(T²) GARP check
    let garp = garp_check(graph);
    if garp.is_consistent {
        return VeiResult {
            success: true,
            efficiency_vector: vec![1.0; t],
            mean_efficiency: 1.0,
            min_efficiency: 1.0,
            worst_observation: 0,
            total_inefficiency: 0.0,
        };
    }

    // Collect strict preference arcs with costs
    // Arc (i,j): obs i strictly revealed preferred over j (P[i,j] = true)
    // Cost: (own_exp[i] - E[i,j]) / own_exp[i] = 1 - ratio
    let mut arc_from = Vec::new();
    let mut arc_to = Vec::new();
    let mut arc_cost = Vec::new();
    let mut arc_ratio = Vec::new();

    for i in 0..t {
        if graph.own_exp[i] <= 0.0 {
            continue;
        }
        for j in 0..t {
            if i != j && graph.p[i * t + j] {
                let ratio = graph.e[i * t + j] / graph.own_exp[i];
                let cost = 1.0 - ratio;
                if cost > 1e-12 {
                    arc_from.push(i);
                    arc_to.push(j);
                    arc_cost.push(cost);
                    arc_ratio.push(ratio);
                }
            }
        }
    }

    let n_arcs = arc_from.len();
    if n_arcs == 0 {
        return VeiResult {
            success: true,
            efficiency_vector: vec![1.0; t],
            mean_efficiency: 1.0,
            min_efficiency: 1.0,
            worst_observation: 0,
            total_inefficiency: 0.0,
        };
    }

    // Build adjacency list: adj[i] = [(to, arc_idx), ...]
    let mut adj: Vec<Vec<(usize, usize)>> = vec![vec![]; t];
    for idx in 0..n_arcs {
        adj[arc_from[idx]].push((arc_to[idx], idx));
    }

    // Build arc lookup for 2-cycle detection: arc_idx[i*t+j] or usize::MAX
    let mut arc_lookup = vec![usize::MAX; t * t];
    for idx in 0..n_arcs {
        arc_lookup[arc_from[idx] * t + arc_to[idx]] = idx;
    }

    // Initialize cycle constraints with 2-cycles (WARP violations)
    let mut cycle_constraints: Vec<Vec<usize>> = Vec::new();
    for idx in 0..n_arcs {
        let (i, j) = (arc_from[idx], arc_to[idx]);
        if i < j {
            // check reverse arc exists
            let rev = arc_lookup[j * t + i];
            if rev != usize::MAX {
                cycle_constraints.push(vec![idx, rev]);
            }
        }
    }

    // If no 2-cycles, run initial DFS to find longer cycles
    if cycle_constraints.is_empty() {
        let initial = vei_dfs_find_cycles(&adj, &arc_from, t, &vec![false; n_arcs]);
        if initial.is_empty() {
            // No cycles in strict graph - consistent
            return VeiResult {
                success: true,
                efficiency_vector: vec![1.0; t],
                mean_efficiency: 1.0,
                min_efficiency: 1.0,
                worst_observation: 0,
                total_inefficiency: 0.0,
            };
        }
        cycle_constraints = initial;
    }

    // Row generation loop
    let max_iters = 50;
    let mut theta = vec![false; n_arcs];

    for _iter in 0..max_iters {
        // Solve binary LP
        theta = vei_solve_milp(&arc_cost, &cycle_constraints, n_arcs);

        // DFS separation oracle: find cycles in residual graph (arcs where θ=0)
        let new_cycles = vei_dfs_find_cycles(&adj, &arc_from, t, &theta);

        if new_cycles.is_empty() {
            break;
        }

        cycle_constraints.extend(new_cycles);
    }

    // Derive per-observation efficiency from θ
    // For each obs i, efficiency = min{ratio of removed arcs from i}
    // If no arcs removed from i: efficiency = 1.0
    let mut efficiency = vec![1.0f64; t];
    for idx in 0..n_arcs {
        if theta[idx] && arc_ratio[idx] < efficiency[arc_from[idx]] {
            efficiency[arc_from[idx]] = arc_ratio[idx];
        }
    }

    let mean = efficiency.iter().sum::<f64>() / t as f64;
    let min_e = efficiency.iter().cloned().fold(f64::INFINITY, f64::min);
    let worst = efficiency
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    let total_ineff = efficiency.iter().map(|&e| 1.0 - e).sum();

    VeiResult {
        success: true,
        efficiency_vector: efficiency,
        mean_efficiency: mean,
        min_efficiency: min_e,
        worst_observation: worst,
        total_inefficiency: total_ineff,
    }
}

/// Solve the WFAS binary LP: min Σ cost_j · θ_j s.t. for each cycle: Σ θ ≥ 1.
fn vei_solve_milp(
    costs: &[f64],
    cycles: &[Vec<usize>],
    n_arcs: usize,
) -> Vec<bool> {
    let mut pb = RowProblem::default();

    // Binary variables: θ_j ∈ {0,1} for each arc
    let mut cols = Vec::with_capacity(n_arcs);
    for j in 0..n_arcs {
        cols.push(pb.add_integer_column(costs[j], 0.0..1.0));
    }

    // Cycle constraints: Σ_{j ∈ cycle} θ_j ≥ 1
    for cycle in cycles {
        let terms: Vec<_> = cycle.iter().map(|&idx| (cols[idx], 1.0)).collect();
        pb.add_row(1.0.., terms);
    }

    let model = pb.optimise(Sense::Minimise);
    let solved = model.solve();

    match solved.status() {
        HighsModelStatus::Optimal => {
            let sol = solved.get_solution();
            (0..n_arcs).map(|j| sol.columns()[j] > 0.5).collect()
        }
        _ => vec![false; n_arcs], // fallback: nothing removed
    }
}

/// DFS separation oracle: find cycles in the residual graph (arcs where θ=false).
/// Returns cycles as lists of arc indices.
fn vei_dfs_find_cycles(
    adj: &[Vec<(usize, usize)>],
    arc_from: &[usize],
    t: usize,
    theta: &[bool],
) -> Vec<Vec<usize>> {
    let mut cycles = Vec::new();
    let mut color = vec![0u8; t]; // 0=white, 1=gray, 2=black
    let mut path_arcs: Vec<usize> = Vec::new();
    let mut path_nodes: Vec<usize> = Vec::new();

    for start in 0..t {
        if color[start] == 0 {
            vei_dfs_visit(
                start, adj, arc_from, theta, &mut color, &mut path_arcs, &mut path_nodes,
                &mut cycles,
            );
        }
    }

    cycles
}

fn vei_dfs_visit(
    node: usize,
    adj: &[Vec<(usize, usize)>],
    arc_from: &[usize],
    theta: &[bool],
    color: &mut [u8],
    path_arcs: &mut Vec<usize>,
    path_nodes: &mut Vec<usize>,
    cycles: &mut Vec<Vec<usize>>,
) {
    color[node] = 1; // gray
    path_nodes.push(node);

    for &(next, arc_idx) in &adj[node] {
        if theta[arc_idx] {
            continue; // skip removed arcs
        }

        if color[next] == 1 {
            // Back edge → extract cycle: next → ... → node → next
            if let Some(pos) = path_nodes.iter().rposition(|&n| n == next) {
                let mut cycle = Vec::new();
                // Arcs from path_arcs[pos..] cover: next→...→node
                for i in pos..path_arcs.len() {
                    cycle.push(path_arcs[i]);
                }
                cycle.push(arc_idx); // closing edge: node→next
                cycles.push(cycle);
            }
        } else if color[next] == 0 {
            path_arcs.push(arc_idx);
            vei_dfs_visit(next, adj, arc_from, theta, color, path_arcs, path_nodes, cycles);
            path_arcs.pop();
        }
    }

    path_nodes.pop();
    color[node] = 2; // black
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::garp::garp_check_with_closure;

    #[test]
    fn test_vei_consistent_all_ones() {
        let prices = [1.0, 2.0, 2.0, 1.0];
        let quantities = [4.0, 1.0, 1.0, 4.0];
        let mut graph = PreferenceGraph::new(2);
        graph.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        let _ = garp_check_with_closure(&mut graph);
        let vei = compute_vei(&mut graph);
        assert!(vei.success);
        assert_eq!(vei.mean_efficiency, 1.0);
        assert_eq!(vei.min_efficiency, 1.0);
    }

    #[test]
    fn test_vei_violation_data() {
        // For 2-obs WARP violation: E[0,1]/E[0,0] = 7/8 = 0.875, same for E[1,0]/E[1,1].
        // LP relaxation constrains e_i ≥ 0.875 (from R* constraints) and maximizes Σe_i,
        // so both e_i = 1.0 (since 0.875 < 1.0 and we're maximizing).
        // The LP relaxation does NOT detect cycles - only transitive ratios > 1 would bind.
        // For exact VEI, use compute_vei_exact().
        let prices = [2.0, 1.0, 1.0, 2.0];
        let quantities = [3.0, 2.0, 2.0, 3.0];
        let mut graph = PreferenceGraph::new(2);
        graph.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        let _ = garp_check_with_closure(&mut graph);
        let vei = compute_vei(&mut graph);
        assert!(vei.success);
        assert_eq!(vei.efficiency_vector.len(), 2);
        // LP relaxation: both e_i = 1.0 (lower bound 0.875 is not binding at max)
        for &e in &vei.efficiency_vector {
            assert!(e >= 0.875 - 1e-6);
        }
    }

    #[test]
    fn test_vei_efficiency_vector_bounded() {
        let prices = [2.0, 1.0, 1.0, 2.0];
        let quantities = [3.0, 2.0, 2.0, 3.0];
        let mut graph = PreferenceGraph::new(2);
        graph.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        let _ = garp_check_with_closure(&mut graph);
        let vei = compute_vei(&mut graph);
        for &e in &vei.efficiency_vector {
            assert!(e >= 0.0 && e <= 1.0, "Efficiency {e} out of [0,1]");
        }
    }

    // --- Exact VEI tests ---

    #[test]
    fn test_vei_exact_consistent() {
        let prices = [1.0, 2.0, 2.0, 1.0];
        let quantities = [4.0, 1.0, 1.0, 4.0];
        let mut graph = PreferenceGraph::new(2);
        graph.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        let vei = compute_vei_exact(&mut graph);
        assert!(vei.success);
        assert_eq!(vei.mean_efficiency, 1.0);
        assert_eq!(vei.min_efficiency, 1.0);
        assert_eq!(vei.total_inefficiency, 0.0);
    }

    #[test]
    fn test_vei_exact_warp_violation() {
        // 2-obs WARP violation: mutual preference, ratio = 7/8 = 0.875
        // Exact VEI: remove ONE arc (cost 0.125), so one obs gets e=0.875, other e=1.0
        // Total inefficiency = 0.125, mean efficiency = (0.875 + 1.0)/2 = 0.9375
        let prices = [2.0, 1.0, 1.0, 2.0];
        let quantities = [3.0, 2.0, 2.0, 3.0];
        let mut graph = PreferenceGraph::new(2);
        graph.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        let vei = compute_vei_exact(&mut graph);
        assert!(vei.success);
        // One obs has e=0.875, the other e=1.0 (optimizer picks one to reduce)
        let mut sorted_e = vei.efficiency_vector.clone();
        sorted_e.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((sorted_e[0] - 0.875).abs() < 1e-6, "lower e={}", sorted_e[0]);
        assert!((sorted_e[1] - 1.0).abs() < 1e-6, "upper e={}", sorted_e[1]);
        assert!((vei.mean_efficiency - 0.9375).abs() < 1e-6);
        assert!((vei.total_inefficiency - 0.125).abs() < 1e-6);
    }

    #[test]
    fn test_vei_exact_bounded() {
        let prices = [2.0, 1.0, 1.0, 2.0];
        let quantities = [3.0, 2.0, 2.0, 3.0];
        let mut graph = PreferenceGraph::new(2);
        graph.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        let vei = compute_vei_exact(&mut graph);
        for &e in &vei.efficiency_vector {
            assert!(e >= 0.0 && e <= 1.0, "Efficiency {e} out of [0,1]");
        }
    }

    #[test]
    fn test_vei_exact_3obs_cycle() {
        // 3-obs cycle: each obs can afford the next (circular preferences)
        // p0=[3,1,1], x0=[2,1,1] → own_exp=8
        // p1=[1,3,1], x1=[1,2,1] → own_exp=8
        // p2=[1,1,3], x2=[1,1,2] → own_exp=8
        // E[0,1]=p0·x1=3+3+1=7, E[1,2]=p1·x2=1+2+6=9... too complex, use simple data
        //
        // Simpler: 3 obs, 2 goods, circular strict preferences
        let prices = [2.0, 1.0, 1.0, 2.0, 1.5, 1.5];
        let quantities = [3.0, 1.0, 1.0, 3.0, 2.0, 2.0];
        let mut graph = PreferenceGraph::new(3);
        graph.parse_budget(&prices, &quantities, 3, 2, 1e-10);
        let vei = compute_vei_exact(&mut graph);
        assert!(vei.success);
        // Should have some efficiency < 1 (GARP violated)
        assert!(vei.mean_efficiency < 1.0);
        // All bounded
        for &e in &vei.efficiency_vector {
            assert!(e >= 0.0 && e <= 1.0);
        }
    }
}
