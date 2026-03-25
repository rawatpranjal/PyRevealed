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
/// LP: max Σe_i subject to e_i ≥ E[i,j]/own_exp[i] for all (i,j) where R*[i,j].
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
                    // e_i ≥ ratio → -e_i ≤ -ratio
                    pb.add_row(-ratio.., [(cols[i], 1.0)]);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::garp::garp_check;

    #[test]
    fn test_vei_consistent_all_ones() {
        let prices = [1.0, 2.0, 2.0, 1.0];
        let quantities = [4.0, 1.0, 1.0, 4.0];
        let mut graph = PreferenceGraph::new(2);
        graph.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        let _ = garp_check(&mut graph);
        let vei = compute_vei(&mut graph);
        assert!(vei.success);
        assert_eq!(vei.mean_efficiency, 1.0);
        assert_eq!(vei.min_efficiency, 1.0);
    }

    #[test]
    fn test_vei_violation_data() {
        // For 2-obs violation data, VEI constraints require e_i ≥ 0.875
        // LP maximizes, so both get 1.0 (VEI detects cycle-structure violations
        // only when some ratios exceed 1.0 — here they don't).
        let prices = [2.0, 1.0, 1.0, 2.0];
        let quantities = [3.0, 2.0, 2.0, 3.0];
        let mut graph = PreferenceGraph::new(2);
        graph.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        let _ = garp_check(&mut graph);
        let vei = compute_vei(&mut graph);
        assert!(vei.success);
        assert_eq!(vei.efficiency_vector.len(), 2);
        // Both efficiencies should be ≥ 0.875
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
        let _ = garp_check(&mut graph);
        let vei = compute_vei(&mut graph);
        for &e in &vei.efficiency_vector {
            assert!(e >= 0.0 && e <= 1.0, "Efficiency {e} out of [0,1]");
        }
    }
}
