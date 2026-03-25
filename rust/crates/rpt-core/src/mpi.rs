use crate::graph::PreferenceGraph;

/// Compute Money Pump Index using Karp's max-mean-weight cycle algorithm.
///
/// The MPI measures the maximum per-step exploitation rate across all
/// preference cycles. For a cycle i₁→i₂→…→iₙ→i₁, the per-step pump is:
///   w[iₖ,iₖ₊₁] = (own_exp[iₖ] - E[iₖ,iₖ₊₁]) / own_exp[iₖ]
///
/// MPI = max mean weight over all cycles = max_cycle { mean(w on cycle edges) }
///
/// Uses Karp's O(T³) min-mean-weight-cycle on negated weights to find the
/// max-mean cycle. This matches the theoretical definition in Chambers &
/// Echenique (2016) Chapter 5.
///
/// Requires: graph has R and expenditure computed.
pub fn mpi_karp(graph: &PreferenceGraph) -> f64 {
    let t = graph.t;
    if t < 2 {
        return 0.0;
    }

    let inf = f64::MAX / 2.0;

    // Build NEGATED weight matrix: neg_w[i,j] = -(own_exp[i] - E[i,j]) / own_exp[i]
    // on R edges, +inf elsewhere.
    // Karp's finds min-mean cycle on neg_w, then MPI = -min_mean.
    let mut neg_w = vec![inf; t * t];
    for i in 0..t {
        if graph.own_exp[i] <= 0.0 {
            continue;
        }
        for j in 0..t {
            if i != j && graph.r[i * t + j] {
                let savings = (graph.own_exp[i] - graph.e[i * t + j]) / graph.own_exp[i];
                neg_w[i * t + j] = -savings;
            }
        }
    }

    // Karp's algorithm:
    // d[k][v] = minimum cost of a path of exactly k edges ending at v
    // d[0][v] = 0 for all v
    //
    // min_mean = min_v { max_{k=0..T-1} { (d[T][v] - d[k][v]) / (T - k) } }

    // d is (T+1) × T matrix, stored flat
    let mut d = vec![inf; (t + 1) * t];

    // d[0][v] = 0 for all v
    for v in 0..t {
        d[0 * t + v] = 0.0;
    }

    // Fill d[k][v] for k = 1..T
    for k in 1..=t {
        for v in 0..t {
            let mut best = inf;
            for u in 0..t {
                if neg_w[u * t + v] < inf / 2.0 {
                    let candidate = d[(k - 1) * t + u] + neg_w[u * t + v];
                    if candidate < best {
                        best = candidate;
                    }
                }
            }
            d[k * t + v] = best;
        }
    }

    // Compute min mean weight
    let mut min_mean = inf;
    for v in 0..t {
        if d[t * t + v] >= inf / 2.0 {
            continue; // v is not reachable via T edges
        }
        let mut max_ratio = f64::NEG_INFINITY;
        for k in 0..t {
            if d[k * t + v] < inf / 2.0 {
                let ratio = (d[t * t + v] - d[k * t + v]) / ((t - k) as f64);
                if ratio > max_ratio {
                    max_ratio = ratio;
                }
            }
        }
        if max_ratio < min_mean {
            min_mean = max_ratio;
        }
    }

    if min_mean >= inf / 2.0 {
        return 0.0; // No cycles exist
    }

    // MPI = -min_mean (negate back)
    (-min_mean).max(0.0)
}

/// Fast approximate MPI: max per-edge savings across violation pairs.
///
/// This is NOT the theoretical MPI (which uses cycle means), but a quick
/// upper bound. Use `mpi_karp` for the theory-correct value.
///
/// Requires: graph has closure, P, and E computed (call garp_check first).
pub fn mpi_fast(graph: &PreferenceGraph) -> f64 {
    let t = graph.t;
    let mut max_mpi = 0.0f64;

    for i in 0..t {
        for j in 0..t {
            if i == j {
                continue;
            }
            if graph.r_star[i * t + j] && graph.p[j * t + i] {
                let own_j = graph.e[j * t + j];
                if own_j > 0.0 {
                    let savings = (own_j - graph.e[j * t + i]) / own_j;
                    if savings > max_mpi {
                        max_mpi = savings;
                    }
                }
            }
        }
    }

    max_mpi.max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::garp::garp_check;

    #[test]
    fn test_mpi_karp_zero_when_consistent() {
        let prices = [1.0, 2.0, 2.0, 1.0];
        let quantities = [4.0, 1.0, 1.0, 4.0];
        let mut graph = PreferenceGraph::new(2);
        graph.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        let mpi = mpi_karp(&graph);
        assert_eq!(mpi, 0.0);
    }

    #[test]
    fn test_mpi_karp_positive_when_violation() {
        let prices = [2.0, 1.0, 1.0, 2.0];
        let quantities = [3.0, 2.0, 2.0, 3.0];
        let mut graph = PreferenceGraph::new(2);
        graph.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        let mpi = mpi_karp(&graph);
        assert!(mpi > 0.0, "MPI should be positive for violation data, got {mpi}");
        // For this 2-cycle data, Karp's MPI should be close to the cycle average
        assert!(mpi < 0.5, "MPI should be moderate, got {mpi}");
    }

    #[test]
    fn test_mpi_karp_le_mpi_fast() {
        // Karp's cycle-mean MPI should be ≤ the fast per-edge max
        let prices = [2.0, 1.0, 1.0, 2.0];
        let quantities = [3.0, 2.0, 2.0, 3.0];
        let mut graph = PreferenceGraph::new(2);
        graph.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        let _ = garp_check(&mut graph); // need closure for mpi_fast
        let karp = mpi_karp(&graph);
        let fast = mpi_fast(&graph);
        assert!(
            karp <= fast + 1e-10,
            "Karp MPI ({karp}) should be <= fast MPI ({fast})"
        );
    }

    #[test]
    fn test_mpi_fast_backward_compat() {
        let prices = [2.0, 1.0, 1.0, 2.0];
        let quantities = [3.0, 2.0, 2.0, 3.0];
        let mut graph = PreferenceGraph::new(2);
        graph.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        let garp = garp_check(&mut graph);
        assert!(!garp.is_consistent);
        let mpi = mpi_fast(&graph);
        assert!(mpi > 0.0);
    }
}
