use crate::graph::PreferenceGraph;
use crate::types::GarpResult;

/// Result of a WARP-LA (Limited Attention) check.
pub struct AttentionResult {
    pub is_rationalizable: bool,
    pub n_attention_violations: u32,
}

/// Check WARP-LA: Weak Axiom under Limited Attention (Masatlioglu+ 2012).
///
/// A choice function satisfies WARP-LA if there exists an attention filter
/// Γ (mapping each menu to a "consideration set" subset) and a linear order
/// > over items such that:
///   c(B) = max_{>} Γ(B)  for all menus B
///
/// The test: choices are WARP-LA rationalizable iff there are no "attention
/// cycles" - cases where x is chosen over y in one menu, y is chosen over z
/// in another, but z is chosen over x in a third, AND all three were
/// demonstrably in each other's consideration sets.
///
/// Simplified implementation: check if WARP violations can be explained by
/// limited attention (some violations are "excusable" if the rejected item
/// might not have been noticed).
///
/// Requires: graph parsed with parse_menu().
pub fn warp_la_check(graph: &PreferenceGraph) -> AttentionResult {
    let t = graph.t; // n_items

    // A WARP violation (x,y): x chosen over y AND y chosen over x.
    // Under limited attention, this is excusable if in one of the menus,
    // the "winner" might not have been in the consideration set.
    //
    // Simple test: count WARP violations that CANNOT be explained by
    // any attention filter.

    let mut n_warp_violations = 0u32;
    let mut n_inexcusable = 0u32;

    for x in 0..t {
        for y in (x + 1)..t {
            if graph.r[x * t + y] && graph.r[y * t + x] {
                n_warp_violations += 1;

                // Check if this violation is "excusable":
                // If there exists any z such that z > x and z > y (both dominated
                // by z), then the attention filter could exclude one.
                // Simplified: if either x or y is dominated by some third item,
                // the violation might be excusable.
                let mut excusable = false;
                for z in 0..t {
                    if z == x || z == y { continue; }
                    // z dominates both x and y → attention might exclude x or y
                    if graph.r[z * t + x] && graph.r[z * t + y] {
                        excusable = true;
                        break;
                    }
                }

                if !excusable {
                    n_inexcusable += 1;
                }
            }
        }
    }

    AttentionResult {
        is_rationalizable: n_inexcusable == 0,
        n_attention_violations: n_warp_violations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_warp_la_consistent() {
        // No WARP violations → trivially rationalizable with full attention
        let mut graph = PreferenceGraph::new(3);
        let menus = vec![vec![0, 1, 2], vec![1, 2]];
        let choices = vec![0, 1];
        graph.parse_menu(&menus, &choices, 3);
        let result = warp_la_check(&graph);
        assert!(result.is_rationalizable);
    }

    #[test]
    fn test_warp_la_with_violation() {
        // WARP violation: 0 > 1 and 1 > 0
        let mut graph = PreferenceGraph::new(2);
        let menus = vec![vec![0, 1], vec![0, 1]];
        let choices = vec![0, 1];
        graph.parse_menu(&menus, &choices, 2);
        let result = warp_la_check(&graph);
        // With only 2 items, no third item can "excuse" the violation
        assert!(!result.is_rationalizable);
        assert_eq!(result.n_attention_violations, 1);
    }
}
