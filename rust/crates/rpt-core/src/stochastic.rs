use highs::{HighsModelStatus, RowProblem, Sense};

/// Result of a RUM (Random Utility Model) consistency test.
#[derive(Debug, Clone)]
pub struct RumResult {
    pub is_consistent: bool,
    pub n_items: usize,
    pub n_orderings: usize,  // K! — number of possible orderings
}

/// Result of a regularity test.
#[derive(Debug, Clone)]
pub struct RegularityResult {
    pub is_regular: bool,
    pub n_violations: u32,
}

/// Test RUM (Random Utility Model) consistency via LP.
///
/// Given choice frequencies over menus, test whether there exists a
/// probability distribution over all K! preference orderings that
/// generates the observed frequencies. This is the nonparametric test —
/// if it fails, NO discrete choice model (logit, probit, mixed logit)
/// can rationalize the data.
///
/// The LP:
///   Variables: w_π >= 0 for each ordering π (one of K! permutations)
///   Constraints: for each (menu B, item x):
///     sum_{π : x is top-ranked in B under π} w_π = freq(x | B)
///   Also: sum_π w_π = 1 (probability distribution)
///
/// Tractable for K <= 6 (720 orderings). Returns infeasible for K > 6.
///
/// Args:
///   menus: list of menus, each a sorted vec of item indices
///   frequencies: for each menu, a vec of (item, frequency) pairs
///   n_items: total number of distinct items
pub fn rum_consistency_check(
    menus: &[Vec<usize>],
    frequencies: &[Vec<(usize, f64)>],
    n_items: usize,
) -> RumResult {
    if n_items > 6 {
        return RumResult {
            is_consistent: false, // Can't test — too many orderings
            n_items,
            n_orderings: 0,
        };
    }

    // Generate all K! orderings
    let orderings = generate_permutations(n_items);
    let n_ord = orderings.len();

    let mut pb = RowProblem::default();

    // Variables: w_0 ... w_{K!-1}, each in [0, 1]
    let cols: Vec<_> = (0..n_ord).map(|_| pb.add_column(0.0, 0.0..1.0)).collect();

    // Constraint: sum(w_π) = 1
    let unity: Vec<_> = cols.iter().map(|&c| (c, 1.0)).collect();
    pb.add_row(1.0..=1.0, unity);

    // For each (menu, item): sum of w_π where π ranks item as best in menu = freq
    for (menu_idx, menu) in menus.iter().enumerate() {
        let freq_map: std::collections::HashMap<usize, f64> = frequencies[menu_idx]
            .iter()
            .cloned()
            .collect();

        // Normalize frequencies to probabilities
        let total: f64 = freq_map.values().sum();
        if total <= 0.0 {
            continue;
        }

        for &item in menu {
            let target_prob = freq_map.get(&item).copied().unwrap_or(0.0) / total;

            // Find orderings where `item` is the top-ranked item in this menu
            let mut coeffs: Vec<(highs::Col, f64)> = Vec::new();
            for (ord_idx, ordering) in orderings.iter().enumerate() {
                // In ordering `ordering`, which item in `menu` is ranked highest?
                let top = top_in_menu(ordering, menu);
                if top == item {
                    coeffs.push((cols[ord_idx], 1.0));
                }
            }

            if !coeffs.is_empty() || target_prob > 1e-10 {
                pb.add_row(target_prob..=target_prob, coeffs);
            }
        }
    }

    let mut model = pb.optimise(Sense::Minimise);
    let solved = model.solve();

    let is_consistent = matches!(solved.status(), HighsModelStatus::Optimal);

    RumResult {
        is_consistent,
        n_items,
        n_orderings: n_ord,
    }
}

/// Test regularity: does P(x|B) >= P(x|B') when B ⊆ B'?
///
/// Regularity is a necessary condition for RUM. If adding items to a menu
/// increases the choice probability of an existing item, the data violates
/// regularity (and therefore cannot be a RUM).
///
/// No LP needed — just pairwise comparison of frequencies across nested menus.
pub fn regularity_check(
    menus: &[Vec<usize>],
    frequencies: &[Vec<(usize, f64)>],
) -> RegularityResult {
    let n_menus = menus.len();
    let mut n_violations = 0u32;

    for i in 0..n_menus {
        for j in 0..n_menus {
            if i == j {
                continue;
            }
            // Check if menu_i ⊆ menu_j (i is a subset of j)
            let is_subset = menus[i].iter().all(|item| menus[j].contains(item));
            if !is_subset {
                continue;
            }

            // For each item in menu_i: P(item | menu_i) should >= P(item | menu_j)
            let total_i: f64 = frequencies[i].iter().map(|(_, f)| f).sum();
            let total_j: f64 = frequencies[j].iter().map(|(_, f)| f).sum();
            if total_i <= 0.0 || total_j <= 0.0 {
                continue;
            }

            for &(item, freq_i) in &frequencies[i] {
                let prob_i = freq_i / total_i;
                let freq_j = frequencies[j]
                    .iter()
                    .find(|(it, _)| *it == item)
                    .map(|(_, f)| *f)
                    .unwrap_or(0.0);
                let prob_j = freq_j / total_j;

                if prob_j > prob_i + 1e-10 {
                    n_violations += 1;
                }
            }
        }
    }

    RegularityResult {
        is_regular: n_violations == 0,
        n_violations,
    }
}

/// Generate all permutations of 0..n (Heap's algorithm).
fn generate_permutations(n: usize) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    let mut arr: Vec<usize> = (0..n).collect();
    let mut c = vec![0usize; n];

    result.push(arr.clone());

    let mut i = 1;
    while i < n {
        if c[i] < i {
            if i % 2 == 0 {
                arr.swap(0, i);
            } else {
                arr.swap(c[i], i);
            }
            result.push(arr.clone());
            c[i] += 1;
            i = 1;
        } else {
            c[i] = 0;
            i += 1;
        }
    }

    result
}

/// Find which item in `menu` is ranked highest by `ordering`.
/// ordering[0] is most preferred, ordering[K-1] is least.
fn top_in_menu(ordering: &[usize], menu: &[usize]) -> usize {
    for &item in ordering {
        if menu.contains(&item) {
            return item;
        }
    }
    menu[0] // fallback
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rum_consistent() {
        // 3 items, 1 menu {0,1,2}, frequencies consistent with ordering 0>1>2
        let menus = vec![vec![0, 1, 2]];
        let freqs = vec![vec![(0, 60.0), (1, 30.0), (2, 10.0)]];
        let result = rum_consistency_check(&menus, &freqs, 3);
        assert!(result.is_consistent);
        assert_eq!(result.n_orderings, 6); // 3! = 6
    }

    #[test]
    fn test_regularity_consistent() {
        // P(0|{0,1}) = 0.6, P(0|{0,1,2}) = 0.5 — regularity holds (0.6 >= 0.5)
        let menus = vec![vec![0, 1], vec![0, 1, 2]];
        let freqs = vec![
            vec![(0, 60.0), (1, 40.0)],
            vec![(0, 50.0), (1, 30.0), (2, 20.0)],
        ];
        let result = regularity_check(&menus, &freqs);
        assert!(result.is_regular);
    }

    #[test]
    fn test_regularity_violation() {
        // P(0|{0,1}) = 0.4, P(0|{0,1,2}) = 0.6 — violation! (adding item 2 increased P(0))
        let menus = vec![vec![0, 1], vec![0, 1, 2]];
        let freqs = vec![
            vec![(0, 40.0), (1, 60.0)],
            vec![(0, 60.0), (1, 20.0), (2, 20.0)],
        ];
        let result = regularity_check(&menus, &freqs);
        assert!(!result.is_regular);
    }

    #[test]
    fn test_permutations() {
        assert_eq!(generate_permutations(3).len(), 6);   // 3! = 6
        assert_eq!(generate_permutations(4).len(), 24);  // 4! = 24
    }
}
