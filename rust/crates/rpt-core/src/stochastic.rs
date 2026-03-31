use highs::{HighsModelStatus, RowProblem, Sense};
use rand::seq::SliceRandom;
use rand::thread_rng;

/// Result of a RUM (Random Utility Model) consistency test.
#[derive(Debug, Clone)]
pub struct RumResult {
    pub is_consistent: bool,
    pub n_items: usize,
    pub n_orderings: usize,
    pub distance: f64,       // L1 distance to nearest RUM (0 if consistent)
    pub n_iterations: u32,   // Column generation iterations used
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
/// probability distribution over preference orderings that generates
/// the observed frequencies. This is the nonparametric test: if it
/// fails, NO discrete choice model can rationalize the data.
///
/// The LP:
///   Variables: w_pi >= 0 for each ordering pi
///   Constraints: for each (menu B, item x):
///     sum_{pi : x is top-ranked in B under pi} w_pi = freq(x | B)
///   Also: sum_pi w_pi = 1 (probability distribution)
///
/// Uses exact enumeration for K <= 6 (720 orderings) and column
/// generation for K > 6 to avoid factorial blowup.
///
/// Reference: Block & Marschak (1960); Kitamura & Stoye (2018).
pub fn rum_consistency_check(
    menus: &[Vec<usize>],
    frequencies: &[Vec<(usize, f64)>],
    n_items: usize,
) -> RumResult {
    if n_items <= 6 {
        rum_exact(menus, frequencies, n_items)
    } else {
        rum_column_generation(menus, frequencies, n_items, 100, 1e-6)
    }
}

/// Exact RUM test: enumerate all K! orderings and solve LP.
fn rum_exact(
    menus: &[Vec<usize>],
    frequencies: &[Vec<(usize, f64)>],
    n_items: usize,
) -> RumResult {
    let orderings = generate_permutations(n_items);
    let result = solve_rum_lp(menus, frequencies, &orderings);

    RumResult {
        is_consistent: result.0,
        n_items,
        n_orderings: orderings.len(),
        distance: result.1,
        n_iterations: 1,
    }
}

/// Column generation RUM test for K > 6.
///
/// Starts with a small set of initial orderings, solves the restricted LP,
/// and iteratively adds new orderings (columns) until the LP is feasible
/// or no improving column can be found.
///
/// Ported from Python `_test_rum_column_generation` (stochastic.py:846-952).
fn rum_column_generation(
    menus: &[Vec<usize>],
    frequencies: &[Vec<(usize, f64)>],
    n_items: usize,
    max_iterations: u32,
    _tolerance: f64,
) -> RumResult {
    let mut active_orderings = generate_initial_orderings(frequencies, n_items);
    let mut rng = thread_rng();

    for iteration in 0..max_iterations {
        let (feasible, distance) = solve_rum_lp(menus, frequencies, &active_orderings);

        if feasible {
            return RumResult {
                is_consistent: true,
                n_items,
                n_orderings: active_orderings.len(),
                distance: 0.0,
                n_iterations: iteration + 1,
            };
        }

        // Pricing: generate random candidate orderings.
        // Same heuristic as Python (stochastic.py:986-997).
        let mut items: Vec<usize> = (0..n_items).collect();
        let mut found_new = false;
        for _ in 0..10 {
            items.shuffle(&mut rng);
            let candidate = items.clone();
            if !active_orderings.contains(&candidate) {
                active_orderings.push(candidate);
                found_new = true;
                break;
            }
        }

        if !found_new {
            break;
        }
    }

    // Column generation did not find a feasible solution
    let (_, distance) = solve_rum_lp_with_slack(menus, frequencies, &active_orderings);
    RumResult {
        is_consistent: false,
        n_items,
        n_orderings: 0,
        distance,
        n_iterations: max_iterations,
    }
}

/// Solve the RUM LP: check if observed frequencies can be decomposed
/// as a mixture of the given orderings.
///
/// Returns (is_feasible, 0.0) if feasible, (false, 1.0) if not.
fn solve_rum_lp(
    menus: &[Vec<usize>],
    frequencies: &[Vec<(usize, f64)>],
    orderings: &[Vec<usize>],
) -> (bool, f64) {
    let n_ord = orderings.len();
    if n_ord == 0 {
        return (false, 1.0);
    }

    let mut pb = RowProblem::default();

    // Variables: w_0 ... w_{n_ord-1}, each in [0, 1]
    let cols: Vec<_> = (0..n_ord).map(|_| pb.add_column(0.0, 0.0..1.0)).collect();

    // Constraint: sum(w) = 1
    let unity: Vec<_> = cols.iter().map(|&c| (c, 1.0)).collect();
    pb.add_row(1.0..=1.0, unity);

    // For each (menu, item): sum of w where ordering ranks item best = prob
    for (menu_idx, menu) in menus.iter().enumerate() {
        let freq_map: std::collections::HashMap<usize, f64> = frequencies[menu_idx]
            .iter()
            .cloned()
            .collect();
        let total: f64 = freq_map.values().sum();
        if total <= 0.0 {
            continue;
        }

        for &item in menu {
            let target_prob = freq_map.get(&item).copied().unwrap_or(0.0) / total;

            let mut coeffs: Vec<(highs::Col, f64)> = Vec::new();
            for (ord_idx, ordering) in orderings.iter().enumerate() {
                if top_in_menu(ordering, menu) == item {
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

    let is_feasible = matches!(solved.status(), HighsModelStatus::Optimal);
    (is_feasible, if is_feasible { 0.0 } else { 1.0 })
}

/// Solve RUM LP with slack variables to compute L1 distance to nearest RUM.
///
/// Ported from Python `_compute_rum_distance` (stochastic.py:754-843).
fn solve_rum_lp_with_slack(
    menus: &[Vec<usize>],
    frequencies: &[Vec<(usize, f64)>],
    orderings: &[Vec<usize>],
) -> (bool, f64) {
    let n_ord = orderings.len();
    if n_ord == 0 {
        return (false, 1.0);
    }

    // Count constraints (one per menu-item pair)
    let mut n_constraints = 0usize;
    let mut targets: Vec<f64> = Vec::new();
    let mut constraint_menus: Vec<(usize, usize)> = Vec::new(); // (menu_idx, item)

    for (menu_idx, menu) in menus.iter().enumerate() {
        let freq_map: std::collections::HashMap<usize, f64> = frequencies[menu_idx]
            .iter()
            .cloned()
            .collect();
        let total: f64 = freq_map.values().sum();
        if total <= 0.0 {
            continue;
        }
        for &item in menu {
            targets.push(freq_map.get(&item).copied().unwrap_or(0.0) / total);
            constraint_menus.push((menu_idx, item));
            n_constraints += 1;
        }
    }

    let mut pb = RowProblem::default();

    // Variables: w_0..w_{n_ord-1} (ordering weights) + s+_i, s-_i (slack pairs)
    let w_cols: Vec<_> = (0..n_ord).map(|_| pb.add_column(0.0, 0.0..1.0)).collect();
    let sp_cols: Vec<_> = (0..n_constraints).map(|_| pb.add_column(1.0, 0.0..)).collect(); // s+ (cost 1)
    let sm_cols: Vec<_> = (0..n_constraints).map(|_| pb.add_column(1.0, 0.0..)).collect(); // s- (cost 1)

    // Constraint: sum(w) = 1
    let unity: Vec<_> = w_cols.iter().map(|&c| (c, 1.0)).collect();
    pb.add_row(1.0..=1.0, unity);

    // For each (menu, item): sum_w + s+ - s- = target_prob
    for (ci, &(menu_idx, item)) in constraint_menus.iter().enumerate() {
        let target = targets[ci];

        let mut coeffs: Vec<(highs::Col, f64)> = Vec::new();
        for (ord_idx, ordering) in orderings.iter().enumerate() {
            if top_in_menu(ordering, &menus[menu_idx]) == item {
                coeffs.push((w_cols[ord_idx], 1.0));
            }
        }
        coeffs.push((sp_cols[ci], 1.0));
        coeffs.push((sm_cols[ci], -1.0));

        pb.add_row(target..=target, coeffs);
    }

    let mut model = pb.optimise(Sense::Minimise);
    let solved = model.solve();

    if matches!(solved.status(), HighsModelStatus::Optimal) {
        let sol = solved.get_solution();
        let vals = sol.columns();
        // Distance = sum of all slack variables
        let distance: f64 = vals[n_ord..].iter().sum();
        let is_consistent = distance < 1e-6;
        (is_consistent, distance)
    } else {
        (false, 1.0)
    }
}

/// Generate initial orderings for column generation.
///
/// Ported from Python `_generate_initial_orderings` (stochastic.py:955-983).
fn generate_initial_orderings(
    frequencies: &[Vec<(usize, f64)>],
    n_items: usize,
) -> Vec<Vec<usize>> {
    let mut orderings: Vec<Vec<usize>> = Vec::new();

    // Count total choices per item across all menus
    let mut counts = vec![0.0f64; n_items];
    for freq_list in frequencies {
        for &(item, freq) in freq_list {
            if item < n_items {
                counts[item] += freq;
            }
        }
    }

    // Ordering 1: items sorted by frequency (descending)
    let mut items: Vec<usize> = (0..n_items).collect();
    items.sort_by(|&a, &b| counts[b].partial_cmp(&counts[a]).unwrap_or(std::cmp::Ordering::Equal));
    orderings.push(items.clone());

    // Ordering 2: reverse
    let mut rev = items.clone();
    rev.reverse();
    orderings.push(rev);

    // Orderings 3-7: random permutations
    let mut rng = thread_rng();
    for _ in 0..5.min(n_items) {
        let mut perm: Vec<usize> = (0..n_items).collect();
        perm.shuffle(&mut rng);
        if !orderings.contains(&perm) {
            orderings.push(perm);
        }
    }

    orderings
}

/// Test regularity: does P(x|B) >= P(x|B') when B is a subset of B'?
///
/// Regularity is a necessary condition for RUM. If adding items to a menu
/// increases the choice probability of an existing item, the data violates
/// regularity (and therefore cannot be a RUM).
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
            let is_subset = menus[i].iter().all(|item| menus[j].contains(item));
            if !is_subset {
                continue;
            }

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
    fn test_rum_consistent_exact() {
        // 3 items, 1 menu {0,1,2}, frequencies consistent with ordering 0>1>2
        let menus = vec![vec![0, 1, 2]];
        let freqs = vec![vec![(0, 60.0), (1, 30.0), (2, 10.0)]];
        let result = rum_consistency_check(&menus, &freqs, 3);
        assert!(result.is_consistent);
        assert_eq!(result.n_orderings, 6); // 3! = 6
        assert!(result.distance < 1e-6);
    }

    #[test]
    fn test_rum_consistent_two_menus() {
        // 3 items, 2 menus, consistent with a mixture
        let menus = vec![vec![0, 1, 2], vec![0, 1]];
        let freqs = vec![
            vec![(0, 50.0), (1, 30.0), (2, 20.0)],
            vec![(0, 60.0), (1, 40.0)],
        ];
        let result = rum_consistency_check(&menus, &freqs, 3);
        assert!(result.is_consistent);
    }

    #[test]
    fn test_rum_column_generation_small() {
        // 8 items (K > 6, triggers column gen), simple case
        // One menu with 8 items, all frequency on item 0
        let menus = vec![vec![0, 1, 2, 3, 4, 5, 6, 7]];
        let freqs = vec![vec![
            (0, 100.0),
            (1, 0.0),
            (2, 0.0),
            (3, 0.0),
            (4, 0.0),
            (5, 0.0),
            (6, 0.0),
            (7, 0.0),
        ]];
        let result = rum_consistency_check(&menus, &freqs, 8);
        // Should be consistent: any ordering with 0 first works
        assert!(result.is_consistent);
    }

    #[test]
    fn test_regularity_consistent() {
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
        // P(0|{0,1}) = 0.4, P(0|{0,1,2}) = 0.6 - violation
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
        assert_eq!(generate_permutations(3).len(), 6);
        assert_eq!(generate_permutations(4).len(), 24);
    }
}
