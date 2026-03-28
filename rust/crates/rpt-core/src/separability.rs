use crate::lp::solve_afriat_lp;
use crate::expenditure::build_expenditure;

/// Weak separability test result.
pub struct SeparabilityResult {
    pub is_separable: bool,
    pub group_a_consistency: f64,  // CCEI within group A
    pub group_b_consistency: f64,  // CCEI within group B
}

/// Test weak separability between two groups of goods.
///
/// Group A and Group B are separable if utility can be written as
/// U(x_A, x_B) = W(V_A(x_A), V_B(x_B)) - i.e., the sub-utility of
/// group A doesn't depend on group B's quantities, and vice versa.
///
/// Test: run Afriat LP independently on each group's sub-data.
/// If both groups satisfy GARP independently, separability is supported.
pub fn test_separability(
    prices: &[f64],       // T x K flat
    quantities: &[f64],   // T x K flat
    t: usize,
    k: usize,
    group_a: &[usize],   // indices of goods in group A
    group_b: &[usize],   // indices of goods in group B
    tolerance: f64,
) -> SeparabilityResult {
    let ka = group_a.len();
    let kb = group_b.len();

    // Extract sub-prices and sub-quantities for each group
    let mut p_a = vec![0.0; t * ka];
    let mut q_a = vec![0.0; t * ka];
    let mut p_b = vec![0.0; t * kb];
    let mut q_b = vec![0.0; t * kb];

    for obs in 0..t {
        for (gi, &g) in group_a.iter().enumerate() {
            p_a[obs * ka + gi] = prices[obs * k + g];
            q_a[obs * ka + gi] = quantities[obs * k + g];
        }
        for (gi, &g) in group_b.iter().enumerate() {
            p_b[obs * kb + gi] = prices[obs * k + g];
            q_b[obs * kb + gi] = quantities[obs * k + g];
        }
    }

    // Build expenditure matrices for each group
    let mut e_a = vec![0.0; t * t];
    let mut own_a = vec![0.0; t];
    build_expenditure(&p_a, &q_a, t, ka, &mut e_a, &mut own_a);

    let mut e_b = vec![0.0; t * t];
    let mut own_b = vec![0.0; t];
    build_expenditure(&p_b, &q_b, t, kb, &mut e_b, &mut own_b);

    // Test Afriat feasibility on each group independently
    let a_feasible = solve_afriat_lp(&e_a, &own_a, t, tolerance).is_some();
    let b_feasible = solve_afriat_lp(&e_b, &own_b, t, tolerance).is_some();

    // Consistency scores (1.0 if feasible, approximate otherwise)
    let ccei_a = if a_feasible { 1.0 } else { estimate_ccei(&e_a, &own_a, t, tolerance) };
    let ccei_b = if b_feasible { 1.0 } else { estimate_ccei(&e_b, &own_b, t, tolerance) };

    SeparabilityResult {
        is_separable: a_feasible && b_feasible,
        group_a_consistency: ccei_a,
        group_b_consistency: ccei_b,
    }
}

/// Quick CCEI estimate via binary search over 10 discrete levels.
fn estimate_ccei(e: &[f64], own_exp: &[f64], t: usize, tolerance: f64) -> f64 {
    let levels = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50, 0.30, 0.10];
    for &eff in &levels {
        // Build R at efficiency level
        let mut r = vec![false; t * t];
        let mut p = vec![false; t * t];
        for i in 0..t {
            for j in 0..t {
                r[i * t + j] = eff * own_exp[i] >= e[i * t + j] - tolerance;
                p[i * t + j] = eff * own_exp[i] > e[i * t + j] + tolerance;
            }
            p[i * t + i] = false;
        }
        // Quick check: any violation?
        let mut closure = r.clone();
        for i in 0..t { closure[i * t + i] = true; }
        for k in 0..t {
            for i in 0..t {
                if closure[i * t + k] {
                    for j in 0..t {
                        if closure[k * t + j] { closure[i * t + j] = true; }
                    }
                }
            }
        }
        let mut has_violation = false;
        for i in 0..t {
            for j in 0..t {
                if closure[i * t + j] && p[j * t + i] { has_violation = true; break; }
            }
            if has_violation { break; }
        }
        if !has_violation { return eff; }
    }
    0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trivial_separability() {
        // 2 goods, each in its own group. Consistent data → separable.
        let prices = [1.0, 2.0, 2.0, 1.0];
        let quantities = [4.0, 1.0, 1.0, 4.0];
        let r = test_separability(&prices, &quantities, 2, 2, &[0], &[1], 1e-10);
        assert!(r.is_separable);
    }
}
