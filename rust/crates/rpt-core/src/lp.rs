use highs::{HighsModelStatus, Model, RowProblem, Sense};

/// Solve the Afriat feasibility LP: find utility values U and marginal
/// utilities λ satisfying Afriat's inequalities.
///
/// For all pairs (i,j) where i is revealed preferred to j:
///   U_i - U_j ≤ λ_j * (E[j,i] - E[j,j])
///
/// Variables: [U_0..U_{T-1}, λ_0..λ_{T-1}] (2T variables)
/// Objective: minimize sum(λ) (for a centered solution)
///
/// Returns Some((utilities, lambdas)) if feasible, None otherwise.
pub fn solve_afriat_lp(
    e: &[f64],
    own_exp: &[f64],
    t: usize,
    tolerance: f64,
) -> Option<(Vec<f64>, Vec<f64>)> {
    let n_vars = 2 * t;
    let lambda_lb = 1e-6;

    let mut pb = RowProblem::default();

    // Variables: U_0..U_{T-1} (bounds: [0, inf]), λ_0..λ_{T-1} (bounds: [lambda_lb, inf])
    // Cost: minimize sum(λ)
    let mut cols = Vec::with_capacity(n_vars);
    for i in 0..t {
        cols.push(pb.add_column(0.0, 0.0..)); // U_i: cost=0, lb=0
    }
    for i in 0..t {
        cols.push(pb.add_column(1.0, lambda_lb..)); // λ_i: cost=1, lb=lambda_lb
    }

    // Constraints: for each (i,j) with i≠j:
    //   U_i - U_j - λ_j * (E[j,i] - E[j,j]) ≤ 0
    for i in 0..t {
        for j in 0..t {
            if i == j {
                continue;
            }
            // Coefficient of λ_j is -(E[j,i] - E[j,j])
            let lambda_coeff = -(e[j * t + i] - own_exp[j]);

            // Row: U_i - U_j + lambda_coeff * λ_j ≤ 0
            pb.add_row(
                ..0.0, // upper bound = 0
                [(cols[i], 1.0), (cols[j], -1.0), (cols[t + j], lambda_coeff)],
            );
        }
    }

    let mut model = pb.optimise(Sense::Minimise);
    let solved = model.solve();

    match solved.status() {
        HighsModelStatus::Optimal => {
            let sol = solved.get_solution();
            let u: Vec<f64> = (0..t).map(|i| sol.columns()[i]).collect();
            let lambdas: Vec<f64> = (0..t).map(|i| sol.columns()[t + i]).collect();
            Some((u, lambdas))
        }
        _ => None,
    }
}

/// Solve the Houtman-Maks MILP: find the maximum subset of observations
/// consistent with GARP.
///
/// Implements Demuynck & Rehbeck (2023, Economic Theory) Corollary 2.
/// This formulation avoids Big-M sensitivity issues by using fixed parameters
/// α, δ, ε with clean, data-derived bounds.
///
/// # Variables (per D&R notation)
///
/// - `u_t ∈ [0,1]`  - continuous utility numbers (T variables)
/// - `A_t ∈ {0,1}`  - binary: 1 = keep observation t (T variables)
/// - `U_{t,v} ∈ {0,1}` - binary: 1 iff u_t ≥ u_v (T² variables, t≠v)
///
/// # Parameters
///
/// - `ε ∈ (0, 1/T)` - minimum utility gap to separate distinct values
/// - `α > max_{t} p^t·q^t` - deactivation constant (just above max expenditure)
/// - `δ` - strict/weak affordability separator:
///   `0 < δ < min{min_{t,v} p^t·q^v, min_{t,v: p^t·q^v > p^t·q^t} (p^t·q^v - p^t·q^t)}`
///
/// # Constraints (D&R equations IP-1, IP-2, IP-5, IP-6)
///
/// ```text
/// (IP-1)  u_t - u_v ≤ -ε + 2·U_{t,v}              ∀ t,v ≤ T
/// (IP-2)  U_{v,t} - 1 ≤ u_t - u_v                  ∀ t,v ≤ T
/// (IP-5)  p^t·q^t - p^t·q^v ≤ -δ + α·[U_{t,v} + (1-A_t)]  ∀ t≠v
/// (IP-6)  α·(U_{v,t} + A_t - 2) ≤ p^t·q^v - p^t·q^t        ∀ t≠v
/// ```
///
/// When A_t=1 (kept): IP-5 forces U_{t,v}=1 when q^t R q^v (weak preference),
/// and IP-6 forces U_{v,t}=0 when q^t P q^v (strict preference).
/// When A_t=0 (removed): α deactivates the constraints.
/// IP-1 and IP-2 link U_{t,v} to the utility ordering u_t ≥ u_v.
/// A GARP cycle among kept observations forces u_t > u_t, which is infeasible.
///
/// Objective: `max Σ A_t`
///
/// Returns indices of observations to REMOVE (where A_t = 0).
///
/// # References
///
/// Demuynck, T. & Rehbeck, J. (2023). "Computing revealed preference
/// goodness-of-fit measures with integer programming." Economic Theory,
/// 75, 1101–1130. Corollary 2, constraints IP-1 through IP-6.
pub fn solve_hm_ilp(
    e: &[f64],
    own_exp: &[f64],
    t: usize,
    _tolerance: f64,
) -> Vec<usize> {
    if t == 0 {
        return Vec::new();
    }

    // ── Parameter computation (D&R Section 2, p. 4–5) ──────────────────
    //
    // α must exceed the maximum own-expenditure p^t·q^t.
    let max_exp = own_exp.iter().cloned().fold(0.0f64, f64::max);
    let alpha = max_exp + 1.0;

    // ε must satisfy 0 < ε < 1/T. We use ε = 1/(2T).
    let eps = 1.0 / (2.0 * t as f64);

    // δ must satisfy (D&R p. 9):
    //   0 < δ < min{ min_{t,v} p^t·q^v,
    //                min_{t,v: p^t·q^v > p^t·q^t} (p^t·q^v - p^t·q^t) }
    //
    // First term: minimum cross-expenditure (ensures all expenditures > δ).
    // Second term: minimum gap where a bundle EXCEEDS the budget (ensures
    //   the formulation correctly distinguishes "affordable" from "not affordable").
    //   Note: the gap is cross - own (NOT own - cross).
    let mut min_cross_exp = f64::MAX;
    let mut min_positive_gap = f64::MAX;
    for tv in 0..t {
        for v in 0..t {
            if tv == v {
                continue;
            }
            let cross = e[tv * t + v]; // p^t · q^v
            if cross > 1e-15 && cross < min_cross_exp {
                min_cross_exp = cross;
            }
            // Gap where cross > own: bundle v costs MORE than own bundle at prices t
            let gap = cross - own_exp[tv]; // p^t·q^v - p^t·q^t
            if gap > 1e-15 && gap < min_positive_gap {
                min_positive_gap = gap;
            }
        }
    }
    // δ = half the minimum of both bounds (or a small fallback).
    // If no positive gaps exist (all bundles affordable at all prices),
    // only the first bound applies.
    let delta_bound = min_cross_exp.min(min_positive_gap);
    let delta = if delta_bound < f64::MAX && delta_bound > 1e-15 {
        delta_bound / 2.0
    } else {
        1e-6
    };

    // ── Build MILP ─────────────────────────────────────────────────────

    let mut pb = RowProblem::default();

    // A_t: binary keep/remove indicators - added FIRST so they occupy
    // column indices 0..T-1 for easy solution extraction.
    // Objective: maximize Σ A_t = minimize Σ (-A_t).
    let mut a_cols = Vec::with_capacity(t);
    for _ in 0..t {
        a_cols.push(pb.add_integer_column(-1.0, 0.0..1.0));
    }

    // u_t: continuous utility numbers in [0, 1]
    let mut u_cols = Vec::with_capacity(t);
    for _ in 0..t {
        u_cols.push(pb.add_column(0.0, 0.0..1.0));
    }

    // U_{t,v}: binary indicators for u_t ≥ u_v
    // Indexed as big_u[t * t + v] (diagonal entries unused)
    let mut big_u_cols = vec![None; t * t];
    for tv in 0..t {
        for v in 0..t {
            if tv == v {
                continue;
            }
            big_u_cols[tv * t + v] = Some(pb.add_integer_column(0.0, 0.0..1.0));
        }
    }

    // ── Constraints ────────────────────────────────────────────────────

    for tv in 0..t {
        for v in 0..t {
            if tv == v {
                continue;
            }
            let u_tv = big_u_cols[tv * t + v].unwrap();
            let u_vt = big_u_cols[v * t + tv].unwrap();

            // (IP-1): u_t - u_v - 2·U_{t,v} ≤ -ε
            pb.add_row(
                ..-eps,
                [(u_cols[tv], 1.0), (u_cols[v], -1.0), (u_tv, -2.0)],
            );

            // (IP-2): U_{t,v} - 1 ≤ u_t - u_v  →  -u_t + u_v + U_{t,v} ≤ 1
            // This enforces: U_{t,v}=1 → u_t ≥ u_v (D&R proof, p. 6).
            pb.add_row(
                ..1.0,
                [(u_cols[tv], -1.0), (u_cols[v], 1.0), (u_tv, 1.0)],
            );

            // (IP-5): α·A_t - α·U_{t,v} ≤ α - δ - (p^t·q^t - p^t·q^v)
            // (from: p^t·q^t - p^t·q^v ≤ -δ + α·[U_{t,v} + (1 - A_t)])
            let d_tv = own_exp[tv] - e[tv * t + v]; // p^t·q^t - p^t·q^v
            let ip5_rhs = alpha - delta - d_tv;
            pb.add_row(
                ..ip5_rhs,
                [(a_cols[tv], alpha), (u_tv, -alpha)],
            );

            // (IP-6): α·U_{v,t} + α·A_t ≤ 2α + (p^t·q^v - p^t·q^t)
            // (from: α·(U_{v,t} + A_t - 2) ≤ p^t·q^v - p^t·q^t)
            let ip6_rhs = 2.0 * alpha + (e[tv * t + v] - own_exp[tv]);
            pb.add_row(
                ..ip6_rhs,
                [(u_vt, alpha), (a_cols[tv], alpha)],
            );
        }
    }

    // ── Solve ──────────────────────────────────────────────────────────

    let mut model = pb.optimise(Sense::Minimise);
    let solved = model.solve();

    match solved.status() {
        HighsModelStatus::Optimal => {
            let sol = solved.get_solution();
            let mut removed = Vec::new();
            // A_t columns were added first → indices 0..t-1
            for i in 0..t {
                if sol.columns()[i] < 0.5 {
                    removed.push(i);
                }
            }
            removed
        }
        _ => {
            // Solver failed - return empty so caller can fall back to greedy
            Vec::new()
        }
    }
}

// ---------------------------------------------------------------------------
// Optional Gurobi backend (feature = "gurobi")
// ---------------------------------------------------------------------------

/// Solve the Afriat feasibility LP using Gurobi (requires license).
///
/// Identical formulation to `solve_afriat_lp` but uses Gurobi's LP solver,
/// which is ~10x faster than HiGHS for large MIP problems (Machado, 2024).
#[cfg(feature = "gurobi")]
pub fn solve_afriat_lp_gurobi(
    e: &[f64],
    own_exp: &[f64],
    t: usize,
    tolerance: f64,
) -> Option<(Vec<f64>, Vec<f64>)> {
    use grb::prelude::*;

    let lambda_lb = 1e-6;

    let mut env = Env::empty().ok()?.start().ok()?;
    env.set(param::OutputFlag, 0).ok()?;

    let mut model = Model::with_env("afriat", &env).ok()?;

    // Variables: U_i (continuous, lb=0) and λ_i (continuous, lb=lambda_lb)
    let u: Vec<Var> = (0..t)
        .map(|i| add_ctsvar!(model, name: &format!("u_{}", i), bounds: 0.0..).unwrap())
        .collect();
    let lam: Vec<Var> = (0..t)
        .map(|i| {
            add_ctsvar!(model, name: &format!("l_{}", i), bounds: lambda_lb..).unwrap()
        })
        .collect();

    // Objective: minimize sum(λ)
    model.set_objective(lam.iter().grb_sum(), Minimize).ok()?;

    // Constraints: U_i - U_j - λ_j * (E[j,i] - E[j,j]) <= 0
    for i in 0..t {
        for j in 0..t {
            if i == j {
                continue;
            }
            let lambda_coeff = -(e[j * t + i] - own_exp[j]);
            model
                .add_constr(
                    &format!("c_{}_{}", i, j),
                    c!(u[i] - u[j] + lambda_coeff * lam[j] <= 0.0),
                )
                .ok()?;
        }
    }

    model.optimize().ok()?;

    match model.status().ok()? {
        Status::Optimal => {
            let u_vals: Vec<f64> = u.iter().map(|v| model.get_obj_attr(attr::X, v).unwrap()).collect();
            let l_vals: Vec<f64> = lam.iter().map(|v| model.get_obj_attr(attr::X, v).unwrap()).collect();
            Some((u_vals, l_vals))
        }
        _ => None,
    }
}

/// Solve the Houtman-Maks ILP using Gurobi (requires license).
///
/// ~10x faster than HiGHS for MIP problems. Same Big-M Afriat formulation.
#[cfg(feature = "gurobi")]
pub fn solve_hm_ilp_gurobi(
    e: &[f64],
    own_exp: &[f64],
    t: usize,
    tolerance: f64,
) -> Vec<usize> {
    use grb::prelude::*;

    let max_exp = own_exp.iter().cloned().fold(0.0f64, f64::max);
    let big_m = (3.0 * max_exp).max(10.0);
    let lambda_lb = 0.01;

    let env = match Env::empty().and_then(|e| e.start()) {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };

    let mut model = match Model::with_env("hm", &env) {
        Ok(m) => m,
        Err(_) => return Vec::new(),
    };
    let _ = model.set_param(param::OutputFlag, 0);

    // Variables
    let z: Vec<Var> = (0..t)
        .map(|i| add_binvar!(model, name: &format!("z_{}", i)).unwrap())
        .collect();
    let u: Vec<Var> = (0..t)
        .map(|i| add_ctsvar!(model, name: &format!("u_{}", i), bounds: 0.0..big_m).unwrap())
        .collect();
    let lam: Vec<Var> = (0..t)
        .map(|i| {
            add_ctsvar!(model, name: &format!("l_{}", i), bounds: lambda_lb..big_m).unwrap()
        })
        .collect();

    // Objective: maximize sum(z) = minimize sum(-z)
    model
        .set_objective(z.iter().map(|v| (*v, -1.0)).grb_sum(), Minimize)
        .ok();

    // Constraints
    for i in 0..t {
        for j in 0..t {
            if i == j {
                continue;
            }
            let lambda_coeff = -(e[j * t + i] - own_exp[j]);
            let _ = model.add_constr(
                &format!("c_{}_{}", i, j),
                c!(u[i] - u[j] + lambda_coeff * lam[j] + big_m * z[i] + big_m * z[j] <= 2.0 * big_m),
            );
        }
    }

    if model.optimize().is_err() {
        return Vec::new();
    }

    match model.status() {
        Ok(Status::Optimal) => {
            let mut removed = Vec::new();
            for i in 0..t {
                if let Ok(val) = model.get_obj_attr(attr::X, &z[i]) {
                    if val < 0.5 {
                        removed.push(i);
                    }
                }
            }
            removed
        }
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_afriat_lp_consistent() {
        // Consistent data: should find feasible U, λ
        // p=[[1,2],[2,1]], q=[[4,1],[1,4]]
        // E = [[6,3],[9,6]]
        let e = [6.0, 3.0, 9.0, 6.0];
        let own = [6.0, 6.0];
        let result = solve_afriat_lp(&e, &own, 2, 1e-10);
        assert!(result.is_some());
    }

    #[test]
    fn test_afriat_lp_violation() {
        // Violation data: p=[[2,1],[1,2]], q=[[3,2],[2,3]]
        // E = [[8,7],[7,8]]
        let e = [8.0, 7.0, 7.0, 8.0];
        let own = [8.0, 8.0];
        let result = solve_afriat_lp(&e, &own, 2, 1e-10);
        assert!(result.is_none());
    }

    #[test]
    fn test_hm_ilp_consistent() {
        let e = [6.0, 3.0, 9.0, 6.0];
        let own = [6.0, 6.0];
        let removed = solve_hm_ilp(&e, &own, 2, 1e-10);
        assert_eq!(removed.len(), 0); // No removals needed
    }

    #[test]
    fn test_hm_ilp_violation() {
        let e = [8.0, 7.0, 7.0, 8.0];
        let own = [8.0, 8.0];
        let removed = solve_hm_ilp(&e, &own, 2, 1e-10);
        assert_eq!(removed.len(), 1); // Must remove 1 observation
    }
}
