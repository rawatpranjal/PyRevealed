use crate::closure::scc_transitive_closure;
use crate::expenditure::build_expenditure;

/// Universal intermediate representation for revealed preference analysis.
///
/// Replaces the old Scratchpad with state-tracked lazy computation.
/// Each Rayon thread gets one PreferenceGraph, reused across users via `reset()`.
///
/// Computation is lazy: `ensure_*` methods skip work if already done.
/// Downstream state is invalidated when upstream changes (e.g., new R invalidates closure).
pub struct PreferenceGraph {
    // Expenditure data (budget/production parsers fill these)
    pub e: Vec<f64>,              // T×T expenditure matrix
    pub own_exp: Vec<f64>,        // T diagonal

    // Boolean preference relations
    pub r: Vec<bool>,             // T×T weak revealed preference
    pub p: Vec<bool>,             // T×T strict revealed preference
    pub r_star: Vec<bool>,        // T×T transitive closure of R

    // SCC decomposition
    pub scc_labels: Vec<u32>,     // T SCC component IDs

    // Edge weights (HARP, quasilinear)
    pub edge_weights: Vec<f64>,   // T×T log(own_exp[i]/e[i,j])
    pub max_product: Vec<f64>,    // T×T max log-product paths

    // Dimensions
    pub t: usize,
    pub capacity: usize,

    // State flags
    pub has_expenditure: bool,
    pub has_r: bool,
    pub has_closure: bool,
    pub has_weights: bool,
    pub has_max_product: bool,

    // Cached diagnostics from closure
    pub n_components: usize,
    pub max_scc_size: usize,
    pub tolerance: f64,
}

impl PreferenceGraph {
    pub fn new(max_t: usize) -> Self {
        let n2 = max_t * max_t;
        PreferenceGraph {
            e: vec![0.0; n2],
            own_exp: vec![0.0; max_t],
            r: vec![false; n2],
            p: vec![false; n2],
            r_star: vec![false; n2],
            scc_labels: vec![0; max_t],
            edge_weights: vec![0.0; n2],
            max_product: vec![f64::NEG_INFINITY; n2],
            t: 0,
            capacity: max_t,
            has_expenditure: false,
            has_r: false,
            has_closure: false,
            has_weights: false,
            has_max_product: false,
            n_components: 0,
            max_scc_size: 0,
            tolerance: 1e-10,
        }
    }

    /// Grow buffers if needed for a graph with T nodes.
    pub fn ensure_capacity(&mut self, t: usize) {
        if t > self.capacity {
            let n2 = t * t;
            self.e.resize(n2, 0.0);
            self.own_exp.resize(t, 0.0);
            self.r.resize(n2, false);
            self.p.resize(n2, false);
            self.r_star.resize(n2, false);
            self.scc_labels.resize(t, 0);
            self.edge_weights.resize(n2, 0.0);
            self.max_product.resize(n2, f64::NEG_INFINITY);
            self.capacity = t;
        }
    }

    /// Clear all state for reuse with a new user. Does NOT deallocate.
    pub fn reset(&mut self) {
        self.t = 0;
        self.has_expenditure = false;
        self.has_r = false;
        self.has_closure = false;
        self.has_weights = false;
        self.has_max_product = false;
        self.n_components = 0;
        self.max_scc_size = 0;
    }

    /// Compute expenditure matrix E[i,j] = p_i · x_j. No-op if already done.
    pub fn ensure_expenditure(&mut self, prices: &[f64], quantities: &[f64], t: usize, k: usize) {
        if self.has_expenditure && self.t == t {
            return;
        }
        self.ensure_capacity(t);
        self.t = t;
        build_expenditure(prices, quantities, t, k, &mut self.e, &mut self.own_exp);
        self.has_expenditure = true;
        // Invalidate downstream
        self.has_r = false;
        self.has_closure = false;
        self.has_weights = false;
        self.has_max_product = false;
    }

    /// Build R (weak) and P (strict) from expenditure at e=1.0. No-op if already done.
    pub fn ensure_r(&mut self, tolerance: f64) {
        if self.has_r {
            return;
        }
        let t = self.t;
        for i in 0..t {
            for j in 0..t {
                let idx = i * t + j;
                self.r[idx] = self.own_exp[i] >= self.e[idx] - tolerance;
                self.p[idx] = self.own_exp[i] > self.e[idx] + tolerance;
            }
            self.p[i * t + i] = false;
        }
        self.tolerance = tolerance;
        self.has_r = true;
        // Invalidate downstream
        self.has_closure = false;
    }

    /// Build R and P at a given efficiency level e. Used by CCEI binary search.
    /// Always rebuilds (efficiency varies per call). Invalidates closure.
    pub fn build_r_at_efficiency(&mut self, efficiency: f64, tolerance: f64) {
        let t = self.t;
        for i in 0..t {
            let deflated = efficiency * self.own_exp[i];
            for j in 0..t {
                let idx = i * t + j;
                self.r[idx] = deflated >= self.e[idx] - tolerance;
                self.p[idx] = deflated > self.e[idx] + tolerance;
            }
            self.p[i * t + i] = false;
        }
        self.has_r = true;
        self.has_closure = false;
    }

    /// Compute SCC-optimized transitive closure of R. No-op if already done.
    pub fn ensure_closure(&mut self) {
        if self.has_closure {
            return;
        }
        let t = self.t;
        let (n_comp, max_scc) = scc_transitive_closure(
            &self.r[..t * t],
            t,
            &mut self.r_star[..t * t],
            &mut self.scc_labels[..t],
        );
        self.n_components = n_comp;
        self.max_scc_size = max_scc;
        self.has_closure = true;
    }

    /// Build log-ratio edge weights from E for HARP. No-op if already done.
    pub fn ensure_weights(&mut self) {
        if self.has_weights {
            return;
        }
        let t = self.t;
        let tol = self.tolerance;
        for i in 0..t {
            for j in 0..t {
                let idx = i * t + j;
                let e_ij = self.e[idx];
                if e_ij > tol && self.own_exp[i] > tol {
                    self.edge_weights[idx] = self.own_exp[i].ln() - e_ij.ln();
                } else {
                    self.edge_weights[idx] = f64::NEG_INFINITY;
                }
            }
            self.edge_weights[i * t + i] = 0.0;
        }
        self.has_weights = true;
        self.has_max_product = false;
    }

    /// Run max-product Floyd-Warshall on edge weights. No-op if already done.
    pub fn ensure_max_product(&mut self) {
        if self.has_max_product {
            return;
        }
        let t = self.t;

        // Initialize from direct edges
        for i in 0..t {
            for j in 0..t {
                let idx = i * t + j;
                if self.r[idx] {
                    self.max_product[idx] = self.edge_weights[idx];
                } else {
                    self.max_product[idx] = f64::NEG_INFINITY;
                }
            }
            self.max_product[i * t + i] = 0.0;
        }

        // Modified Floyd-Warshall: maximize log-sum
        for k in 0..t {
            for i in 0..t {
                for j in 0..t {
                    let via_k = self.max_product[i * t + k] + self.max_product[k * t + j];
                    if via_k > self.max_product[i * t + j] {
                        self.max_product[i * t + j] = via_k;
                    }
                }
            }
        }

        self.has_max_product = true;
    }

    // ---- Parsers: fill the graph from different data types ----

    /// Parse budget data: prices × quantities → expenditure → R/P.
    pub fn parse_budget(
        &mut self,
        prices: &[f64],
        quantities: &[f64],
        t: usize,
        k: usize,
        tolerance: f64,
    ) {
        self.ensure_expenditure(prices, quantities, t, k);
        self.ensure_r(tolerance);
    }

    /// Parse menu data: menus + choices → item-space R/P.
    /// Graph dimension is n_items (not n_observations).
    pub fn parse_menu(
        &mut self,
        menus: &[Vec<usize>],
        choices: &[usize],
        n_items: usize,
    ) {
        self.ensure_capacity(n_items);
        self.t = n_items;
        let t = n_items;

        // Zero out R and P
        for idx in 0..t * t {
            self.r[idx] = false;
            self.p[idx] = false;
        }

        // R[choice, item] = true for all non-chosen items in each menu
        for (menu, &choice) in menus.iter().zip(choices) {
            for &item in menu {
                if item != choice {
                    self.r[choice * t + item] = true;
                    self.p[choice * t + item] = true; // menu choice is strict
                }
            }
        }

        self.has_expenditure = false; // menus have no expenditure
        self.has_r = true;
        self.has_closure = false;
        self.has_weights = false;
        self.has_max_product = false;
    }

    /// Parse production data: input/output prices × quantities → profit comparison → R/P.
    pub fn parse_production(
        &mut self,
        input_prices: &[f64],
        input_quantities: &[f64],
        output_prices: &[f64],
        output_quantities: &[f64],
        t: usize,
        n_inputs: usize,
        n_outputs: usize,
        tolerance: f64,
    ) {
        self.ensure_capacity(t);
        self.t = t;

        // Compute actual profits: revenue[i] - cost[i]
        // revenue[i] = output_prices[i] · output_quantities[i]
        // cost[i] = input_prices[i] · input_quantities[i]
        // Counterfactual profit: output_prices[i] · output_quantities[j] - input_prices[i] · input_quantities[j]

        // Build counterfactual profit matrix and actual profits
        for i in 0..t {
            let mut actual_rev = 0.0f64;
            let mut actual_cost = 0.0f64;
            for g in 0..n_outputs {
                actual_rev += output_prices[i * n_outputs + g] * output_quantities[i * n_outputs + g];
            }
            for g in 0..n_inputs {
                actual_cost += input_prices[i * n_inputs + g] * input_quantities[i * n_inputs + g];
            }
            self.own_exp[i] = actual_rev - actual_cost; // actual profit

            for j in 0..t {
                let mut cf_rev = 0.0f64;
                let mut cf_cost = 0.0f64;
                for g in 0..n_outputs {
                    cf_rev += output_prices[i * n_outputs + g] * output_quantities[j * n_outputs + g];
                }
                for g in 0..n_inputs {
                    cf_cost += input_prices[i * n_inputs + g] * input_quantities[j * n_inputs + g];
                }
                self.e[i * t + j] = cf_rev - cf_cost; // counterfactual profit
            }
        }
        self.has_expenditure = true;

        // R[i,j] = actual_profit[i] >= counterfactual_profit[i,j]
        for i in 0..t {
            for j in 0..t {
                let idx = i * t + j;
                self.r[idx] = self.own_exp[i] >= self.e[idx] - tolerance;
                self.p[idx] = self.own_exp[i] > self.e[idx] + tolerance;
            }
            self.p[i * t + i] = false;
        }
        self.tolerance = tolerance;
        self.has_r = true;
        self.has_closure = false;
        self.has_weights = false;
        self.has_max_product = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_budget_sets_flags() {
        let prices = [1.0, 2.0, 2.0, 1.0];
        let quantities = [4.0, 1.0, 1.0, 4.0];
        let mut g = PreferenceGraph::new(2);
        g.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        assert!(g.has_expenditure);
        assert!(g.has_r);
        assert!(!g.has_closure);
    }

    #[test]
    fn test_ensure_closure_lazy() {
        let prices = [1.0, 2.0, 2.0, 1.0];
        let quantities = [4.0, 1.0, 1.0, 4.0];
        let mut g = PreferenceGraph::new(2);
        g.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        g.ensure_closure();
        assert!(g.has_closure);
        // Second call is a no-op
        g.ensure_closure();
        assert!(g.has_closure);
    }

    #[test]
    fn test_reset_clears_flags() {
        let prices = [1.0, 2.0, 2.0, 1.0];
        let quantities = [4.0, 1.0, 1.0, 4.0];
        let mut g = PreferenceGraph::new(2);
        g.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        g.ensure_closure();
        g.reset();
        assert!(!g.has_expenditure);
        assert!(!g.has_r);
        assert!(!g.has_closure);
    }

    #[test]
    fn test_parse_menu() {
        let menus = vec![vec![0, 1, 2], vec![0, 1]];
        let choices = [0, 1];
        let mut g = PreferenceGraph::new(3);
        g.parse_menu(&menus, &choices, 3);
        assert_eq!(g.t, 3);
        assert!(g.has_r);
        assert!(!g.has_expenditure);
        // 0 preferred to 1, 2 from menu 1; 1 preferred to 0 from menu 2
        assert!(g.r[0 * 3 + 1]); // 0 > 1
        assert!(g.r[0 * 3 + 2]); // 0 > 2
        assert!(g.r[1 * 3 + 0]); // 1 > 0
        assert!(!g.r[2 * 3 + 0]); // 2 not revealed preferred to anything
    }

    #[test]
    fn test_expenditure_reuse() {
        let prices = [1.0, 2.0, 2.0, 1.0];
        let quantities = [4.0, 1.0, 1.0, 4.0];
        let mut g = PreferenceGraph::new(2);
        g.ensure_expenditure(&prices, &quantities, 2, 2);
        assert!(g.has_expenditure);

        // Build R from cached E
        g.ensure_r(1e-10);
        assert!(g.has_r);

        // Ensure weights from cached E
        g.ensure_weights();
        assert!(g.has_weights);

        // E was computed only once (both ensure_r and ensure_weights used cached E)
    }
}
