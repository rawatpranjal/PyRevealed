/// Test exponential discounting consistency.
///
/// For each choice pair, derives bounds on the discount factor delta.
/// If all bounds are compatible (delta_lower <= delta_upper), the choices
/// are consistent with exponential discounting.
///
/// No LP needed - just constraint propagation on delta bounds.
pub struct IntertemporalResult {
    pub is_consistent: bool,
    pub delta_lower: f64,
    pub delta_upper: f64,
    pub n_constraints: u32,
}

/// Check if a sequence of intertemporal choices is consistent with
/// exponential discounting U = sum(delta^t * u(c_t)).
///
/// Each choice is: (amount_chosen, time_chosen, amount_rejected, time_rejected).
/// If chosen at later time: delta >= (rejected/chosen)^(1/(t_chosen - t_rejected))
/// If chosen at earlier time: delta <= (rejected/chosen)^(1/(t_chosen - t_rejected))
pub fn check_exponential_discounting(
    choices: &[(f64, f64, f64, f64)],  // (amt_chosen, t_chosen, amt_rejected, t_rejected)
    tolerance: f64,
) -> IntertemporalResult {
    let mut delta_lower = 0.0f64;
    let mut delta_upper = 1.0f64;
    let mut n_constraints = 0u32;

    for &(amt_chosen, t_chosen, amt_rejected, t_rejected) in choices {
        if amt_chosen <= 0.0 || amt_rejected <= 0.0 {
            continue;
        }

        let dt = t_chosen - t_rejected;
        if dt.abs() < tolerance {
            continue; // Same time - no discounting constraint
        }

        let ratio = amt_rejected / amt_chosen;
        if ratio <= 0.0 {
            continue;
        }

        let bound = ratio.powf(1.0 / dt);
        n_constraints += 1;

        if dt > 0.0 {
            // Chose later option → patient → delta must be high enough
            if bound > delta_lower {
                delta_lower = bound;
            }
        } else {
            // Chose earlier option → impatient → delta must be low enough
            if bound < delta_upper {
                delta_upper = bound;
            }
        }
    }

    IntertemporalResult {
        is_consistent: delta_lower <= delta_upper + tolerance,
        delta_lower,
        delta_upper,
        n_constraints,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_patient_chooser() {
        // Always picks the larger-later option → high delta
        let choices = vec![
            (200.0, 1.0, 100.0, 0.0),  // 200 at t=1 over 100 at t=0 → delta >= 0.5
            (150.0, 2.0, 100.0, 0.0),  // 150 at t=2 over 100 at t=0 → delta >= sqrt(2/3)
        ];
        let r = check_exponential_discounting(&choices, 1e-10);
        assert!(r.is_consistent);
        assert!(r.delta_lower > 0.0);
    }

    #[test]
    fn test_inconsistent_discounting() {
        // Picks later in one case, earlier in another → contradiction
        let choices = vec![
            (200.0, 10.0, 100.0, 0.0),  // Very patient (delta >= 0.5^0.1 ≈ 0.93)
            (100.0, 0.0, 200.0, 1.0),   // Very impatient (delta <= 0.5)
        ];
        let r = check_exponential_discounting(&choices, 1e-10);
        assert!(!r.is_consistent);
    }
}
