use numpy::PyReadonlyArray2;

/// Extract flat f64 slice, T, and K from a numpy 2D array.
/// Copies data since it needs to be Send for Rayon threads.
pub fn extract_user_data(
    prices: &PyReadonlyArray2<f64>,
    quantities: &PyReadonlyArray2<f64>,
) -> (Vec<f64>, Vec<f64>, usize, usize) {
    let p = prices.as_array();
    let q = quantities.as_array();
    let t = p.nrows();
    let k = p.ncols();
    let p_flat: Vec<f64> = p.iter().cloned().collect();
    let q_flat: Vec<f64> = q.iter().cloned().collect();
    (p_flat, q_flat, t, k)
}
