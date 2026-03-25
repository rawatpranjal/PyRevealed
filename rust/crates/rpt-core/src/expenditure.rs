/// Build the T x T expenditure matrix E[i,j] = p_i @ x_j.
///
/// All matrices are row-major flat arrays: prices is T*K, quantities is T*K,
/// output e is T*T, own_exp is T.
pub fn build_expenditure(
    prices: &[f64],
    quantities: &[f64],
    t: usize,
    k: usize,
    e: &mut [f64],
    own_exp: &mut [f64],
) {
    for i in 0..t {
        for j in 0..t {
            let mut sum = 0.0f64;
            for g in 0..k {
                sum += prices[i * k + g] * quantities[j * k + g];
            }
            e[i * t + j] = sum;
        }
    }
    for i in 0..t {
        own_exp[i] = e[i * t + i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diagonal_is_own_expenditure() {
        // 2 obs, 2 goods: p=[[1,2],[3,1]], q=[[4,1],[1,3]]
        let p = [1.0, 2.0, 3.0, 1.0];
        let q = [4.0, 1.0, 1.0, 3.0];
        let mut e = [0.0; 4];
        let mut own = [0.0; 2];
        build_expenditure(&p, &q, 2, 2, &mut e, &mut own);
        // E[0,0] = 1*4+2*1 = 6
        assert!((own[0] - 6.0).abs() < 1e-10);
        // E[1,1] = 3*1+1*3 = 6
        assert!((own[1] - 6.0).abs() < 1e-10);
        // E[0,1] = 1*1+2*3 = 7
        assert!((e[0 * 2 + 1] - 7.0).abs() < 1e-10);
    }
}
