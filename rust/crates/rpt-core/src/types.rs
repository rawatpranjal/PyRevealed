/// Result of a GARP consistency check.
#[derive(Debug, Clone)]
pub struct GarpResult {
    pub is_consistent: bool,
    pub n_violations: u32,
    pub max_scc_size: u32,
    pub n_components: u32,
}

/// Result of a CCEI computation.
#[derive(Debug, Clone)]
pub struct CceiResult {
    pub ccei: f64,
    pub iterations: u32,
    pub is_perfectly_consistent: bool,
}

/// Result of a HARP consistency check.
#[derive(Debug, Clone)]
pub struct HarpResult {
    pub is_consistent: bool,
    pub max_cycle_product: f64,
}

/// Combined analysis result for one user.
#[derive(Debug, Clone)]
pub struct UserResult {
    pub is_garp: bool,
    pub n_violations: u32,
    pub ccei: f64,
    pub mpi: f64,
    pub is_harp: bool,
    pub max_scc_size: u32,
    pub compute_time_us: u64,
}
