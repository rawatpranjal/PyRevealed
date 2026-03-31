use pyo3::prelude::*;

mod convert;
mod batch;
mod generators;
#[cfg(feature = "parquet")]
mod parquet_reader;

/// PrefGraph Rust core.
#[pymodule]
fn _rust_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(batch::analyze_batch, m)?)?;
    m.add_function(wrap_pyfunction!(batch::analyze_menu_batch, m)?)?;
    m.add_function(wrap_pyfunction!(batch::rum_consistency_batch, m)?)?;
    m.add_function(wrap_pyfunction!(batch::build_preference_graph, m)?)?;
    m.add_function(wrap_pyfunction!(generators::generate_random_budgets, m)?)?;
    m.add_function(wrap_pyfunction!(generators::generate_random_menus, m)?)?;
    m.add_function(wrap_pyfunction!(generators::generate_random_production, m)?)?;
    m.add_function(wrap_pyfunction!(generators::generate_random_intertemporal, m)?)?;
    // Champion vs Challenger benchmarks
    m.add_function(wrap_pyfunction!(batch::benchmark_mpi, m)?)?;
    m.add_function(wrap_pyfunction!(batch::benchmark_hm, m)?)?;
    m.add_function(wrap_pyfunction!(batch::benchmark_closure, m)?)?;
    #[cfg(feature = "parquet")]
    m.add_function(wrap_pyfunction!(batch::analyze_parquet_file, m)?)?;
    Ok(())
}
