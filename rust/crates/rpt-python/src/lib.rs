use pyo3::prelude::*;

mod convert;
mod batch;

/// PyRevealed Rust core.
#[pymodule]
fn _rust_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(batch::analyze_batch, m)?)?;
    m.add_function(wrap_pyfunction!(batch::build_preference_graph, m)?)?;
    Ok(())
}
