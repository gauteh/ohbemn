#![allow(non_snake_case)]
use pyo3::prelude::*;

pub mod integrators;
pub mod solver;

/// A Python module implemented in Rust.
#[pymodule]
fn ohbemn(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(integrators::l_2d_py, m)?)?;
    Ok(())
}

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}
