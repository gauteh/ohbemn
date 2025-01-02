#![allow(non_snake_case)]
use ndarray::{Array, ArrayView, Dim};
use pyo3::prelude::*;

pub mod geometry;
pub mod integrators;
pub mod solver;

pub type A2<'a> = ArrayView<'a, f64, Dim<[usize; 1]>>;
pub type A2O = Array<f64, Dim<[usize; 1]>>;

#[pyclass]
#[derive(Debug, Clone)]
pub enum Orientation {
    Interior,
    Exterior,
}

/// A Python module implemented in Rust.
#[pymodule]
fn ohbemn(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(integrators::l_2d_py, m)?)?;
    m.add_class::<Orientation>()?;
    Ok(())
}

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}
