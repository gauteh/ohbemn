#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
use ndarray::{Array, ArrayView, Dim};
use pyo3::prelude::*;

extern crate blas_src;

pub mod boundary;
pub mod geometry;
pub mod integrators;
pub mod solver;

use boundary::{BoundaryCondition, BoundaryIncidence, Region};
use solver::{BoundarySolution, Solver};

pub type A2<'a> = ArrayView<'a, f64, Dim<[usize; 1]>>;
pub type A2N<'a> = ArrayView<'a, f64, Dim<[usize; 2]>>;
pub type A2O = Array<f64, Dim<[usize; 1]>>;

#[pyclass]
#[derive(Debug, Clone, Copy)]
pub enum Orientation {
    Interior,
    Exterior,
}

/// Ocean Boundary Element Method implemented in Rust.
#[pymodule]
fn ohbemn(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(integrators::l_2d_py, m)?)?;
    m.add_class::<Orientation>()?;

    m.add_class::<Region>()?;
    m.add_class::<BoundaryIncidence>()?;
    m.add_class::<BoundaryCondition>()?;

    m.add_class::<Solver>()?;
    m.add_class::<BoundarySolution>()?;
    Ok(())
}
