use std::f64::consts::PI;
use std::mem::uninitialized;

use ndarray::{Array, Array1, Array2, ArrayView, Dim};
use ndarray_linalg::Norm;
use num::Complex;
use numpy::{Complex64, IntoPyArray, PyArray, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn};
use pyo3::prelude::*;

use crate::geometry::normal_2d;
use crate::{Orientation, A2, A2N};

#[pyclass]
#[derive(Debug, Clone)]
struct Region {
    edges: Array2<f64>,
    vertices: Array2<f64>
}

#[pymethods]
impl Region {
    pub fn len(&self) -> usize {
        self.edges.shape()[0]
    }
}

#[pyclass]
#[derive(Debug, Clone)]
struct Solver {
    region: Region,
}

#[pymethods]
impl Solver {
    #[new]
    pub fn new(region: Region) -> Solver {
        Solver { region }
    }

    pub fn len(&self) -> usize {
        self.region.len()
    }

    pub fn compute_boundary_matrices(
        &self,
        k: f64,
        mu: Complex64,
        orientation: Orientation,
    ) -> (numpy::PyArray2<Complex64>, Array2<Complex64>) {
        unimplemented!()
    }
}
