use std::f64::consts::PI;

use ndarray::{
    array, concatenate, s, Array, Array1, Array2, Array3, ArrayView, ArrayView2, Axis, Dim,
};
use ndarray_linalg::Norm;
use num::Complex;
use numpy::{
    Complex64, IntoPyArray, PyArray, PyArray2, PyArray3, PyArrayDyn, PyArrayMethods,
    PyReadonlyArray2, PyReadonlyArrayDyn, PyUntypedArrayMethods, ToPyArray,
};
use pyo3::prelude::*;

use crate::geometry::normal_2d;
use crate::{Orientation, A2, A2N};

#[pyclass]
#[derive(Debug, Clone)]
struct Region {
    edges: Array2<usize>,
    vertices: Array2<f64>,
    normals: Array2<f64>,
    lengths: Array1<f64>,
}

impl Region {
    pub fn centers(&self) -> Array2<f64> {
        let mut centers = Array2::<f64>::zeros((self.len(), 2));

        for (i, edge) in self.edges.outer_iter().enumerate() {
            let v0 = self.vertices.index_axis(Axis(0), edge[0]);
            let v1 = self.vertices.index_axis(Axis(0), edge[1]);

            let c = (v0.to_owned() + v1) / 2.;
            centers.slice_mut(s![i, ..]).assign(&c);
        }

        centers
    }

    pub fn normals(&self) -> Array2<f64> {
        self.normals.clone()
    }

    pub fn edge(&self, edge: usize) -> Array2<f64> {
        let edge = self.edges.index_axis(Axis(0), edge);
        let v0 = self.vertices.index_axis(Axis(0), edge[0]);
        let v1 = self.vertices.index_axis(Axis(0), edge[1]);
        concatenate(
            Axis(0),
            &[
                v0.to_shape((1, 2)).unwrap().view(),
                v1.to_shape((1, 2)).unwrap().view(),
            ],
        )
        .unwrap()
    }
}

#[pymethods]
impl Region {
    #[new]
    pub fn new<'py>(
        vertices: PyReadonlyArray2<'py, f64>,
        edges: PyReadonlyArray2<'py, usize>,
    ) -> Region {
        assert_eq!(vertices.shape(), edges.shape());

        let vertices = vertices.to_owned_array();
        let edges = edges.to_owned_array();

        let mut normals = Array2::<f64>::zeros((vertices.shape()[0], 2));
        let mut lengths = Array1::<f64>::zeros((vertices.shape()[0],));

        for (i, edge) in edges.outer_iter().enumerate() {
            let v0 = vertices.index_axis(Axis(0), edge[0]);
            let v1 = vertices.index_axis(Axis(0), edge[1]);

            let (n, l) = normal_2d(v1, v0);
            normals.slice_mut(s![i, ..]).assign(&n);
            lengths.slice_mut(s![i]).assign(&array![l]);
        }

        Region {
            vertices,
            edges,
            normals,
            lengths,
        }
    }

    #[pyo3(name = "edge")]
    pub fn edge_py<'py>(&self, py: Python<'py>, edge: usize) -> Bound<'py, PyArray2<f64>> {
        self.edge(edge).to_pyarray(py)
    }

    pub fn edges<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        let mut edges = Array3::zeros((self.len(), 2, 2));

        for i in 0..self.len() {
            edges.slice_mut(s![i, .., ..]).assign(&self.edge(i));
        }

        edges.to_pyarray(py)
    }

    pub fn vertices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.vertices.to_pyarray(py)
    }

    #[pyo3(name = "centers")]
    pub fn centers_py<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.centers().to_pyarray(py)
    }

    #[pyo3(name = "normals")]
    pub fn normals_py<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.normals().to_pyarray(py)
    }

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
}

impl Solver {
    pub fn compute_boundary_matrices(
        &self,
        k: f64,
        mu: Complex64,
        orientation: Orientation,
    ) -> (numpy::PyArray2<Complex64>, Array2<Complex64>) {
        let A = Array2::<Complex64>::zeros((self.len(), self.len()));
        let B = Array2::<Complex64>::zeros((self.len(), self.len()));

        let centers = self.region.centers();
        let normals = self.region.normals();

        unimplemented!()
    }
}
