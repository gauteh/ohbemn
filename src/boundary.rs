use ndarray::{array, s, Array1, Array2, Array3, Axis};
use numpy::{
    AllowTypeChange, Complex64, PyArray1, PyArray2, PyArray3, PyArrayLike2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods, ToPyArray
};
use pyo3::prelude::*;

use crate::geometry::normal_2d;

#[pyclass]
#[derive(Debug, Clone)]
pub struct Region {
    edges: Array2<usize>,
    vertices: Array2<f64>,
    normals: Array2<f64>,
    // lengths: Array1<f64>,
}

impl Region {
    pub fn new(vertices: Array2<f64>, edges: Array2<usize>) -> Region {
        let mut normals = Array2::<f64>::zeros((vertices.shape()[0], 2));
        // let mut lengths = Array1::<f64>::zeros((vertices.shape()[0],));

        for (i, edge) in edges.outer_iter().enumerate() {
            let v0 = vertices.index_axis(Axis(0), edge[0]);
            let v1 = vertices.index_axis(Axis(0), edge[1]);

            let (n, _l) = normal_2d(v0, v1);
            normals.slice_mut(s![i, ..]).assign(&n);
            // lengths[i] = l;
        }

        Region {
            vertices,
            edges,
            normals,
            // lengths,
        }
    }

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

    pub fn edge(&self, edge: usize) -> (Array1<f64>, Array1<f64>) {
        let edge = self.edges.index_axis(Axis(0), edge);
        let v0 = self.vertices.index_axis(Axis(0), edge[0]);
        let v1 = self.vertices.index_axis(Axis(0), edge[1]);
        // concatenate(
        //     Axis(0),
        //     &[
        //         v0.to_shape((1, 2)).unwrap().view(),
        //         v1.to_shape((1, 2)).unwrap().view(),
        //     ],
        // )
        // .unwrap()
        (v0.to_owned(), v1.to_owned())
    }
}

#[pymethods]
impl Region {
    #[new]
    pub fn new_py<'py>(
        vertices: PyArrayLike2<'py, f64, AllowTypeChange>,
        edges: PyArrayLike2<'py, usize, AllowTypeChange>,
    ) -> Region {
        assert_eq!(vertices.shape(), edges.shape());

        let vertices = vertices.to_owned_array();
        let edges = edges.to_owned_array();

        Region::new(vertices, edges)
    }

    #[staticmethod]
    pub fn square(scale: Option<f64>) -> Region {
        let vertices = scale.unwrap_or(1.0)
            * 10.0
            * array![
                [0.00, 0.0000],
                [0.00, 0.0125],
                [0.00, 0.0250],
                [0.00, 0.0375],
                [0.00, 0.0500],
                [0.00, 0.0625],
                [0.00, 0.0750],
                [0.00, 0.0875],
                [0.0000, 0.10],
                [0.0125, 0.10],
                [0.0250, 0.10],
                [0.0375, 0.10],
                [0.0500, 0.10],
                [0.0625, 0.10],
                [0.0750, 0.10],
                [0.0875, 0.10],
                [0.10, 0.1000],
                [0.10, 0.0875],
                [0.10, 0.0750],
                [0.10, 0.0625],
                [0.10, 0.0500],
                [0.10, 0.0375],
                [0.10, 0.0250],
                [0.10, 0.0125],
                [0.1000, 0.00],
                [0.0875, 0.00],
                [0.0750, 0.00],
                [0.0625, 0.00],
                [0.0500, 0.00],
                [0.0375, 0.00],
                [0.0250, 0.00],
                [0.0125, 0.00]
            ];

        let edges = array![
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 8],
            [8, 9],
            [9, 10],
            [10, 11],
            [11, 12],
            [12, 13],
            [13, 14],
            [14, 15],
            [15, 16],
            [16, 17],
            [17, 18],
            [18, 19],
            [19, 20],
            [20, 21],
            [21, 22],
            [22, 23],
            [23, 24],
            [24, 25],
            [25, 26],
            [26, 27],
            [27, 28],
            [28, 29],
            [29, 30],
            [30, 31],
            [31, 0]
        ];

        Region::new(vertices, edges)
    }

    // #[pyo3(name = "edge")]
    // pub fn edge_py<'py>(
    //     &self,
    //     py: Python<'py>,
    //     edge: usize,
    // ) -> Bound<'py, (PyArray1<f64>, PyArray1<f64>)> {
    //     let (qa, qb) = self.edge(edge);
    //     PyTuple::(qa.to_pyarray(py), qb.to_pyarray(py))
    // }

    pub fn edges<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        let mut edges = Array3::zeros((self.len(), 2, 2));

        for i in 0..self.len() {
            let (qa, qb) = self.edge(i);
            edges.slice_mut(s![i, 0, ..]).assign(&qa);
            edges.slice_mut(s![i, 1, ..]).assign(&qb);
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

    /// Boundary conditions
    pub fn boundary_condition(&self) -> BoundaryCondition {
        BoundaryCondition::new(self.len())
    }

    pub fn neumann_boundary_condition(&self) -> BoundaryCondition {
        let mut bc = BoundaryCondition::new(self.len());
        bc.alpha.fill(0.0.into());
        bc.beta.fill(1.0.into());
        bc.f.fill(0.0.into());

        bc
    }

    pub fn dirichlet_boundary_condition(&self) -> BoundaryCondition {
        let mut bc = BoundaryCondition::new(self.len());
        bc.alpha.fill(1.0.into());
        bc.beta.fill(0.0.into());
        bc.f.fill(1.0.into());

        bc
    }

    /// Boundary incidence
    pub fn boundary_incidence(&self) -> BoundaryIncidence {
        BoundaryIncidence::new(self.len())
    }
}

/// Boundary conditions.
///
/// Robin boundary conditions:
///
/// alpha*phi + beta*v = f
///
/// Dirichlet:
///
/// alpha = 1
/// beta = 0
///
/// Neumann:
///
/// alpha = 0
/// beta = 1
#[pyclass]
#[derive(Clone, Debug)]
pub struct BoundaryCondition {
    pub alpha: Array1<Complex64>,
    pub beta: Array1<Complex64>,
    pub f: Array1<Complex64>,
}

#[pymethods]
impl BoundaryCondition {
    #[new]
    pub fn new(size: usize) -> BoundaryCondition {
        let alpha = Array1::zeros(size);
        let beta = Array1::zeros(size);
        let f = Array1::zeros(size);

        BoundaryCondition { alpha, beta, f }
    }

    #[getter]
    pub fn get_alpha<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray1<Complex64>> {
        let array = &this.borrow().alpha;
        unsafe { PyArray1::borrow_from_array(array, this.into_any()) }
    }

    #[getter]
    pub fn get_beta<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray1<Complex64>> {
        let array = &this.borrow().beta;
        unsafe { PyArray1::borrow_from_array(array, this.into_any()) }
    }

    #[getter]
    pub fn get_f<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray1<Complex64>> {
        let array = &this.borrow().f;
        unsafe { PyArray1::borrow_from_array(array, this.into_any()) }
    }

    pub fn len(&self) -> usize {
        self.f.len()
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct BoundaryIncidence {
    pub phi: Array1<Complex64>,
    pub v: Array1<Complex64>,
}

#[pymethods]
impl BoundaryIncidence {
    #[new]
    pub fn new(size: usize) -> BoundaryIncidence {
        let phi = Array1::zeros(size);
        let v = Array1::zeros(size);

        BoundaryIncidence { phi, v }
    }

    #[getter]
    pub fn get_phi<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray1<Complex64>> {
        let array = &this.borrow().phi;
        unsafe { PyArray1::borrow_from_array(array, this.into_any()) }
    }

    #[getter]
    pub fn get_v<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray1<Complex64>> {
        let array = &this.borrow().v;
        unsafe { PyArray1::borrow_from_array(array, this.into_any()) }
    }

    pub fn len(&self) -> usize {
        self.phi.len()
    }
}
