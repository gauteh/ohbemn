use std::f64::consts::PI;
use std::sync::Arc;

use ndarray::{
    array, concatenate, s, Array, Array1, Array2, Array3, ArrayView, ArrayView2, Axis, Dim,
};
use ndarray_linalg::Norm;
use num::Complex;
use numpy::{
    Complex64, IntoPyArray, PyArray, PyArray1, PyArray2, PyArray3, PyArrayDyn, PyArrayMethods,
    PyReadonlyArray2, PyReadonlyArrayDyn, PyUntypedArrayMethods, ToPyArray,
};
use pyo3::prelude::*;

use crate::boundary::*;
use crate::geometry::normal_2d;
use crate::integrators as int;
use crate::{Orientation, A2, A2N};

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
    ) -> (Array2<Complex64>, Array2<Complex64>) {
        let mut A = Array2::<Complex64>::zeros((self.len(), self.len()));
        let mut B = Array2::<Complex64>::zeros((self.len(), self.len()));

        let centers = self.region.centers();
        let normals = self.region.normals();

        for i in 0..self.len() {
            let center = centers.index_axis(Axis(0), i);
            let normal = normals.index_axis(Axis(0), i);

            for j in 0..self.len() {
                let (qa, qb) = self.region.edge(j);

                let l = int::l_2d(k, center, qa.view(), qb.view(), i == j);
                let m = int::m_2d(k, center, qa.view(), qb.view(), i == j);
                let mt = int::mt_2d(k, center, qa.view(), qb.view(), i == j);
                let n = int::n_2d(k, center, normal, qa.view(), qb.view(), i == j);

                let a = l + mu * mt;
                let b = m + mu * n;

                A[[i, j]] = a;
                B[[i, j]] = b;
            }

            match orientation {
                Orientation::Interior => {
                    let a = A[[i, i]] - 0.5 * mu;
                    let b = B[[i, i]] + 0.5;

                    A[[i, i]] = a;
                    B[[i, i]] = b;
                }
                Orientation::Exterior => {
                    let a = A[[i, i]] + 0.5 * mu;
                    let b = B[[i, i]] - 0.5;

                    A[[i, i]] = a;
                    B[[i, i]] = b;
                }
            }
        }

        (A, B)
    }

    pub fn solve_boundary(
        self: Arc<Self>,
        orientation: Orientation,
        k: f32,
        celerity: f32,
        boundary_condition: BoundaryCondition,
        boundary_incidence: BoundaryIncidence,
        mu: Option<Complex64>,
    ) -> BoundarySolution {
        unimplemented!()
    }
}

#[pyclass]
pub struct BoundarySolution {
    solver: Arc<Solver>,
    k: f32,
    celerity: f32,
    orientation: Orientation,
    boundary_condition: BoundaryCondition,

    /// Phi at boundary elements.
    phis: Array1<Complex64>,

    /// Normal velocity at boundary elements.
    velocities: Array1<Complex64>,
}

impl BoundarySolution {
    pub fn new(
        solver: Arc<Solver>,
        orientation: Orientation,
        bc: BoundaryCondition,
        k: f32,
        celerity: f32,
        phis: Array1<Complex64>,
        velocities: Array1<Complex64>,
    ) -> BoundarySolution {
        assert_eq!(phis.len(), velocities.len());

        BoundarySolution {
            solver,
            orientation,
            boundary_condition: bc,
            k,
            celerity,
            phis,
            velocities,
        }
    }

    /// $\eta$ at boundary elements.
    pub fn eta(&self) -> Array1<Complex64> {
        unimplemented!()
    }

    pub fn solve_samples(
        self: Arc<BoundarySolution>,
        incident_phi: Array1<Complex64>,
        points: Array2<f64>,
    ) -> SampleSolution {
        unimplemented!()
    }
}

#[pymethods]
impl BoundarySolution {}

#[pyclass]
pub struct SampleSolution {
    boundary_solution: Arc<BoundarySolution>,
    phis: Array1<Complex64>,
}

impl SampleSolution {
    pub fn new(bs: Arc<BoundarySolution>, phis: Array1<Complex64>) -> SampleSolution {
        SampleSolution {
            boundary_solution: bs,
            phis,
        }
    }
}
