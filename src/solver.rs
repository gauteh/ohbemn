use std::f64::consts::PI;
use std::mem::swap;
use std::sync::Arc;

use ndarray::{
    array, concatenate, prelude::*, s, Array, Array1, Array2, Array3, ArrayView, ArrayView2, Axis,
    Dim,
};
use ndarray_linalg::{Norm, Solve};
use num::{complex::ComplexFloat, Complex};
use numpy::{
    Complex64, IntoPyArray, PyArray, PyArray1, PyArray2, PyArray3, PyArrayDyn, PyArrayMethods,
    PyReadonlyArray2, PyReadonlyArrayDyn, PyUntypedArrayMethods, ToPyArray,
};
use pyo3::prelude::*;

use crate::boundary::*;
use crate::geometry::normal_2d;
use crate::integrators::{self as int, l_2d};
use crate::{Orientation, A2, A2N};

#[pyclass]
#[derive(Debug, Clone)]
pub struct Solver {
    pub region: Region,
}

#[pymethods]
impl Solver {
    #[new]
    pub fn new(region: Region) -> Solver {
        Solver { region }
    }

    pub fn solve_boundary(
        &self,
        orientation: Orientation,
        k: f64,
        celerity: f64,
        boundary_condition: BoundaryCondition,
        boundary_incidence: BoundaryIncidence,
        mu: Option<Complex64>,
    ) -> BoundarySolution {
        let mu = mu.unwrap_or(Complex64::new(0., 1.) / Complex64::new(k + 1., 0.));

        assert_eq!(boundary_condition.len(), self.len());
        assert_eq!(boundary_incidence.len(), self.len());

        let (A, B) = self.compute_boundary_matrices(k, mu, orientation);

        let c = boundary_incidence.phi + mu * boundary_incidence.v;
        let c = match orientation {
            Orientation::Interior => c,
            Orientation::Exterior => -c,
        };

        let (phi, v) = solve_linear_equation(
            A,
            B,
            c,
            &boundary_condition.alpha,
            &boundary_condition.beta,
            &boundary_condition.f,
        );

        BoundarySolution::new(
            self.clone(),
            orientation,
            boundary_condition,
            k,
            celerity,
            phi,
            v,
        )
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

    pub fn solve_samples(
        &self,
        solution: &BoundarySolution,
        incident_phis: Array1<Complex64>,
        points: Array2<f64>,
        orientation: Orientation,
    ) -> Array1<Complex64> {
        assert_eq!(incident_phis.len(), points.shape()[0]);

        let N = incident_phis.len();

        let mut result = Array1::<Complex64>::zeros(N);

        for i in 0..N {
            let p = points.index_axis(Axis(0), i);
            let mut sum = incident_phis[i];

            for j in 0..solution.len() {
                let (qa, qb) = self.region.edge(j);

                let l = int::l_2d(solution.k, p, qa.view(), qb.view(), false);
                let m = int::m_2d(solution.k, p, qa.view(), qb.view(), false);

                match orientation {
                    Orientation::Interior => {
                        sum += l * solution.velocities[j] - m * solution.phis[j];
                    }
                    Orientation::Exterior => {
                        sum -= l * solution.velocities[j] - m * solution.phis[j];
                    }
                }
            }

            result[i] = sum;
        }

        result
    }
}

fn solve_linear_equation(
    mut A: Array2<Complex64>,
    mut B: Array2<Complex64>,
    mut c: Array1<Complex64>,
    alpha: &Array1<Complex64>,
    beta: &Array1<Complex64>,
    f: &Array1<Complex64>,
) -> (Array1<Complex64>, Array1<Complex64>) {
    let mut x = Array1::<Complex64>::zeros(c.len());
    let mut y = Array1::<Complex64>::zeros(c.len());

    let N = c.len();
    let gamma = B.norm_max() / A.norm_max();

    let mut swapXY = Array1::<bool>::default(c.len());
    for i in 0..N {
        if beta[i].abs() >= gamma * alpha[i].abs() {
            swapXY[i] = true;
        }
    }

    for i in 0..N {
        if swapXY[i] {
            for j in 0..alpha.len() {
                c[j] += f[i] * B[[j, i]] / beta[i];
                B[[j, i]] = -alpha[i] * B[[j, i]] / beta[i];
            }
        } else {
            for j in 0..alpha.len() {
                c[j] -= f[i] * A[[j, i]] / alpha[i];
                A[[j, i]] = -beta[i] * A[[j, i]] / alpha[i];
            }
        }
    }

    A = A - B;
    y = A.solve_into(c).unwrap();

    for i in 0..N {
        if swapXY[i] {
            x[i] = (f[i] - alpha[i] * y[i]) / beta[i];
        } else {
            x[i] = (f[i] - beta[i] * y[i]) / alpha[i];
        }
    }

    for i in 0..N {
        if swapXY[i] {
            swap(&mut x[i], &mut y[i]);
        }
    }

    (x, y)
}

#[pyclass]
#[derive(Clone)]
pub struct BoundarySolution {
    solver: Solver,
    k: f64,
    celerity: f64,
    orientation: Orientation,
    boundary_condition: BoundaryCondition,

    /// Phi at boundary elements.
    phis: Array1<Complex64>,

    /// Normal velocity at boundary elements.
    velocities: Array1<Complex64>,
}

impl BoundarySolution {
    pub fn new(
        solver: Solver,
        orientation: Orientation,
        bc: BoundaryCondition,
        k: f64,
        celerity: f64,
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

    pub fn len(&self) -> usize {
        self.phis.len()
    }

    pub fn solve_samples(
        &self,
        incident_phis: Array1<Complex64>,
        points: Array2<f64>,
    ) -> SampleSolution {
        let phis = self
            .solver
            .solve_samples(&self, incident_phis, points, self.orientation);
        SampleSolution::new(self.clone(), phis)
    }
}

#[pymethods]
impl BoundarySolution {}

#[pyclass]
pub struct SampleSolution {
    boundary_solution: BoundarySolution,
    phis: Array1<Complex64>,
}

impl SampleSolution {
    pub fn new(bs: BoundarySolution, phis: Array1<Complex64>) -> SampleSolution {
        SampleSolution {
            boundary_solution: bs,
            phis,
        }
    }
}
