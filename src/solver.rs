use std::mem::swap;

use ndarray::{par_azip, Array1, Array2, Axis};
use ndarray_linalg::{Norm, Solve};
use num::{complex::ComplexFloat, Complex};
use numpy::{AllowTypeChange, PyArrayLike1, PyArrayLike2};
use numpy::{Complex64, PyArray1, PyArrayMethods};
use pyo3::prelude::*;

use crate::boundary::*;
use crate::integrators as int;
use crate::Orientation;

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
            B,
            A,
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
                let mt = int::mt_2d(k, center, normal, qa.view(), qb.view(), i == j);
                let n = int::n_2d(k, center, normal, qa.view(), qb.view(), i == j);

                A[[i, j]] = l + mu * mt;
                B[[i, j]] = m + mu * n;
            }

            match orientation {
                Orientation::Interior => {
                    A[[i, i]] = A[[i, i]] - 0.5 * mu;
                    B[[i, i]] = B[[i, i]] + 0.5;
                }
                Orientation::Exterior => {
                    A[[i, i]] = A[[i, i]] + 0.5 * mu;
                    B[[i, i]] = B[[i, i]] - 0.5;
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

        par_azip!((sum in &mut result, p in points.outer_iter(), phi in &incident_phis) {
            *sum = *phi;

            for j in 0..solution.len() {
                let (qa, qb) = self.region.edge(j);

                let l = int::l_2d(solution.k, p, qa.view(), qb.view(), false);
                let m = int::m_2d(solution.k, p, qa.view(), qb.view(), false);

                match orientation {
                    Orientation::Interior => {
                        *sum += l * solution.velocities[j] - m * solution.phis[j];
                    }
                    Orientation::Exterior => {
                        *sum -= l * solution.velocities[j] - m * solution.phis[j];
                    }
                }
            }
        });

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

    let N = c.len();
    let gamma = B.norm_max() / A.norm_max();

    let mut swapXY = Array1::<bool>::default(c.len());
    for i in 0..N {
        if beta[i].abs() < (gamma * alpha[i].abs()) {
            swapXY[i] = false;
        } else {
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
    let mut y = A.solve(&c).unwrap();

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
}

impl BoundarySolution {
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
impl BoundarySolution {
    pub fn len(&self) -> usize {
        self.phis.len()
    }

    #[pyo3(name = "solve_samples")]
    pub fn solve_samples_py<'py>(
        &self,
        incident_phis: PyArrayLike1<'py, Complex<f64>, AllowTypeChange>,
        points: PyArrayLike2<'py, f64, AllowTypeChange>,
    ) -> SampleSolution {
        let incident_phis = incident_phis.to_owned_array();
        let points = points.to_owned_array();
        self.solve_samples(incident_phis, points)
    }

    #[getter]
    pub fn get_phis<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray1<Complex64>> {
        let array = &this.borrow().phis;
        unsafe { PyArray1::borrow_from_array(array, this.into_any()) }
    }

    #[getter]
    pub fn get_velocities<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray1<Complex64>> {
        let array = &this.borrow().velocities;
        unsafe { PyArray1::borrow_from_array(array, this.into_any()) }
    }

    // /// $\eta$ at boundary elements.
    // pub fn eta(&self) -> Array1<Complex64> {
    //     unimplemented!()
    // }
}

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

#[pymethods]
impl SampleSolution {
    #[getter]
    pub fn get_phis<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray1<Complex64>> {
        let array = &this.borrow().phis;
        unsafe { PyArray1::borrow_from_array(array, this.into_any()) }
    }

    // #[getter]
    // pub fn get_velocities<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray1<Complex64>> {
    //     let array = &this.borrow().velocities;
    //     unsafe { PyArray1::borrow_from_array(array, this.into_any()) }
    // }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, azip, s};
    use num::complex::c64;
    use std::f64::consts::PI;

    #[test]
    fn test_internal_2d_1() {
        let I = Complex64::I;
        let frequency = 400.0;
        let k = 2.0 * PI * frequency / 344.0;

        let region = Region::square(Some(0.1));
        let mut bc = region.dirichlet_boundary_condition();

        let f = (k / f64::sqrt(2.0) * region.centers().slice(s![.., 0]).to_owned())
            .mapv(|v| c64(v.sin(), 0.0))
            * (k / f64::sqrt(2.0) * region.centers().slice(s![.., 1]).to_owned())
                .mapv(|v| c64(v.sin(), 0.0));

        bc.f.slice_mut(s![..]).assign(&f);

        let bi = region.boundary_incidence();

        let solver = Solver::new(region);
        let bs = solver.solve_boundary(Orientation::Interior, k, 344.0, bc, bi, None);

        println!("{}", bs.phis);

        let expected = array![
            0.0000e00 + 0.0000e00 * I,
            0.0000e00 + 0.0000e00 * I,
            0.0000e00 + 0.0000e00 * I,
            0.0000e00 + 0.0000e00 * I,
            0.0000e00 + 0.0000e00 * I,
            0.0000e00 + 0.0000e00 * I,
            0.0000e00 + 0.0000e00 * I,
            0.0000e00 + 0.0000e00 * I,
            0.1595e-01 + 0.0000e00 * I,
            0.4777e-01 + 0.0000e00 * I,
            0.7940e-01 + 0.0000e00 * I,
            0.1107e00 + 0.0000e00 * I,
            0.1415e00 + 0.0000e00 * I,
            0.1718e00 + 0.0000e00 * I,
            0.2013e00 + 0.0000e00 * I,
            0.2300e00 + 0.0000e00 * I,
            0.2300e00 + 0.0000e00 * I,
            0.2013e00 + 0.0000e00 * I,
            0.1718e00 + 0.0000e00 * I,
            0.1415e00 + 0.0000e00 * I,
            0.1107e00 + 0.0000e00 * I,
            0.7940e-01 + 0.0000e00 * I,
            0.4777e-01 + 0.0000e00 * I,
            0.1595e-01 + 0.0000e00 * I,
            0.0000e00 + 0.0000e00 * I,
            0.0000e00 + 0.0000e00 * I,
            0.0000e00 + 0.0000e00 * I,
            0.0000e00 + 0.0000e00 * I,
            0.0000e00 + 0.0000e00 * I,
            0.0000e00 + 0.0000e00 * I,
            0.0000e00 + 0.0000e00 * I,
            0.0000e00 + 0.0000e00 * I,
        ];

        azip!((a in &bs.phis, b in &expected) {
            println!("a={a} == b={b}");
            approx::assert_abs_diff_eq!(a, b, epsilon = 1e-4);
        });

        let points = array![
            [0.0250, 0.0250],
            [0.0750, 0.0250],
            [0.0250, 0.0750],
            [0.0750, 0.0750],
            [0.0500, 0.0500],
        ];
        let incident = Array1::zeros(points.shape()[0]);

        let samples = bs.solve_samples(incident, points);

        let expected = array![
            0.1589e-01 + 0.1183e-03 * I,
            0.4818e-01 + 0.4001e-04 * I,
            0.4818e-01 + 0.4001e-04 * I,
            0.1434e00 - 0.2577e-03 * I,
            0.6499e-01 - 0.1422e-04 * I,
        ];

        println!("{}", samples.phis);
        azip!((a in &samples.phis, b in &expected) {
            println!("a={a} == b={b}");
            approx::assert_abs_diff_eq!(a, b, epsilon = 1e-4);
        });
    }
}
