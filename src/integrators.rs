use std::sync::LazyLock;

use gauss_quad::GaussLegendre;
use ndarray::{Array, Array1, Dim};
use ndarray_linalg::{normalize, Norm, NormalizeAxis as Axis};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn, PyArrayMethods};
use num::Complex;
use pyo3::prelude::*;
use scilib::math::bessel::h1_nu;

pub type A2 = Array<f64, Dim<[usize; 2]>>;

static quad: LazyLock<GaussLegendre> = LazyLock::new(|| GaussLegendre::new(8).unwrap());

pub fn complex_quad_2d<F>(a: A2, b: A2, integrand: F) -> Complex<f64>
where
    F: Fn(A2) -> Complex<f64>,
{
    let v = b - a.clone();

    // quad.as_node_weight_pairs()
    //     .iter()
    //     .map(|(x_val, w_val)| {
    //         // integrand(Self::argument_transformation(*x_val, a, b)) * w_val)
    //         let x = a.clone() + *x_val * v.clone();
    //         integrand(x) * w_val
    //     })
    //     .sum()

    let nw = [
        [0.980144928249, 5.061426814519e-02],
        [0.898333238707, 0.111190517227],
        [0.762766204958, 0.156853322939],
        [0.591717321248, 0.181341891689],
        [0.408282678752, 0.181341891689],
        [0.237233795042, 0.156853322939],
        [0.101666761293, 0.111190517227],
        [1.985507175123e-02, 5.061426814519e-02],
    ];

    nw.iter()
        .map(|nw| {
            let x_val = nw[0];
            let w_val = nw[1];
            let x = a.clone() + x_val * v.clone();
            integrand(x) * w_val
        })
        .sum()
}
// def l_2d(k, p, qa, qb, p_on_element):
//     qab = qb - qa
//     if p_on_element:
//         if k == 0.0:
//             ra = norm(p - qa)
//             rb = norm(p - qb)
//             re = norm(qab)
//             result = 0.5 / np.pi * (re - (ra * np.log(ra) + rb * np.log(rb)))
//         else:

//             def func(x):
//                 R = norm(p - x)
//                 return 0.5 / np.pi * np.log(R) + 0.25j * hankel1(0, k * R)

//             result = (complex_quad_2d(func, qa, p) +
//                       complex_quad_2d(func, p, qb) +
//                       l_2d(0.0, p, qa, qb, True))
//     else:
//         if k == 0.0:
//             result = -0.5 / np.pi * complex_quad_2d(
//                 lambda q: np.log(norm(p - q)), qa, qb)
//         else:
//             result = 0.25j * complex_quad_2d(
//                 lambda q: hankel1(0, k * norm(p - q)), qa, qb)

//     return result

/// Compute L matrix.
///
/// k: wave number
/// p: point (center of edge)
/// qa: first vertex of edge
/// qb: second vertex of edge
pub fn l_2d(k: f64, p: A2, qa: A2, qb: A2, p_on_element: bool) {
    assert!(k > 0.0, "wavenumber==0 not supported");

    // TODO: Possibly put in a different function.
    if p_on_element {
        let int = |x: A2| -> Complex<f64> {
            let R = (&p - x).norm_l2();
            return 0.5 * std::f64::consts::PI * f64::log2(R)
                + Complex::new(0.0, 0.25) * h1_nu(0.0, Complex::new(k, 0.0) * R);
        };

        // complex quad
        complex_quad_2d(qa, p.clone(), int);
    } else {
        let int = |x: Array1<f64>| -> Complex<f64> {
            let R = (p - x).norm_l2();
            return h1_nu(0.0, Complex::new(k, 0.0) * R);
        };

        // complex quad
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quad_node_weights() {
        // [
        //     [0.980144928249, 5.061426814519e-02],
        //     [0.898333238707, 0.111190517227],
        //     [0.762766204958, 0.156853322939],
        //     [0.591717321248, 0.181341891689],
        //     [0.408282678752, 0.181341891689],
        //     [0.237233795042, 0.156853322939],
        //     [0.101666761293, 0.111190517227],
        //     [1.985507175123e-02, 5.061426814519e-02],
        // ],

        for (n, w) in quad.as_node_weight_pairs() {
            println!("{n}: {w}");
        }
    }
}
