use std::f64::consts::PI;
use std::sync::LazyLock;

use gauss_quad::GaussLegendre;
use ndarray::{Array, Array1, ArrayView, Dim};
use ndarray_linalg::{normalize, Norm, NormalizeAxis as Axis};
use num::Complex;
use numpy::{Complex64, IntoPyArray, PyArray, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn};
use pyo3::prelude::*;

use crate::geometry::normal_2d;
use crate::A2;

static quad: LazyLock<GaussLegendre> = LazyLock::new(|| GaussLegendre::new(8).unwrap());

// h1_nu from scilab is inaccurate.
pub fn hankel1(order: f64, z: Complex64) -> Complex64 {
    use complex_bessel_rs::bessel_j::bessel_j;
    use complex_bessel_rs::bessel_y::bessel_y;

    bessel_j(order, z).unwrap_or(Complex64::new(f64::NAN, f64::NAN))
        + Complex64::new(0.0, 1.0)
            * bessel_y(order, z).unwrap_or(Complex64::new(f64::NAN, f64::NAN))
}

pub fn complex_quad_2d<F>(a: &A2, b: &A2, integrand: F) -> Complex<f64>
where
    F: Fn(A2) -> Complex<f64>,
{
    let v = b - a;

    // not sure how these are computed. does not match weights from quad-gauss.
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
            let x = a + x_val * v.clone();
            let i = integrand(x.view()) * w_val;

            // println!("{x}, {w_val}, {i}");

            i
        })
        .sum::<Complex64>()
        * v.norm_l2()
}

/// Compute L matrix.
///
/// k: wave number
/// p: point (center of edge)
/// qa: first vertex of edge
/// qb: second vertex of edge
pub fn l_2d(k: f64, p: A2, qa: A2, qb: A2, p_on_element: bool) -> Complex<f64> {
    assert!(k > 0.0, "wavenumber==0 not supported");

    // TODO: Possibly put in a different function.
    if p_on_element {
        let int = |x: A2| -> Complex<f64> {
            let R = (p.to_owned() - x).norm_l2();
            return 0.5 / PI * f64::log2(R)
                + Complex::new(0.0, 0.25) * hankel1(0.0, Complex::new(k, 0.0) * R);
        };

        let ra = (&p - &qa).norm_l2();
        let rb = (&p - &qb).norm_l2();
        let re = (&qb - &qa).norm_l2();
        let l = 0.5 / PI * (re - (ra * f64::log2(ra) + rb * f64::log2(rb)));

        // complex quad
        complex_quad_2d(&qa, &p, int) + complex_quad_2d(&p, &qb, int) + l
    } else {
        // complex quad
        Complex::new(0.0, 0.25)
            * complex_quad_2d(&qa, &qb, |q: A2| {
                hankel1(0.0, Complex::new(k, 0.0) * (p.to_owned() - q).norm_l2())
            })
    }
}

/// Compute M matrix.
///
/// k: wave number
/// p: point (center of edge)
/// qa: first vertex of edge
/// qb: second vertex of edge
pub fn m_2d(k: f64, p: A2, qa: A2, qb: A2, p_on_element: bool) -> Complex<f64> {
    assert!(k > 0.0, "wavenumber==0 not supported");

    let vecq = normal_2d(qa, qb);

    if p_on_element {
        Complex::new(0.0, 0.0)
    } else {
        let int = |x: A2| {
            let r = p.to_owned() - x;
            let R = r.norm_l2();
            return hankel1(1.0, Complex::new(k, 0.) * R) * r.dot(&vecq) / R;
        };

        Complex::new(0., 0.25) * k * complex_quad_2d(&qa, &qb, int)
    }
}

/// Compute Mt matrix.
///
/// k: wave number
/// p: point (center of edge)
/// qa: first vertex of edge
/// qb: second vertex of edge
pub fn mt_2d(k: f64, p: A2, qa: A2, qb: A2, p_on_element: bool) -> Complex<f64> {
    assert!(k > 0.0, "wavenumber==0 not supported");

    let vecq = normal_2d(qa, qb);

    if p_on_element {
        Complex::new(0.0, 0.0)
    } else {
        let int = |x: A2| {
            let r = p.to_owned() - x;
            let R = r.norm_l2();
            return hankel1(1.0, Complex::new(k, 0.) * R) * r.dot(&vecq) / R;
        };

        Complex::new(0., -0.25) * k * complex_quad_2d(&qa, &qb, int)
    }
}

/// Compute Mt matrix.
///
/// k: wave number
/// p: point (center of edge)
/// qa: first vertex of edge
/// qb: second vertex of edge
pub fn n_2d(k: f64, p: A2, vecp: A2, qa: A2, qb: A2, p_on_element: bool) -> Complex<f64> {
    assert!(k > 0.0, "wavenumber==0 not supported");

    let qab = qb.to_owned() - qa;
    let vecq = normal_2d(qa, qb);

    if p_on_element {
        let ra = (p.to_owned() - qa).norm_l2();
        let rb = (p.to_owned() - qb).norm_l2();

        let ksq = k * k;

        let int = |x: A2| {
            let r = p.to_owned() - x;
            let R2 = r.dot(&r);
            let R = R2.sqrt();

            let drdudrdn = -r.dot(&vecq) * r.dot(&vecp) / R2;
            let dpnu = vecp.dot(&vecq);
            let hkr = hankel1(1., Complex::new(k, 0.) * R);

            let c1 = Complex::new(0., 0.25) * k / R * hkr - 0.5 / (PI * R2);

            let c2 = Complex::new(0., 0.5) * k / R * hkr
                - Complex::new(0., 0.25) * ksq * hankel1(0., Complex::new(k, 0.) * R)
                - 1.0 / (PI * R2);

            let c3 = -0.25 * ksq * R.log2() / PI;

            return c1 * dpnu + c2 * drdudrdn + c3;
        };

        let n_0 = -(1.0 / ra + 1.0 / rb) / (2.0 * PI);

        let ra = (&p - &qa).norm_l2();
        let rb = (&p - &qb).norm_l2();
        let re = (&qb - &qa).norm_l2();
        let l_0 = 0.5 / PI * (re - (ra * f64::log2(ra) + rb * f64::log2(rb)));
        let l_0 = -0.5 * ksq * l_0;

        let i = complex_quad_2d(&qa, &p, int) + complex_quad_2d(&p, &qb, int);

        return i + n_0 + l_0;
    } else {
        let un = vecp.dot(&vecq);

        let int = |x: A2| {
            let r = p.to_owned() - x;
            let R2 = r.dot(&r);
            let drdudrdn = -r.dot(&vecq) * r.dot(&vecp) / R2;
            let R = r.norm_l2();

            let k = Complex::new(k, 0.);
            return hankel1(1., k * R) / R * (un + 2. * drdudrdn)
                - k * hankel1(0., k * R) * drdudrdn;
        };

        return Complex::new(0., 0.25) * k * complex_quad_2d(&qa, &qb, int);
    }
}

#[pyfunction]
#[pyo3(name = "l_2d")]
pub fn l_2d_py<'py>(
    py: Python<'py>,
    k: f64,
    p: PyReadonlyArrayDyn<'py, f64>,
    qa: PyReadonlyArrayDyn<'py, f64>,
    qb: PyReadonlyArrayDyn<'py, f64>,
    p_on_element: bool,
) -> Complex64 {
    let p = p.as_array().into_shape((2,)).unwrap();
    let qa = qa.as_array().into_shape((2,)).unwrap();
    let qb = qb.as_array().into_shape((2,)).unwrap();

    l_2d(k, p, qa, qb, p_on_element)
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

    #[test]
    fn test_hankel1() {
        for j in ndarray::Array::range(0.0, 10.0, 1.0) {
            // let h = h1_nu(0.0, Complex::new(10., 0.0) * j);
            let h2 = hankel1(0.0, Complex::new(10., 0.0) * j);
            // println!("{j} = {h:?}");
            println!("{j} = {h2:?}");
        }
    }

    #[test]
    fn test_complex_quad() {
        use ndarray::array;

        let k = 10.;
        let qa = array![0.0, 0.0].into_shape((2,)).unwrap();
        let qb = array![10.0, 10.0].into_shape((2,)).unwrap();
        let p = array![1.0, 1.0].into_shape((2,)).unwrap();

        println!("h1 = {:?}", hankel1(0.0, Complex::new(5., 0.) * 3.));

        let r = complex_quad_2d(&qa.view(), &qb.view(), |q: A2| {
            dbg!(&q);
            dbg!(&p);
            let h = hankel1(0.0, Complex::new(k, 0.0) * (p.to_owned() - q).norm_l2());
            dbg!(&h);
            h
        });
        dbg!(r);
    }
}
