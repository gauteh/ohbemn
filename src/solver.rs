use std::f64::consts::PI;

use ndarray::{Array, Array1, ArrayView, Dim};
use ndarray_linalg::Norm;
use num::Complex;
use numpy::{Complex64, IntoPyArray, PyArray, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn};
use pyo3::prelude::*;

use crate::geometry::normal_2d;
use crate::A2;
