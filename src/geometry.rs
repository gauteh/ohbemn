use ndarray::array;
use ndarray_linalg::Norm;

use crate::{A2, A2O};

/// Calculate the normal vector to a line from qa to qb.
pub fn normal_2d(qa: A2, qb: A2) -> A2O {
    let diff = qa.to_owned() - qb;
    let L = diff.norm_l2();
    return array![diff[1] / L, -diff[0] / L];
}
