use std::sync::LazyLock;

use pyo3::prelude::*;

#[pyfunction]
pub fn complex_quad() {
    use gauss_quad::GaussLegendre;

    static G: LazyLock<GaussLegendre> = LazyLock::new(|| GaussLegendre::new(8).unwrap());

    unimplemented!()
}
