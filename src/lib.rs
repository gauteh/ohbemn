use pyo3::prelude::*;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn ohbemn(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}

#[pyfunction]
fn complex_quad() -> PyResult<f32> {
    x = [
        0.980144928249,
        0.898333238707,
        0.762766204958,
        0.591717321248,
        0.408282678752,
        0.237233795042,
        0.101666761293,
        1.985507175123e-2,
    ];

    w = [
        5.061426814519e-02,
        0.111190517227,
        0.156853322939,
        0.181341891689,
        0.181341891689,
        0.156853322939,
        0.111190517227,
        5.061426814519e-02,
    ];

    unimplemented!()
}
