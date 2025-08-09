use std::collections::HashSet;
use std::sync::atomic::AtomicUsize;
use anyhow::Error;
use ndarray::Array2;
use ndarray_rand::RandomExt;
use ndarray_rand::rand::distributions::Uniform;

pub fn interpolated_range(a: f32, b: f32, r: usize) -> Vec<f32> {
    assert!(r >= 2);

    let step = (b - a) / (r - 1) as f32;
    (0..r).map(|i| a + i as f32 * step).collect()
}

pub fn apply_epsilon(h: &mut Array2<f64>, epsilon: f64) {
    for val in h.iter_mut() {
        if *val < epsilon {
            *val = 0.0
        }
    }
}

pub fn normalize_to_global(array: &mut Array2<f64>) {
    let max_val = array.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    if max_val > 0.0 {
        for val in array.iter_mut() {
            *val /= max_val;
        }
    }
}

pub fn matrix_from_vecs(matrix_vec: Vec<Vec<f64>>) -> Result<Array2<f64>, Error> {
    let flat_vec: Vec<f64> = matrix_vec.clone().into_iter().flatten().collect();

    let rows = matrix_vec.len();
    let cols = if rows > 0 { matrix_vec[0].len() } else { 0 };
    let shape = (rows, cols);

    return Ok(ndarray::ArrayView2::from_shape(shape, &flat_vec)?
            .to_owned());
}

/// data is V, dimensioned (m, n)
/// basis is W, dimensioned (m, r)
/// return value is h, dimensioned (r, n)
/// 
/// see update rule for PGD NNLS in:
/// https://angms.science/doc/NMF/nnls_pgd.pdf
/// description of NNLS for quadratic programming:
/// https://en.wikipedia.org/wiki/Non-negative_least_squares
/// 
/// PGD NNLS update rule: hk = [hk−1 − t(Qhk−1 − p)]
/// objective (quadratic form NNLS):
/// min (x>=0) (1/2)(x^T Q * x - 2p^T x)
///
/// our objective is:
/// min (H>=0) ||(1/2)(Wh-V)||2_F <-- frob norm
///
/// some math later (chatgpt), you get that subbing
/// Q = W^T W and p = W^T V
/// to the original gives
/// h <- h - t(W^T Wh - W^T V)
///
/// you can calculate the gradient above without explicitly storing
/// W^T W or W^T V by doing W^T(Wh-V) which is equivalent via
/// distribution, saving precious memory. lovely!
pub fn pgd_nnls(
    data: &Array2<f64>,
    basis: &Array2<f64>,
    iters: usize,
    step: f64,
) -> Array2<f64> {
    let (m1, n) = data.dim();
    let (m2, r) = basis.dim();

    assert_eq!(m1, m2);

    let mut h = Array2::<f64>::zeros((r, n));

    let wt = basis.t();

    for i in 0..iters {
        let wh = basis.dot(&h);
        let grad = wt.dot(&(wh - data));
        h = &h - &(grad * step);
        h.mapv_inplace(|x| x.max(0.0));
        println!("{}", i);
    }

    h
}
