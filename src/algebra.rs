use std::time::Instant;

use anyhow::Error;
use ndarray::{Array2, ArrayView2};
use ocl::{Buffer, ProQue};

static KERNEL: &str = include_str!("pgd.ocl");

pub fn interpolated_range(a: f32, b: f32, r: usize) -> Vec<f32> {
    assert!(r >= 2);

    let step = (b - a) / (r - 1) as f32;
    (0..r).map(|i| a + i as f32 * step).collect()
}

pub fn round_to(h: &mut Array2<f32>, decimals: usize) {
    for val in h.iter_mut() {
        *val = (*val * 10f32.powi(decimals as i32)).round() / 10f32.powi(decimals as i32)
    }
}

pub fn normalize_to_minus_plus(array: &mut Array2<f32>) {
    let min_val = array.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = array.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let range = max_val - min_val;

    if range > 0.0 {
        for val in array.iter_mut() {
            *val = 2.0 * (*val - min_val) / range - 1.0;
        }
    } else {
        for val in array.iter_mut() {
            *val = 0.0;
        }
    }
}

pub fn normalize_to_global(array: &mut Array2<f32>) {
    let max_val = array.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    if max_val > 0.0 {
        for val in array.iter_mut() {
            *val /= max_val;
        }
    }
}

pub fn dynamic_range(array: &mut Array2<f32>, gamma: f32) {
    for x in array.iter_mut() {
        *x = x.powf(gamma);
    }
}

pub fn matrix_from_vecs(matrix_vec: Vec<Vec<f32>>) -> Result<Array2<f32>, Error> {
    let flat_vec: Vec<f32> = matrix_vec.clone().into_iter().flatten().collect();

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
pub fn cpu_pgd_nnls(
    data: ArrayView2<f32>,
    basis: ArrayView2<f32>,
    iters: usize,
    step: f32,
) -> Array2<f32> {
    let (m1, n) = data.dim();
    let (m2, r) = basis.dim();

    assert_eq!(m1, m2);

    let mut h = Array2::<f32>::zeros((r, n));

    let wt = basis.t();

    for i in 0..iters {
        let start = Instant::now();
        let wh = basis.dot(&h);
        let grad = wt.dot(&(wh - data));
        h = &h - &(grad * step);
        h.mapv_inplace(|x| x.max(0.0));
        println!("iter {}, elapsed: {}s", i, start.elapsed().as_secs());
    }

    h
}

pub fn pgd_nnls(
    data: ArrayView2<f32>,
    basis: ArrayView2<f32>,
    iters: usize,
    step: f32,
) -> Array2<f32> {
    let (m1, n) = data.dim();
    let (m2, r) = basis.dim();

    assert_eq!(m1, m2);

    // row-major
    let v: Vec<f32> = data.iter().cloned().collect();
    let w: Vec<f32> = basis.iter().cloned().collect();
    let mut h: Vec<f32> = vec![0.0; r * n];

    let mut w_t = vec![0.0f32; r * m1];
    for i in 0..m1 {
        for j in 0..r {
            w_t[j * m1 + i] = basis[(i, j)];
        }
    }

    let pq = ProQue::builder()
        .src(KERNEL)
        .dims((r.max(m1), n))
        .build()
        .unwrap();

    let buffer_w = Buffer::<f32>::builder()
        .queue(pq.queue().clone())
        .flags(ocl::flags::MEM_READ_ONLY)
        .len(w.len())
        .copy_host_slice(&w)
        .build()
        .unwrap();

    let buffer_w_t = Buffer::<f32>::builder()
        .queue(pq.queue().clone())
        .flags(ocl::flags::MEM_READ_ONLY)
        .len(w_t.len())
        .copy_host_slice(&w_t)
        .build()
        .unwrap();

    let buffer_v = Buffer::<f32>::builder()
        .queue(pq.queue().clone())
        .flags(ocl::flags::MEM_READ_ONLY)
        .len(v.len())
        .copy_host_slice(&v)
        .build()
        .unwrap();

    let buffer_h = Buffer::<f32>::builder()
        .queue(pq.queue().clone())
        .len(h.len())
        .copy_host_slice(&h)
        .build()
        .unwrap();

    let buffer_whv = Buffer::<f32>::builder()
        .queue(pq.queue().clone())
        .len(m1 * n)
        .build()
        .unwrap();

    let buffer_grad = Buffer::<f32>::builder()
        .queue(pq.queue().clone())
        .len(r * n)
        .build()
        .unwrap();

    let k_whv = pq.kernel_builder("gemm_whv")
        .global_work_size((m1, n))
        .arg(&buffer_w)
        .arg(&buffer_h)
        .arg(&buffer_v)
        .arg(&buffer_whv)
        .arg(m1 as u32)
        .arg(n as u32)
        .arg(r as u32)
        .build()
        .unwrap();

    let k_grad = pq.kernel_builder("gemm_grad")
        .global_work_size((r, n))
        .arg(&buffer_w_t)
        .arg(&buffer_whv)
        .arg(&buffer_grad)
        .arg(r as u32)
        .arg(n as u32)
        .arg(m1 as u32)
        .build()
        .unwrap();

    let k_update = pq.kernel_builder("update_h")
        .global_work_size((r, n))
        .arg(&buffer_h)
        .arg(&buffer_grad)
        .arg(step)
        .arg(r as u32)
        .arg(n as u32)
        .build()
        .unwrap();

    for i in 0..iters {
        let start = Instant::now();
        unsafe { k_whv.enq().unwrap(); }
        pq.finish().unwrap();
        println!("whv done: {}ms", start.elapsed().as_millis());
        let start = Instant::now();
        unsafe { k_grad.enq().unwrap(); }
        pq.finish().unwrap();
        println!("grad: {}ms", start.elapsed().as_millis());
        let start = Instant::now();
        unsafe { k_update.enq().unwrap(); }
        pq.finish().unwrap();
        println!("update: {}ms", start.elapsed().as_millis());
        println!("iter {}, elapsed: {}ms", i, start.elapsed().as_millis());
    }

    println!("reading...");
    buffer_h.read(&mut h).enq().unwrap();

    println!("read! cpu");
    Array2::from_shape_vec((r, n), h).unwrap()
}

