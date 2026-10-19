//! B0 retest: burn 0.20.1 / cubecl 0.9 on a Pascal GPU.
//! `matmul` is batched [2, 128, 2048] x [2, 2048, 2048].
//! `topk` is burn's `topk_with_indices` (sort + select in this release).

use burn_cuda::{Cuda, CudaDevice};
use burn_tensor::{Distribution, Int, Tensor};

type B = Cuda<f32, i32>;

fn main() {
    let op = std::env::args().nth(1).unwrap_or_else(|| "matmul".to_string());
    println!(
        "op={op} burn-cuda=0.20.1 cubecl=0.9 CUDA_VISIBLE_DEVICES={} CUBECL_AUTOTUNE_LEVEL={}",
        std::env::var("CUDA_VISIBLE_DEVICES").unwrap_or_else(|_| "(unset)".into()),
        std::env::var("CUBECL_AUTOTUNE_LEVEL").unwrap_or_else(|_| "(unset)".into()),
    );
    let device = CudaDevice::default();
    println!("device={device:?}");
    match op.as_str() {
        "matmul" => matmul(&device),
        "topk" => topk(&device),
        other => {
            eprintln!("unknown op {other} (expected matmul or topk)");
            std::process::exit(2);
        }
    }
}

fn matmul(device: &CudaDevice) {
    let lhs = Tensor::<B, 3>::random([2, 128, 2048], Distribution::Default, device);
    let rhs = Tensor::<B, 3>::random([2, 2048, 2048], Distribution::Default, device);
    let out = lhs.matmul(rhs);
    let dims = out.dims();
    let data = out.to_data();
    let values = data
        .as_slice::<f32>()
        .unwrap_or_else(|err| panic!("matmul to_data was not f32: {err:?}"));
    println!(
        "matmul PASS dims={dims:?} numel={} first={}",
        values.len(),
        values.first().copied().unwrap_or(f32::NAN)
    );
}

fn topk(device: &CudaDevice) {
    let scores = Tensor::<B, 2>::random([2, 128], Distribution::Default, device);
    let (vals, idx) = scores.topk_with_indices(8, 1);
    let v = vals.to_data();
    let i = idx.to_data();
    let vf = v
        .as_slice::<f32>()
        .unwrap_or_else(|err| panic!("topk values were not f32: {err:?}"));
    let ii = i
        .as_slice::<i32>()
        .unwrap_or_else(|err| panic!("topk indices were not i32: {err:?}"));
    println!(
        "topk PASS nvals={} nidx={} idx0={}",
        vf.len(),
        ii.len(),
        ii.first().copied().unwrap_or(-1)
    );
}
