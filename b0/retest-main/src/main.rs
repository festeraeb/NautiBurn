//! B0 retest on burn main (cubecl pin in that checkout).
//! Same shapes as the 0.20.1 binary. Set BURN_DEVICE=cuda.

use burn::tensor::{Distribution, Tensor};

fn main() {
    let op = std::env::args().nth(1).unwrap_or_else(|| "matmul".to_string());
    println!(
        "op={op} burn=main CUDA_VISIBLE_DEVICES={} BURN_DEVICE={} CUBECL_AUTOTUNE_LEVEL={}",
        std::env::var("CUDA_VISIBLE_DEVICES").unwrap_or_else(|_| "(unset)".into()),
        std::env::var("BURN_DEVICE").unwrap_or_else(|_| "(unset)".into()),
        std::env::var("CUBECL_AUTOTUNE_LEVEL").unwrap_or_else(|_| "(unset)".into()),
    );
    match op.as_str() {
        "matmul" => matmul(),
        "topk" => topk(),
        other => {
            eprintln!("unknown op {other}");
            std::process::exit(2);
        }
    }
}

fn matmul() {
    let device = Default::default();
    println!("device={device:?}");
    let lhs = Tensor::<3>::random([2, 128, 2048], Distribution::Default, &device);
    let rhs = Tensor::<3>::random([2, 2048, 2048], Distribution::Default, &device);
    let out = lhs.matmul(rhs);
    let dims = out.dims();
    let data = out.to_data();
    let values = data
        .as_slice::<f32>()
        .unwrap_or_else(|err| panic!("matmul to_data was not f32: {err}"));
    println!(
        "matmul PASS dims={dims:?} dtype={:?} numel={} first={}",
        data.dtype,
        values.len(),
        values.first().copied().unwrap_or(f32::NAN)
    );
}

fn topk() {
    let device = Default::default();
    println!("device={device:?}");
    let scores = Tensor::<2>::random([2, 128], Distribution::Default, &device);
    let (vals, idx) = scores.topk_with_indices(8, 1);
    let v = vals.to_data();
    let i = idx.to_data();
    let vf = v
        .as_slice::<f32>()
        .unwrap_or_else(|err| panic!("topk values were not f32: {err}"));
    println!(
        "topk PASS nvals={} value_dtype={:?} index_dtype={:?} index_bytes={}",
        vf.len(),
        v.dtype,
        i.dtype,
        i.bytes.len()
    );
}
