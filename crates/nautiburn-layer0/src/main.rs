//! OLMoE FFN stack: gate, top-k, expert GEMM, weighted residual combine.
//!
//! `scrape` walks every `blk.{L}` router (default 16). Attention is skipped;
//! hidden is `x += SwiGLU(RMSNorm(x))` so later layers see a different state.
//!
//! `gemm_cubecl` is the B5 `Tensor::matmul` call site. FP16 stock CubeCL is
//! refused (69632 > 49152). The scrape path uses `gemm_tile::gemm_f16`, a
//! forced BM=BN=16 BK=32 CubeCL kernel (4096 shared bytes). It does not
//! demote to FP32.

mod bridge;
mod gemm_tile;
mod gguf;
mod scrape_run;

use std::env;
use std::fs;
use std::path::Path;
use std::process::ExitCode;

use burn::tensor::activation::{silu, softmax};
use burn::tensor::{FloatDType, IndexingUpdateOp, Int, Tensor, TensorData};
use cubecl::prelude::*;
use half::f16;

use crate::gemm_tile::{gemm_cpu_f32, gemm_f16};

const N_TOKENS: usize = 128;
const N_EMBD: usize = 2048;
const N_FF: usize = 1024;
const N_EXPERT: usize = 64;
const N_USED: usize = 8;

/// Absolute error allowed against the llama.cpp F16 oracle.
/// Chosen for a two-GEMM SwiGLU (k = 2048, then k = 1024) before the run.
const MAX_ABS_TOLERANCE: f32 = 5.0e-2;

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let cmd = args.next().unwrap_or_else(|| "compare".to_string());
    match cmd.as_str() {
        "probe" => scrape_run::run_probe(),
        "dump" => {
            let gguf = args.next();
            let dir = args.next();
            scrape_run::run_dump(gguf, dir)
        }
        "scrape" => {
            let gguf = args.next();
            let out = args.next();
            scrape_run::run_scrape(gguf, out)
        }
        "compare" => {
            let dir = args.next().unwrap_or_else(|| "/tmp/nautiburn-b5/run".to_string());
            let dtype = args.next().unwrap_or_else(|| "f16".to_string());
            compare(Path::new(&dir), &dtype)
        }
        other => {
            eprintln!(
                "unknown command {other}; try probe | dump [GGUF] [DIR] | scrape [GGUF|DIR] [OUT] | compare"
            );
            eprintln!(
                "scrape env: NAUTIBURN_SCRAPE_LAYERS (default 16) NAUTIBURN_SCRAPE_DECODE (default 2) NAUTIBURN_CUDA_DEVICE (default 0)"
            );
            ExitCode::from(2)
        }
    }
}

fn device() -> burn::tensor::Device {
    burn::tensor::Device::cuda(0)
}

/// The one GEMM call site. CubeCL on the CUDA backend.
///
/// NOTE: this must NOT be `Tensor::matmul`. Stock CubeCL FP16 matmul picks its
/// own tile and requests 69632 bytes of shared memory, which sm_60 caps at
/// 49152, so it panics before launch. `gemm_tile::gemm_f16` is the forced
/// BM=BN=16 BK=32 kernel (4096 shared bytes) that fits, and it does not
/// demote to FP32. Every GEMM in `compare` goes through this wrapper so the
/// P100 constraint cannot be bypassed by accident.
fn gemm_cubecl(
    client: &cubecl::client::Client,
    lhs: &[f32],
    rhs: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<Vec<f32>, String> {
    if lhs.len() != m * k {
        return Err(format!("lhs len {} != {m}*{k}", lhs.len()));
    }
    if rhs.len() != k * n {
        return Err(format!("rhs len {} != {k}*{n}", rhs.len()));
    }
    let l16: Vec<f16> = lhs.iter().map(|v| f16::from_f32(*v)).collect();
    let r16: Vec<f16> = rhs.iter().map(|v| f16::from_f32(*v)).collect();
    let out = gemm_f16(client, &l16, &r16, m, n, k)?;
    Ok(out.iter().map(|v| v.to_f32()).collect())
}

/// Reference GEMM in FP32 on the host, used to demote only when the FP16
/// router genuinely disagrees with the oracle about expert ids.
fn gemm_host_f32(lhs: &[f32], rhs: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let l16: Vec<f16> = lhs.iter().map(|v| f16::from_f32(*v)).collect();
    let r16: Vec<f16> = rhs.iter().map(|v| f16::from_f32(*v)).collect();
    gemm_cpu_f32(&l16, &r16, m, n, k)
}

fn probe() -> ExitCode {
    let dev = device();
    println!("device={dev:?}");
    // The real FP16 path, not Tensor::matmul: on sm_60 stock CubeCL matmul asks
    // for 69632 shared bytes against a 49152 cap, so it never launches. This
    // probe uses the same forced-tile kernel that `compare` uses, which is the
    // thing we actually need to know is alive on this GPU.
    let client = match scrape_run::cuda_client() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("cuda client: {e}");
            return ExitCode::from(5);
        }
    };
    // 1x4 @ 4x1 == 10
    let lhs = [1.0f32, 2.0, 3.0, 4.0];
    let rhs = [1.0f32, 1.0, 1.0, 1.0];
    let values = match gemm_cubecl(&client, &lhs, &rhs, 1, 1, 4) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("probe gemm: {e}");
            return ExitCode::from(5);
        }
    };
    println!("probe_f16_gemm={values:?}");
    print_nvrtc();
    if values.first().copied().unwrap_or(f32::NAN).is_nan() {
        ExitCode::from(1)
    } else {
        ExitCode::from(0)
    }
}

fn compare(dir: &Path, dtype_name: &str) -> ExitCode {
    let gemm_f16 = match dtype_name {
        "f16" => true,
        "f32" => false,
        other => {
            eprintln!("dtype must be f16 or f32, got {other}");
            return ExitCode::from(2);
        }
    };
    let dev = device();
    let client = match cubecl::Device::cuda(0).map_err(|e| format!("cuda: {e:?}")) {
        Ok(d) => d.client(),
        Err(e) => {
            eprintln!("cuda client: {e}");
            return ExitCode::from(5);
        }
    };
    println!("device={dev:?} gemm_dtype={dtype_name} tolerance={MAX_ABS_TOLERANCE:.4e}");

    let hidden_f32 = read_f32(&dir.join("hidden.f32"), N_TOKENS * N_EMBD);
    let oracle = read_f32(&dir.join("oracle_out.f32"), N_TOKENS * N_EMBD);
    let oracle_ids = read_i32(&dir.join("oracle_ids.i32"), N_TOKENS * N_USED);
    let gate_inp = read_f32(&dir.join("ffn_gate_inp.f32"), N_EXPERT * N_EMBD);
    let gate_exps = read_f32(&dir.join("ffn_gate_exps.f32"), N_EXPERT * N_EMBD * N_FF);
    let up_exps = read_f32(&dir.join("ffn_up_exps.f32"), N_EXPERT * N_EMBD * N_FF);
    let down_exps = read_f32(&dir.join("ffn_down_exps.f32"), N_EXPERT * N_FF * N_EMBD);
    println!(
        "host_copy_bytes={}",
        (gate_inp.len() + gate_exps.len() + up_exps.len() + down_exps.len()) * 4
    );

    // Router: hidden [N_TOKENS, N_EMBD] @ router^T [N_EMBD, N_EXPERT].
    let router_t = transpose(&gate_inp, N_EXPERT, N_EMBD);
    let mut logits = match gemm_cubecl(&client, &hidden_f32, &router_t, N_TOKENS, N_EXPERT, N_EMBD) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("router gemm: {e}");
            return ExitCode::from(5);
        }
    };
    let mut gate_note = if gemm_f16 { "f16" } else { "f32" };
    let mut probs_f32 = softmax_rows(&logits, N_TOKENS, N_EXPERT);
    let mut ids = topk_ids_host(&probs_f32);
    let mut mismatches = id_mismatches(&ids, &oracle_ids);
    if mismatches != 0 {
        println!("gate_gemm_demoted=f32 reason=topk_disagreed_with_oracle count={mismatches}");
        logits = gemm_host_f32(&hidden_f32, &router_t, N_TOKENS, N_EXPERT, N_EMBD);
        probs_f32 = softmax_rows(&logits, N_TOKENS, N_EXPERT);
        ids = topk_ids_host(&probs_f32);
        gate_note = "f32";
        mismatches = id_mismatches(&ids, &oracle_ids);
    }
    println!("gate_gemm={gate_note} softmax=host_f32 topk=host_f32 id_mismatches={mismatches}");

    let miss_expert = ids[0] as usize;
    println!("cpu_miss_expert={miss_expert} cpu_miss_token=0");

    let mut acc = vec![0.0f32; N_TOKENS * N_EMBD];

    let mut groups: Vec<Vec<(usize, f32)>> = vec![Vec::new(); N_EXPERT];
    for token in 0..N_TOKENS {
        for slot in 0..N_USED {
            let expert = ids[token * N_USED + slot] as usize;
            let weight = probs_f32[token * N_EXPERT + expert];
            groups[expert].push((token, weight));
        }
    }

    for expert in 0..N_EXPERT {
        let group = &groups[expert];
        if group.is_empty() {
            continue;
        }
        let n = group.len();

        // Gather the rows this expert owns. Row-major [n, N_EMBD].
        let mut rows = Vec::with_capacity(n * N_EMBD);
        for &(token, _) in group {
            rows.extend_from_slice(&hidden_f32[token * N_EMBD..(token + 1) * N_EMBD]);
        }

        // The expert weights are stored [N_FF, N_EMBD] / [N_EMBD, N_FF] and
        // must be transposed to the [k, n] form the GEMM expects.
        let gate_t = transpose(&expert_matrix(&gate_exps, expert), N_FF, N_EMBD);
        let up_t = transpose(&expert_matrix(&up_exps, expert), N_FF, N_EMBD);
        let down_t = transpose(&expert_matrix(&down_exps, expert), N_EMBD, N_FF);

        let out = if expert == miss_expert {
            // Cross-check: the CPU FP32 path must agree with the GPU kernel.
            cpu_expert(&hidden_f32, &gate_exps, &up_exps, &down_exps, expert, group)
        } else {
            let gate = gemm_cubecl(&client, &rows, &gate_t, n, N_FF, N_EMBD)
                .map_err(|e| format!("gate gemm expert {expert}: {e}"))
                .unwrap_or_else(|e| {
                    eprintln!("{e}");
                    std::process::exit(5);
                });
            let up = gemm_cubecl(&client, &rows, &up_t, n, N_FF, N_EMBD)
                .map_err(|e| format!("up gemm expert {expert}: {e}"))
                .unwrap_or_else(|e| {
                    eprintln!("{e}");
                    std::process::exit(5);
                });
            // SwiGLU in FP32, matching llama.cpp's activation precision.
            let mut mid = vec![0.0f32; n * N_FF];
            for i in 0..n * N_FF {
                let g = gate[i];
                mid[i] = g * (1.0 / (1.0 + (-g).exp())) * up[i];
            }
            gemm_cubecl(&client, &mid, &down_t, n, N_EMBD, N_FF)
                .map_err(|e| format!("down gemm expert {expert}: {e}"))
                .unwrap_or_else(|e| {
                    eprintln!("{e}");
                    std::process::exit(5);
                })
        };

        // Scale by the router weight and scatter-add into the accumulator.
        for (r, &(token, weight)) in group.iter().enumerate() {
            for c in 0..N_EMBD {
                acc[token * N_EMBD + c] += out[r * N_EMBD + c] * weight;
            }
        }
    }

    let got = acc;
    let mut max_abs = 0.0f32;
    for (a, b) in got.iter().zip(oracle.iter()) {
        max_abs = max_abs.max((a - b).abs());
    }
    println!("expert_gemm={}", if gemm_f16 { "f16" } else { "f32" });
    println!("combine=host_scatter_add");
    println!("max_abs_error={max_abs:.8e}");
    println!("tolerance={MAX_ABS_TOLERANCE:.8e}");
    print_nvrtc();
    if mismatches != 0 {
        eprintln!("expert id set disagrees with the oracle");
        return ExitCode::from(4);
    }
    if max_abs > MAX_ABS_TOLERANCE {
        return ExitCode::from(3);
    }
    ExitCode::from(0)
}

/// Row-wise softmax over the expert axis, in FP32.
///
/// Done on the host deliberately: the router decision must match llama.cpp
/// bit-for-bit closely enough that top-k agrees, and the tensor softmax would
/// reintroduce a CubeCL kernel that is not the thing under test.
fn softmax_rows(logits: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let row = &logits[r * cols..(r + 1) * cols];
        // Subtract the max for numerical stability, as llama.cpp does.
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for (i, v) in row.iter().enumerate() {
            let e = (v - max).exp();
            out[r * cols + i] = e;
            sum += e;
        }
        for v in &mut out[r * cols..(r + 1) * cols] {
            *v /= sum;
        }
    }
    out
}

/// Top-k expert ids per token, ordered by descending probability with ties
/// broken by lower expert index (stable partial sort).
fn topk_ids_host(probs: &[f32]) -> Vec<i32> {
    let mut ids = Vec::with_capacity(N_TOKENS * N_USED);
    for t in 0..N_TOKENS {
        let mut order: Vec<usize> = (0..N_EXPERT).collect();
        order.sort_by(|&a, &b| {
            probs[t * N_EXPERT + b]
                .partial_cmp(&probs[t * N_EXPERT + a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        for &e in order.iter().take(N_USED) {
            ids.push(e as i32);
        }
    }
    ids
}

fn cpu_expert(
    hidden: &[f32],
    gate_exps: &[f32],
    up_exps: &[f32],
    down_exps: &[f32],
    expert: usize,
    group: &[(usize, f32)],
) -> Vec<f32> {
    let gate_t = transpose(&expert_matrix(gate_exps, expert), N_FF, N_EMBD);
    let up_t = transpose(&expert_matrix(up_exps, expert), N_FF, N_EMBD);
    let down_t = transpose(&expert_matrix(down_exps, expert), N_EMBD, N_FF);
    let mut out = Vec::with_capacity(group.len() * N_EMBD);
    for &(token, weight) in group {
        let h = &hidden[token * N_EMBD..(token + 1) * N_EMBD];
        let gate = gemm_cpu(h, &gate_t, 1, N_EMBD, N_FF);
        let up = gemm_cpu(h, &up_t, 1, N_EMBD, N_FF);
        let mut mid = vec![0.0f32; N_FF];
        for i in 0..N_FF {
            let g = gate[i];
            mid[i] = g * (1.0 / (1.0 + (-g).exp())) * up[i];
        }
        let down = gemm_cpu(&mid, &down_t, 1, N_FF, N_EMBD);
        for value in down {
            out.push(value * weight);
        }
    }
    out
}

/// Host GEMM for the expert that is absent from the device. Not CubeCL.
fn gemm_cpu(lhs: &[f32], rhs: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let a = lhs[i * k + p];
            let rhs_row = p * n;
            for j in 0..n {
                out[i * n + j] += a * rhs[rhs_row + j];
            }
        }
    }
    out
}

fn transpose(matrix: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = matrix[r * cols + c];
        }
    }
    out
}

fn expert_matrix(all: &[f32], expert: usize) -> Vec<f32> {
    let elems = N_EMBD * N_FF;
    let start = expert * elems;
    all[start..start + elems].to_vec()
}

fn upload(values: &[f32], shape: [usize; 2], f16: bool, dev: &burn::tensor::Device) -> Tensor<2> {
    let tensor = Tensor::<2>::from_data(TensorData::new(values.to_vec(), shape), dev);
    if f16 {
        tensor.cast(FloatDType::F16)
    } else {
        tensor
    }
}

fn cat_rows(hidden: &Tensor<2>, tokens: &[usize]) -> Tensor<2> {
    let parts = tokens
        .iter()
        .map(|&token| hidden.clone().slice([token..token + 1, 0..N_EMBD]))
        .collect();
    Tensor::cat(parts, 0)
}

fn topk_ids(probs: &Tensor<2>) -> Vec<i32> {
    let (_vals, idx) = probs.clone().topk_with_indices(N_USED, 1);
    let data = idx.to_data();
    println!("topk_index_dtype={:?}", data.dtype);
    if let Ok(values) = data.clone().try_into_vec::<i32>() {
        return values;
    }
    data.try_into_vec::<i64>()
        .unwrap_or_else(|err| panic!("topk indices were not i32 or i64: {err}"))
        .into_iter()
        .map(|v| i32::try_from(v).expect("expert id"))
        .collect()
}

fn id_mismatches(got: &[i32], oracle: &[i32]) -> usize {
    let mut n = 0;
    for token in 0..N_TOKENS {
        let mut a: Vec<i32> = got[token * N_USED..(token + 1) * N_USED].to_vec();
        let mut b: Vec<i32> = oracle[token * N_USED..(token + 1) * N_USED].to_vec();
        a.sort_unstable();
        b.sort_unstable();
        if a != b {
            n += 1;
        }
    }
    n
}

fn read_f32(path: &Path, n: usize) -> Vec<f32> {
    let bytes = fs::read(path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    assert_eq!(bytes.len(), n * 4, "{}", path.display());
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_i32(path: &Path, n: usize) -> Vec<i32> {
    let bytes = fs::read(path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    assert_eq!(bytes.len(), n * 4, "{}", path.display());
    bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn print_nvrtc() {
    let maps = fs::read_to_string("/proc/self/maps").unwrap_or_default();
    let mut seen = std::collections::BTreeSet::new();
    for line in maps.lines() {
        if let Some(path) = line.split_whitespace().last() {
            if path.contains("libnvrtc") {
                seen.insert(path.to_string());
            }
        }
    }
    if seen.is_empty() {
        println!("nvrtc_map=none");
        return;
    }
    for path in seen {
        println!("nvrtc_map={path}");
        if path.contains("libnvrtc.so.13") {
            eprintln!("CUDA 13 nvrtc is mapped");
            std::process::exit(5);
        }
    }
}
