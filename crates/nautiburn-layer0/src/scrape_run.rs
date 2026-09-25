//! Run one OLMoE layer-0 expert block on the forced FP16 tile and write scrape JSON.

use std::path::Path;
use std::process::ExitCode;
use std::time::{SystemTime, UNIX_EPOCH};

use cubecl::prelude::*;
use half::f16;
use nautiburn_engine::{
    scheduled_compute_dtype, top_k_expert_ids, write_scrape, ActivationRow, ScrapeDocument,
    FORCED_TILE_SHARED_BYTES, P100_SHARED_CAP_BYTES,
};

use crate::gemm_tile::{forced_tile_shared_bytes, gemm_cpu_f32, gemm_f16};
use crate::gguf::Gguf;

const N_EMBD: usize = 2048;
const N_FF: usize = 1024;
const N_EXPERT: usize = 64;
const N_USED: usize = 8;
const N_TOKENS: usize = 8;
const LAYER: u16 = 0;

const DEFAULT_GGUF: &str = "/home/cesarops/models/olmoe-1b-7b-0924-f16.gguf";

pub fn run_probe() -> ExitCode {
    match probe_inner() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("{err}");
            ExitCode::from(1)
        }
    }
}

pub fn run_scrape(gguf: Option<String>, out: Option<String>) -> ExitCode {
    match scrape_inner(
        gguf.unwrap_or_else(|| DEFAULT_GGUF.to_string()),
        out.unwrap_or_else(|| "/tmp/nautiburn-fp16-scrape/scrape.json".to_string()),
    ) {
        Ok(path) => {
            println!("scrape_json={path}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("{err}");
            ExitCode::from(1)
        }
    }
}

fn cuda_client() -> Result<cubecl::client::Client, String> {
    let device = cubecl::Device::cuda(0).map_err(|err| format!("cuda device: {err:?}"))?;
    println!("device={device:?}");
    Ok(device.client())
}

fn probe_inner() -> Result<(), String> {
    print_nvrtc();
    let shared = forced_tile_shared_bytes();
    println!(
        "probe dtype=f16 shared_bytes={shared} cap={P100_SHARED_CAP_BYTES} scheduled={}",
        scheduled_compute_dtype()
    );
    if scheduled_compute_dtype() != "f16" {
        return Err(format!(
            "scheduled_compute_dtype must stay f16, got {}",
            scheduled_compute_dtype()
        ));
    }
    let client = cuda_client()?;
    let m = 64usize;
    let n = 64usize;
    let k = 64usize;
    let lhs: Vec<f16> = (0..m * k)
        .map(|i| f16::from_f32(((i % 17) as f32) * 0.05 - 0.4))
        .collect();
    let rhs: Vec<f16> = (0..k * n)
        .map(|i| f16::from_f32(((i % 13) as f32) * 0.04 - 0.2))
        .collect();
    let got = gemm_f16(&client, &lhs, &rhs, m, n, k)?;
    let expect = gemm_cpu_f32(&lhs, &rhs, m, n, k);
    let mut max_abs = 0.0f32;
    for (a, b) in got.iter().zip(expect.iter()) {
        max_abs = max_abs.max((a.to_f32() - *b).abs());
    }
    println!("probe_max_abs={max_abs:.6e} dtype=f16 launched=yes shared_bytes={shared}");
    print_nvrtc();
    if max_abs > 5.0e-2 {
        return Err(format!("probe abs error {max_abs} too large"));
    }
    Ok(())
}

fn scrape_inner(gguf_path: String, out_path: String) -> Result<String, String> {
    print_nvrtc();
    if scheduled_compute_dtype() != "f16" {
        return Err("scheduled_compute_dtype must stay f16".to_string());
    }
    let shared = forced_tile_shared_bytes();
    if shared != FORCED_TILE_SHARED_BYTES {
        return Err(format!(
            "tile shared {shared} != constant {FORCED_TILE_SHARED_BYTES}"
        ));
    }
    println!(
        "scrape gguf={gguf_path} dtype=f16 shared_bytes={shared} cap={P100_SHARED_CAP_BYTES}"
    );

    let mut gguf = Gguf::open(&gguf_path)?;
    let router_f32 = gguf.read_f32("blk.0.ffn_gate_inp.weight")?;
    if router_f32.len() != N_EXPERT * N_EMBD {
        return Err(format!(
            "router len {} != {}",
            router_f32.len(),
            N_EXPERT * N_EMBD
        ));
    }
    // File layout is [experts, hidden]. GEMM wants rhs [hidden, experts].
    let mut router_t = vec![f16::from_f32(0.0); N_EMBD * N_EXPERT];
    for e in 0..N_EXPERT {
        for h in 0..N_EMBD {
            router_t[h * N_EXPERT + e] = f16::from_f32(router_f32[e * N_EMBD + h]);
        }
    }

    let hidden = synthetic_hidden(N_TOKENS);
    let client = cuda_client()?;
    let scores = gemm_f16(&client, &hidden, &router_t, N_TOKENS, N_EXPERT, N_EMBD)?;
    println!("router_gemm=f16 launched=yes shared_bytes={shared}");

    let mut activations = Vec::with_capacity(N_TOKENS);
    let mut token_ids = Vec::with_capacity(N_TOKENS);
    let mut needed = std::collections::BTreeSet::new();
    for token in 0..N_TOKENS {
        let row: Vec<f32> = (0..N_EXPERT)
            .map(|e| scores[token * N_EXPERT + e].to_f32())
            .collect();
        let ids = top_k_expert_ids(&row, N_USED);
        if ids.len() != N_USED {
            return Err(format!("token {token} top-k {}", ids.len()));
        }
        for id in &ids {
            needed.insert(*id);
        }
        println!("token_pos={token} expert_ids={ids:?}");
        activations.push(ActivationRow {
            layer: LAYER,
            token_pos: token as u32,
            expert_ids: ids,
        });
        token_ids.push(token as u32 + 1);
    }

    let per = N_EMBD * N_FF;
    let gate_bytes = (per * 2) as u64;
    for &expert in &needed {
        let off = expert as u64 * gate_bytes;
        let gate = bytes_to_f16(gguf.read_f16_slice(
            "blk.0.ffn_gate_exps.weight",
            off,
            per * 2,
        )?);
        let up = bytes_to_f16(gguf.read_f16_slice("blk.0.ffn_up_exps.weight", off, per * 2)?);
        let down = bytes_to_f16(gguf.read_f16_slice(
            "blk.0.ffn_down_exps.weight",
            off,
            per * 2,
        )?);
        // gate/up stored [ff, hidden] → rhs [hidden, ff]
        let gate_t = transpose_f16(&gate, N_FF, N_EMBD);
        let up_t = transpose_f16(&up, N_FF, N_EMBD);
        let down_t = transpose_f16(&down, N_EMBD, N_FF);
        let g = gemm_f16(&client, &hidden, &gate_t, N_TOKENS, N_FF, N_EMBD)?;
        let u = gemm_f16(&client, &hidden, &up_t, N_TOKENS, N_FF, N_EMBD)?;
        let mut mid = vec![f16::from_f32(0.0); N_TOKENS * N_FF];
        for i in 0..g.len() {
            let gv = g[i].to_f32();
            mid[i] = f16::from_f32(silu(gv) * u[i].to_f32());
        }
        let _down = gemm_f16(&client, &mid, &down_t, N_TOKENS, N_EMBD, N_FF)?;
        println!(
            "expert={expert} gate_up_down=f16 launched=yes shared_bytes={shared}"
        );
    }

    let request_id = format!(
        "nb-fp16-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    );
    let doc = ScrapeDocument::new(
        "olmoe-1b-7b-0924-f16",
        request_id,
        "olmoe-layer0-forced-tile",
        "prefill",
        "f16",
        activations,
        token_ids,
    );
    write_scrape(&out_path, &doc).map_err(|err| format!("write {out_path}: {err}"))?;
    print_nvrtc();
    println!(
        "wrote={} activations={} dtype={} shared_bytes={shared} launched=yes",
        Path::new(&out_path).display(),
        doc.activations.len(),
        doc.dtype
    );
    Ok(out_path)
}

fn synthetic_hidden(tokens: usize) -> Vec<f16> {
    (0..tokens * N_EMBD)
        .map(|i| {
            let token = i / N_EMBD;
            let h = i % N_EMBD;
            f16::from_f32(
                (((h + 17 * token) % 97) as f32) / 97.0 * 0.5
                    - 0.25
                    + 0.01 * (token as f32),
            )
        })
        .collect()
}

fn bytes_to_f16(bytes: Vec<u8>) -> Vec<f16> {
    bytes
        .chunks_exact(2)
        .map(|c| f16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn transpose_f16(matrix: &[f16], rows: usize, cols: usize) -> Vec<f16> {
    let mut out = vec![f16::from_f32(0.0); rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = matrix[r * cols + c];
        }
    }
    out
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn print_nvrtc() {
    let maps = std::fs::read_to_string("/proc/self/maps").unwrap_or_default();
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
