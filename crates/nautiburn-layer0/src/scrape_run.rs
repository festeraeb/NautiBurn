//! Run the OLMoE FFN stack on the forced FP16 tile and write scrape JSON.
//!
//! Default: all 16 layers, 8 prefill tokens, 2 synthetic decode tokens.
//! Attention is skipped. Residual is FFN-only so later routers see a
//! different hidden. Not llama.cpp.

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
const N_PREFILL: usize = 8;
const N_LAYERS_DEFAULT: usize = 16;
const N_DECODE_DEFAULT: usize = 2;
const RMS_EPS: f32 = 1e-6;

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

pub fn run_dump(gguf: Option<String>, dir: Option<String>) -> ExitCode {
    match dump_inner(
        gguf.unwrap_or_else(|| DEFAULT_GGUF.to_string()),
        dir.unwrap_or_else(|| "/tmp/nautiburn-fp16-scrape/weights".to_string()),
    ) {
        Ok(path) => {
            println!("weights_dir={path}");
            ExitCode::SUCCESS
        }
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

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or(default)
}

fn cuda_index() -> usize {
    env_usize("NAUTIBURN_CUDA_DEVICE", 0)
}

/// Public so `main::probe` can drive the same CUDA client the scrape path uses.
pub fn cuda_client() -> Result<cubecl::client::Client, String> {
    let index = cuda_index();
    let device = cubecl::Device::cuda(index).map_err(|err| format!("cuda device: {err:?}"))?;
    println!("device={device:?} cuda_index={index}");
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
    let n_layers = env_usize("NAUTIBURN_SCRAPE_LAYERS", N_LAYERS_DEFAULT).min(16);
    let n_decode = env_usize("NAUTIBURN_SCRAPE_DECODE", N_DECODE_DEFAULT);
    println!(
        "scrape source={gguf_path} dtype=f16 shared_bytes={shared} cap={P100_SHARED_CAP_BYTES} \
         layers={n_layers} prefill_tokens={N_PREFILL} decode_tokens={n_decode} \
         attention=skipped residual=ffn_only"
    );

    let mut weights = LayerWeights::load(&gguf_path)?;
    let client = cuda_client()?;
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let mut hidden = synthetic_hidden(N_PREFILL, 0);
    let (prefill_acts, layers_done) =
        run_stack(&client, &mut weights, &mut hidden, n_layers, N_PREFILL, 0, shared)?;
    let prefill_ids: Vec<u32> = (1..=N_PREFILL as u32).collect();
    let prefill_doc = ScrapeDocument::new(
        "olmoe-1b-7b-0924-f16",
        format!("nb-fp16-{stamp}-prefill"),
        "olmoe-multilayer-forced-tile",
        "prefill",
        "f16",
        prefill_acts,
        prefill_ids,
    );
    write_scrape(&out_path, &prefill_doc).map_err(|err| format!("write {out_path}: {err}"))?;
    println!(
        "wrote={} activations={} layers_done={} dtype={} shared_bytes={shared} launched=yes phase=prefill",
        Path::new(&out_path).display(),
        prefill_doc.activations.len(),
        layers_done,
        prefill_doc.dtype
    );

    if n_decode > 0 && layers_done > 0 {
        let mut dec_hidden = synthetic_hidden(n_decode, N_PREFILL);
        let (decode_acts, dec_layers) = run_stack(
            &client,
            &mut weights,
            &mut dec_hidden,
            layers_done,
            n_decode,
            N_PREFILL,
            shared,
        )?;
        let decode_ids: Vec<u32> = ((N_PREFILL as u32 + 1)..=(N_PREFILL as u32 + n_decode as u32))
            .collect();
        let decode_path = decode_out_path(&out_path);
        let decode_doc = ScrapeDocument::new(
            "olmoe-1b-7b-0924-f16",
            format!("nb-fp16-{stamp}-decode"),
            "olmoe-multilayer-forced-tile",
            "decode",
            "f16",
            decode_acts,
            decode_ids,
        );
        write_scrape(&decode_path, &decode_doc)
            .map_err(|err| format!("write {decode_path}: {err}"))?;
        println!(
            "wrote={} activations={} layers_done={} dtype={} shared_bytes={shared} launched=yes phase=decode",
            Path::new(&decode_path).display(),
            decode_doc.activations.len(),
            dec_layers,
            decode_doc.dtype
        );
    }

    print_nvrtc();
    if layers_done < n_layers {
        return Err(format!(
            "stopped after {layers_done}/{n_layers} layers; prefill JSON still written"
        ));
    }
    Ok(out_path)
}

fn decode_out_path(prefill_out: &str) -> String {
    if let Some(stem) = prefill_out.strip_suffix(".json") {
        format!("{stem}-decode.json")
    } else {
        format!("{prefill_out}.decode.json")
    }
}

/// Residual FFN-only stack. Attention is skipped; `ffn_norm` + router +
/// selected expert SwiGLU run on the forced FP16 tile. Hidden is updated
/// `x += expert_combine(RMSNorm(x))` so later layers see a different state.
fn run_stack(
    client: &cubecl::client::Client,
    weights: &mut LayerWeights,
    hidden: &mut Vec<f16>,
    n_layers: usize,
    n_tokens: usize,
    token_base: usize,
    shared: u32,
) -> Result<(Vec<ActivationRow>, usize), String> {
    let mut activations = Vec::with_capacity(n_layers * n_tokens);
    let mut layers_done = 0usize;
    for layer in 0..n_layers as u16 {
        match run_layer(client, weights, hidden, layer, n_tokens, token_base, shared) {
            Ok(rows) => {
                activations.extend(rows);
                layers_done += 1;
                println!(
                    "layer={layer} tokens={n_tokens} activations={} dtype=f16 launched=yes shared_bytes={shared}",
                    n_tokens
                );
            }
            Err(err) => {
                eprintln!("layer={layer} stopped: {err}");
                break;
            }
        }
    }
    if layers_done == 0 {
        return Err("no layer launched".to_string());
    }
    Ok((activations, layers_done))
}

fn run_layer(
    client: &cubecl::client::Client,
    weights: &mut LayerWeights,
    hidden: &mut [f16],
    layer: u16,
    n_tokens: usize,
    token_base: usize,
    shared: u32,
) -> Result<Vec<ActivationRow>, String> {
    let norm_w = weights.ffn_norm(layer)?;
    let normed = rmsnorm(hidden, &norm_w, n_tokens);
    let router_f32 = weights.router(layer)?;
    if router_f32.len() != N_EXPERT * N_EMBD {
        return Err(format!(
            "layer {layer} router len {} != {}",
            router_f32.len(),
            N_EXPERT * N_EMBD
        ));
    }
    let mut router_t = vec![f16::from_f32(0.0); N_EMBD * N_EXPERT];
    for e in 0..N_EXPERT {
        for h in 0..N_EMBD {
            router_t[h * N_EXPERT + e] = f16::from_f32(router_f32[e * N_EMBD + h]);
        }
    }
    let scores = gemm_f16(client, &normed, &router_t, n_tokens, N_EXPERT, N_EMBD)?;

    let mut rows = Vec::with_capacity(n_tokens);
    let mut needed: std::collections::BTreeMap<u16, Vec<(usize, f32)>> =
        std::collections::BTreeMap::new();
    for token in 0..n_tokens {
        let row: Vec<f32> = (0..N_EXPERT)
            .map(|e| scores[token * N_EXPERT + e].to_f32())
            .collect();
        let probs = softmax(&row);
        let ids = top_k_expert_ids(&row, N_USED);
        if ids.len() != N_USED {
            return Err(format!("layer {layer} token {token} top-k {}", ids.len()));
        }
        for &id in &ids {
            needed.entry(id).or_default().push((token, probs[id as usize]));
        }
        rows.push(ActivationRow {
            layer,
            token_pos: (token_base + token) as u32,
            expert_ids: ids,
        });
    }

    let mut delta = vec![0.0f32; n_tokens * N_EMBD];
    for (&expert, group) in &needed {
        let (gate, up, down) = weights.expert(layer, expert)?;
        let gate_t = transpose_f16(&gate, N_FF, N_EMBD);
        let up_t = transpose_f16(&up, N_FF, N_EMBD);
        let down_t = transpose_f16(&down, N_EMBD, N_FF);
        let g = gemm_f16(client, &normed, &gate_t, n_tokens, N_FF, N_EMBD)?;
        let u = gemm_f16(client, &normed, &up_t, n_tokens, N_FF, N_EMBD)?;
        let mut mid = vec![f16::from_f32(0.0); n_tokens * N_FF];
        for i in 0..g.len() {
            mid[i] = f16::from_f32(silu(g[i].to_f32()) * u[i].to_f32());
        }
        let down_out = gemm_f16(client, &mid, &down_t, n_tokens, N_EMBD, N_FF)?;
        for &(token, weight) in group {
            let off = token * N_EMBD;
            for h in 0..N_EMBD {
                delta[off + h] += weight * down_out[off + h].to_f32();
            }
        }
    }
    let _ = shared;
    for i in 0..hidden.len() {
        hidden[i] = f16::from_f32(hidden[i].to_f32() + delta[i]);
    }
    Ok(rows)
}

fn synthetic_hidden(tokens: usize, token_base: usize) -> Vec<f16> {
    (0..tokens * N_EMBD)
        .map(|i| {
            let token = token_base + i / N_EMBD;
            let h = i % N_EMBD;
            f16::from_f32(
                (((h + 17 * token) % 97) as f32) / 97.0 * 0.5 - 0.25 + 0.01 * (token as f32),
            )
        })
        .collect()
}

fn rmsnorm(hidden: &[f16], weight: &[f32], n_tokens: usize) -> Vec<f16> {
    let mut out = vec![f16::from_f32(0.0); n_tokens * N_EMBD];
    for token in 0..n_tokens {
        let off = token * N_EMBD;
        let mut ms = 0.0f32;
        for h in 0..N_EMBD {
            let v = hidden[off + h].to_f32();
            ms += v * v;
        }
        let inv = 1.0 / (ms / N_EMBD as f32 + RMS_EPS).sqrt();
        for h in 0..N_EMBD {
            let w = weight.get(h).copied().unwrap_or(1.0);
            out[off + h] = f16::from_f32(hidden[off + h].to_f32() * inv * w);
        }
    }
    out
}

fn softmax(row: &[f32]) -> Vec<f32> {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let ex: Vec<f32> = row.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = ex.iter().sum();
    if sum == 0.0 {
        return vec![0.0; row.len()];
    }
    ex.into_iter().map(|v| v / sum).collect()
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

struct LayerWeights {
    kind: WeightSource,
}

enum WeightSource {
    Gguf(Gguf),
    Dir(std::path::PathBuf),
}

impl LayerWeights {
    fn load(path: &str) -> Result<Self, String> {
        let p = Path::new(path);
        if p.is_dir() {
            return Ok(Self {
                kind: WeightSource::Dir(p.to_path_buf()),
            });
        }
        Ok(Self {
            kind: WeightSource::Gguf(Gguf::open(path)?),
        })
    }

    fn router(&mut self, layer: u16) -> Result<Vec<f32>, String> {
        match &mut self.kind {
            WeightSource::Gguf(gguf) => {
                gguf.read_f32(&format!("blk.{layer}.ffn_gate_inp.weight"))
            }
            WeightSource::Dir(dir) => {
                let named = dir.join(format!("l{layer}_router.f32"));
                if named.exists() {
                    read_f32_file(&named, N_EXPERT * N_EMBD)
                } else if layer == 0 {
                    read_f32_file(&dir.join("router.f32"), N_EXPERT * N_EMBD)
                } else {
                    Err(format!("missing {}", named.display()))
                }
            }
        }
    }

    fn ffn_norm(&mut self, layer: u16) -> Result<Vec<f32>, String> {
        match &mut self.kind {
            WeightSource::Gguf(gguf) => gguf.read_f32(&format!("blk.{layer}.ffn_norm.weight")),
            WeightSource::Dir(dir) => {
                let named = dir.join(format!("l{layer}_ffn_norm.f32"));
                if named.exists() {
                    read_f32_file(&named, N_EMBD)
                } else {
                    Ok(vec![1.0; N_EMBD])
                }
            }
        }
    }

    fn expert(
        &mut self,
        layer: u16,
        expert: u16,
    ) -> Result<(Vec<f16>, Vec<f16>, Vec<f16>), String> {
        let per = N_EMBD * N_FF;
        match &mut self.kind {
            WeightSource::Gguf(gguf) => {
                let off = expert as u64 * (per * 2) as u64;
                let gate = bytes_to_f16(gguf.read_f16_slice(
                    &format!("blk.{layer}.ffn_gate_exps.weight"),
                    off,
                    per * 2,
                )?);
                let up = bytes_to_f16(gguf.read_f16_slice(
                    &format!("blk.{layer}.ffn_up_exps.weight"),
                    off,
                    per * 2,
                )?);
                let down = bytes_to_f16(gguf.read_f16_slice(
                    &format!("blk.{layer}.ffn_down_exps.weight"),
                    off,
                    per * 2,
                )?);
                Ok((gate, up, down))
            }
            WeightSource::Dir(dir) => {
                let prefix = if dir.join(format!("l{layer}_e{expert}_gate.f16")).exists() {
                    format!("l{layer}_e{expert}")
                } else if layer == 0 {
                    format!("e{expert}")
                } else {
                    return Err(format!(
                        "missing {}/l{layer}_e{expert}_gate.f16",
                        dir.display()
                    ));
                };
                let gate = read_f16_file(&dir.join(format!("{prefix}_gate.f16")), per)?;
                let up = read_f16_file(&dir.join(format!("{prefix}_up.f16")), per)?;
                let down = read_f16_file(&dir.join(format!("{prefix}_down.f16")), per)?;
                Ok((gate, up, down))
            }
        }
    }
}

fn dump_inner(gguf_path: String, dir: String) -> Result<String, String> {
    let mut gguf = Gguf::open(&gguf_path)?;
    std::fs::create_dir_all(&dir).map_err(|err| format!("mkdir {dir}: {err}"))?;
    let router = gguf.read_f32("blk.0.ffn_gate_inp.weight")?;
    if router.len() != N_EXPERT * N_EMBD {
        return Err(format!("router len {}", router.len()));
    }
    write_f32_file(&Path::new(&dir).join("router.f32"), &router)?;
    let per = N_EMBD * N_FF;
    for expert in 0..N_EXPERT as u16 {
        let (gate, up, down) = {
            let off = expert as u64 * (per * 2) as u64;
            let gate = gguf.read_f16_slice("blk.0.ffn_gate_exps.weight", off, per * 2)?;
            let up = gguf.read_f16_slice("blk.0.ffn_up_exps.weight", off, per * 2)?;
            let down = gguf.read_f16_slice("blk.0.ffn_down_exps.weight", off, per * 2)?;
            (gate, up, down)
        };
        std::fs::write(Path::new(&dir).join(format!("e{expert}_gate.f16")), &gate)
            .map_err(|err| err.to_string())?;
        std::fs::write(Path::new(&dir).join(format!("e{expert}_up.f16")), &up)
            .map_err(|err| err.to_string())?;
        std::fs::write(Path::new(&dir).join(format!("e{expert}_down.f16")), &down)
            .map_err(|err| err.to_string())?;
    }
    println!("dumped experts=64 dir={dir}");
    Ok(dir)
}

fn read_f32_file(path: &Path, n: usize) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|err| format!("read {}: {err}", path.display()))?;
    if bytes.len() != n * 4 {
        return Err(format!("{} len {} != {}", path.display(), bytes.len(), n * 4));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn write_f32_file(path: &Path, values: &[f32]) -> Result<(), String> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, bytes).map_err(|err| format!("write {}: {err}", path.display()))
}

fn read_f16_file(path: &Path, n: usize) -> Result<Vec<f16>, String> {
    let bytes = std::fs::read(path).map_err(|err| format!("read {}: {err}", path.display()))?;
    if bytes.len() != n * 2 {
        return Err(format!("{} len {} != {}", path.display(), bytes.len(), n * 2));
    }
    Ok(bytes_to_f16(bytes))
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
