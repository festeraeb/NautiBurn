# Kernels we actually have

Burn does not wrap Candle. Burn does not wrap WGPU as a GGUF runner.
Burn's GPU path is CubeCL (CUDA / WGPU / Metal / HIP). Candle is a
separate crate with its own `Tensor`, CUDA/Metal kernels, and GGUF
loader. Cake is Candle, not Burn.

## In this repo

| Piece | What it is | Launched a GPU kernel? |
|---|---|---|
| `nautiburn-layer0` `gemm_cubecl` | `Tensor::matmul` on Burn CUDA → CubeCL | Yes, once. B5 on cesarops2 P100. FP16 asked 69632 bytes shared, P100 cap 49152, so the measured block was FP32. Host `*.f32` dumps, not a GGUF loader. |
| `nautiburn-layer0` `gemm_cpu` | Host FP32 miss path | CPU only |
| `nautiburn-wgpu` | `type_name::<burn::backend::Wgpu>()` | No. No device, no GGUF. |
| `nautiburn-engine` | Scrape, dtype policy, JSON handoff | No. Refuses the known 69632 FP16 launch. Does not rewrite to FP32. |
| `nautiburn-predict` | Expert-id predictors on the B1 trace | No |
| llama.cpp on the fleet | Live GGUF servers | Yes. Not a Burn kernel. Leave Mixtral / Tiel-Coder / embedder / SmolLM up. |

OpenVINO on this repo is CPU-only on the Xeons.

## Upstream we can call (no PTX from us)

**Burn / CubeCL** (`upstream/burn`, `upstream/cubecl`): matmul, softmax,
SiLU, RMSNorm, RoPE, multi-head attention, PTQ dequant (Burn's own
2/4/8-bit blocks, not GGUF `Q4_K`). Safetensors via `burn-store`. No
GGUF in that checkout. P100 is sm_60: 49152 shared, no tensor cores, no
BF16, no `cp.async`. CubeCL CMMA is not a Pascal path.

**Candle** (`candle-core` 0.10.2 on this machine): `Device` is Cpu /
Cuda / Metal only — no WGPU. GGUF loader in `quantized/gguf_file.rs`.
`GgmlDType`: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2K,
Q3K, Q4K, Q5K, Q6K, Q8K. Quantized examples include Llama, Mixtral
8x7B, Qwen3 MoE. Not OLMoE. Wrapping Candle is not every GGUF: each
arch needs a `from_gguf` path. "Tensor something" here is Candle's
`Tensor` plus the `safetensors` crate (and/or the GGUF loader).

**Cake** ([evilsocket/cake](https://github.com/evilsocket/cake), Simone
Margaritelli): Candle 0.9 + `safetensors`. Layer-shards a HF checkpoint
across machines. CUDA/Metal are Candle. Vulkan is Cake's own (ash;
wgpu is an optional dep). No GGUF in `cake-core`. Text arches:
Llama/SmolLM2, Qwen2/3/3.5 (+ MoE), Phi-4, Mistral (dense), Gemma 3,
Falcon3, OLMo 2, EXAONE 4. No Mixtral, no OLMoE. Pascal needs
`candle-kernels` patches. Reuse the *split idea*, not Burn WGPU
kernels.

**llama.cpp**: already runs the fleet GGUFs. Not Burn.

ExLlama / EXL2 is a different format. No Burn WGPU loader here.

## What to write vs what to call

Writing a Pascal FP16 CubeCL tile that stays under 49152 is one kernel
(or a tile-config change). Writing a full GGUF MoE forward (dequant +
attention + expert GEMM + router) is a new engine. Call Candle or
llama.cpp for that.

First honest "load a real file and do a forward": Candle GGUF for an
arch it already has (quantized Llama / Mixtral / Qwen3-MoE) on CUDA
12.8 — or Burn `Tensor` + CubeCL matmul on extracted F16 expert
weights (B5 already did layer-0 in FP32). Not "almost every model."
Not 80% / 235B / two cards. Do not start that on busy P100s.
