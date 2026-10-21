# dp4a for the fleet's Pascal and mixed GPUs — research (2026-10-21)

Question (operator, via Warp): CUDA-for-Rust, PTX, or a custom shim to get int8 `dp4a` dot products
into the NautiBurn build, and how that ties to the cesarops inference engine, Nauti-Virtualizer
and nauti-fabric in a pure-Rust build. (The Warp job for this returned 19 tokens and never posted;
this note replaces it.)

## Finding: no shim needed — CubeCL already has dp4a

Checked in this repo's pinned `upstream/cubecl` (rev 313f41b):

| Where | What |
|---|---|
| `cubecl-core/src/frontend/operation/dp4a.rs` | `dp4a(a: i32, b: i32, c: i32) -> i32`: `c + Σ a_i·b_i` over four packed signed bytes, usable inside `#[cube]` kernels |
| `cubecl-cuda/src/runtime.rs:138` | `dp4a: arch_version >= 61` — the device capability flag |
| `cubecl-cpp/src/cuda/binary.rs` | emits `__dp4a(a, b, c)` when the flag is set, otherwise `expand_dp4a_polyfill` (byte-wise multiply-add) |
| doc comment in `dp4a.rs` | Vulkan/SPIR-V lowers to `OpSDot` + add; other backends use the polyfill |
| `cubecl-ir/src/dialect/math.rs` | constant folding + unit tests for the op |

So one Rust `#[cube]` kernel gets:

| Card | Arch | Path |
|---|---|---|
| GTX 1070 / P106 | sm_61 | native `__dp4a` |
| T10 / 2060 / 2060S | sm_75 | native `__dp4a` |
| P100 | sm_60 | polyfill (software byte multiply-add) |
| RX 6700 XT | Vulkan | `OpSDot` (needs the Vulkan/WGPU-SPIR-V CubeCL backend) |

**Nothing calls it yet.** No NautiBurn crate and no Burn upstream code uses `dp4a` (grep, 2026-10-21).
`docs/kernels-we-have.md` is accurate: the only Burn kernel ever launched here is an FP32 matmul.

## What this means for the three routes the operator named

- **Custom PTX / shim**: not needed for dp4a. CubeCL generates the CUDA C, NVRTC compiles it per
  arch, and the arch flag picks native vs polyfill. Inline PTX stays a fallback only if a kernel
  needs an instruction CubeCL does not expose.
- **CUDA-for-Rust (cudarc / Rust-CUDA)**: a second toolchain beside CubeCL. Only worth it for a
  microbenchmark outside Burn; for NautiBurn kernels, stay in CubeCL so CUDA and Vulkan share code.
- **P100 (sm_60)**: the polyfill is roughly what llama.cpp's fallback does. The P100's real
  strength is packed FP16 (HFMA2, ~2x FP32 rate) and full-rate FP64. Expectation to test, not a
  fact: Q8 → FP16 dequant + half2 math beats the dp4a polyfill on P100.

## The numbers we need (forge tasks, one function each)

All in a new crate `crates/nautiburn-dp4a-bench` (binary + lib), CubeCL CUDA backend:

1. `q8_dot_dp4a` kernel: dot product of two `i32`-packed int8 vectors using `dp4a`, block of 32
   (Q8_0 layout: int8 values + one f16 scale per 32). Test: matches a CPU reference exactly on
   random seeded data.
2. `q8_dot_f16` kernel: same inputs, dequantize to f16 and accumulate in f32. Test: matches the
   CPU reference within 1e-2 relative.
3. `q8_matvec` wrapper for both: an M×K Q8_0 matrix times a K vector, at real shapes
   (K = 2816 and 4096; M = 2816, 4096, 14336). Test: matches CPU reference.
4. Bench binary: runs 1–3 on the visible device, prints arch, path (native/polyfill), GB/s and
   tokens-equivalent per second, as JSON, 3 warm-up + 20 timed runs.
5. Fleet run (judge, not forge): P100 (c2/c4), P106/1070 (c3), T10/2060 (c4), RX 6700 XT via
   Vulkan when that backend builds. Results go in `docs/kernels-we-have.md` as measured rows.

Success is a table of measured numbers per card and path. If the polyfill loses to FP16 on P100,
P100 kernels use FP16; sm_61+ use dp4a. The choice is made by the numbers, per card.

## How it connects to the rest

- **cesarops-inference** (own wgpu/ash engine; `backend_cuda.rs` is a placeholder): the same
  measured result decides its Q8 path. The engine's WGSL path can use `dot4I8Packed` on
  hardware that has it; that is a separate check.
- **Nauti-Virtualizer / nauti-fabric**: already discovers GPUs (NVML + wgpu). Add the compute
  capability and the measured dp4a/FP16 rates to each node's resource record, so fabric leases
  place quantized-matmul work on cards where the native path wins, and FP16 work on P100s.
  This is a resource-model field, not new kernels.
- **Pure-Rust build**: CubeCL (CUDA + Vulkan) for kernels, fabric for placement, NautiBurn for
  the expert/MoE schedule. No C++ shims in the hot path.

## Open questions for the operator

- OK to add `crates/nautiburn-dp4a-bench` to the workspace and give the forge tasks 1–4?
- Which P100 node can be reserved for the bench run (needs no other GPU load during timing)?
