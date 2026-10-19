# Sprint 2026-10-19 — judging log

Coder: **Swift 1.5 IQ2_XS** (Nauti-Strata, cesarops2:8083, 32k context) through
`cesarops-forge-runner` + Aider (diff edit format). Gate judge: Bonsai 27B
(cesarops4om:8090). Final judge: Claude, who re-ran every test and read the code.
Accepted work lands on `sprint/2026-10-19`, never on a live checkout.

## Accepted

### N4 — `crates/nautiburn-residency` (runs 05fc88f7b4aa → dca0b03130be)

- 5 coder commits over 2 runs, 3 trips in the second. `cargo test -p nautiburn-residency`:
  7/7 pass, re-run by the judge.
- LRU is a per-device `BTreeSet<(last_used, key…)>`; eviction skips pinned and
  in-use keys; `OverBudget` when evictable bytes cannot cover the need; Hot →
  Warm eviction cascades Warm → Cold when the host budget is full.
- The randomized test is real: 10,000 ops, xorshift, budget and pin/in-use
  invariants asserted after every step.
- Follow-ups (not blockers):
  - `request`/`commit_load` on an unregistered key invents a 1-byte entry
    (`unwrap_or(1)`); it should refuse.
  - `commit_load` duplicates `request`.
  - `prefetch` silently ignores an unknown device.
  - `register` silently drops over-budget placements (the signature has no `Result`).

### N3 — `crates/nautiburn-tile-parity` (run 5d418a287eb6)

One `#[cube]` FP16 tile (BM=BN=16, BK=32, 4096 B shared, identical to
`nautiburn-layer0`'s apart from typed literals) launched on CUDA and on WGPU.
WGPU feeds the RX 6700 XT CubeCL's WGSL; no hand-written SPIR-V or WGSL, and not
the `wgpu-spirv` feature. Tests: 5/5 pass, including the lavapipe (CPU Vulkan)
parity test on ragged shapes. The 2×3·3×2 hand example was checked by hand.

Hardware, seed 7, atol 0.05, rtol 0.02, CPU f32 reference:

| shape m×n×k | P100 (CUDA, t440 GPU1) | RX 6700 XT (WGPU/Vulkan RADV, cesarops4om) |
|---|---|---|
| 16×16×32 | pass, max_abs 0.00162 | pass, max_abs 0.00162 |
| 17×33×65 | pass, 0.00191 | pass, 0.00191 |
| 64×2048×1024 | pass, 0.01557, 6.5 GFLOPS | pass, 0.01557, 15.4 GFLOPS |
| 128×2048×2048 | pass, 0.02799, 14.6 GFLOPS | pass, 0.02799, 28.4 GFLOPS |
| 512×4096×4096 | pass, 0.03124, 68 GFLOPS | pass, 0.03124, 119 GFLOPS |

- The errors are bit-identical across backends: one kernel source, the same f32
  accumulation order.
- The GFLOPS include upload and readback every call, and the tile computes one
  output per thread, so these are parity numbers, not a speed claim. Register
  tiling is the next kernel step.
- The AMD run was pinned with `VK_ICD_FILENAMES=radeon_icd.json`; the
  `gpu_busy_percent` peak was 74% on Navi 22 (88:00.0).
- The cesarops4om binary was built in `rust:1-bookworm` (glibc 2.36) because that
  host has glibc 2.39 and no Rust.

Not done yet from the N3 brief: parity on real OLMoE expert weights and
expert-id mismatches per backend (needs the layer0 dumps as input files).

## Harness faults found and fixed (cesaropsserver/cesarops-forge-runner)

- Aider sent no `max_tokens` for an unknown model, so Swift reasoned 29k tokens
  into an empty answer. The runner now writes per-run litellm metadata plus aider
  model settings: diff edits, output cap = 3/8 of the context, and
  `reasoning_budget_tokens` 4096.
- The worker ran the coder once. It now re-dispatches the same worktree with the
  test failure until `max_trips` commits, stopping on a round with no commit.
- A URL in a task made Aider scrape it and try to apt-install Playwright (it ran
  `apt-get update`; nothing was installed). The runner now passes
  `--no-detect-urls --no-browser`.
- Aider died on a symlink loop in cesaropsserver (227 unreadable tracked paths).
  The runner now passes a per-run `--aiderignore`.
- Swift's server config got `"fit_max_tokens": true`, so prompt plus cap overflow
  clamps instead of returning a 400.

## Gate-judge observations (Bonsai)

- Run 05fc88f7b4aa: Bonsai wrote "compiles and passes the listed tests" while the
  test gate showed a compile failure.
- On passing runs it returned `on_ask: false` with an empty issues list.
- It is not reliable as the only judge. The test gate and a second judge carry the
  decision until the role-qualification suite has scored it.
