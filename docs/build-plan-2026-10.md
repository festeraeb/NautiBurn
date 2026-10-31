# NautiBurn build plan — MoE routing for sm60 + AMD (2026-10-19)

Status: plan. Nothing here is measured unless it cites a file or a telemetry endpoint.
Builds on [json-frankenmoe-roadmap.md](json-frankenmoe-roadmap.md) (M0–M9) and
[stage-build-plan.md](stage-build-plan.md) (Stages 1–3); those stay the guardrails.
Briefs live in [`../briefs/`](../briefs/). One brief, one agent, one pass/fail.

## What we are building

An expert-routing runtime for MoE models on hardware PyTorch has left behind:
Pascal `sm_60` (P100s) and AMD cards (RX 6700 XT on cesarops4om, Polaris on cesarops3).
The runtime keeps the model's own gate authoritative and makes the expensive part —
getting the right expert weights next to the work — fast.

Five ideas, each a measured feature rather than a promise:

| Idea | What it means here | Lands in |
|---|---|---|
| **CubeCL kernels** | One expert-GEMM kernel source, compiled for CUDA `sm_60` and WGPU/Vulkan (AMD). Pascal tile already exists (Brief 2: `BM=BN=16, BK=32`, 4096 B shared). | N3 |
| **Lookahead + predetermined layers** | Per layer, a *pinned* hot set chosen offline from histodb (the "predetermined" part), plus L+1/L+2 lookahead from the predictor ensemble for the rest. Pinned sets never move; lookahead only prefetches. | N2, N6 |
| **JIT expert loading** | Exact-expert residency manager (hot = VRAM, warm = host RAM, cold = disk). A miss loads the exact expert on demand; prediction only orders and pre-starts loads. | N4, N6 |
| **Pass the work to the experts** | Expert parallelism: ship the token's activations to the device that already holds the expert instead of moving the expert. OLMoE: 4096 B activation per token vs ~12.5 MB per FP16 expert (`/Nauti-Burn/api/engine`). | N7 |
| **FrankenMoE** | Same-family expert composition only, behind a compatibility checker that fails closed. Cross-family → JSON specialist routing. | N8 |

## Baseline (measured, today)

- Prediction hit rate **4.21/8** experts (≈53%) over the last hour; belief latency 212–223 µs
  (`cesarops.com/Nauti-Burn/api/overview`, `/api/updates`). 11.5k requests, 99.7M firings in histodb.
- GLM-5.3-Flash REAP50 via PolyStrata on cesarops4om's 4 NVIDIA cards: 12.9 tok/s @32K, ~11.8 @64K;
  GPUs 14–23% busy during decode, ~half of 6192 experts live on CPU, 31% read cross-NUMA. The
  bottleneck NautiBurn should attack is expert placement, not GPU math.
- B6 ~5 tok/s (one P100, experts in RAM), B7 ~1.4 tok/s (Mixtral, 3×P100) — single runs, re-measure before quoting.
- **Mixtral 8x7B Instruct Q4_K_M (merged `_exps` GGUF, 28.4 GB) placement on cesarops4om, 2026-10-19**
  (llama.cpp mainline sm_60+75, layer split, same prompt, 300 tokens, greedy, 2 runs each):

  | Placement | prompt tok/s | decode tok/s |
  |---|---|---|
  | T10 + RTX 2060, experts of 6 layers on CPU (`--n-cpu-moe 6`) | 26.6 | 5.2 |
  | same, 5 layers on CPU | 23.3 | 8.5 |
  | same, 4 layers on CPU | 27.1 | 8.15 |
  | + 2 layers on t440 P100 over ggml-rpc (TCP, 25 GbE, 0.16 ms RTT), 3 layers on CPU | 44.3 | 10.9 |
  | **all 32 layers on GPUs: T10 + 2060 + both local P100s (shared with Tiel), no CPU experts** | **72.6** | **21.2** |
  | same, with Tiel generating at the same time | — | 17.9 (Tiel 28.6) |

  Lessons for N5–N7: every expert layer left on the CPU costs far more than an extra GPU stage, even a
  network one; a 25 GbE hop added ~0.3 ms/token and still won by freeing VRAM. Cards sat at 13–18 %
  utilisation with CPU experts present. The T10's idle PCIe Gen 1 reading was ASPM power saving; it runs
  Gen 3 x16 under load. Operator's own test: for single-token decode, computing a cold expert on the CPU
  beat loading it over PCIe Gen 3 (≈200 MB per Mixtral expert pair ≈ 17 ms at ~12 GB/s).

## Hardware lanes

| Node | Devices | Role in the build |
|---|---|---|
| cesarops4om | 2× P100 (sm_60), T10 + RTX 2060 (sm_75), RX 6700 XT (Vulkan, RADV) | Primary test bench: one box covers sm_60, sm_75 and AMD. GPUs mostly hold GLM + Bonsai — kernel tests must fit in free VRAM or get a scheduled window. |
| t440 | 2× P100 | Second sm_60 node; histodb + engine services (`nautiburn-engine :9119`). |
| cesarops3 | AMD Polaris | Second AMD target (older GCN, Vulkan only). |
| cesarops2 | P100s / RTX 2060 SUPER | Remote expert host for N9 (currently unreachable). |

## Phases and briefs

**Phase 1 — local correctness and data (no GPU contention, can run in parallel)**
- **N0 Truth audit** — label every claim in B0–B8 / Briefs 1–7 implemented / measured / estimated / proposed. Docs only.
- **N1 Durable observations** — `nautiburn-engine` HTTP keeps a fresh in-memory `Engine`; make `/scrape` ingest into histodb with idempotent event IDs and a restart test.
- **N4 Residency inventory** — host-side hot/warm/cold state machine keyed by (model, checkpoint, layer, expert), byte budgets, pinning, deterministic eviction tests. No GPU copies.

**Phase 2 — kernels and policy**
- **N2 Predetermined layers + held-out eval** — choose pinned sets from histodb per (model, layer, phase) under a byte budget; evaluate pinned ∪ lookahead on held-out requests: slot hit, exact-set, false-prefetch bytes, deadline-ready rate.
- **N3 One kernel, two backends** — run the Pascal FP16 expert tile through CubeCL on CUDA `sm_60` (P100) and WGPU/Vulkan (RX 6700 XT); parity vs CPU reference on real OLMoE expert weights, max abs error and expert-id mismatches reported per backend.
- **N5 Transfer benchmark** — disk→RAM, RAM→VRAM, VRAM→VRAM (same node) p50/p95/p99 per device class.

**Phase 3 — JIT loading and expert parallelism on one node**
- **N6 JIT fill** — exact-expert on-demand load behind a disabled-by-default flag; then predictor-ordered prefetch. Pass only with output parity under hit / miss / eviction / concurrency.
- **N7 Expert-parallel dispatch** — experts placed across cesarops4om's cards (NVIDIA + AMD); route activations to the holding device, gather results. Compare tok/s and stall time against weight-moving (N6) at equal VRAM.

**Phase 4 — composition and fabric**
- **N8 FrankenMoE checker + one static same-family artifact** (roadmap M6).
- **N9 Remote experts over nauti fabric** — N7's activation-shipping over the existing Iroh/QUIC `nauti agent-serve`, private LAN only, local path stays the correctness fallback.
- **N10 Virtualizer peer worker** — run an N9 expert host inside a Cloud Hypervisor microVM (Nauti-Virtualizer M9), registered like any other node.

Federation to outside users waits until N9/N10 have a threat model and measured LAN latency (roadmap M9 gate).

## How the build is driven

1. Each brief in `briefs/` is dispatched to a local agent through **Forge V3** `POST /cluster/agent/run`
   (tool-using agent, project root `/codebase/repos/cesaropsserver`). **Paddler** (`:8080`) supplies the
   seat: katcoder (cesarops4om `:8310`) for Rust, GLM for heavy reasoning, NIM Kimi K3 as reviewer.
2. Briefs name allowed files, commands and the exact evidence to return. Agents work on a branch
   `nb/<brief-id>`; they do not merge.
3. Every result is reviewed before the next brief in its chain is issued: diff read, tests re-run by the
   reviewer, numbers re-measured on hardware. **A result with simulated numbers, hardcoded outputs, or a
   test that cannot fail is a failed brief** — the agent is told why and the brief is re-issued.
4. Progress is posted to the buzz board so the human in the loop can see each gate open and close.

## Out of scope for now

Cross-family hidden-state grafting, WAN token-level MoE, tensor-parallel decode over the network,
public federation. Each needs its own gated track after Phase 4.
