# Lemonade Developer Challenge — Submission
## CESARops: A Complete AI Stack, Tested Piece by Piece

**Original submission**: https://cesarops.com/lemonade/index.html
**GitHub**: github.com/festeraeb
**Built with**: Lemonade SDK, Cake, FreeToken, Burn/CubeCL, Vulkan (no ROCm)

---

## The Stack (Top to Bottom)

### 1. forge_lemon-cake — The NVIDIA Inference Engine
FreeToken fork with MoE offloading, expert caching, GGUF loading, CPU fallback, JIT compilation, and nauti-monitor fleet telemetry.

**Tested**: ✅ Running live. OLMoE, Tiel (Qwen3.6-35B), KAT-Coder all serving on P100s with live activation scraping.

### 2. iced-lemcake — The AMD Inference Path
evilsocket's Cake (distributed inference, layer-sharding across devices) meshed with Lemonade (Vulkan serving, no ROCm). Cake shards model layers across heterogeneous devices. Lemonade serves on AMD via Vulkan.

**Tested**: ✅ Lemonade verified on AMD Radeon Pro WX 5100. Bonsai-1.7B serving via Vulkan. Nauti-Virtualizer LemonadeAdapter discovers it automatically.

### 3. Nauti-Virtualizer — The Unified Fabric
Topology- and policy-aware virtual resource fabric. Registry, leases, capability filtering. Every device — NVIDIA, AMD, CPU — appears in one scene. No SSH. No tmux. Agents see one resource pool.

**Tested**: ✅ LemonadeAdapter proven on WX 5100. GPU discovery via all-smi + sysfs. Lease model and adapter architecture documented.

### 4. Nauti-Burn — The Predictor and Learner
Predicts which MoE experts will fire BEFORE the router decides. Learns from every request via SQLite histograms. Four prediction methods: frequency (35.8%), prev-token copy (55.8%), cross-layer bigram (41.5%), ensemble (65.2%).

**Tested**: ✅ 65.2% hit rate measured on OLMoE-1B-7B (64 experts, top-8, 16 layers). Histodb live on 3 models. Sub-13ms prediction time.

### 5. The Federation — Work to the Model, Not Model to the Work
Traditional approach: load experts onto the GPU where the user is. Our approach: route the token to wherever the expert already lives. GPU expert → GPU compute. CPU expert → AVX2 fallback. Remote expert → fabric transport.

**Tested**: ✅ CPU expert fallback measured (B6): 72 tok/s resident vs 26 tok/s offload. Q4 copy + GEMM beats CPU (0.56ms vs 4.22ms). CubeCL GEMM on P100 matches oracle (err 2.7e-3).

### 6. Cognitive Contracts — Models Earn Their Grade
Specialist writes code. Reviewer judges it. cargocheck verifies it compiles. Judge scores the chain. No model grades its own work. Every grade backed by measured evidence.

**Tested**: 🔶 Schema complete (10 tests pass). Scorecard design complete. Grading pipeline not yet wired end-to-end.

### 7. Wayfinder — The User-Facing App
ADHD helper for students and programmers. Tauri 2.0 desktop app. File scanning, embeddings, clustering, semantic search, Git assistant. Currently uses koboldcpp for local LLM.

**Tested**: ✅ Desktop app working. Wayfinder Campus (static HTML/CSS/JS for Raspberry Pi/Chromebook) scaffolded.

---

## Keeping Old Hardware Useful

This stack runs on hardware that the market has abandoned:

| Device | Era | GPU | What It Does | Status |
|---|---|---|---|---|
| Tesla P100 ×6 | Pascal (2016) | GP100, 16GB HBM2 | Primary inference (Tiel, Kat, OLMoE) | ✅ Live |
| ThinkPad P51 | Polaris-era | AMD/Intel | Reviewer node | Next to join |
| AMD WX 5100 | Polaris (2016) | Ellesmere, 8GB | Reviewer/Specialist via Lemonade | ✅ Live |
| GTX 1070 | Pascal (2016) | GP104, 8GB | SmolLM reviewer | ✅ Live |
| RTX 2060 | Turing (2019) | TU106, 8GB | Thinker role | ✅ Live |

The P100 is the only Pascal chip with true native FP16 at 2× FP32 rate (21.2 TFLOPS). Everyone else threw these away. We built an inference engine that exploits what they're actually good at.

---

## How ROCm and Lemonade Enhance the Stack

Lemonade is already built into the stack via Nauti-Virtualizer's LemonadeAdapter:

- **No ROCm needed** for consumer AMD: mesa radv Vulkan driver handles WX 5100, RX 580, and older
- **ROCm optional** for datacenter AMD: if ROCm is available, Lemonade can use it
- **Same federation**: AMD nodes and NVIDIA nodes share the same histodb, same predictor, same grading
- **Cake integration**: evilsocket's Cake shards layers across AMD + NVIDIA + CPU devices in one cluster

The P51 joins via Lemonade Vulkan. The WX 5100 is already serving. Future AMD hardware (Strix Halo, RDNA 4) plugs in the same way.

---

## What's Tested vs What's Claimed

### Tested (measured on real hardware)
- ✅ Lemonade on WX 5100 (Vulkan, serving Bonsai-1.7B)
- ✅ CubeCL GEMM on P100 sm_60 (err 2.7e-3, 0 mismatches)
- ✅ Expert predictors at 65.2% hit rate (OLMoE, 227 rows)
- ✅ Histodb live on 3 models (Tiel, Kat-Coder, OLMoE)
- ✅ Q4 resident 72 tok/s, CPU offload 26 tok/s (P100)
- ✅ Mixtral Q4 on 3× P100 at 22.6 tok/s
- ✅ nauti-monitor fleet telemetry (MCP surface, WASM HUD)
- ✅ Wayfinder desktop app (Tauri 2.0)
- ✅ nautiburn-schema (10 tests pass)
- ✅ nautiburn-histodb (3 migrations, CLI, AllenAI import)

### Claimed (architecture complete, not yet wired end-to-end)
- 🔶 Cognitive contract grading pipeline (schema done, wiring pending)
- 🔶 Live prefetch from predictor (shadow mode working, live swap pending)
- 🔶 Wayfinder + Lemonade backend (adapter exists, swap pending)
- 🔶 Federation node protocol (nauti-nodes coordinator working, protocol pending)
- 🔶 Cake + Lemonade integration (both working separately, mesh pending)

---

## What Strix Halo Would Mean

Right now the stack runs on Polaris-era (WX 5100, P51) and Maxwell/Pascal-era (P100, GTX 1070) hardware. These GPUs have 8-16GB VRAM, PCIe Gen3 bandwidth, and no tensor cores.

A Strix Halo (Ryzen AI Max+ 395, 128GB unified memory, NPU) would change the system fundamentally:

| Capability | Current (Polaris + Pascal) | With Strix Halo |
|---|---|---|
| Max model size | 35B MoE (offload to CPU) | 70B+ dense (fits in unified memory) |
| Expert caching | 4-16 experts in VRAM | Hundreds in unified memory |
| NPU inference | Not available | Lemonade NPU backend |
| Federation role | Worker nodes | Leader node (orchestrator + judge) |
| Cognitive contracts | Judge via cloud API | Judge runs locally |
| Wayfinder AI | koboldcpp on NVIDIA | Lemonade on AMD, any model |
| Cake sharding | Across P100s (PCIe) | Across NPU + GPU + remote (unified memory) |
| Classroom deployment | Needs NVIDIA GPU | One Strix Halo serves entire classroom via Lemonade |

The Strix Halo doesn't replace the old hardware. It leads the federation. The P100s become expert offload nodes. The WX 5100 becomes a reviewer. The P51 becomes a staging box. Every piece earns its role by measured performance.

---

## Measured Performance

| Hardware | Model | tok/s | Notes |
|---|---|---|---|
| Tesla P100 (resident Q4) | Qwen1.5-MoE-2.7B | 72.0 | llama-server |
| Tesla P100 (CPU offload) | Qwen1.5-MoE-2.7B | 26.4 | --cpu-moe |
| 3× Tesla P100 | Mixtral-8x7B Q4 | 22.6 | layer split |
| CubeCL GEMM on P100 | OLMoE layer 0 | err 2.7e-3 | NautiBurn |
| AMD WX 5100 (Vulkan) | Bonsai-1.7B | serving | Lemonade |
| Predictor ensemble | OLMoE-1B-7B | 65.2% hit | NautiBurn |

---

## Source Code
```
forge_lemon-cake/       ← NVIDIA inference (FreeToken + MoE offload)
iced-lemcake/           ← AMD inference (Cake + Lemonade)
NautiBurn/              ← Predictor + histodb + CubeCL GEMM
Nauti-Virtualizer/      ← Unified fabric + LemonadeAdapter
Wayfinder/              ← ADHD helper (Tauri 2.0)
wayfinder-Campus/       ← School version (static HTML/CSS/JS)
nauti-nodes/            ← Federation coordinator
nauti-monitor/          ← Fleet telemetry
```

---

## Team
CESARops — Civilian Emergency Search and Rescue Operations Platform
Solo developer. ADHD + dyslexia. Builds tools for people like him. Measures everything because simulated results cost time.
github.com/festeraeb · cesarops.com
