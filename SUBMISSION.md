# Lemonade Developer Challenge — Submission
## CESARops: From a Missing Sailboat to a Federated AI Platform

**Original submission**: https://cesarops.com/lemonade/index.html
**GitHub**: github.com/festeraeb
**Built with**: Lemonade SDK, Cake, FreeToken, Burn/CubeCL, Vulkan (no ROCm)

---

## How This Started

A sailboat named Rosa went missing off Milwaukee. The Coast Guard wouldn't run drift models. So I built them.

CESARops is a civilian search and rescue platform that grew from that one refusal into a full AI orchestration system. It processes side-scan sonar, runs hydrocarbon analysis, classifies hillshade imagery from .bag files, models Great Lakes drift patterns, and coordinates search missions across a fleet of volunteer vessels.

Lemonade serves CESARops. It runs the vision models that classify sonar imagery. It runs the drift models. It serves the inference layer that Forge v3 dispatches specialist workers to. It's not a demo — it's the backend for a system that helps find missing people.

---

## The Stack

### forge_lemon-cake — NVIDIA Inference Engine
FreeToken fork with MoE offloading, expert caching, GGUF loading, JIT compilation. Runs Qwen3.6-35B, KAT-Coder, OLMoE on Tesla P100s. nauti-monitor fleet telemetry with MCP surface for agents.

**Status**: ✅ Live. 3 models serving. Activation scraping feeding histograms.

### iced-lemcake — AMD Inference Path
Cake (evilsocket's distributed inference, layer-sharding across devices) meshed with Lemonade's Vulkan serving. No ROCm needed. Verified on AMD Radeon Pro WX 5100.

**Status**: ✅ Lemonade verified. Cake integration scaffolded. Forge plugin slot exists.

### NautiBurn — Predictor and Learner
Predicts which MoE experts will fire BEFORE the router decides. 65.2% hit rate on real OLMoE traces. SQLite histograms learn from every request. CubeCL GEMM proven on P100.

**Status**: ✅ Predictor live. Histodb on 3 models. CubeCL GEMM matched oracle (err 2.7e-3).

### Nauti-Virtualizer — Unified Fabric
Topology-aware resource fabric. LemonadeAdapter for AMD. Registry, leases, capability filtering. Every device in one scene. No SSH. No tmux.

**Status**: ✅ LemonadeAdapter proven on WX 5100.

### Forge v3 — Orchestrator
Converts human intent into validated contracts. Dispatches specialist workers to inference backends. Thom translator (local llama-server → NIM fallback). Plugin architecture for inference engines (lemonade, llama.cpp, NIM, experimental Vulkan).

**Status**: ✅ Production. Serving CESARops missions.

### Cognitive Contracts
Models earn grades by measured performance. Specialist writes. Reviewer judges. cargocheck verifies. Judge scores. No model grades its own work. User contracts travel with them through Forge — describes who they are, how they work, what they need.

**Status**: 🔶 Schema complete (10 tests). Grading pipeline wiring pending.

### Wayfinder
ADHD helper for students and programmers. Tauri 2.0 desktop app. File scanning, embeddings, clustering, Git assistant.

**Status**: ✅ Working. Wayfinder Campus (Raspberry Pi/Chromebook) scaffolded.

---

## What Lemonade Does in CESARops

Lemonade isn't just a model server. It's the inference layer that powers:

- **Vision models** on .bag file hillshade classifications (side-scan sonar imagery)
- **Hydrocarbon analysis** for pollution detection
- **Drift models** for missing vessel search patterns
- **Thom translator** — intent translation for mission planning
- **Specialist workers** — code review, planning, analysis dispatched by Forge
- **AMD hardware** — WX 5100, RX 580, and older cards via Vulkan (no ROCm)

The cognitive contract travels with every request. A model stepping in to help knows: this user built this system because the Coast Guard said no. They measure everything. They don't trust simulated results. Push back on what's not real.

---

## Keeping Old Hardware Useful

| Device | Era | Role | Status |
|---|---|---|---|
| Tesla P100 ×6 | Pascal (2016) | Primary inference (Tiel, Kat, OLMoE) | ✅ Live |
| AMD WX 5100 | Polaris (2016) | Reviewer via Lemonade | ✅ Live |
| GTX 1070 | Pascal (2016) | SmolLM reviewer | ✅ Live |
| RTX 2060 | Turing (2019) | Thinker | ✅ Live |
| ThinkPad P51 | Maxwell-era | Reviewer (next to join) | 🔗 |

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

## What Strix Halo Would Mean

The Polaris-era and Pascal-era hardware has been carrying this system. A Strix Halo (128GB unified memory, NPU) would:

1. Run 70B+ models locally (no offloading)
2. Serve as federation leader (orchestrator + judge)
3. Run cognitive contract grading locally (no cloud API)
4. Serve an entire classroom via Lemonade + Wayfinder Campus
5. Enable Cake to shard across NPU + GPU + remote workers

---

## Source Code
```
forge_lemon-cake/       ← NVIDIA inference (FreeToken + MoE)
iced-lemcake/           ← AMD inference (Cake + Lemonade)
NautiBurn/              ← Predictor + histodb + CubeCL
Nauti-Virtualizer/      ← Unified fabric
Wayfinder/              ← ADHD helper (Tauri 2.0)
wayfinder-Campus/       ← School version
Forge/                  ← Orchestrator, gates, plugins
nauti-nodes/            ← Federation coordinator
nauti-monitor/          ← Fleet telemetry
CESARops/               ← SAR platform
```

---

## Team
CESARops — Civilian Emergency Search and Rescue Operations Platform
Solo developer. ADHD + dyslexia. Built this because the Coast Guard said no.
github.com/festeraeb · cesarops.com
