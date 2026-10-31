# Lemonade Developer Challenge — Submission
## Wayfinder + Lemonade: Local AI for ADHD Students on Any Hardware

**Original submission**: https://cesarops.com/lemonade/index.html
**Current repo**: https://github.com/festeraeb/Wayfinder
**Built with**: Lemonade SDK, Tauri 2.0, Vulkan (no ROCm)

---

## What's Running Right Now

### Wayfinder (Tauri 2.0 desktop app)
ADHD helper for students and programmers. Scans files, generates embeddings, clusters projects, semantic search, Git assistant. Currently uses koboldcpp for local LLM inference.

### Lemonade on AMD WX 5100 (verified, serving)
Bonsai-1.7B via Vulkan on AMD Radeon Pro WX 5100. No ROCm. Mesa radv. Discovered by Nauti-Virtualizer LemonadeAdapter.

### NautiBurn testing lab (live on 3 models)
Expert predictor (65.2% hit rate) + histodb (SQLite histograms) + CubeCL GEMM on P100. Measured on real hardware.

---

## The Quick Win: Wayfinder + Lemonade

Wayfinder already runs local LLMs via koboldcpp. Swap in Lemonade as the backend and Wayfinder works on AMD hardware for free:

| Before (koboldcpp) | After (Lemonade) |
|---|---|
| NVIDIA only | AMD, NVIDIA, Intel, any Vulkan GPU |
| Manual setup | `lemonade pull model && lemonade load model` |
| No ROCm AMD support | Vulkan backend, no ROCm needed |
| Single GPU | Future: Cake sharding across devices |

### What This Means for Students
A student with a school Chromebook, a Raspberry Pi, or an old laptop with an AMD GPU can run Wayfinder's AI features locally. No cloud. No API keys. No cost.

Wayfinder Campus (static HTML/CSS/JS for Raspberry Pi and Chromebook) connects to a Lemonade server on the same network. One Lemonade server in the classroom, every student connects.

---

## The Bigger Picture: Federation as Test Lab

NautiBurn's federation isn't just for running models. It's a **test lab for hardware and models**.

### How It Works
Every node in the federation runs NautiBurn's histodb. Every request feeds the histogram. Every model gets graded by cognitive contracts (specialist writes, reviewer judges, cargocheck verifies, judge scores).

### What This Enables
- **AMD tests a new GPU**: Ship it as a federation node. Real workloads, real grading, real measurements. No synthetic benchmarks.
- **A model provider tests a new model**: Deploy it as a specialist. The scorecard grades it against every other model on every task class.
- **A school tests hardware**: Add a Chromebook as a node. Wayfinder Campus connects. The histodb tracks what works and what doesn't.
- **No public release needed**: The sandbox protects everyone. Results are measured, not claimed.

### The Contract
Every node signs a cognitive contract:
- **Role**: what the node does (specialist, reviewer, etc.)
- **Grade**: earned by measured performance over time
- **Sandbox**: the node's work stays in its sandbox until graded
- **No simulated results**: every number from real hardware

---

## What's Live vs What's Next

### Live Now (submit this)
- ✅ Wayfinder desktop app (Tauri 2.0, Rust + React)
- ✅ Lemonade on AMD WX 5100 (Vulkan, no ROCm)
- ✅ NautiBurn histodb on 3 models
- ✅ Expert predictor at 65.2% hit rate
- ✅ CubeCL GEMM on P100 (err 2.7e-3)
- ✅ Fleet telemetry (nauti-monitor)

### Next (what the foundation enables)
- Wayfinder + Lemonade backend swap
- Wayfinder Campus + classroom Lemonade server
- Federation node protocol (any device joins)
- Cognitive contract grading pipeline
- Cake integration for cross-device sharding

---

## Measured Performance

| Hardware | Model | tok/s | Notes |
|---|---|---|---|
| Tesla P100 (resident Q4) | Qwen1.5-MoE-2.7B | 72.0 | llama-server |
| Tesla P100 (CPU offload) | Qwen1.5-MoE-2.7B | 26.4 | --cpu-moe |
| 3× Tesla P100 | Mixtral-8x7B Q4 | 22.6 | layer split |
| CubeCL GEMM on P100 | OLMoE layer 0 | err 2.7e-3 | NautiBurn |
| AMD WX 5100 (Vulkan) | Bonsai-1.7B | serving | Lemonade |

---

## What the Strix Halo Enables
1. 128GB unified memory → 70B+ models locally
2. NPU + GPU hybrid via Lemonade
3. Wayfinder runs on it with full AI features
4. Federation leader — other devices connect to it
5. Local judge model — cognitive contracts without cloud

---

## Source Code
```
Wayfinder/              ← ADHD helper (Tauri 2.0, Rust + React)
wayfinder-Campus/       ← School version (static HTML/CSS/JS)
NautiBurn/              ← Testing lab (predictor + histodb)
iced-lemcake/           ← Cake + Lemonade (AMD path)
forge_lemon-cake/       ← FreeToken + NautiBurn (NVIDIA path)
```

---

## Team
CESARops — Civilian Emergency Search and Rescue Operations Platform
Solo developer. ADHD + dyslexia. Builds tools for people like him.
github.com/festeraeb · cesarops.com
