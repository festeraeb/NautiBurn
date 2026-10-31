# NautiBurn: JSON Routing + Burn/CubeCL + FrankenMoE Roadmap

Status: planning document; no implementation or fresh benchmark is claimed here.

## Objective

Develop two related but deliberately separate capabilities:

1. **JSON model/role routing**: route a request between complete, possibly heterogeneous models using a versioned JSON contract and readable text/context handoffs.
2. **Model-internal MoE and FrankenMoE**: run one defined model correctly, preserve its true router decisions, predict upcoming expert residency, and later compose experts only when their weights are demonstrably compatible.

Tensor parallelism and cross-node federation are later, separately gated tracks. A JSON control-plane message is not a hidden-state tensor, and a predictor is not the model's MoE gate.

## Current repository baseline

The project is already a useful sidecar foundation, but it is not yet the proposed Burn inference runtime:

- The root Cargo workspace contains schema, prediction, engine, histodb, and shadow crates. The layer0 CubeCL experiment is a separate workspace. See the root [Cargo workspace](../Cargo.toml#L1-L9).
- `nautiburn-engine` has typed `ExpertPass`/`NextExperts`, prediction selection, byte/copy estimates, and an HTTP surface. The HTTP service creates a fresh in-memory `Engine`; `/scrape` writes a scrape document rather than ingesting it into persistent history. See [HTTP service](../crates/nautiburn-engine/src/http.rs#L24-L46) and [scrape handler](../crates/nautiburn-engine/src/http.rs#L101-L130).
- `nautiburn-shadow` has the predictor ensemble, pre-prediction, JSON belief output, and a device-map scheduler. The scheduler explicitly logs only: it does not dispatch or move weights. Its ensemble is a union of predictor candidates, not a model-output ensemble and not necessarily the model's top-k set. See [shadow contract](../crates/nautiburn-shadow/README.md#L1-L26) and [scheduler contract](../crates/nautiburn-shadow/src/schedule.rs#L1-L12).
- `nautiburn-layer0` runs a selected OLMoE FFN/MoE experiment with a custom CubeCL FP16 tile. It skips attention and is not end-to-end LLM inference. See [layer0 implementation](../crates/nautiburn-layer0/src/main.rs#L1-L16).
- The existing architecture note itself says the engine does not depend on Burn and the WGPU crate only links a backend type; it also describes the 12 GB/s and 13 ms figures as planning assumptions, not measured copies. See [Burn engine note](burn-expert-engine.md#L15-L29).
- Tensor split must not be reported as proven: the B7 note says tensor mode failed because flash attention was required and the run that stayed up used layer split. Reconcile the contradictory B7 prose before using it as a performance baseline. See [B7 run evidence](../B7.md#L64-L78).

## Architecture review

The broad direction is viable, with important boundaries:

- **Keep actual routing authoritative.** At inference time, the model's gate chooses the exact experts. NautiBurn's predictors may prefetch/cache those likely experts, but must not silently replace the gate's choices with a “nearest resident” expert. On a miss, compute the correct expert, wait, or fail explicitly; changing expert identity changes the model.
- **JSON routing is coarse-grained model/agent orchestration.** A different model receives serialized text/context, tokenizes it itself, and returns a structured result. This works across unrelated model families, but it is not token-level MoE and adds a complete model-call boundary.
- **FrankenMoE needs a compatibility contract.** Arbitrary experts from different models cannot be dropped into one another merely because matrix dimensions match. The safe initial case is experts from the same checkpoint family/base, with matching architecture/config, layer, tensor layout, dtype and router semantics. Cross-base grafting requires an explicit alignment/training method and evaluation; otherwise treat those models as JSON-routed specialists.
- **Predictor ensemble is not a model ensemble.** Keep its union as a prefetch candidate set and measure its coverage, size, predictor cost, false prefetch bytes, and deadline hit rate. Do not confuse `|predicted ∩ actual|/k` with exact-set accuracy or a router decision.
- **Tensor parallel is not expert routing.** It shards matrix math and needs frequent collectives. Validate it independently from MoE placement. Do not target WAN tensor-parallel decode as an initial feature; dependent per-layer round trips make latency a first-order constraint.
- **Hot/warm/cold is a measured cache policy, not a preset.** Measure actual device-to-host, host-to-device, storage-to-host and remote transfer latency, concurrency, and memory pressure on the target hardware. Async overlap, pinned memory, stream/event safety, peer access, and CubeCL backend support must be demonstrated rather than assumed.

## Target data/control flow

### Model-internal MoE path

`inference runner -> true gate/top-k -> actual expert IDs -> NautiBurn observation`

In parallel, the predictor proposes a bounded, confidence-ranked prefetch set for the next layer(s). A local residency manager can move exact expert weights among hot (device memory), warm (host RAM), and cold (local storage). The executor always uses the gate's actual IDs. Record whether each prefetched expert was used, ready by deadline, evicted, or wasted.

### Heterogeneous JSON model routing

`request -> policy/router -> selected complete model -> validated JSON result -> optional next model`

Each hop has a model identity/version, request ID, input schema, output schema, deadline, timeout, and fallback policy. The payload contains task-relevant text/context and structured metadata, never a claim that hidden vectors are portable across model families.

### Future FrankenMoE path

Start with compatible experts assembled offline into a versioned checkpoint/package. Verify its layer/expert map and compare it with a reference. Only after that, allow the residency manager to cache/move those packaged experts. Runtime weight blending is a later optimization, not the first proof of correctness.

## Milestones and exit gates

### M0 — Reconcile repository truth and choose the target

- Resolve B7's tensor-vs-layer-split status and the Brief 4 claim versus shadow-only scheduler behavior.
- Choose OLMoE-1B-7B / its precise checkpoint and quantization as the first architecture, since the repository already has OLMoE router traces and selected tensor experiments.
- Record the reference runtime, checkpoint hash, tokenizer/config identity, GPU/CUDA/CubeCL versions, and baseline outputs/performance.
- Decide whether milestone one is a Burn FFN/MoE block embedded beside a reference runtime or a complete Burn model. Do not call the FFN-only experiment end-to-end inference.

**Gate:** reproducible baseline and written scope; no performance claim based only on estimates.

### M1 — Single-model correctness, true router, reference parity

- Make one model's router and expert block deterministic and testable on real activations from the reference run.
- Compare selected expert IDs exactly; compare expert outputs/logits with a predeclared dtype-specific tolerance and report top-token agreement.
- If pursuing full Burn inference, add missing architecture components in dependency order (weight loading, embeddings, norm/attention/position handling, MoE blocks, KV cache, tokenizer/sampling) and gate every block against the reference. First prove the current custom CubeCL FP16 expert kernel on the actual target path.

**Gate:** repeatable parity on fixed inputs, and an honest statement of whether the artifact is an expert-block prototype or full model.

### M2 — Make observation, JSON, and history reliable

- Define versioned, validated observation and prediction schemas. Include model/checkpoint identity, layer, phase, request/token position, actual expert IDs, predicted IDs, and prediction source/confidence.
- Make the engine service consume observations into persistent histodb state or clearly delegate that operation to a single durable component; make restart/recovery behavior explicit.
- Add idempotency by request/event ID, model/top-k/range validation, unknown/empty semantics, and backward-compatible schema tests.
- Keep activation tensors out of JSON. If later needed, define a separate typed binary transport.

**Gate:** process restart preserves history; replay and HTTP return equivalent routes; malformed/unknown-model data is rejected safely.

### M3 — Calibrate the ensemble as a prefetch predictor

- Keep frequency, previous-token, bigram, and their union separate from the model gate.
- Evaluate per model, layer, and prefill/decode phase on held-out requests, not only the same observations used to form predictions.
- Report slot hit, exact-set rate, any-overlap, candidate count, predictor latency, false-prefetch bytes, and deadline-ready rate at horizons L+1/L+2. Use a fixed prefetch-byte budget.
- Add confidence/calibration and an explicit no-prefetch option. A larger union can improve coverage while increasing transfer cost; score that trade-off directly.

**Gate:** a repeatable baseline table and a policy that never changes the model's selected expert IDs.

### M4 — Local exact-expert residency, before networking

- Implement a model/layer/expert/checkpoint-keyed inventory and byte budget for hot, warm, and cold tiers. Define pinning, eviction, concurrent request ownership, and version invalidation.
- First implement synchronous exact-expert fills and misses; test that the requested weights and outputs are identical to the always-resident path.
- Measure storage→RAM and RAM→GPU transfers, p50/p95/p99, cache hit rate, stall time, and wasted-prefetch bytes on the target card. Then investigate pinned buffers, async copies, CUDA/CubeCL stream/event ordering, and double-buffered publication.
- Only enable prediction-driven prefetch if it beats the baseline end-to-end. Avoid claiming “zero stall” until measured.

**Gate:** correct outputs under eviction/miss/concurrency tests and a measured local performance result.

### M5 — JSON routing across complete models (heterogeneous ensemble)

- Define a model registry and router policy that selects complete specialist models by task/capability/health/cost/deadline.
- Use JSON input/output schemas with strict validation, request IDs, bounded context, timeout/cancellation, retries and explicit fallback. Preserve provenance so each result can be audited.
- Build the ensemble at the response/text boundary (one model proposes, another reviews/transforms, etc.). Do not pass hidden-state tensors between different model families.
- Ensure the local NautiBurn expert-prediction endpoint remains a separate service/API from model-role dispatch.

**Gate:** deterministic contract tests using local mock endpoints, then a measured local-model smoke; failure of a specialist must not corrupt or hang the request.

### M6 — Define and validate a compatible FrankenMoE

- Pick a precise definition: first, same-family expert replacement/composition in a static artifact, not an arbitrary cross-model tensor graft.
- Add a compatibility checker for base/checkpoint identity, architecture/config, tokenizer, layer/expert indices, dimensions, tensor names/layouts, dtype and quantization.
- Build one small offline composition, validate loadability, exact router semantics and reference outputs, then evaluate quality on task suites and regression prompts.
- If sources have different bases, require a separately specified alignment/projector training/evaluation track; otherwise expose them as JSON-routed complete models.

**Gate:** artifact manifest plus correctness/quality report; unsupported grafts fail closed.

### M7 — Dynamic packaged-expert caching and prefetch

- Reuse the M4 exact-expert residency API for the M6 packaged artifact; do not dynamically blend full weight matrices per token.
- Add a cache-aware execution schedule and measured prefetch windows. Preserve the gate's chosen experts and support a precise miss path.
- Explore on-device fusion/weight blending only for cases where weights are already resident and a benchmark proves benefit over selecting a prebuilt expert.

**Gate:** parity with the static FrankenMoE artifact and a measured improvement over no prefetch.

### M8 — Local multi-GPU tensor/expert placement

- Separately benchmark layer parallel, tensor parallel, and expert placement on known-good hardware/runtime settings. Reconcile current B7 evidence first.
- Add explicit topology/capability discovery (PCIe/NVLink, device memory, backend, dtype support), collectives and transfer benchmarks; test correctness against the single-device reference.
- Use local PCIe/NVLink or LAN first. MoE all-to-all and tensor-parallel collectives are different protocols and need separate measurements.

**Gate:** successful supported-mode launch, exact routing, reference parity and end-to-end latency/throughput results.

### M9 — Federated nodes, only after local behavior is stable

- Begin with JSON task-level dispatch to opted-in worker nodes; use authenticated node registration, signed model/artifact manifests, capabilities, leases, health, quotas, deadlines, cancellation and audit logs.
- If remote same-model expert execution is later justified, pin exact model/checkpoint/layer/expert identity and specify the activation wire format, dtype, shape, batching, encryption, integrity and timeout. Keep a correct local path; do not silently substitute a different expert.
- Treat WAN token-level MoE as a research experiment, not the initial production path. Measure the full serial latency and failure rate, not just payload size.

**Gate:** threat model, protocol tests, controlled private-network benchmark and an explicit rollback path before public participation.

## Forge-ready small work orders

Use the repository's existing **one brief, one agent, one pass/fail** convention from [BRIEF_LEASE](../BRIEF_LEASE.md#L1-L7). Each task below should be submitted separately; do not ask a worker to implement the whole roadmap in one prompt.

1. **Truth audit:** reconcile B7/Brief 4 statuses and create a source-linked inventory; no code changes. Pass when every claim is labeled implemented, measured, estimated, or proposed.
2. **Target contract:** write a short OLMoE model/checkpoint and parity-test contract. No runtime changes. Pass when shapes, IDs, dtype, reference and tolerances are explicit.
3. **JSON schema:** add versioned observation/prediction types and validation tests only. Pass when bad IDs, model mismatches, and empty routes have defined behavior.
4. **History integration:** connect the HTTP observation path to the agreed persistent store, with idempotency and restart tests. Do not alter predictor math.
5. **Ensemble evaluation:** add held-out evaluation and the requested coverage/cost metrics. Do not alter inference routing or GPU behavior.
6. **Single expert parity:** make a minimal real-activation test for one OLMoE expert on the CubeCL path against the reference. Do not expand to full-model inference in this brief.
7. **Resident inventory:** implement the host-side cache state machine and deterministic eviction tests only; no GPU copies yet.
8. **Transfer benchmark:** measure storage/RAM/GPU transfer paths and report percentiles on a specified idle device; do not claim asynchronous overlap.
9. **Local prefetch:** add exact-expert transfers behind a disabled-by-default feature flag; pass only with parity under hit/miss/eviction/concurrency tests.
10. **JSON specialist router:** implement mockable whole-model selection and validated request/response envelopes; no hidden-state transport.
11. **FrankenMoE compatibility checker:** reject incompatible manifests and test same-family artifacts; do not create cross-base projectors.
12. **Static FrankenMoE proof:** build and evaluate one declared compatible composition; no runtime weight blending.
13. **Tensor-parallel feasibility:** reconcile B7, run one supported local topology, and report correctness/latency; no federated work.
14. **Federation protocol design:** threat model and protocol spec only, after M5/M8 gates; do not expose public workers yet.

Every implementation brief must name allowed files, dependencies, tests/commands, expected evidence, and explicit non-goals. Ask Forge to return a concise change summary, test output, measured evidence, and unresolved risks. Review each result here before issuing the next brief.

## Tooling / orchestration note

The repository documents Forge's staged brief workflow, which is suitable for this breakdown. The mounted drive contains local Forge and Warp source trees (`/mnt/t440/home/cesarops_push/forge` and `/mnt/t440/codebase/repos/cesaropsserver/warp-grid-standalone`), but this chat has no direct Forge or `/warp` dispatch tool. I have not submitted jobs or verified that a live dispatch endpoint is available. These are prepared Forge-ready prompts rather than submitted jobs. I can act as reviewer/orchestrator when you decide to launch them: split work, check scope, inspect diffs and evidence, polish follow-up prompts, and gate each milestone before the next.
