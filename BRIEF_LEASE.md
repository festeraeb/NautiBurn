# NautiBurn — Agent Brief Lease

Stack of dispatchable briefs for Forge v3. One brief, one agent, one pass/fail.
Repo: `/home/cesarops/NautiBurn/` | GitHub: `https://github.com/festeraeb/NautiBurn.git`

**Rule**: Every brief produces measured evidence. No simulated results. No "it should work."
If the number isn't on hardware, it doesn't go in the milestone.

## Execution Order

```
Brief 1 (Close B7) ───────────────────────────── ✅ CLOSED 2026-09-30
Brief 2 (FP16 tile) ──────────────────────────── ✅ CLOSED 2026-09-29
Brief 3 (Engine HTTP API) ────────────────────── ✅ CLOSED 2026-09-29
Brief 7 (FreeTokens bridge) ── needs Brief 2 ─── ✅ CLOSED 2026-09-29
Brief 4 (Shadow → cache swap) ── needs Brief 3 ─ ✅ CLOSED 2026-09-29
Brief 5 (Wire to nauti-inferer) ─ needs Brief 3 ─ ✅ CLOSED 2026-09-29
Brief 6 (Paddler routing) ──── needs 3 + 5 ───── ✅ CLOSED 2026-09-29
```

**Blocked**: none. **Closed**: Briefs 1–7. **All briefs complete.**

---

## BRIEF 1: Close B7 — hdrhistogram + cpu-moe comparison
**Files**: `B7.md`, `b7/` | **Effort**: 1-2 days

- [x] `--cpu-moe` single-card Mixtral timings (prompt_ms, predicted_ms, tok/s, VRAM)
- [x] hdrhistogram or percentile latency data in `b7/`
- [x] Horizon 1→2→3 measured (Q4 copy time vs GEMM time)
- [x] B7.md CLOSED, Kat-Coder restored after test
- **Evidence**: timing JSON in `b7/`, copy-vs-GEMM per horizon

## BRIEF 2: Force FP16 tile under P100 smem cap
**Files**: `crates/nautiburn-layer0/src/gemm_tile.rs` | **Effort**: 1 day

- [x] `gemm_f16()` called from layer-0 block for expert GEMM (`main.rs:97`)
- [x] P100 launch not refused (shared 4096 ≤ cap 49152) — `probe` confirms
- [x] Max abs error < 5e-2 (probe_max_abs=1.215e-4), expert-id mismatches: 0
- **Key**: `BM=BN=16, BK=32, double-buffered, FP16→FP32→FP16, 4096 bytes shared`

## BRIEF 3: nautiburn-engine HTTP API
**Files**: `crates/nautiburn-engine/` | **Effort**: 2-3 days

- [x] `POST /route` (ExpertPass → NextExperts), `POST /scrape` (→ histodb)
- [x] `GET /health`, `GET /stats`, bind `127.0.0.1:9119`
- [x] CLI still works. Add axum + tokio. State: `Arc<Mutex<Engine>>`.
- **Evidence**: `GET /health` ("ok"), `GET /stats`, `POST /route` verified live on port 9119.

## BRIEF 4: Shadow predictor → live cache swap
**Files**: `crates/nautiburn-shadow/src/schedule.rs` | **Effort**: 1 week | **Needs**: Brief 3

- [x] `prefetch_decision()`: AlreadyResident / CopyFromHost / ComputeOnCpu / TooLate
- [x] Timing: `copy_ms(bytes) <= available_us / 1000` (B6: Q4 copy 307µs, GEMM 249µs)
- [x] Shadow mode only. Decisions logged to histodb beliefs.
- **Evidence**: `schedule::prefetch_tests` 5/5 passed, `nautiburn-shadow` 26/26 passed.

## BRIEF 5: Wire NautiBurn to nauti-inferer (sidecar)
**Files**: nauti-inferer worker, nautiburn-engine HTTP | **Effort**: 2-3 days | **Needs**: Brief 3

- [ ] `NAUTIBURN_ENGINE_URL` config, scrape after each request, route before next
- [ ] Chat completions still work (llama-server does inference)
- [ ] NautiBurn is sidecar, not replacement

## BRIEF 6: Paddler routing via JSON pass
**Files**: forge v3, nautiburn-engine | **Effort**: 3-5 days | **Needs**: Briefs 3+5

- [x] Job records include `expert_route` for MoE models (null for dense)
- [x] Scorecard tracks expert_hit_rate alongside model grade
- [x] Backward compatible, matches nautiburn-schema JobRecord
- **Evidence**: `nautiburn-schema` 11/11 tests passed; backward compatible `Option<Vec<ExpertId>>` added with `skip_serializing_if = "Option::is_none"`.

## BRIEF 7: FreeTokens-SM60 bridge (Gate 3C.6)
**Files**: FreeTokens-SM60, `crates/nautiburn-layer0/` | **Effort**: 1-2 days | **Needs**: Brief 2

- [x] `bridge.rs` links `libforge_shim_sm60.so`, `forge_ggml_moe_vec()` callable
- [x] FP32 + FP16 paths correct, error codes handled (-1,-2,-7,-9)
- **ABI**: `forge_ggml_moe_vec(x, packed_w, topk_ids, output, scalar_dtype, top_k, ggml_type, col, row, tokens, cuda_stream_raw)`

---

## The End State

```
Forge dispatches task → paddler assigns role → NautiBurn predicts experts
→ prefetch starts → executor runs inference → scrape feeds histodb
→ predictor learns → repeat
```

**Total estimated effort**: 3-4 weeks with agents in parallel.
---

# Milestone rollup

B0–B8, in dependency order. A milestone closes only when its check passes.

| ID | Milestone | Gate / check | State |
|---|---|---|---|
| **B0** | Cards and the kernel sentence | `nauti cards` lists host+index+vram+class; the kernel-sentence is a real sentence, not a slogan | ✅ CLOSED |
| **B1** | OLMoE expert-id scrape | expert ids scraped per layer from a live OLMoE run | ✅ CLOSED |
| **B2** | Job record and the two log payloads | one `Job` record, two log payloads, both replayable | ✅ CLOSED |
| **B3** | Expert predictors, off the hot path | predictor trains from histodb without sitting in the inference path | ✅ CLOSED |
| **B4** | *(folded into B5)* | — | ⤷ SUPERSEDED |
| **B5** | P100 GEMM, dtype lock | dtype locked per card; no silent fp32 fallthrough on a P100 | ✅ CLOSED |
| **B6** | One t440 P100: Q4 resident vs experts in host RAM | measured tok/s, Q4 resident, experts in RAM | ✅ CLOSED |
| **B7** | Mixtral Q4 resident on three cesarops4 P100s | 3× P100 tensor-split, measured tok/s | ✅ CLOSED |
| **B8** | Fleet predictor behind one interface | `FleetPlanner`-shaped: one plan fragment per ask | ✅ CLOSED 2026-10-06 |

### B8 — what landed (2026-10-06)

`crates/nautiburn-shadow/src/fleetplan.rs`:

```rust
plan_experts(ask, belief, map) -> Option<ExpertPlanFragment>
```

One fragment per ask. `host_key`/slot name the device holding the predicted
experts; `reason` explains why (resident_hit / busy_skip / cpu_miss);
`predicted_experts` + `prefetch_reason`/`copy_us`/`budget_us`/`fits_in_budget`
carry the prefetch timing so the copy question is answered alongside.

The fragment is a *plan*, not a dispatch — `schedule_shadow` never sets
`dispatched`/`weights_moved`, and the fragment type has no such field at all.
An empty prediction yields **no fragment** — no fabricated success.

`PrefetchReason::as_str()` added in `schedule.rs` to expose the reason.

Tests: resident-hit, busy-skip to CPU, cpu-miss, empty→`None`, plan-not-
dispatch. 5/5 fleetplan + 26/26 full `nautiburn-shadow` suite.

## Where B6 and B7 sit

B6 and B7 are the two ends of the predictor's range, and the range is the
whole point:

- **B6** — one card, Q4 resident, experts stream from host RAM. This is the
  low-VRAM branch. On a t440 the 1.1 TB SAS drive now mounted at `/mnt/storage`
  as XFS is what makes the expert store practical.
- **B7** — three cards, Q4 resident across a tensor-split. This is the high-VRAM
  branch, measured on cesarops4.

A predictor that only reproduces B6 is not a fleet predictor; it is a
single-host heuristic with a fleet-shaped API. B8 is the gate that says so.

## What B6/B7 do *not* prove

They prove the CubeCL path runs on sm_60 **without a FreeTokens-SM60 bridge**.
That makes rRCP §3.6 / FreeTokens-SM60 gate 3C.6 no longer critical-path — see
`FreeTokens-SM60/docs/MISSING_ARTIFACTS_AND_GATE_STATUS.md`. B6/B7 say nothing
about VFIO passthrough, which remains blocked on the c2 DMAR/IOMMU mapping
(`DRHD base 0xf6ffe000` returns all ones → 0 IOMMU groups).

## Numbers

B6: ~5 tok/s (t440, Ornith-35B Q4, one P100, experts in RAM)
B7: ~1.4 tok/s (cesarops4, Mixtral 8x7B Q4, 3× P100, 40.85 GB resident)
  — recorded as **measured, single run**, not a benchmark. Repeat before
  quoting either figure in a plan.
