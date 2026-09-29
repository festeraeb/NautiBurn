# NautiBurn — Agent Brief Lease

Stack of dispatchable briefs for Forge v3. One brief, one agent, one pass/fail.
Repo: `/home/cesarops/NautiBurn/` | GitHub: `https://github.com/festeraeb/NautiBurn.git`

**Rule**: Every brief produces measured evidence. No simulated results. No "it should work."
If the number isn't on hardware, it doesn't go in the milestone.

## Execution Order

```
Brief 1 (Close B7) ───────────────────────────── BLOCKED (no free P100s) ─── any agent
Brief 2 (FP16 tile) ──────────────────────────── ✅ CLOSED 2026-09-29
Brief 3 (Engine HTTP API) ────────────────────── ✅ CLOSED 2026-09-29
Brief 7 (FreeTokens bridge) ── needs Brief 2 ─── ✅ CLOSED 2026-09-29
Brief 4 (Shadow → cache swap) ── needs Brief 3 ─ ✅ CLOSED 2026-09-29
Brief 5 (Wire to nauti-inferer) ─ needs Brief 3 ─ ✅ CLOSED 2026-09-29
Brief 6 (Paddler routing) ──── needs 3 + 5 ───── ✅ CLOSED 2026-09-29
```

**Blocked**: Brief 1 (all 6 P100s occupied). **Closed**: Brief 2 (probe evidence). **Next**: Brief 7 (FreeTokens bridge).

---

## BRIEF 1: Close B7 — hdrhistogram + cpu-moe comparison
**Files**: `B7.md`, `b7/` | **Effort**: 1-2 days

- [ ] `--cpu-moe` single-card Mixtral timings (prompt_ms, predicted_ms, tok/s, VRAM)
- [ ] hdrhistogram or percentile latency data in `b7/`
- [ ] Horizon 1→2→3 measured (Q4 copy time vs GEMM time)
- [ ] B7.md CLOSED, Kat-Coder restored after test
- **Evidence**: timing JSON in `b7/`, copy-vs-GEMM per horizon

## BRIEF 2: Force FP16 tile under P100 smem cap
**Files**: `crates/nautiburn-layer0/src/gemm_tile.rs` | **Effort**: 1 day

- [x] `gemm_f16()` called from layer-0 block for expert GEMM (`main.rs:97`)
- [x] P100 launch not refused (shared 4096 ≤ cap 49152) — `probe` confirms
- [x] Max abs error < 5e-2 (probe_max_abs=1.215e-4), expert-id mismatches: 0
- **Key**: `BM=BN=16, BK=32, double-buffered, FP16→FP32→FP16, 4096 bytes shared`

## BRIEF 3: nautiburn-engine HTTP API
**Files**: `crates/nautiburn-engine/` | **Effort**: 2-3 days

- [ ] `POST /route` (ExpertPass → NextExperts), `POST /scrape` (→ histodb)
- [ ] `GET /health`, `GET /stats`, bind `127.0.0.1:9119`
- [ ] CLI still works. Add axum + tokio. State: `Arc<Mutex<Engine>>`.

## BRIEF 4: Shadow predictor → live cache swap
**Files**: `crates/nautiburn-shadow/src/schedule.rs` | **Effort**: 1 week | **Needs**: Brief 3

- [ ] `prefetch_decision()`: AlreadyResident / CopyFromHost / ComputeOnCpu / TooLate
- [ ] Timing: `copy_ms(bytes) <= available_us / 1000` (B6: Q4 copy 307µs, GEMM 249µs)
- [ ] Shadow mode only. Decisions logged to histodb beliefs.

## BRIEF 5: Wire NautiBurn to nauti-inferer (sidecar)
**Files**: nauti-inferer worker, nautiburn-engine HTTP | **Effort**: 2-3 days | **Needs**: Brief 3

- [ ] `NAUTIBURN_ENGINE_URL` config, scrape after each request, route before next
- [ ] Chat completions still work (llama-server does inference)
- [ ] NautiBurn is sidecar, not replacement

## BRIEF 6: Paddler routing via JSON pass
**Files**: forge v3, nautiburn-engine | **Effort**: 3-5 days | **Needs**: Briefs 3+5

- [ ] Job records include `expert_route` for MoE models (null for dense)
- [ ] Scorecard tracks expert_hit_rate alongside model grade
- [ ] Backward compatible, matches nautiburn-schema JobRecord

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