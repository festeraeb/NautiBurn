# S1 — Real per-token expert telemetry from Nauti-Strata (live agent traffic → histodb)

Plan: [docs/build-plan-2026-10.md](../docs/build-plan-2026-10.md) · Phase 1 · needs N1 for the ingest endpoint · repo `/codebase/repos/Nauti-Strata`

## Why
Mapping metrics today come only from `scripts/moe_histogram_scrape_loop.sh`: CPU-only (`-ngl 0`) side runs on
fixed prompts. Agent work served by Strata/PolyStrata seats records nothing. Nauti-Strata's prefetch commits
(929af723, 9416bd5a) do not fix that:
- `Engine::prefetch_expert()` (`src/poly/engine.cpp`) is defined but **never called**.
- `serve/server.py:3966` `prefetch_metrics()` reads a `cdll` the server never has (the engine is a separate
  process), so `/v1/telemetry/prefetch` always returns a **hard-coded fallback** (`effective_hit_rate_pct: 100.0`).

## Setup
```bash
cd /codebase/repos/Nauti-Strata
git worktree add /codebase/repos/Nauti-Strata-wt/S1 -b nb/S1 HEAD
cd /codebase/repos/Nauti-Strata-wt/S1
```

## Task
1. In the poly engine, where the gate's top-k is known for each (layer, token), append the **actual** expert ids
   to a bounded ring (layer, token index, phase prefill/decode, expert ids). Gate output stays authoritative;
   nothing about routing changes.
2. Expose it over the engine's existing control channel (the stdout control lines `server.py` already parses),
   not through ctypes. `server.py` drains it per request.
3. `server.py`: when `NAUTIBURN_INGEST_URL` is set, POST each finished request's observations to it
   (N1's `/scrape` format, `request_id` = the server's request id) off the request thread; failures are logged,
   never block or fail the request.
4. Replace the fake fallback: with no data, `/v1/telemetry/prefetch` returns `{"status": "no_data"}` and HTTP 503.
   Never return invented rates.
5. Tests: a recorded OLMoE or Qwen-MoE run's expert ids from the ring equal the ids the engine used (compare
   against the engine's existing debug dump for the same prompt); overhead on decode tok/s measured with
   telemetry on vs off on one P100.

## Allowed files
`src/poly/engine.{cpp,hpp}`, `src/poly/nauti_prefetch.hpp`, `serve/server.py`, new tests under `tests/`.

## Evidence to return
- Build log line for `-DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="60;75"`.
- Ring-vs-engine id comparison output (mismatch count must be 0).
- tok/s on vs off, same prompt, same card, 3 runs each.
- Commit hash on `nb/S1`.

## Non-goals
No prefetch policy, no eviction changes, no deployment to fleet seats (that is S2, an A/B on one seat).
