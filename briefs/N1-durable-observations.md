# N1 — Durable observations through the engine HTTP API

Plan: [docs/build-plan-2026-10.md](../docs/build-plan-2026-10.md) · Phase 1 · no dependencies · CPU only

## Setup
```bash
cd /home/cesarops/NautiBurn
git worktree add /home/cesarops/NautiBurn-wt/N1 -b nb/N1 HEAD   # never edit the main checkout: it holds uncommitted WIP
cd /home/cesarops/NautiBurn-wt/N1
export CARGO_TARGET_DIR=/data/cargo-target/nautiburn-N1
```

## Problem
`crates/nautiburn-engine/src/http.rs` builds one in-memory `Engine`; `POST /scrape` only calls
`write_scrape()`. Restarting `nautiburn-engine.service` loses all observations, and the HTTP path and
`nautiburn-histodb` are two sources of truth.

## Task
1. Give `serve` an optional `--histodb <path>`. When set, `POST /scrape` also calls
   `HistDb::insert_scrape` (`crates/nautiburn-histodb/src/lib.rs:226`) and bumps the histogram.
2. Idempotency: a scrape carries a `request_id`; posting the same `request_id` twice must not double-count.
   Return `{"ingested": n, "duplicate": bool}`.
3. Validation: reject (HTTP 422, JSON error) an unknown model, an expert id ≥ the model's expert count,
   a layer ≥ the model's layer count, and an empty expert list.
4. Tests in `crates/nautiburn-engine/tests/http_histodb.rs` using a temp-file histodb:
   - ingest → stop engine → reopen db → counts unchanged (restart survives)
   - duplicate `request_id` → counts unchanged, `duplicate: true`
   - each invalid case → 422
   - `/route` after ingest returns the same `NextExperts` as the CLI `route` on the same db

## Allowed files
`crates/nautiburn-engine/src/{http.rs,main.rs,lib.rs}`, `crates/nautiburn-engine/Cargo.toml`,
`crates/nautiburn-engine/tests/http_histodb.rs` (new). Read-only: `crates/nautiburn-histodb`.

## Evidence to return
- `cargo test -p nautiburn-engine` full output (pass counts, no filtered-out tests).
- The diff stat and commit hash on `nb/N1`.

## Non-goals
Do not touch the live service, `/var/lib/nautiburn/histogram/sidechannel.sqlite`, predictor math,
or histodb's schema. Do not change `/route` behaviour when `--histodb` is absent.
