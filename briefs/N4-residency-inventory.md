# N4 — Exact-expert residency inventory (host-side state machine)

Plan: [docs/build-plan-2026-10.md](../docs/build-plan-2026-10.md) · Phase 1 · no dependencies · CPU only

## Setup
```bash
cd /home/cesarops/NautiBurn
git worktree add /home/cesarops/NautiBurn-wt/N4 -b nb/N4 HEAD   # never edit the main checkout: it holds uncommitted WIP
cd /home/cesarops/NautiBurn-wt/N4
export CARGO_TARGET_DIR=/data/cargo-target/nautiburn-N4
```

## Task
New crate `crates/nautiburn-residency` (add to the root workspace `members`). It tracks where each expert
lives; it never copies bytes (that is N6).

- Key: `ExpertKey { model_id, checkpoint_sha, layer, expert }`. Tier: `Hot { device }`, `Warm`, `Cold`.
- Per-device and host byte budgets. `pin(key)` makes a key ineligible for eviction (the
  "predetermined layers" set from N2 will use this).
- `request(key) -> Decision` returns `Resident`, or `Load { from, to, evict: Vec<ExpertKey> }` choosing
  evictions by LRU among unpinned keys; returns `Err(OverBudget)` if pinned keys alone exceed the budget.
- `prefetch(key, deadline)` records intent without changing tier until `commit_load(key)`.
- Ownership: a key in use by a request (`acquire`/`release`) is never evicted.
- Changing `checkpoint_sha` for a model invalidates every key of that model.

## Tests (deterministic, in the crate)
LRU order, pin never evicted, in-use never evicted, over-budget error, checkpoint invalidation,
prefetch-then-commit, and a 10k-op randomized sequence (fixed seed) asserting budgets are never exceeded
and pinned/in-use keys never leave their tier.

## Allowed files
`crates/nautiburn-residency/**` (new), root `Cargo.toml` (members line only).

## Evidence to return
- `cargo test -p nautiburn-residency` full output.
- Diff stat and commit hash on `nb/N4`.

## Non-goals
No GPU, no I/O, no async runtime, no predictor coupling. No changes to other crates.
