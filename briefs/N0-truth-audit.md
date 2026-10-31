# N0 — Truth audit of B0–B8 and Briefs 1–7

Plan: [docs/build-plan-2026-10.md](../docs/build-plan-2026-10.md) · Phase 1 · no dependencies · docs only

## Setup
```bash
cd /home/cesarops/NautiBurn
git worktree add /home/cesarops/NautiBurn-wt/N0 -b nb/N0 HEAD   # never edit the main checkout: it holds uncommitted WIP
cd /home/cesarops/NautiBurn-wt/N0
```

## Task
Write `docs/truth-audit.md`: one table row per claim made in `B0.md`…`B7.md`, `BRIEF_LEASE.md`,
`docs/burn-expert-engine.md`, `docs/kernels-we-have.md`. Columns: claim · source file:line ·
label (**implemented** / **measured** / **estimated** / **proposed**) · evidence (file:line of code, or
path of the timing/log artifact) · note.

Specifically resolve, with file:line evidence:
1. B7: tensor-split vs layer-split — which actually ran (`B7.md`, `b7/`).
2. Brief 4 "Shadow → cache swap" CLOSED vs `crates/nautiburn-shadow/src/schedule.rs` saying it only logs.
3. Whether `nautiburn-engine` `/scrape` persists anything (`crates/nautiburn-engine/src/http.rs`).

## Allowed files
`docs/truth-audit.md` (new). Nothing else.

## Evidence to return
- Path of the new file and the row count per label.
- The three resolutions above, each with file:line.
- `git -C /home/cesarops/NautiBurn-wt/N0 log --oneline -1` after committing on `nb/N0`.

## Non-goals
No code changes, no re-running benchmarks, no new claims. "I could not find evidence" is a valid label note.
