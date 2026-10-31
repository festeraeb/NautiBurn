# NautiBurn staged build plan

Status: repo-local build plan for the Rust-only staged rollout.

## Goal

Build the system in stages, with the local MoE routing and prediction path first, then a same-family FrankenMoE layer, then a federated nauti-nodes expansion. Keep the real model gate authoritative at every step.

## Ground rules

- Rust-native first, using Burn + CubeCL for local compute where practical.
- Reuse known Rust crates and adapt them rather than building a custom runtime from scratch.
- No silent replacement of a real model expert with a different expert unless compatibility is proven.
- Keep the prediction engine distinct from the actual gate; prediction is a prefetch, not a model decision.
- Treat federation as a later milestone, not the first build.

## Stage 1 — standard layered MoE predictor engine

This is the first production gate.

### Objective

Build a dependable local MoE prediction engine that predicts the next expert(s) ahead of time and improves expert residency without changing the actual model output path.

### Scope

- Keep a real model or reference model as the truth source.
- Implement or stabilize the NautiBurn predictor layer for:
  - frequency
  - previous-token copy
  - bigram / cross-layer joint
  - ensemble union
  - L+1 / L+2 lookahead
- Integrate this with the existing shadow-style planner.
- Use the prediction layer to decide likely resident experts and prefetch ordering.
- Require the true gate to remain authoritative.

### Deliverables

- prediction crate/service boundary
- JSON observation schema for actual expert IDs + predicted IDs
- local belief / schedule output
- exact gate-vs-prediction comparison metrics
- hot/warm/cold residency decision layer

### Acceptance criteria

- prediction does not change output
- actual expert IDs remain from the model gate
- prefetch decisions are measured and logged
- local timing and hit-rate data are captured
- no silent expert substitution

### Repo anchors

- [README.md](../README.md)
- [crates/nautiburn-shadow/README.md](../crates/nautiburn-shadow/README.md)
- [crates/nautiburn-engine/src/lib.rs](../crates/nautiburn-engine/src/lib.rs)
- [crates/nautiburn-shadow/src/schedule.rs](../crates/nautiburn-shadow/src/schedule.rs)

## Stage 2 — same-family FrankenMoE

This is the second milestone.

### Objective

Add a compatible expert composition layer only for proven same-family experts.

### Scope

- same checkpoint family or exact compatibility manifest
- matching dimensions, dtype, layer layout, router semantics
- compatibility check before composition
- static packaged expert composition only, not arbitrary cross-model swap

### Deliverables

- compatibility manifest
- expert packaging format
- same-family composition validator
- evaluated parity test against the reference model

### Acceptance criteria

- composition only happens after compatibility checks pass
- no composition across truly different model families without an explicit adapter path
- if compatibility fails, route via JSON model handoff instead

### Non-goals

- arbitrary hidden-state transfer between unrelated models
- runtime blending of unrelated experts without a defined projector or training path
- federated composition before local compatibility is proven

## Stage 3 — federated nauti-nodes expansion

This is the expansion stage after local correctness is stable.

### Objective

Allow remote expert hosting and routing only after the local path is sound.

### Scope

- node registration + capability map
- signed expert manifests
- authenticated workers
- local hot/warm/cold policy extended to remote nodes
- private local network first, public federation second

### Deliverables

- node registry
- capability metadata
- model/expert identity manifest
- remote expert lease or fetch protocol
- failure, retry, and rollback rules

### Acceptance criteria

- remote experts are only used when exact identity and compatibility are known
- network latency is measured and budgeted
- no remote substitution of a different model or wrong expert occurs
- local run remains the fallback correctness path

## Stage order summary

1. Build standard layered MoE prediction engine
2. Add same-family FrankenMoE compatibility and composition
3. Expand to federated nauti-nodes with explicit remote capability rules

## Correctness guardrails

- The actual model gate remains the source of truth.
- Prediction is a prefetch hint only.
- Expert replacement is never allowed by default.
- If compatibility is unproven, do JSON specialist routing instead.
- Federation is not the first milestone.

## Recommended Rust stack

- Burn
- CubeCL
- axum
- tokio
- serde / serde_json
- parking_lot
- sqlite / sqlx or rusqlite for observation persistence
- quinn or tokio transport later for federation

This keeps the implementation fully Rust-native and aligned with the existing repo architecture.

## Milestone summary

- Milestone 1: stable local MoE predictor and routing improvement
- Milestone 2: compatible same-family FrankenMoE composition
- Milestone 3: federated nauti-nodes expert hosting and remote routing

This is the staged build we should commit to.
