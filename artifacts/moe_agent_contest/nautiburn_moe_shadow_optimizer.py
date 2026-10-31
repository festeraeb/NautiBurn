Let me analyze this task carefully. I need to produce a complete, self-contained, runnable Python implementation script for a Nauti-Burn MoE routing optimization task. The script must:

1. Connect to SQLite at `/mnt/raid0/nautiburn-histodb/live-fleet.sqlite`.
2. Discover all models from the `models` table.
3. Compute layer transition matrices (L to L+1, L+2, L+3).
4. Run an ensemble predictor (Histogram Prior + Transition Matrix + Softmax Ranker).
5. Compute expected utility vs latency budget (prefetch vs CPU compute).
6. Evaluate on a 70/30 train/held-out split reporting Recall@1, Recall@4, Recall@8, and coverage.
7. Emit sidecar JSON to `/home/cesarops/NautiBurn/artifacts/moe_agent_contest/shadow_route_recommendations.json`.

The output should be clean Python code between ```python and ```.

Let me think about the architecture:

- The core idea: a mixture-of-experts routing system where we have a history database of routing decisions (which expert/model was chosen at each layer for each request). We want to predict the next routing decision based on the current state, to prefetch the expert weights (avoiding the latency of loading them from disk/CPU compute).

- The ensemble predictor combines:
  - Histogram Prior: frequency of each expert being chosen at a given layer.
  - Transition Matrix: P(next expert | current expert) learned from consecutive layers (L to L+1, L+2, L+3).
  - Softmax Ranker: softmax over scores to rank candidates.

- Expected utility vs latency budget: if we prefetch (load weights ahead of time), we save latency; if we don't, we pay CPU compute cost. We decide based on expected utility whether prefetching is worth it given a latency budget (13ms mentioned in the prompt as GLM 5.3 Flash's 13ms latency-budget).

- The evaluation: 70/30 train/held-out split, Recall@1, Recall@4, Recall@8, coverage.

- Sidecar JSON: zero-mutation contract (don't mutate the DB), sparse-data fallback (handle cases with little data).

Let me design the schema assumptions. Since we're connecting to a SQLite DB at a specific path, we need to handle the case where the table schema might vary. The prompt says "Discover all models from the `models` table" and "Compute layer transition matrices (L to L+1, L+2, L+3)". So there's a `models` table and presumably a routing history table.

Let me think about what tables might exist:
- `models`: model registry with model IDs, names, etc.
- Some history table with routing decisions: probably something like `routing_history`, `moe_routing`, `layer_decisions`, or similar. Since we can't know the exact schema, we should write robust discovery code that tries multiple candidate table names and column names.

Actually, given this is a synthetic benchmark task, the DB likely has tables like:
- `models` (model_id, name, ...)
- `routing_events` or `layer_transitions` or `expert_selections` with columns like (request_id, layer, expert_id/model_id, timestamp, latency, ...)

Since we must be robust, I'll write a schema-discovery routine that:
1. Lists all tables via `sqlite_master`.
2. Finds the models table (named `models` or containing model-like columns).
3. Finds the routing/decision history table by trying candidate names and by scanning tables for columns that look like (layer, expert/model, request/session id, sequence/step