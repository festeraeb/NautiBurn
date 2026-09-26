# nautiburn-shadow

Shadow expert guesser. The four B3 methods run **beside** the real router. They do not change tokens and do not replace the gate.

`prepredict` names the next expert set from the **kept ensemble** (L+1 / L+2) using the histogram/belief, compares that set to a scrape, and writes misses back. `schedule` logs which device already holds those experts (shadow mode: no dispatch, no weight move, no SSH).

Default scoring is **next-expert**: given state at layer `L` / token `t`, predict the next expert set (`L+1`, and `L+2` / `t+1` when the scrape has those positions). The 13 ms budget is lookahead vs that next weight load, not a kernel target.

Methods (all four, every next-expert horizon):

1. `frequency` — phase histogram / bar at the **target** layer
2. `prev_token` — previous-token copy at the **target** layer (`(L+1, t-1) → (L+1, t)`; for `t+1`, `(L, t) → (L, t+1)`)
3. `bigram` — cross-layer joint from the current layer (`predict_ahead` hops for `L+2`)
4. `ensemble` — copy ∪ bigram (B3 either-hit; denominator stays `k`)

`same_step_copy` is the old metric: `(L, t-1) → (L, t)` at the current gate. It is measured and printed, and it is **not** a next-expert winner.

A method that cannot produce a prediction (no previous token, no previous layer) is recorded as `n_scored=0` / `n/a`, not a fabricated hit rate.

`--same-step` still prints the old current-gate table (`score_events`) for comparison.

## Metric

**slot_hit** = `|predicted ∩ actual| / k` averaged over scored positions (B3 `hits/slots`). Also printed: `exact_set` (sets equal) and `any_overlap` (overlap > 0). Times are microseconds of `predict` (ensemble is copy + bigram + union).

Keep the ensemble only if it **beats** the best single method on slot_hit **and** mean µs is under the 13 ms planning lead (`PLANNING_LEAD_US = 13000`). That 13 ms is a planning budget for an expert-weight copy over Gen3 PCIe x16, not a kernel target.

## Persist

Each scored position writes histodb `beliefs` (predicted set + `latency_us`) and `disagreements` (`overlap_k`, scraped ids). Replay uses `record_disagreement_ex(..., bump_histogram=false)` so four methods do not multiply an already-ingested scrape into `histogram_counts`.

## Pre-prediction

`prepredict` names L+1 / L+2 from the kept ensemble **before** treating the scrape as truth.

- Prior rows for the same `model_id` (excluding this `request_id`) seed the frequency bar and bigram joint, then freeze them. `prev_token` still copies this run at the target layer.
- No prior histogram → in-run causal walk (same as `score`). That names the next layer before that scrape step; it does not name the whole run before it started.
- Ensemble disagreements write back with `bump_histogram=true` (belief-loop miss). Only the ensemble is persisted, so four methods do not quadruple the bump.
- Replay of a JSON scrape measures predict microseconds. It does **not** measure wall-ahead versus a live generate. Do not read 13 ms into a replay.

`--also-decode FILE` ingests the prefill scrape first, freezes that belief, names the decode scrape, then also prints a combined causal next-expert table (comparable L+1 / L+2 slot_hit).

`--belief-out FILE` writes the named ensemble set (`request_id`, `expert_ids`, `by_layer`) for `schedule`.

## Shadow scheduler

`schedule` picks a device that **already holds** the predicted experts. Weights stay put. An idle GPU that does not hold the set is not a destination. CPU is the OpenVINO miss-path stub (`holds_all`).

Reasons:

- `resident_hit` — idle device holds every predicted expert
- `busy_skip` — a device holds them but every such device is busy (live Tiel / Mixtral / coder). Logged to CPU; not sent to the busy box
- `cpu_miss` — no device holds the full set

The stub map is `crates/nautiburn-shadow/fixtures/devices.stub.json`. It is an **example layout**, not a lock and not a hot-cut policy (do not read “first N experts” into it). Busy flags mark live Tiel-Coder T440 `:5230`, Mixtral cesarops4 `:5017`, embedder, SmolLM, and coder jobs. Output always has `dispatched=no` and `weights_moved=no`.

## CLI

```bash
nautiburn-shadow score --db histodb.sqlite --scrape scrape.json --label fp16
nautiburn-shadow score --db histodb.sqlite --scrape scrape.json --also-decode scrape-decode.json
nautiburn-shadow score --db histodb.sqlite --trace b1/trace.tsv --label b1
nautiburn-shadow score --db histodb.sqlite --scrape scrape.json --also-b1 b1/trace.tsv --label fp16
nautiburn-shadow score --db histodb.sqlite --scrape scrape.json --same-step

nautiburn-shadow prepredict --db histodb.sqlite --scrape scrape.json
nautiburn-shadow prepredict --db histodb.sqlite --scrape scrape.json --also-decode scrape-decode.json --label fp16 --belief-out belief.json

nautiburn-shadow schedule --belief belief.json --devices crates/nautiburn-shadow/fixtures/devices.stub.json
nautiburn-shadow schedule --belief belief.json --devices stub
```

`--devices stub` loads the shipped stub map. `--request-id ID` overrides the belief's id.

Binary: `nautiburn-shadow` (`crates/nautiburn-shadow/src/main.rs`). Predictors live in `nautiburn-predict::online` — reused, not rewritten.
