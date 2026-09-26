# nautiburn-shadow

Shadow expert guesser. The four B3 methods run **beside** the real router. They do not change tokens and do not replace the gate. Job-to-device scheduling and Q-routing are out of scope.

Methods (all four, every run):

1. `frequency` — phase histogram / bar (prefill updates after each token; decode frozen)
2. `prev_token` — previous-token copy (prefill copies `t-1`; decode copies last prefill)
3. `bigram` — cross-layer bigram (layer `L` from the same token’s layer `L-1`)
4. `ensemble` — copy ∪ bigram (B3 either-hit; denominator stays `k`)

A method that cannot produce a prediction (no previous token, no previous layer) is recorded as `n_scored=0` / `n/a`, not a fabricated hit rate.

## Metric

**slot_hit** = `|predicted ∩ actual| / k` averaged over scored positions (B3 `hits/slots`). Also printed: `exact_set` (sets equal) and `any_overlap` (overlap > 0). Times are microseconds of `predict` (ensemble is copy + bigram + union).

Keep the ensemble only if it **beats** the best single method on slot_hit **and** mean µs is under the 13 ms planning lead (`PLANNING_LEAD_US = 13000`). That 13 ms is a planning budget for an expert-weight copy over Gen3 PCIe x16, not a kernel target.

## Persist

Each scored position writes histodb `beliefs` (predicted set + `latency_us`) and `disagreements` (`overlap_k`, scraped ids). Replay uses `record_disagreement_ex(..., bump_histogram=false)` so four methods do not multiply an already-ingested scrape into `histogram_counts`.

## CLI

```bash
nautiburn-shadow score --db histodb.sqlite --scrape scrape.json --label fp16
nautiburn-shadow score --db histodb.sqlite --trace b1/trace.tsv --label b1
nautiburn-shadow score --db histodb.sqlite --scrape scrape.json --also-b1 b1/trace.tsv --label fp16
```

Binary: `nautiburn-shadow` (`crates/nautiburn-shadow/src/main.rs`). Predictors live in `nautiburn-predict::online` — reused, not rewritten.
