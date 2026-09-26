# nautiburn-histodb

SQLite histogram store for NautiBurn activation scrapes and belief updates.

Scrape JSON schema version: **1**. Database schema: **3** (`meta.schema_version`; `migrations/001_schema_v1.sql`, `002_requests_dtype.sql`, `003_beliefs_latency.sql`). Histograms are keyed by `model_id`, so Mixtral and Qwen never share a row. Do not tokenize counts. `request_tokens` holds integer token ids only; it is not an embedding index. `beliefs.latency_us` is the shadow predictor's microseconds; overlap stays on `disagreements.overlap_k`.

## Open / ingest

```text
HistDb::open("histodb.sqlite")
upsert_model(...)
insert_scrape(ScrapeRecord)   # or insert_scrape_json / ingest_scrape_file
query_histogram(model, class, phase, layer)
activations_for(request_id)
record_belief(...) / record_disagreement(belief_id, actual_ids)
record_disagreement_ex(belief_id, actual_ids, bump_histogram)  # shadow replay: false
import_allenai_olmoe(path)    # C4 JSONL exp_ids, source=allenai_olmoe
import_allenai_mixtral(path)  # C4 JSONL exp_ids, base Mixtral only
import_continuum_counts(path) # Instruct histogram only; no per-token rows
bump_histogram(...)           # same path continuum uses internally
```

A miss (`record_disagreement` when predicted ≠ actual) increments `histogram_counts`; it does not move placement. Re-ingesting the same `request_id` replaces that request's contribution instead of adding it again. `requests.dtype` stores the scrape dtype (`f16`, …).

CLI:

```bash
nautiburn-histodb ingest --db histodb.sqlite scrape.json
nautiburn-histodb ingest --db histodb.sqlite --allenai-olmoe traces/          # or c4_results.jsonl
nautiburn-histodb ingest --db histodb.sqlite --allenai-mixtral traces/        # base Mixtral only
nautiburn-histodb ingest --db histodb.sqlite --continuum-counts importance.activation_count.json
```

AllenAI files are C4 JSONL: `input_ids` plus `exp_ids` shaped `[n_tokens][top_k][n_layers]` (OLMoE `(N, 8, 16)`, Mixtral `(N, 2, 32)`). Filed under `olmoe-1b-7b-0924` or `mixtral-8x7b-v0.1` (base — never Instruct). Class is `c4`, phase `prefill`. v1 scrape JSON in a directory is still accepted.

Continuum Instruct counts go to `histogram_counts` only (`source=continuum_counts`, phase `mixed`, class `unknown`). The `model` field must be Instruct. Do not mix with `allenai_mixtral`.

## Scrape JSON v1

```json
{
  "schema_version": 1,
  "model_id": "olmoe-1b-7b-0924-f16",
  "request_id": "uuid",
  "class_id": "code_refactor",
  "source": "scrape",
  "phase": "prefill",
  "n_tokens": 128,
  "dtype": "f16",
  "activations": [
    { "layer": 0, "token_pos": 0, "expert_ids": [3, 1, 7, 2, 9, 4, 11, 0] }
  ],
  "token_ids": [1, 2, 3]
}
```

`expert_ids` length must equal that model's `top_k`. Prefill and decode are separate posts. `source` defaults to `scrape` when omitted. `dtype` is stored on `requests.dtype`.

## Synthetic ingest (tests)

Empty DB → one scrape → `histogram_counts` matches the expert-id multiset. Mixtral-base rows (`source=allenai_mixtral`) are refused under an Instruct `model_id`. AllenAI JSONL fixtures and continuum Instruct counts are covered by unit tests.
