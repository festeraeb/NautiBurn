# nautiburn-histodb

SQLite histogram store for NautiBurn activation scrapes and belief updates.

Schema version: **1** (`meta.schema_version`). Histograms are keyed by `model_id`, so Mixtral and Qwen never share a row. Do not tokenize counts. `request_tokens` holds integer token ids only; it is not an embedding index.

## Open / ingest

```text
HistDb::open("histodb.sqlite")
upsert_model(...)
insert_scrape(ScrapeRecord)   # or insert_scrape_json / ingest_scrape_file
```

CLI:

```bash
nautiburn-histodb ingest --db histodb.sqlite scrape.json
nautiburn-histodb ingest --db histodb.sqlite --allenai-olmoe traces/
nautiburn-histodb ingest --db histodb.sqlite --allenai-mixtral traces/
```

AllenAI directory files must already be v1 scrape JSON. Use a **base** Mixtral `model_id` for `--allenai-mixtral`. Continuum Instruct counts go through `bump_histogram` only (`source=continuum_counts`) and must use an Instruct `model_id`.

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

`expert_ids` length must equal that model's `top_k`. Prefill and decode are separate posts.

## Synthetic ingest (tests)

Empty DB → one scrape → `histogram_counts` matches the expert-id multiset. Mixtral-base rows (`source=allenai_mixtral`) are refused under an Instruct `model_id`.
