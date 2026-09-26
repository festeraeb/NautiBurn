# moe-histodb-scrape

Side-channel llama.cpp MoE expert scrape → histodb scrape JSON v1.

Same gate as `olmoe_expert_scrape.cpp`: `cb_eval` copies `ffn_moe_topk-*`
(or first-k of `ffn_moe_argsort-*`). Adapter only — not a Burn engine.

```bash
bash b1/build_moe_histodb_scrape.sh
NAUTIBURN_MODEL_ID=qwen3.6-35b-a3b NAUTIBURN_OUT=/tmp/run.json \
  ./b1/moe-histodb-scrape -m MODEL.gguf -p 'prompt' -ngl 0 -c 64 -t 8 -fa off
```

On cesarops4, rebuild against `/home/cesarops/llama-cuda60` (T440-linked
binaries can SIGILL). Q8_0 CPU GEMM on c4's `libggml-cpu` SIGILLs; use GPU
(`-ngl 99`) or `-ncmoe 99` with CUDA visible.

Continuous loops: `scripts/llama_moe_shadow_loop.sh`,
`scripts/kat_c4_scrape_loop.sh`, `scripts/kat_t440_consume_loop.sh`.

## Live-server hook (env-gated)

`b1/nautiburn_moe_server_scrape.{h,cpp}` — same tensor names, installed into
`llama.cpp` via `scripts/apply_llama_server_scrape_hook.sh`. When
`NAUTIBURN_SCRAPE_DIR` is set, `llama-server` writes scrape JSON v1 for real
request streams (no second weight load). Optional:
`NAUTIBURN_MODEL_ID`, `NAUTIBURN_CLASS_ID`, `NAUTIBURN_DTYPE`.

```bash
bash scripts/apply_llama_server_scrape_hook.sh /path/to/llama.cpp
# rebuild llama-common + llama-server-impl + llama-server (sm_60 on T440/c4)
# Tiel drop-in: scripts/systemd/cesarops-tilecoder-scrape.conf
# Kat on c4:    bash scripts/deploy_kat_server_scrape_c4.sh
```

T440 consumer `nautiburn-kat-shadow` rsyncs `/tmp/nautiburn-llama-scrape/kat/`
and runs histodb ingest + `nautiburn-shadow prepredict`.
