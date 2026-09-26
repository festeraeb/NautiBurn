# moe-histodb-scrape

Side-channel llama.cpp MoE expert scrape → histodb scrape JSON v1.

Same gate as `olmoe_expert_scrape.cpp`: `cb_eval` copies `ffn_moe_topk-*`
(or first-k of `ffn_moe_argsort-*`). Adapter only — not a Burn engine and not
inside `llama-server`.

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

**Remaining hook for true live-server capture:** env-gated `cb_eval` inside
`llama-server` (`NAUTIBURN_SCRAPE_DIR`) so expert ids follow real requests
without a second weight load.
