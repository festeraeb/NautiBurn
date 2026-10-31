---
cursor:
  subagentId: "worker-moe-shadow-fleet"
---

# MoE shadow fleet status (T440 + c4)

**Host:** t440cesarops · **When:** 2026-09-26 ~07:17Z  
**Fleet hard rule:** T440 + cesarops4 only. Do not touch c2/c3.

## Live now

| Stream | Where | Status |
|---|---|---|
| **OLMoE Burn FP16** | tmux `nautiburn-olmoe-shadow` → `/tmp/nautiburn-shadow-live/continuous-loop.sh` | **Healthy** (GPU1). scrape → histodb → `prepredict` → `schedule`. DB `/tmp/nautiburn-histodb/live-olmoe-shadow.sqlite` |
| **Tiel / Qwen3.6-35B-A3B** | `:5230` `cesarops-tilecoder` + tmux `nautiburn-tiel-shadow` / `nautiburn-tiel-live` | **Live-server scrape ON** (`NAUTIBURN_SCRAPE_DIR=/tmp/nautiburn-llama-scrape/tiel-live`). Sidechannel still running. Live consume → `live-tiel-shadow.sqlite` |
| **Kat-Coder Q8** | c4 tmux `kat-coder` → `:5017` + T440 `nautiburn-kat-shadow` | **Serving + live-server scrape ON** while holding all 3 P100s. JSON → `/tmp/nautiburn-llama-scrape/kat/` → rsync consume → `live-kat-shadow.sqlite` |
| Mixtral `:5017` | c4 | **Dropped** (Kat scrapeable) |

## Live-server hook (landed)

Env-gated `cb_eval` inside `llama-server` (adapter only — not a Burn engine):

- Source: `NautiBurn/b1/nautiburn_moe_server_scrape.{h,cpp}`
- Apply: `bash scripts/apply_llama_server_scrape_hook.sh`
- Gate: same tensor names as `b1/moe_histodb_scrape.cpp` (`ffn_moe_topk-*` / `ffn_moe_argsort-*`)
- Env: `NAUTIBURN_SCRAPE_DIR` (+ `NAUTIBURN_MODEL_ID` / `CLASS_ID` / `DTYPE`)
- Flush: layer-wrap + `send_final_response`
- T440 build: `llama.cpp-mainline/build-cuda-60` (sm_60)
- c4: libs rsynced into `/home/cesarops/llama-cuda60/bin` via `scripts/deploy_kat_server_scrape_c4.sh`
- Tiel systemd drop-in: `/etc/systemd/system/cesarops-tilecoder.service.d/20-nautiburn-scrape.conf` (from `scripts/systemd/cesarops-tilecoder-scrape.conf`)

### Proof (this pass)

| Model | Live request | scrape JSON | histodb ingest | `prepredict` |
|---|---|---|---|---|
| Tiel `:5230` | chat completion 16 tok | 40 layers, prefill+decode under `tiel-live/` | `live-tiel-server-shadow.sqlite` | winner **ensemble** |
| Kat `:5017` | chat completion 12 tok | 40 layers under `kat/` while GPUs held | `live-kat-shadow.sqlite` | winner **ensemble** |

Continuous Kat consume: tmux `nautiburn-kat-shadow` (15s poll). Continuous Tiel live: tmux `nautiburn-tiel-live`.

## Smallest scrape path (sidechannel, still available)

- Binary: `NautiBurn/b1/moe-histodb-scrape`
- Loops: `scripts/llama_moe_shadow_loop.sh`, `scripts/kat_c4_scrape_loop.sh` (CPU Q8 SIGILL — superseded by server hook for Kat)

## Daemons / do not stop

- `tmux nautiburn-olmoe-shadow` (T440 GPU1)
- `tmux nautiburn-tiel-shadow` (T440 CPU sidechannel)
- `tmux nautiburn-tiel-live` (T440 live-server consume)
- `tmux kat-coder` on c4 `:5017` (scrape-enabled)
- `tmux nautiburn-kat-shadow` (T440 Kat consumer)
- Do not start NautiBurn on c2/c3

## Related paths

- Adapter: `/home/cesarops/NautiBurn/b1/nautiburn_moe_server_scrape.cpp`
- Tiel live scrapes: `/tmp/nautiburn-llama-scrape/tiel-live/`
- Kat live scrapes: c4 `/tmp/nautiburn-llama-scrape/kat/` (rsync to T440 same path)
- Branch: `cursor/moe-llama-scrape-6889`
- Prior notes: `internal/moe-shadow-scrape-status.md`
