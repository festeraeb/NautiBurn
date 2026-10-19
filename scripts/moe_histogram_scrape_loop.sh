#!/usr/bin/env bash
# MoE activation-histogram sidechannel scrape.
#
# Runs the prebuilt b1/moe-histodb-scrape (cb_eval on ffn_moe_topk-*) against
# EVERY scrapable MoE GGUF in the fleet (round-robin, one prompt per cycle) and
# writes histodb scrape JSON v1. This is the OUT-OF-PROCESS path: it does NOT
# patch or rebuild llama-server, so it survives lineup swaps and llama.cpp
# rebuilds, and it never disturbs a serving seat (weights load to host RAM,
# -ngl 0). histodb keys everything by model_id, so one shared OUTDIR is fine.
#
# Prompts are rotated so the histogram sees varied routing, not one shape.
set -uo pipefail

ROOT="${ROOT:-/home/cesarops/NautiBurn}"
BIN="${BIN:-$ROOT/b1/moe-histodb-scrape}"
OUTDIR="${OUTDIR:-/tmp/nautiburn-llama-scrape/sidechannel}"
INTERVAL="${INTERVAL:-120}"
THREADS="${THREADS:-8}"
CTX="${CTX:-64}"
CLASS_ID="${CLASS_ID:-llama-moe-sidechannel}"

# Scrapable MoE fleet: model_id|path. Probed 2026-10-11 — all five produce
# ffn_moe_topk rows under this scrape binary. Extend this list as new MoE
# GGUFs land; a model that fails 3x in a row is benched for SKIP_CYCLES so a
# bad/unsupported GGUF cannot spin the loop.
MODELS=(
  "qwen3.6-35b-a3b|/mnt/raid0/models/Qwen3.6-35B-A3B-MXFP4_MOE.gguf"
  # Was labelled deepseek-v4-flash-9b after the t440 "deepseek" seat; the file is a Qwen3.6-35B-A3B finetune.
  "qwen3.6-35b-a3b-hauhau-q2kp|/mnt/raid0/models/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q2_K_P.gguf"
  "gemma-4-26b-moe|/mnt/raid0/models/Gemma-4-26B-MoE-IQ4_XS.gguf"
  "gemma-4-26b-a4b|/codebase/models/google_gemma-4-26B-A4B-it-IQ4_XS.gguf"
  "kat-coder-35b-a3b|/codebase/models/Q4_K_M-KAT-Coder-V2.5-Dev-35B-A3B-ABLITERATED-UNCENSORED-PHILADELPHIA-CLASS.gguf"
  "olmoe-1b-7b-0924-f16|/mnt/raid0/models/olmoe-1b-7b-0924-f16.gguf"
  "qwen1.5-moe-a2.7b|/mnt/raid0/models/qwen1.5-moe-a2.7b-chat-q4_k_m.gguf"
)
SKIP_AFTER=3        # consecutive failures before benching a model
SKIP_CYCLES=10      # full-loop cycles a benched model sits out

mkdir -p "$OUTDIR"

# Varied prompts: different domains route to different experts.
PROMPTS=(
  "In one sentence, what is a compile error?"
  "Explain how sonar returns a seabed target in open water."
  "Write a Rust function that merges two sorted vectors."
  "Describe the Great Lakes as a marine surveyor would."
  "What causes class imbalance in shipwreck detection models?"
  "Summarize how batch normalization affects gradient flow."
  "Describe a wrecks database schema for spatial queries."
  "Explain top-k expert routing in a mixture of experts model."
)

declare -A fail_count=()
declare -A bench=()
i=0
cycle=0
while true; do
  cycle=$((cycle + 1))
  for entry in "${MODELS[@]}"; do
    model_id="${entry%%|*}"
    model="${entry#*|}"

    # Honor the bench for a previously flapping model.
    if [ "${bench[$model_id]:-0}" -gt 0 ]; then
      bench[$model_id]=$(( bench[$model_id] - 1 ))
      continue
    fi

    p="${PROMPTS[$((i % ${#PROMPTS[@]}))]}"
    stamp="$(date -u +%Y%m%dT%H%M%S)"
    out="$OUTDIR/run-$stamp-$model_id.json"
    # Per-prompt-domain class_id: histodb keys per-class histograms so the
    # shadow predictor can condition expert prediction on prompt category
    # (2026-10-11 probe: same-prompt routing JSD=0.0, cross-domain 0.48 on
    # all 40 layers — category is the strongest available prior).
    domain="${p%% *}"   # fallback: first word
    case "$p" in
      *compile*) domain="compile" ;;
      *sonar*) domain="sonar" ;;
      *Rust*|*rust*) domain="rust" ;;
      *Great\ Lakes*|*marine*) domain="marine" ;;
      *imbalance*|*normalization*) domain="ml" ;;
      *schema*|*database*) domain="db" ;;
      *expert*routing*|*mixture*) domain="moe" ;;
    esac
    NAUTIBURN_MODEL_ID="$model_id" \
    NAUTIBURN_OUT="$out" \
    NAUTIBURN_CLASS_ID="sidechannel-${model_id}-${domain}" \
    NAUTIBURN_DTYPE="gguf" \
      "$BIN" -m "$model" -p "$p" -ngl 0 -c "$CTX" -t "$THREADS" -fa off \
      >>"$OUTDIR/scrape.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ] && [ -s "$out" ]; then
      echo "[$(date -u +%FT%TZ)] scrape ok $out"
      fail_count[$model_id]=0
    else
      fail_count[$model_id]=$(( ${fail_count[$model_id]:-0} + 1 ))
      echo "[$(date -u +%FT%TZ)] scrape FAILED rc=$rc model=$model_id (consecutive=${fail_count[$model_id]})"
      if [ "${fail_count[$model_id]}" -ge "$SKIP_AFTER" ]; then
        bench[$model_id]=$SKIP_CYCLES
        fail_count[$model_id]=0
        echo "[$(date -u +%FT%TZ)] benching model=$model_id for $SKIP_CYCLES cycles"
      fi
      sleep "$INTERVAL"
    fi
    i=$((i + 1))
    sleep "$INTERVAL"
  done
done
