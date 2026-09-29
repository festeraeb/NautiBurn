#!/usr/bin/env bash
# MoE activation-histogram sidechannel scrape.
#
# Runs the prebuilt b1/moe-histodb-scrape (cb_eval on ffn_moe_topk-*) against a
# MoE GGUF and writes histodb scrape JSON v1. This is the OUT-OF-PROCESS path:
# it does NOT patch or rebuild llama-server, so it survives lineup swaps and
# llama.cpp rebuilds, and it never disturbs a serving seat (weights load to
# host RAM, -ngl 0).
#
# Prompts are rotated so the histogram sees varied routing, not one shape.
set -uo pipefail

ROOT="${ROOT:-/home/cesarops/NautiBurn}"
BIN="${BIN:-$ROOT/b1/moe-histodb-scrape}"
MODEL="${MODEL:-/mnt/raid0/models/Qwen3.6-35B-A3B-MXFP4_MOE.gguf}"
MODEL_ID="${MODEL_ID:-qwen3.6-35b-a3b}"
OUTDIR="${OUTDIR:-/tmp/nautiburn-llama-scrape/sidechannel}"
INTERVAL="${INTERVAL:-120}"
THREADS="${THREADS:-8}"
CTX="${CTX:-64}"
CLASS_ID="${CLASS_ID:-llama-moe-sidechannel}"

mkdir -p "$OUTDIR"

# Varied prompts: different domains route to different experts.
PROMPTS=(
  "In one sentence, what is a compile error?"
  "Explain how sonar returns a seabed target in open water."
  "Write a Rust function that merges two sorted vectors."
  "Describe the Great Lakes as a marine surveyor would."
  "What causes class imbalance in shipwreck detection models?"
  "Summarize how batchy normalization affects gradient flow."
  "Describe a wrecks database schema for spatial queries."
  "Explain top-k expert routing in a mixture of experts model."
)

i=0
while true; do
  p="${PROMPTS[$((i % ${#PROMPTS[@]}))]}"
  stamp="$(date -u +%Y%m%dT%H%M%S)"
  out="$OUTDIR/run-$stamp.json"
  NAUTIBURN_MODEL_ID="$MODEL_ID" \
  NAUTIBURN_OUT="$out" \
  NAUTIBURN_CLASS_ID="$CLASS_ID" \
  NAUTIBURN_DTYPE="gguf" \
    "$BIN" -m "$MODEL" -p "$p" -ngl 0 -c "$CTX" -t "$THREADS" -fa off \
    >>"$OUTDIR/scrape.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ] && [ -s "$out" ]; then
    echo "[$(date -u +%FT%TZ)] scrape ok $out"
  else
    echo "[$(date -u +%FT%TZ)] scrape FAILED rc=$rc" 
    # Do not spin on a persistent failure; back off but keep the loop alive.
    sleep "$((INTERVAL * 2))"
  fi
  i=$((i + 1))
  sleep "$INTERVAL"
done
