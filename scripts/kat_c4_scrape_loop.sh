#!/usr/bin/env bash
# c4: continuous Kat sidechannel scrape (experts on CPU via -ncmoe).
# JSON lands in /tmp/nautiburn-llama-scrape/kat/ for T440 consumer.
set -uo pipefail
export LD_LIBRARY_PATH=/home/cesarops/llama-cuda60/bin:/home/cesarops/llama-cuda60/cuda12
export CUDA_VISIBLE_DEVICES=GPU-c012cc70-3f06-1718-84e2-073aa275f0af
SCRAPE=/home/cesarops/bin/moe-histodb-scrape
MODEL=/home/cesarops/models/Kwaipilot_KAT-Coder-V2.5-Dev-Q8_0.gguf
OUTDIR=/tmp/nautiburn-llama-scrape/kat
LOG=/tmp/nautiburn-llama-scrape/kat-continuous.log
mkdir -p "$OUTDIR"
n=0
echo "kat-scrape-loop start $(date -u +%FT%TZ)" >>"$LOG"
while true; do
  n=$((n+1)); stamp=$(date +%s); out="$OUTDIR/run-${stamp}.json"
  echo "[$(date -u +%FT%TZ)] scrape #$n -> $out" >>"$LOG"
  if ! NAUTIBURN_MODEL_ID=kat-coder-v2.5-dev-q8_0 \
       NAUTIBURN_OUT="$out" NAUTIBURN_CLASS_ID=llama-sidechannel \
       NAUTIBURN_DTYPE=q8_0 NAUTIBURN_DECODE_N=1 \
       "$SCRAPE" -m "$MODEL" -p "Write a short Python function that adds two integers." \
         -ngl 99 -ncmoe 99 -c 64 -t 8 -fa off >>"$LOG" 2>&1; then
    echo "[$(date -u +%FT%TZ)] scrape failed; sleep 120" >>"$LOG"
    sleep 120
    continue
  fi
  echo "[$(date -u +%FT%TZ)] cycle $n done" >>"$LOG"
  sleep 180
done
