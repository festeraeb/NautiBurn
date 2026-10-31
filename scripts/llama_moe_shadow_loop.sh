#!/usr/bin/env bash
# Continuous llama.cpp MoE side-channel scrape → histodb → nautiburn-shadow.
# Uses b1/moe-histodb-scrape (cb_eval on ffn_moe_topk). Does not replace the
# live llama-server; same weights, separate process. Leave OLMoE Burn shadow alone.
set -uo pipefail

SCRAPE_BIN="${SCRAPE_BIN:-/home/cesarops/NautiBurn/b1/moe-histodb-scrape}"
SHADOW="${SHADOW:-/data/cargo-target/nautiburn-b0-main/debug/nautiburn-shadow}"
HISTO="${HISTO:-/data/cargo-target/nautiburn-b0-main/debug/nautiburn-histodb}"
DEVICES="${DEVICES:-/home/cesarops/NautiBurn/crates/nautiburn-shadow/fixtures/devices.stub.json}"

MODEL_PATH="${MODEL_PATH:?MODEL_PATH required}"
MODEL_ID="${MODEL_ID:?MODEL_ID required}"
LABEL="${LABEL:-llama-moe}"
OUTDIR="${OUTDIR:-/mnt/fast/nautiburn/scrape/${LABEL}}"
DB="${DB:-/mnt/fast/nautiburn/state/live-${LABEL}-shadow.sqlite}"
LOG="${LOG:-/mnt/fast/nautiburn/shadow-live/${LABEL}-continuous.log}"
PROMPT="${PROMPT:-Write a short Python function that adds two integers.}"
NGL="${NGL:-0}"
CTX="${CTX:-256}"
THREADS="${THREADS:-8}"
SLEEP_SECS="${SLEEP_SECS:-60}"
DECODE_N="${DECODE_N:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

export LD_LIBRARY_PATH="${LLAMA_BIN:-/home/cesarops/src/llama.cpp-mainline/build-cuda-60/bin}:${LD_LIBRARY_PATH:-}"
mkdir -p "$OUTDIR" "$(dirname "$LOG")" "$(dirname "$DB")"
n=0
echo "llama-moe-shadow-loop start $(date -u +%FT%TZ) model=$MODEL_PATH id=$MODEL_ID ngl=$NGL" >>"$LOG"

while true; do
  n=$((n + 1))
  stamp=$(date +%s)
  out="$OUTDIR/run-${stamp}.json"
  echo "[$(date -u +%FT%TZ)] scrape #$n -> $out" >>"$LOG"
  if ! NAUTIBURN_MODEL_ID="$MODEL_ID" \
       NAUTIBURN_OUT="$out" \
       NAUTIBURN_CLASS_ID="llama-sidechannel" \
       NAUTIBURN_DTYPE="gguf" \
       NAUTIBURN_DECODE_N="$DECODE_N" \
       "$SCRAPE_BIN" -m "$MODEL_PATH" -p "$PROMPT" -ngl "$NGL" -c "$CTX" -t "$THREADS" \
         -fa off $EXTRA_ARGS >>"$LOG" 2>&1; then
    echo "[$(date -u +%FT%TZ)] scrape failed; sleep $SLEEP_SECS" >>"$LOG"
    sleep "$SLEEP_SECS"
    continue
  fi
  pref="$out"
  dec=""
  if [ -f "${out%.json}-decode.json" ]; then
    dec="${out%.json}-decode.json"
  fi
  "$HISTO" ingest --db "$DB" "$pref" >>"$LOG" 2>&1 || true
  if [ -n "$dec" ]; then
    "$HISTO" ingest --db "$DB" "$dec" >>"$LOG" 2>&1 || true
  fi
  belief="/mnt/fast/nautiburn/shadow-live/belief-${LABEL}-${stamp}.json"
  if [ -n "$dec" ]; then
    "$SHADOW" prepredict --db "$DB" --scrape "$pref" --also-decode "$dec" \
      --label "live-${LABEL}-$n" --belief-out "$belief" >>"$LOG" 2>&1 || true
  else
    "$SHADOW" prepredict --db "$DB" --scrape "$pref" \
      --label "live-${LABEL}-$n" --belief-out "$belief" >>"$LOG" 2>&1 || true
  fi
  if [ -f "$belief" ] && [ -f "$DEVICES" ]; then
    "$SHADOW" schedule --belief "$belief" --devices "$DEVICES" --request-id "${LABEL}-$stamp" >>"$LOG" 2>&1 || true
  fi
  echo "[$(date -u +%FT%TZ)] cycle $n done" >>"$LOG"
  sleep "$SLEEP_SECS"
done
