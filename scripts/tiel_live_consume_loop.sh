#!/usr/bin/env bash
# Consume live-server Tiel scrapes (NAUTIBURN_SCRAPE_DIR) → histodb → prepredict.
# Complements (does not replace) the same-weights sidechannel loop.
set -uo pipefail
SHADOW=/data/cargo-target/nautiburn-b0-main/debug/nautiburn-shadow
HISTO=/data/cargo-target/nautiburn-b0-main/debug/nautiburn-histodb
DB=/tmp/nautiburn-histodb/live-tiel-shadow.sqlite
DEVICES=/home/cesarops/NautiBurn/crates/nautiburn-shadow/fixtures/devices.stub.json
LOCAL=/tmp/nautiburn-llama-scrape/tiel-live
LOG=/tmp/nautiburn-shadow-live/tiel-live-continuous.log
seen=/tmp/nautiburn-shadow-live/tiel-live-seen.txt
mkdir -p "$LOCAL" "$(dirname "$LOG")"
touch "$seen"
echo "tiel-live-consume start $(date -u +%FT%TZ)" >>"$LOG"
n=0
while true; do
  for pref in "$LOCAL"/run-*.json; do
    [ -f "$pref" ] || continue
    base=$(basename "$pref")
    grep -qxF "$base" "$seen" && continue
    # Prefer larger prefill frames for prepredict; skip tiny warmup if desired
    n=$((n+1))
    echo "[$(date -u +%FT%TZ)] ingest $base" >>"$LOG"
    "$HISTO" ingest --db "$DB" "$pref" >>"$LOG" 2>&1 || true
    belief=/tmp/nautiburn-shadow-live/belief-tiel-live-${base%.json}.json
    "$SHADOW" prepredict --db "$DB" --scrape "$pref" \
      --label "live-tiel-server-$n" --belief-out "$belief" >>"$LOG" 2>&1 || true
    [ -f "$belief" ] && "$SHADOW" schedule --belief "$belief" --devices "$DEVICES" --request-id "tiel-live-$n" >>"$LOG" 2>&1 || true
    echo "$base" >>"$seen"
  done
  sleep 15
done
