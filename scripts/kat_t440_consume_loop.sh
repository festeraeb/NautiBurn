#!/usr/bin/env bash
set -uo pipefail
SHADOW=/data/cargo-target/nautiburn-b0-main/debug/nautiburn-shadow
HISTO=/data/cargo-target/nautiburn-b0-main/debug/nautiburn-histodb
DB=/tmp/nautiburn-histodb/live-kat-shadow.sqlite
DEVICES=/home/cesarops/NautiBurn/crates/nautiburn-shadow/fixtures/devices.stub.json
LOCAL=/tmp/nautiburn-llama-scrape/kat
LOG=/tmp/nautiburn-shadow-live/kat-continuous.log
mkdir -p "$LOCAL" "$(dirname "$LOG")"
seen=/tmp/nautiburn-shadow-live/kat-seen.txt
touch "$seen"
echo "kat-consume start $(date -u +%FT%TZ)" >>"$LOG"
n=0
while true; do
  # pull new scrapes from c4
  rsync -az --ignore-missing-args cesarops4:/tmp/nautiburn-llama-scrape/kat/ "$LOCAL/" 2>>"$LOG" || true
  for pref in "$LOCAL"/run-*.json; do
    [ -f "$pref" ] || continue
    case "$pref" in *-decode.json) continue ;; esac
    base=$(basename "$pref")
    grep -qxF "$base" "$seen" && continue
    n=$((n+1))
    echo "[$(date -u +%FT%TZ)] ingest $base" >>"$LOG"
    "$HISTO" ingest --db "$DB" "$pref" >>"$LOG" 2>&1 || true
    dec="${pref%.json}-decode.json"
    [ -f "$dec" ] && "$HISTO" ingest --db "$DB" "$dec" >>"$LOG" 2>&1 || true
    belief=/tmp/nautiburn-shadow-live/belief-kat-${base%.json}.json
    if [ -f "$dec" ]; then
      "$SHADOW" prepredict --db "$DB" --scrape "$pref" --also-decode "$dec" \
        --label "live-kat-$n" --belief-out "$belief" >>"$LOG" 2>&1 || true
    else
      "$SHADOW" prepredict --db "$DB" --scrape "$pref" \
        --label "live-kat-$n" --belief-out "$belief" >>"$LOG" 2>&1 || true
    fi
    [ -f "$belief" ] && "$SHADOW" schedule --belief "$belief" --devices "$DEVICES" --request-id "kat-$n" >>"$LOG" 2>&1 || true
    echo "$base" >>"$seen"
  done
  sleep 30
done
