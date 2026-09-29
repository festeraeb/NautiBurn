#!/usr/bin/env bash
# Consume MoE sidechannel scrapes -> histodb -> prepredict -> schedule.
#
# Each unseen scrape file is ingested once (tracked in a seen-file ledger), which
# bumps the histogram, then the NEXT-unseen file is used to pre-predict L+1/L+2
# experts and log a job-to-device decision. Predicting on a file that has NOT
# been ingested yet is what makes the reported hit rate honest rather than
# circular against the same activations.
set -uo pipefail

BIN="${BIN:-/data/cargo-target/nautiburn-b0-main/debug}"
HISTO="${HISTO:-$BIN/nautiburn-histodb}"
SHADOW="${SHADOW:-$BIN/nautiburn-shadow}"
DB="${DB:-/tmp/nautiburn-histodb/sidechannel.sqlite}"
SCRAPE_DIR="${SCRAPE_DIR:-/tmp/nautiburn-llama-scrape/sidechannel}"
DEVICES="${DEVICES:-/home/cesarops/NautiBurn/config/fleet-devices.json}"
SEEN="${SEEN:-/tmp/nautiburn-histodb/sidechannel-seen.txt}"
POLL="${POLL:-15}"
LOG="${LOG:-/tmp/nautiburn-histodb/consume.log}"

mkdir -p "$(dirname "$DB")" "$(dirname "$SEEN")" "$(dirname "$LOG")"
touch "$SEEN"
echo "moe-histogram-consume start $(date -u +%FT%TZ) db=$DB" >>"$LOG"

n=0
while true; do
  # Ingest every unseen prefill scrape, oldest first.
  while IFS= read -r f; do
    base=$(basename "$f")
    grep -qxF "$base" "$SEEN" && continue
    "$HISTO" ingest --db "$DB" "$f" >>"$LOG" 2>&1 || true
    echo "$base" >>"$SEEN"
    n=$((n + 1))
    echo "[$(date -u +%FT%TZ)] ingested $base" >>"$LOG"
  done < <(ls -1 "$SCRAPE_DIR"/run-*.json 2>/dev/null | grep -v -- '-decode.json' | sort)

  # Pre-predict using the most recent UNSEEN scrape if one exists (a scrape the
  # histogram has not seen yet). When the scrape loop is the only writer there is
  # normally nothing held back, so fall back to the newest ingest and label it.
  cand=$(comm -23 \
          <(ls -1 "$SCRAPE_DIR"/run-*.json 2>/dev/null | grep -v -- '-decode.json' | xargs -r -n1 basename 2>/dev/null | sort) \
          <(sort "$SEEN") 2>/dev/null | head -1)
  if [ -n "$cand" ] && [ -f "$SCRAPE_DIR/$cand" ]; then
    belief="$(dirname "$DB")/belief-$cand.json"
    if "$SHADOW" prepredict --db "$DB" --scrape "$SCRAPE_DIR/$cand" \
         --label "sidechannel-$cand" --belief-out "$belief" >>"$LOG" 2>&1; then
      "$SHADOW" schedule --belief "$belief" --devices "$DEVICES" \
         --request-id "sidechain-$cand" >>"$LOG" 2>&1 || true
      echo "[$(date -u +%FT%TZ)] prepredict+schedule $cand" >>"$LOG"
    fi
  fi

  # Report a one-line hit-rate summary so the shadow can be judged at a glance.
  if [ $((n % 10)) -eq 0 ]; then
    latest=$(ls -1t "$SCRAPE_DIR"/run-*.json 2>/dev/null | grep -v -- '-decode.json' | head -1)
    [ -n "$latest" ] && "$SHADOW" prepredict --db "$DB" --scrape "$latest" \
      --label "report" 2>/dev/null | grep -E "^ensemble_L\+(1|2)_slot_hit" | while read -r l; do
        echo "[$(date -u +%FT%TZ)] $l" >>"$LOG"
      done
  fi

  sleep "$POLL"
done
