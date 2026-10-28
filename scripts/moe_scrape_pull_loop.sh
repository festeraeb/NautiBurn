#!/usr/bin/env bash
# Pull live MoE scrape JSON written by hooked llama-servers on a remote node into the local
# SCRAPE_DIR that moe_histogram_consume_loop.sh ingests. The remote hook writes to local disk
# (one file per decode step, too chatty for NFS); this moves the finished files over SSH.
set -uo pipefail

NODE="${NODE:-cesarops4om}"
REMOTE_DIR="${REMOTE_DIR:-/mnt/models/nautiburn-scrape}"
SCRAPE_DIR="${SCRAPE_DIR:-/mnt/fast/nautiburn/scrape/sidechannel}"
INTERVAL="${INTERVAL:-15}"
MIN_AGE="${MIN_AGE:-5}"   # seconds; skip files the hook may still be writing

mkdir -p "$SCRAPE_DIR"
while true; do
  # List files older than MIN_AGE on the remote, then move exactly those.
  list="$(ssh -o BatchMode=yes -o ConnectTimeout=8 "$NODE" \
    "find '$REMOTE_DIR' -maxdepth 1 -name 'run-*.json' -mmin +0 -not -newermt '-${MIN_AGE} seconds' -printf '%f\n'" 2>/dev/null)"
  if [ -n "$list" ]; then
    n=$(printf '%s\n' "$list" | wc -l)
    if printf '%s\n' "$list" | rsync -a --remove-source-files --files-from=- \
         -e "ssh -o BatchMode=yes" "$NODE:$REMOTE_DIR/" "$SCRAPE_DIR/"; then
      echo "[$(date -u +%FT%TZ)] pulled $n from $NODE"
    else
      echo "[$(date -u +%FT%TZ)] pull FAILED from $NODE ($n pending)"
    fi
  fi
  sleep "$INTERVAL"
done
