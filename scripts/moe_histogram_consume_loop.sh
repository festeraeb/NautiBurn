#!/usr/bin/env bash
set -uo pipefail

# ── environment (same defaults as before, plus new ones) ────────────────────
BIN="${BIN:-/data/cargo-target/nautiburn-b0-main/debug}"
HISTO="${HISTO:-$BIN/nautiburn-histodb}"
SHADOW="${SHADOW:-$BIN/nautiburn-shadow}"
DB="${DB:-/tmp/nautiburn-histodb/sidechannel.sqlite}"
SCRAPE_DIR="${SCRAPE_DIR:-/tmp/nautiburn-llama-scrape/sidechannel}"
DEVICES="${DEVICES:-/home/cesarops/NautiBurn/config/fleet-devices.json}"
SEEN="${SEEN:-/tmp/nautiburn-histodb/sidechannel-seen.txt}"
POLL="${POLL:-15}"
LOG="${LOG:-/tmp/nautiburn-histodb/consume.log}"
ARCHIVE_DIR="${ARCHIVE_DIR:-/mnt/raid0/nautiburn/scrape-archive}"
BATCH="${BATCH:-2000}"

# ── ensure directories exist ────────────────────────────────────────────────
mkdir -p "$(dirname "$DB")" "$(dirname "$SEEN")" "$(dirname "$LOG")" "$ARCHIVE_DIR"
touch "$SEEN"

# ── tiny log‑size guard (consume.log already ~141 MB) ───────────────────────
if [ -s "$LOG" ] && [ "$(stat -c %s "$LOG" 2>/dev/null || echo 0)" -gt $((100*1024*1024)) ]; then
    mv "$LOG" "${LOG}.1" 2>/dev/null || true
    >"$LOG"
fi
pack_dir() {
    local dir=$1 day out
    day=$(basename "$dir")
    out="$ARCHIVE_DIR/$day.tar.zst"
    [ -e "$out" ] && out="$ARCHIVE_DIR/$day.$(date -u +%s).tar.zst"
    if tar -cf - -C "$ARCHIVE_DIR" "$day" | zstd -q -o "$out.tmp" && zstd -tq "$out.tmp" && mv "$out.tmp" "$out"; then
        rm -rf "$dir"
        echo "[$(date -u +%FT%TZ)] packed $day -> $(basename "$out")" >>"$LOG"
    else
        rm -f "$out.tmp"
        echo "[$(date -u +%FT%TZ)] pack ERROR for $day (kept)" >>"$LOG"
    fi
}

echo "moe-histogram-consume start $(date -u +%FT%TZ) db=$DB" >>"$LOG"

# ── helpers ─────────────────────────────────────────────────────────────────
# sorted basenames of run‑*.json files (excluding *-decode.json) using find
get_file_list() {
    find "$SCRAPE_DIR" -maxdepth 1 \
         -name 'run-*.json' ! -name '*-decode.json' -printf '%f\n' | sort
}

# latest file by modification time (no glob)
get_latest_file() {
    find "$SCRAPE_DIR" -maxdepth 1 \
         -name 'run-*.json' ! -name '*-decode.json' -printf '%T@ %f\n' |
        sort -nr | head -1 | cut -d' ' -f2-
}

# ── main loop ───────────────────────────────────────────────────────────────
n=0
while true; do
    # load the seen‑ledger once into an associative array (O(m))
    declare -A seen
    while IFS= read -r line; do
        [[ -n "$line" ]] && seen["$line"]=1
    done < "$SEEN"

    # candidate file list (all current scrape files, already sorted)
    file_list=$(get_file_list)

    # ── bounded batch ingestion (max BATCH files per pass) ────────────────
    processed=0
    if [ -n "$file_list" ]; then
        # read the sorted list; stop after BATCH iterations
        while IFS= read -r base && [ $processed -lt $BATCH ]; do
            # skip already‑seen files
            if [[ -v seen["$base"] ]]; then
                # Ingested on an earlier pass (or by the old loop): archive it instead of
                # leaving it in SCRAPE_DIR forever.
                if [ -f "$SCRAPE_DIR/$base" ]; then
                    mkdir -p "$ARCHIVE_DIR/$(date -u +%Y%m%d)"
                    mv "$SCRAPE_DIR/$base" "$ARCHIVE_DIR/$(date -u +%Y%m%d)/"
                fi
                continue
            fi
            f="$SCRAPE_DIR/$base"
            [ ! -f "$f" ] && continue          # already moved/removed

            # ingest
            if "$HISTO" ingest --db "$DB" "$f" >>"$LOG" 2>>"$LOG"; then
                # success → mark seen, count, log
                echo "$base" >>"$SEEN"
                n=$((n + 1))
                echo "[$(date -u +%FT%TZ)] ingested $base" >>"$LOG"

                # move to daily archive dir
                today=$(date -u +%Y%m%d)
                daily_dir="$ARCHIVE_DIR/$today"
                mkdir -p "$daily_dir"
                mv "$f" "$daily_dir/"

                processed=$((processed + 1))
            else
                # ingest failed – keep file, log, do NOT mark seen
                echo "[$(date -u +%FT%TZ)] ingest ERROR for $base" >>"$LOG"
            fi
        done <<< "$file_list"
    fi

    # ── prepredict / schedule using comm (sorted file list vs sorted ledger) ─
    sorted_seen=$(sort "$SEEN")
    # if we have any files left, pick the first unseen via comm
    if [ -n "$file_list" ]; then
        cand=$(comm -23 \
                <(printf '%s\n' "$file_list") \
                <(printf '%s\n' "$sorted_seen") 2>/dev/null | head -1)
    else
        cand=""
    fi

    if [ -n "$cand" ] && [ -f "$SCRAPE_DIR/$cand" ]; then
        belief="$(dirname "$DB")/belief-$cand.json"
        if "$SHADOW" prepredict \
             --db "$DB" --scrape "$SCRAPE_DIR/$cand" \
             --label "sidechannel-$cand" --belief-out "$belief" \
             >>"$LOG" 2>>"$LOG"; then
            "$SHADOW" schedule \
                 --belief "$belief" --devices "$DEVICES" \
                 --request-id "sidechain-$cand" >>"$LOG" 2>>"$LOG" || true
            echo "[$(date -u +%FT%TZ)] prepredict+schedule $cand" >>"$LOG"
        fi
    fi

    # ── every‑10‑pass hit‑rate report (still uses find, no glob) ────────────
    if [ $((n % 10)) -eq 0 ]; then
        latest=$(get_latest_file)
        [ -n "$latest" ] && "$SHADOW" prepredict \
             --db "$DB" --scrape "$SCRAPE_DIR/$latest" \
             --label "report" 2>>"$LOG" |
            grep -E "^ensemble_L\+(1|2)_slot_hit" |
            while read -r l; do
                echo "[$(date -u +%FT%TZ)] $l" >>"$LOG"
            done
    fi

    # ── pack day directories older than a day into .tar.zst ───────────────────
    # A directory is removed only after its archive is written and verified; an existing
    # archive for the same day gets a sibling name instead of being skipped or overwritten.
    find "$ARCHIVE_DIR" -maxdepth 1 -type d -name '????????' -mtime +1 -print0 |
        while IFS= read -r -d '' dir; do pack_dir "$dir"; done

    sleep "$POLL"
done
