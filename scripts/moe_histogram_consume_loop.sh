#!/usr/bin/env bash
set -uo pipefail

# ── environment (same defaults as before, plus new ones) ────────────────────
BIN="${BIN:-/data/cargo-target/nautiburn-b0-main/release}"
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
CHUNK="${CHUNK:-200}"

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
    # Unseen files go to histodb CHUNK at a time (one process and one DB open per chunk; per-file
    # calls on the 7 GB DB managed ~77 files/min). Seen files and ingested files are archived with
    # one bulk mv per chunk.
    processed=0
    if [ -n "$file_list" ]; then
        day_dir="$ARCHIVE_DIR/$(date -u +%Y%m%d)"
        mkdir -p "$day_dir"
        todo=(); already=()
        while IFS= read -r base; do
            [ -n "$base" ] || continue
            if [[ -v seen["$base"] ]]; then already+=("$SCRAPE_DIR/$base"); continue; fi
            [ ${#todo[@]} -lt "$BATCH" ] && todo+=("$SCRAPE_DIR/$base")
        done <<< "$file_list"
        for ((i = 0; i < ${#already[@]}; i += CHUNK)); do
            mv -t "$day_dir" -- "${already[@]:i:CHUNK}" 2>>"$LOG" || true
        done
        for ((i = 0; i < ${#todo[@]}; i += CHUNK)); do
            chunk=("${todo[@]:i:CHUNK}")
            ok=()
            while IFS= read -r line; do
                case "$line" in
                    "ok "*)  path=${line#ok }; ok+=("$path"); base=$(basename "$path")
                             echo "$base" >>"$SEEN"; n=$((n + 1)); processed=$((processed + 1))
                             echo "[$(date -u +%FT%TZ)] ingested $base" >>"$LOG" ;;
                    "err "*) echo "[$(date -u +%FT%TZ)] ingest ERROR ${line#err }" >>"$LOG" ;;
                    *)       echo "$line" >>"$LOG" ;;
                esac
            done < <("$HISTO" ingest --db "$DB" "${chunk[@]}" 2>>"$LOG")
            [ ${#ok[@]} -gt 0 ] && { mv -t "$day_dir" -- "${ok[@]}" 2>>"$LOG" || true; }
        done
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
