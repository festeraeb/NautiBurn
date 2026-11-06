#!/usr/bin/env bash
set -uo pipefail

# ── environment (same defaults as before, plus new ones) ────────────────────
BIN="${BIN:-/data/cargo-target/nautiburn-b0-main/release}"
HISTO="${HISTO:-$BIN/nautiburn-histodb}"
SHADOW="${SHADOW:-$BIN/nautiburn-shadow}"
DB="${DB:-/mnt/fast/nautiburn/state/sidechannel.sqlite}"
SCRAPE_DIR="${SCRAPE_DIR:-/mnt/fast/nautiburn/scrape/sidechannel}"
DEVICES="${DEVICES:-/home/cesarops/NautiBurn/config/fleet-devices.json}"
SEEN="${SEEN:-/mnt/fast/nautiburn/state/sidechannel-seen.txt}"
POLL="${POLL:-15}"
LOG="${LOG:-/mnt/fast/nautiburn/state/consume.log}"
ARCHIVE_DIR="${ARCHIVE_DIR:-/mnt/raid0/nautiburn/scrape-archive}"
BATCH="${BATCH:-2000}"
CHUNK="${CHUNK:-200}"
MAX_RETRIES="${MAX_RETRIES:-5}"
# Operator rule, asked four times before it was put in code (2026-11-04): live scrapes carry no prompt
# context, and the DB already holds far more prompt-less routing than it can use. A live file
# (run-<unix second>-<n>.json) named more than MAX_LIVE_AGE_S ago is deleted unread, not ingested.
MAX_LIVE_AGE_S="${MAX_LIVE_AGE_S:-86400}"
QUARANTINE_DIR="${QUARANTINE_DIR:-$(dirname "$ARCHIVE_DIR")/scrape-quarantine}"

# ── ensure directories exist ────────────────────────────────────────────────
mkdir -p "$(dirname "$DB")" "$(dirname "$SEEN")" "$(dirname "$LOG")" "$ARCHIVE_DIR" "$QUARANTINE_DIR"
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
# basenames of run-*.json files, newest names first: side-channel replays
# (run-YYYYMMDD...) and new live captures land before an old backlog
# (2026-10-24: 526k Oct-21 live-tiel files were starving new scrapes). The
# backlog is still ingested afterwards.
#
# 2026-11-04: *-decode.json is no longer excluded. The exclusion predated
# prompt awareness and was quietly discarding the decode half of every run --
# 237 files had accumulated here, 147 of them from that one day, and the rate
# is ~150/day. They are not an old context-free backlog: a decode scrape now
# carries `prompt` and `task_domain` like any other (verified by ingesting one
# into a scratch DB: 320 histogram counts, phase=decode, class_id
# sidechannel-<model>-<domain>). Decode is where almost all of the tokens are,
# so dropping it was dropping most of the routing evidence.
get_file_list() {
    find "$SCRAPE_DIR" -maxdepth 1 \
         -name 'run-*.json' -printf '%f\n' | sort -r
}

# latest file by modification time (no glob)
get_latest_file() {
    find "$SCRAPE_DIR" -maxdepth 1 \
         -name 'run-*.json' -printf '%T@ %f\n' |
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

    # failure ledger: "basename count" per line, loaded each cycle
    declare -A failures=()
    if [ -f "$SEEN.failures" ]; then
        while IFS= read -r line; do
            [[ -n "$line" ]] && {
                file=${line% *}
                count=${line##* }
                failures["$file"]=$count
            }
        done < "$SEEN.failures"
    fi

    # drop stale live scrapes before listing (see MAX_LIVE_AGE_S)
    cutoff=$(( $(date +%s) - MAX_LIVE_AGE_S ))
    dropped=$(find "$SCRAPE_DIR" "$QUARANTINE_DIR" -maxdepth 1 -regextype posix-extended \
                  -regex '.*/run-[0-9]{10}-[0-9]+(-decode)?\.json' -printf '%f\n' 2>/dev/null |
              awk -v c="$cutoff" -F- '$2+0 < c' |
              while IFS= read -r f; do rm -f -- "${SCRAPE_DIR:?}/${f:?}" "${QUARANTINE_DIR:?}/${f:?}" && echo x; done | wc -l)
    (( dropped > 0 )) && echo "[$(date -u +%FT%TZ)] dropped $dropped live scrape(s) older than ${MAX_LIVE_AGE_S}s, not ingested" >>"$LOG"

    # candidate file list (all current scrape files, already sorted)
    file_list=$(get_file_list)

    # ── bounded batch ingestion (max BATCH files per pass) ────────────────
    # Unseen files go to histodb CHUNK at a time (one process and one DB open per chunk; per-file
    # calls on the 7 GB DB managed ~77 files/min). Seen files and ingested files are archived with
    # one bulk mv per chunk.
    processed=0
    fresh=""   # the first (newest) scrape this pass ingested: the belief step below predicts from it
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
                             echo "$base" >>"$SEEN"; n=$((n + 1)); processed=$((processed + 1)); unset "failures[$base]"
                             [ -z "$fresh" ] && fresh=$base
                             echo "[$(date -u +%FT%TZ)] ingested $base" >>"$LOG" ;;
                    "err "*)
                        path_and_msg=${line#err }
                        path=${path_and_msg%% *}
                        base=$(basename "$path")
                        failures["$base"]=$(( ${failures["$base"]:-0} + 1 ))
                        echo "[$(date -u +%FT%TZ)] ingest ERROR (try ${failures[$base]}/$MAX_RETRIES) $path_and_msg" >>"$LOG"
                        if (( ${failures["$base"]} >= MAX_RETRIES )); then
                            if mv "$path" "$QUARANTINE_DIR/$base"; then
                                echo "[$(date -u +%FT%TZ)] quarantined $base" >>"$LOG"
                                unset "failures[$base]"
                            else
                                echo "[$(date -u +%FT%TZ)] quarantine FAILED for $base (kept in scrape dir)" >>"$LOG"
                            fi
                        fi
                        ;;
                    *)       echo "$line" >>"$LOG" ;;
                esac
            done < <("$HISTO" ingest --db "$DB" "${chunk[@]}" 2>>"$LOG")
            [ ${#ok[@]} -gt 0 ] && { mv -t "$day_dir" -- "${ok[@]}" 2>>"$LOG" || true; }
        done

        # ── bridge the newly ingested scrapes into PAMP's telemetry ────────
        # Expert routing and PAMP were two disconnected streams: everything the
        # scraper captured fed /Nauti-Burn, while PAMP's log held task outcomes
        # and almost nothing else (2026-11-04: 51,092 role_outcome vs 2
        # moe_route). This exports what the ensemble has been missing from the
        # database that already holds it. Watermarked, so it is idempotent and
        # a failure here must never block ingestion.
        if pamp_out=$("$HISTO" pamp-export --db "$DB" --limit "${PAMP_BATCH:-500}" 2>>"$LOG"); then
            case "$pamp_out" in
                *"moe_route=0 "*) : ;;   # nothing new, stay quiet
                *) echo "[$(date -u +%FT%TZ)] $pamp_out" >>"$LOG" ;;
            esac
        else
            echo "[$(date -u +%FT%TZ)] pamp-export failed (ingest unaffected)" >>"$LOG"
        fi
    fi

    # ── prepredict / schedule on the newest scrape this pass ingested ─────────
    # This used to take the first file the ingest above had NOT marked seen. Such a file exists only
    # while ingest has a backlog: once it kept up (2026-11-04 02:16) no belief was written for days
    # and the dashboard's belief counts froze. The file has been archived by now, so look there too.
    cand=$fresh; cand_path=""
    if [ -n "$cand" ]; then
        for p in "$SCRAPE_DIR/$cand" "${day_dir:-}/$cand"; do
            [ -f "$p" ] && { cand_path=$p; break; }
        done
    fi

    if [ -n "$cand_path" ]; then
        belief="$(dirname "$DB")/belief-$cand.json"
        if "$SHADOW" prepredict \
             --db "$DB" --scrape "$cand_path" \
             --label "sidechannel-$cand" --belief-out "$belief" \
             >>"$LOG" 2>>"$LOG"; then
            "$SHADOW" schedule \
                 --belief "$belief" --devices "$DEVICES" \
                 --request-id "sidechain-$cand" >>"$LOG" 2>>"$LOG" || true
            echo "[$(date -u +%FT%TZ)] prepredict+schedule $cand" >>"$LOG"
            # The belief is consumed by schedule right above; keeping one per pass leaked 3,643 files
            # (623 MB) next to the DB by 2026-10-20.
            rm -f "$belief"
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

    # ── write back the failure ledger ───────────────────────────────────────
    for base in "${!failures[@]}"; do
        echo "$base ${failures[$base]}"
    done >"$SEEN.failures.tmp" && mv "$SEEN.failures.tmp" "$SEEN.failures"

    sleep "$POLL"
done
