#!/usr/bin/env bash
# Harness for scripts/moe_histogram_consume_loop.sh with stub histodb/shadow binaries (no real DB).
# Checks: good scrapes are ingested, marked seen and archived; a file that always fails is retried
# at most MAX_RETRIES times, then moved to QUARANTINE_DIR (and not marked seen); nothing else is lost.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
mkdir -p "$T"/{scrape,archive,quarantine,db,bin}

cat >"$T/bin/nautiburn-histodb" <<'EOF'
#!/usr/bin/env bash
# stub: "ingest --db DB f1 [f2 ...]"; one ok/err line per file; files named *BAD* fail
shift 3
for f in "$@"; do
  case "$f" in *BAD*) echo "err $f malformed scrape json";; *) echo "ok $f";; esac
done
EOF
printf '#!/usr/bin/env bash\nexit 0\n' >"$T/bin/nautiburn-shadow"
chmod +x "$T"/bin/*

for i in $(seq 1 50); do echo '{}' >"$T/scrape/run-17924000$(printf %02d "$i")-$i.json"; done
echo '{bad' >"$T/scrape/run-1792400999-BAD.json"

BIN="$T/bin" DB="$T/db/x.sqlite" SCRAPE_DIR="$T/scrape" SEEN="$T/db/seen.txt" LOG="$T/db/consume.log" \
ARCHIVE_DIR="$T/archive" QUARANTINE_DIR="$T/quarantine" MAX_RETRIES=3 BATCH=2000 POLL=1 \
DEVICES=/dev/null timeout 20 bash "$HERE/moe_histogram_consume_loop.sh" >/dev/null 2>&1

fail=0
check() { if eval "$2"; then echo "ok   - $1"; else echo "FAIL - $1"; fail=1; fi; }
archived=$(find "$T/archive" -name 'run-*.json' | wc -l)
check "all 50 good scrapes archived (got $archived)" '[ "$archived" -eq 50 ]'
check "50 good scrapes marked seen" '[ "$(grep -c . "$T/db/seen.txt")" -eq 50 ]'
check "bad file not marked seen" '! grep -q BAD "$T/db/seen.txt"'
check "bad file moved to quarantine" '[ -f "$T/quarantine/run-1792400999-BAD.json" ]'
check "bad file no longer in scrape dir" '[ ! -f "$T/scrape/run-1792400999-BAD.json" ]'
tries=$(grep -c "ingest ERROR.*BAD" "$T/db/consume.log")
check "bad file retried at most 3 times (got $tries)" '[ "$tries" -ge 1 ] && [ "$tries" -le 3 ]'
check "quarantine is logged" 'grep -q "quarantined" "$T/db/consume.log"'
exit $fail
