#!/usr/bin/env bash
# Fleet telemetry tunnel durability check.
#
# The nauti-tunnel-* units are systemd-managed with Restart=always, but systemd
# cannot tell a *listening* ssh forward from a *working* one: if the far end
# stops answering, ssh stays up and the local port still accepts, so the unit
# reports active while /api/telemetry returns nothing. That is how cesarops2
# intermittently showed as unreachable in the fleet rollup.
#
# This checks the far end through each tunnel and restarts the unit only when
# the probe actually fails. Run from the timer every 60s.
set -uo pipefail

LOG=/home/cesarops/logs/tunnel-watchdog.log
mkdir -p "$(dirname "$LOG")"

# unit:port pairs matching the nauti-tunnel-* units.
CHECKS=(
  "nauti-tunnel-cesarops2:19221"
  "nauti-tunnel-cesarops3:19222"
  "nauti-tunnel-cesarops4om:19223"
)

# Repairing a tunnel unit needs root; the oneshot runs as cesarops. Rather than
# running the whole watchdog as root (which would let a bug in this script act on
# any unit), it asks sudo for exactly the two verbs it needs, NOPASSWD, via a
# drop-in in /etc/sudoers.d.
SUDO=""
if [[ "$(id -u)" -ne 0 ]]; then
  SUDO="sudo -n"
fi
systemctl_do() {
  if [[ "$(id -u)" -eq 0 ]]; then
    systemctl "$@"
  else
    sudo -n systemctl "$@" || {
      echo "[$(date -u +%FT%TZ)] sudo denied for: systemctl $*" >>"$LOG"
      return 1
    }
  fi
}

failed=0
for check in "${CHECKS[@]}"; do
  unit="${check%%:*}"
  port="${check##*:}"

  # systemctl is-active "activating" means the unit is crash-looping.
  active=$(systemctl is-active "$unit" 2>/dev/null || echo unknown)

  if [[ "$active" != "active" ]]; then
    echo "[$(date -u +%FT%TZ)] $unit state=$active -> restart" >>"$LOG"
    systemctl_do restart "$unit" >>"$LOG" 2>&1
    failed=$((failed + 1))
    sleep 5
    continue
  fi

  # Probe through the tunnel. /health on nauti-daemon is cheap and answers only
  # when the far end is genuinely reachable.
  body=$(curl -s --max-time 12 "http://127.0.0.1:${port}/health" 2>/dev/null || true)
  if [[ -z "$body" ]]; then
    echo "[$(date -u +%FT%TZ)] $unit ACTIVE but :$port/health empty -> restart" >>"$LOG"
    systemctl_do restart "$unit" >>"$LOG" 2>&1
    failed=$((failed + 1))
    sleep 8
    # Confirm it came back; log but do not loop inside this run.
    if ! curl -s --max-time 12 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      echo "[$(date -u +%FT%TZ)] $unit STILL DOWN after restart" >>"$LOG"
    fi
  fi
done

if [[ $failed -gt 0 ]]; then
  echo "[$(date -u +%FT%TZ)] watchdog repaired $failed tunnel(s)" >>"$LOG"
fi
exit 0
