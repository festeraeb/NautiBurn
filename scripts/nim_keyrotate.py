#!/usr/bin/env python3
"""NIM key rotation with health tracking.

Round-robins across every `NIM_API_KEY*` in the keyring, but a key that fails is
demoted rather than retried blindly. Observed 2026-09-29: one of the two keys
timed out at 40s while the other answered instantly, which is what produced the
HTTP 503 the cloud lane was returning.

State lives beside the keyring as JSON:
    {"healthy": ["NIM_API_KEY_2"], "cooldown_until": {"NIM_API_KEY_1": <epoch>}}

Usage:
  nim_keyrotate.py probe            # health-check every key, update state
  nim_keyrotate.py current          # print the key that would be used now
  nim_keyrotate.py status           # per-key health + cooldown
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time

KEYRING = pathlib.Path(
    os.environ.get("NIM_KEYRING", "/home/cesarops/.config/forge/nim-keys.env")
)
STATE = KEYRING.with_suffix(".state.json")
PROBE_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
COOLDOWN_SECS = 900
TIMEOUT = 25


def load_keys() -> dict[str, str]:
    keys: dict[str, str] = {}
    if not KEYRING.exists():
        return keys
    for line in KEYRING.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, val = line.partition("=")
        name, val = name.strip(), val.strip().strip("'\"")
        # Only the rotating ring, not the legacy single-key aliases.
        if name.startswith("NIM_API_KEY_") and val:
            keys[name] = val
    return keys


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text())
        except json.JSONDecodeError:
            pass
    return {"healthy": [], "cooldown_until": {}}


def save_state(st: dict) -> None:
    STATE.write_text(json.dumps(st, indent=2))
    STATE.chmod(0o600)


def probe(name: str, key: str) -> bool:
    """One tiny request. Timeout is a failure, not an unknown: a key that cannot
    answer in TIMEOUT is not usable for a job run."""
    env = dict(os.environ)
    cmd = [
        "curl", "-sS", "--max-time", str(TIMEOUT),
        "-o", "/dev/null", "-w", "%{http_code}",
        "https://integrate.api.nvidia.com/v1/chat/completions",
        "-H", f"Authorization: Bearer {key}",
        "-H", "Content-Type: application/json",
        "-d", json.dumps({
            "model": PROBE_MODEL,
            "messages": [{"role": "user", "content": "ok"}],
            "max_tokens": 4,
        }),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=TIMEOUT + 5)
    except subprocess.TimeoutExpired:
        return False
    return out.stdout.strip() == "200"


def do_probe() -> int:
    keys = load_keys()
    if not keys:
        print(f"no NIM_API_KEY_* entries in {KEYRING}", file=sys.stderr)
        return 1
    st = load_state()
    now = time.time()
    healthy, cool = [], {}
    print(f"probing {len(keys)} key(s) against {PROBE_MODEL}\n")
    for name, key in keys.items():
        ok = probe(name, key)
        # Key rotation order: preserve the ring order so rotation is stable.
        if ok:
            healthy.append(name)
            print(f"  {name:20} OK")
        else:
            cool[name] = now + COOLDOWN_SECS
            print(f"  {name:20} FAIL -> cooldown {COOLDOWN_SECS}s")
    # Carry forward keys that are still inside a cooldown but were not re-probed.
    for name, until in st.get("cooldown_until", {}).items():
        if name in keys and name not in healthy and name not in cool:
            if until > now:
                cool[name] = until
    st = {"healthy": healthy, "cooldown_until": cool, "checked_at": now}
    save_state(st)
    print(f"\nhealthy: {healthy or 'NONE'}")
    if not healthy:
        print("every key failed — the cloud lane will fall back, not fail the job", file=sys.stderr)
        return 2
    return 0


def usable_keys() -> list[str]:
    st = load_state()
    now = time.time()
    names = load_keys()
    ready = [n for n in st.get("healthy", []) if n in names]
    ready += [n for n in names if n not in ready]
    # Drop anything still cooling down.
    return [n for n in ready if st.get("cooldown_until", {}).get(n, 0) <= now]


def do_current() -> int:
    ok = usable_keys()
    if not ok:
        print("NONE")
        return 2
    print(ok[0])
    return 0


def do_status() -> int:
    keys = load_keys()
    st = load_state()
    now = time.time()
    print(f"keyring: {KEYRING}")
    print(f"state:   {STATE}\n")
    for name in keys:
        h = name in st.get("healthy", [])
        until = st.get("cooldown_until", {}).get(name, 0)
        if h:
            status = "healthy"
        elif until > now:
            status = f"cooldown {int(until - now)}s"
        else:
            status = "unknown (run probe)"
        print(f"  {name:20} {status}")
    print()
    do_current()
    return 0


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "probe":
        return do_probe()
    if cmd == "current":
        return do_current()
    if cmd == "status":
        return do_status()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
