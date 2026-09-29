#!/usr/bin/env python3
"""Two-lane dispatcher for NautiBurn briefs.

Sends one small dispatch to both coding lanes and prints both answers side by
side so a human (or the judge seat) can pick.

  lane A  local  -> paddler :8080/katcoder  (KAT-Coder V2.5 abliterated, c4 P100 #0+#1)
  lane B  cloud  -> paddler :8080/reviewer   (NIM, deepseek-v4-pro class)

Both lanes go through the same OpenAI-compatible surface, so a lane can be
swapped by editing paddler.slots.json + `systemctl reload cesarops-paddler`
without touching this script. That is the point of the JSON overlay.

Usage:
  two_lane_dispatch.py --list-lanes
  two_lane_dispatch.py --task nautiburn-brief3
  two_lane_dispatch.py --prompt "..."
  two_lane_dispatch.py --prompt-file path.md
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
TASKS_JSON = HERE / "dispatch_tasks.json"

PADDLER = os.environ.get("PADDLER_URL", "http://127.0.0.1:8080").rstrip("/")
TIMEOUT = int(os.environ.get("DISPATCH_TIMEOUT", "900"))

LANES = {
    "A": {
        "slot": "katcoder",
        "label": "local  KAT-Coder-V2.5-dev abliterated Q4_K_M (c4 P100 #0+#1)",
    },
    "B": {
        "slot": "lane-b-cloud",
        "label": "cloud  NIM deepseek-v4-pro (no local fallback)",
    },
}


def load_tasks() -> dict:
    with TASKS_JSON.open() as f:
        return json.load(f)


def post(slot: str, prompt: str, max_tokens: int) -> dict:
    url = f"{PADDLER}/{slot}/v1/chat/completions"
    payload = json.dumps(
        {
            "model": slot,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }
    ).encode()
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        body = json.loads(r.read().decode())
    dt = time.time() - t0
    msg = (body.get("choices") or [{}])[0].get("message", {}) or {}
    timings = body.get("timings") or {}
    return {
        "slot": slot,
        "model": body.get("model"),
        "content": msg.get("content") or "",
        "reasoning": msg.get("reasoning_content") or "",
        "secs": round(dt, 1),
        "gen_tps": round(timings.get("predicted_per_second") or 0, 1),
        "finish": (body.get("choices") or [{}])[0].get("finish_reason"),
    }


def list_lanes(tasks: dict) -> int:
    try:
        with urllib.request.urlopen(f"{PADDLER}/v1/models", timeout=15) as r:
            d = json.loads(r.read().decode())
    except Exception as e:  # noqa: BLE001
        print(f"cannot reach paddler at {PADDLER}: {e}", file=sys.stderr)
        return 1
    print(f"{len(d['data'])} slots on {PADDLER}\n")
    ok = True
    for lane, meta in LANES.items():
        hit = next((m for m in d["data"] if m.get("slot") == meta["slot"]), None)
        if hit is None:
            print(f"  lane {lane} {meta['slot']:10} MISSING SLOT")
            ok = False
            continue
        print(f"  lane {lane} {meta['slot']:10} {hit['status']['value']:9} {hit.get('backend','')}")
        ok = ok and hit["status"]["value"] == "loaded"
    print(f"\n  {len(tasks)} dispatch templates: {', '.join(sorted(tasks))}")
    print("  both lanes loaded" if ok else "  ONE OR BOTH LANES NOT LOADED")
    return 0 if ok else 2


def main() -> int:
    tasks = load_tasks()
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt")
    ap.add_argument("--prompt-file")
    ap.add_argument("--task", choices=sorted(tasks))
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--only", choices=["A", "B", "both"], default="both")
    ap.add_argument("--out", help="write both answers as JSON here")
    ap.add_argument("--list-lanes", action="store_true")
    a = ap.parse_args()

    if a.list_lanes:
        return list_lanes(tasks)

    if a.task:
        prompt = tasks[a.task]
    elif a.prompt_file:
        prompt = pathlib.Path(a.prompt_file).read_text()
    elif a.prompt:
        prompt = a.prompt
    else:
        ap.error("need --prompt, --prompt-file, or --task")

    results = {}
    for lane in (["A", "B"] if a.only == "both" else [a.only]):
        meta = LANES[lane]
        print(f"--> lane {lane}: {meta['label']}  (slot {meta['slot']})", flush=True)
        try:
            r = post(meta["slot"], prompt, a.max_tokens)
            results[lane] = r
            print(
                f"<-- lane {lane}: {r['model']} {r['secs']}s {r['gen_tps']} tok/s "
                f"finish={r['finish']} content={len(r['content'])}B",
                flush=True,
            )
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError) as e:
            results[lane] = {"slot": meta["slot"], "error": f"{type(e).__name__}: {e}"}
            print(f"<-- lane {lane}: FAILED {e}", flush=True)

    # A silent fallback makes both lanes the same model and the comparison
    # worthless. Paddler will not tell us it fell back, so check the model id:
    # if both lanes report the same model, the split is not real.
    ids = {
        ln: r.get("model")
        for ln, r in results.items()
        if "error" not in r
    }
    if len(ids) == 2 and len(set(ids.values())) == 1:
        print(
            f"\n!! BOTH LANES REPORTED {ids['A']} - one of them fell back to the\n"
            "   other. The two-lane comparison is invalid for this dispatch.",
            file=sys.stderr,
        )
        results["_warning"] = f"both lanes collapsed to {ids['A']}"

    print("\n" + "=" * 72)
    for lane, r in results.items():
        print(f"\n### LANE {lane} - {LANES[lane]['label']}")
        if "error" in r:
            print(f"ERROR: {r['error']}")
            continue
        print(f"model={r['model']} {r['secs']}s {r['gen_tps']} tok/s finish={r['finish']}")
        print("-" * 72)
        print(r["content"] or "(empty content)")

    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {a.out}")

    return 0 if all("error" not in r for r in results.values()) else 2


if __name__ == "__main__":
    sys.exit(main())
