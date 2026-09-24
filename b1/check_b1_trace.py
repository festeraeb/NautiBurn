#!/usr/bin/env python3
"""Validate a B1 OLMoE fixture trace. Fails closed on missing or invented-looking gaps."""

import sys


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_b1_trace.py TRACE", file=sys.stderr)
        return 2
    header = {}
    rows = []
    llama_tokens = []
    with open(sys.argv[1], encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            if line.startswith("llama_token:"):
                _tag, idx, tok = line.split()
                llama_tokens.append((int(idx), int(tok)))
                continue
            if "\t" not in line and ": " in line:
                key, val = line.split(": ", 1)
                header[key] = val
                continue
            if "\t" not in line:
                continue
            parts = line.split("\t")
            if len(parts) != 6:
                print(f"bad row width: {line}")
                return 1
            phase, layer, token_index, token_id, tensor, experts = parts
            ids = [int(x) for x in experts.split(",") if x != ""]
            rows.append(
                {
                    "phase": phase,
                    "layer": int(layer),
                    "token_index": int(token_index),
                    "token_id": int(token_id),
                    "tensor": tensor,
                    "experts": ids,
                }
            )

    required = ("dtype", "quant", "gpu_uuid", "weight", "cast")
    missing = [k for k in required if k not in header]
    if missing:
        print("missing header: " + ",".join(missing))
        return 1
    if not rows:
        print("no data rows")
        return 1

    layers = sorted({r["layer"] for r in rows if r["phase"] == "prefill"})
    if layers != list(range(16)):
        print(f"prefill layers {layers} != 0..15")
        return 1

    for r in rows:
        if r["phase"] not in ("prefill", "decode"):
            print(f"bad phase {r['phase']}")
            return 1
        if r["tensor"] not in ("ffn_moe_topk", "ffn_moe_argsort"):
            print(f"bad tensor {r['tensor']}")
            return 1
        if len(r["experts"]) != 8:
            print(f"expected 8 experts, got {len(r['experts'])} layer {r['layer']}")
            return 1
        for eid in r["experts"]:
            if eid < 0 or eid >= 64:
                print(f"expert id {eid} outside 0..64")
                return 1
        if len(set(r["experts"])) != len(r["experts"]):
            print(f"duplicate expert in layer {r['layer']} token {r['token_index']}")
            return 1

    print(
        f"ok rows={len(rows)} layers=16 dtype={header['dtype']} "
        f"quant={header['quant']} gpu_uuid={header['gpu_uuid']} "
        f"llama_tokens={len(llama_tokens)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
