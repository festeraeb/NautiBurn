#!/usr/bin/env python3
"""Build OLMoE expert-weight calibration from the F16 GGUF.

The checkpoint has router rows and the three expert projections. It does not
ship a calibration table. This script writes one: per layer, per expert, the
router-row L2 and the RMS of gate, up, and down. Numbers come from the file.
"""

import argparse
import json
import math
import struct
import sys

import numpy as np

GGML_F32 = 0
GGML_F16 = 1
TYPE_SIZE = {GGML_F32: 4, GGML_F16: 2}


def read_string(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8")


def read_value(f, typ):
    if typ == 8:
        return read_string(f)
    if typ == 4:
        return struct.unpack("<I", f.read(4))[0]
    if typ == 5:
        return struct.unpack("<i", f.read(4))[0]
    if typ == 6:
        return struct.unpack("<f", f.read(4))[0]
    if typ == 7:
        return struct.unpack("<?", f.read(1))[0]
    if typ == 10:
        return struct.unpack("<Q", f.read(8))[0]
    if typ == 11:
        return struct.unpack("<q", f.read(8))[0]
    if typ == 12:
        return struct.unpack("<d", f.read(8))[0]
    if typ in (0, 1):
        f.read(1)
        return None
    if typ in (2, 3):
        f.read(2)
        return None
    if typ == 9:
        elem = struct.unpack("<I", f.read(4))[0]
        count = struct.unpack("<Q", f.read(8))[0]
        return [read_value(f, elem) for _ in range(count)]
    raise SystemExit(f"unsupported gguf value type {typ}")


def align_up(pos, alignment):
    return (pos + alignment - 1) // alignment * alignment


def load_header(path):
    with open(path, "rb") as f:
        magic, version = struct.unpack("<II", f.read(8))
        if magic != 0x46554747:
            raise SystemExit(f"not a gguf file, magic={magic:#x}")
        tensor_count, kv_count = struct.unpack("<QQ", f.read(16))
        meta = {}
        for _ in range(kv_count):
            key = read_string(f)
            typ = struct.unpack("<I", f.read(4))[0]
            meta[key] = read_value(f, typ)
        tensors = []
        for _ in range(tensor_count):
            name = read_string(f)
            n_dims = struct.unpack("<I", f.read(4))[0]
            dims = list(struct.unpack("<" + "Q" * n_dims, f.read(8 * n_dims)))
            ggml_type, offset = struct.unpack("<IQ", f.read(12))
            tensors.append(
                {"name": name, "dims": dims, "type": ggml_type, "offset": offset}
            )
        alignment = int(meta.get("general.alignment", 32))
        data_start = align_up(f.tell(), alignment)
    return meta, tensors, data_start


def tensor_bytes(tensor):
    elems = 1
    for dim in tensor["dims"]:
        elems *= dim
    size = TYPE_SIZE.get(tensor["type"])
    if size is None:
        raise SystemExit(f"{tensor['name']} type {tensor['type']} is not f16 or f32")
    return elems, elems * size


def read_tensor(f, data_start, tensor):
    elems, nbytes = tensor_bytes(tensor)
    f.seek(data_start + tensor["offset"])
    raw = f.read(nbytes)
    if len(raw) != nbytes:
        raise SystemExit(f"short read {tensor['name']}: {len(raw)} != {nbytes}")
    if tensor["type"] == GGML_F16:
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32, copy=False)
    return np.frombuffer(raw, dtype=np.float32)


def rms(values):
    if values.size == 0:
        return 0.0
    return float(math.sqrt(float(np.dot(values, values)) / values.size))


def l2(values):
    return float(math.sqrt(float(np.dot(values, values))))


def build(path):
    meta, tensors, data_start = load_header(path)
    by_name = {t["name"]: t for t in tensors}
    layers = int(meta.get("olmoe.block_count", meta.get("llama.block_count", 16)))
    experts = int(meta.get("olmoe.expert_count", meta.get("llama.expert_count", 64)))
    hidden = int(meta.get("olmoe.embedding_length", meta.get("llama.embedding_length", 2048)))
    ff = int(meta.get("olmoe.expert_feed_forward_length", meta.get("llama.expert_feed_forward_length", 1024)))
    rows = []
    with open(path, "rb") as f:
        for layer in range(layers):
            gate_inp = by_name[f"blk.{layer}.ffn_gate_inp.weight"]
            gate = by_name[f"blk.{layer}.ffn_gate_exps.weight"]
            up = by_name[f"blk.{layer}.ffn_up_exps.weight"]
            down = by_name[f"blk.{layer}.ffn_down_exps.weight"]
            router = read_tensor(f, data_start, gate_inp)
            gate_w = read_tensor(f, data_start, gate)
            up_w = read_tensor(f, data_start, up)
            down_w = read_tensor(f, data_start, down)
            per = hidden * ff
            if gate_inp["dims"][-1] != experts or router.size != experts * hidden:
                raise SystemExit(
                    f"layer {layer} router dims={gate_inp['dims']} size={router.size}"
                )
            for name, tensor, values in (
                ("gate", gate, gate_w),
                ("up", up, up_w),
                ("down", down, down_w),
            ):
                if tensor["dims"][-1] != experts or values.size != experts * per:
                    raise SystemExit(
                        f"layer {layer} {name} dims={tensor['dims']} size={values.size}"
                    )
            # ggml ne[0] is contiguous and the expert index is the slowest dim,
            # so each expert is one contiguous block.
            router = router.reshape(experts, hidden)
            gate_w = gate_w.reshape(experts, per)
            up_w = up_w.reshape(experts, per)
            down_w = down_w.reshape(experts, per)
            for expert in range(experts):
                rows.append(
                    {
                        "layer": layer,
                        "expert": expert,
                        "router_l2": l2(router[expert]),
                        "gate_rms": rms(gate_w[expert]),
                        "up_rms": rms(up_w[expert]),
                        "down_rms": rms(down_w[expert]),
                    }
                )
            print(f"layer={layer} experts={experts}", file=sys.stderr)
    return {
        "model": "olmoe-1b-7b-0924",
        "source": "gguf-expert-weights",
        "layers": layers,
        "experts": experts,
        "hidden": hidden,
        "ff": ff,
        "dtype": "f16-projections-f32-router",
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf")
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()
    table = build(args.gguf)
    with open(args.output, "w", encoding="utf-8") as out:
        json.dump(table, out, separators=(",", ":"))
        out.write("\n")
    print(
        f"rows={len(table['rows'])} layers={table['layers']} experts={table['experts']} wrote={args.output}"
    )


if __name__ == "__main__":
    main()
