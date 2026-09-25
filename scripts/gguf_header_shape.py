#!/usr/bin/env python3
"""Read a GGUF header and print expert-shape fields.

Does not read tensor payloads. Byte sizes use the ggml block sizes from
ggml-common.h static_asserts (ggml_half is 2 bytes). Unknown ggml types are
reported without a byte size.
"""

import argparse
import json
import os
import struct
import sys

# (name, type_size, block_size)
GGML_BLOCKS = {
    0: ("f32", 4, 1),
    1: ("f16", 2, 1),
    2: ("q4_0", 2 + 32 // 2, 32),
    3: ("q4_1", 4 + 32 // 2, 32),
    6: ("q5_0", 2 + 4 + 32 // 2, 32),
    7: ("q5_1", 4 + 4 + 32 // 2, 32),
    8: ("q8_0", 2 + 32, 32),
    9: ("q8_1", 4 + 32, 32),
    10: ("q2_K", 4 + 256 // 16 + 256 // 4, 256),
    11: ("q3_K", 2 + 256 // 4 + 256 // 8 + 12, 256),
    12: ("q4_K", 4 + 12 + 256 // 2, 256),
    13: ("q5_K", 4 + 12 + 256 // 2 + 256 // 8, 256),
    14: ("q6_K", 2 + 256 // 16 + 3 * 256 // 4, 256),
    15: ("q8_K", 4 + 256 + (256 // 16) * 2, 256),
    30: ("bf16", 2, 1),
}

KEEP_SUFFIXES = (
    "architecture",
    "block_count",
    "expert_count",
    "expert_used_count",
    "embedding_length",
    "feed_forward_length",
    "expert_feed_forward_length",
    "expert_feed_forward_length_shared",
    "expert_shared_count",
    "expert_shared_feed_forward_length",
    "attention.head_count",
    "attention.head_count_kv",
    "file_type",
    "name",
    "quantization_version",
)


def read_exact(f, n):
    buf = f.read(n)
    if len(buf) != n:
        raise SystemExit(f"short read: wanted {n} bytes, got {len(buf)} at offset {f.tell()}")
    return buf


def read_string(f):
    n = struct.unpack("<Q", read_exact(f, 8))[0]
    return read_exact(f, n).decode("utf-8", "replace")


def scalar_size(typ):
    return {
        0: 1,
        1: 1,
        2: 2,
        3: 2,
        4: 4,
        5: 4,
        6: 4,
        7: 1,
        10: 8,
        11: 8,
        12: 8,
    }.get(typ)


def read_scalar(f, typ):
    size = scalar_size(typ)
    if size is None:
        raise SystemExit(f"unsupported gguf scalar type {typ}")
    raw = read_exact(f, size)
    fmt = {1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 8: None, 10: "<Q", 11: "<q", 12: "<d"}
    if typ == 0:
        return raw[0]
    return struct.unpack(fmt[typ], raw)[0]


def skip_value(f, typ, keep_string):
    if typ == 8:
        text = read_string(f)
        return text if keep_string else None
    if typ == 9:
        elem = struct.unpack("<I", read_exact(f, 4))[0]
        count = struct.unpack("<Q", read_exact(f, 8))[0]
        if elem == 8:
            for _ in range(count):
                n = struct.unpack("<Q", read_exact(f, 8))[0]
                f.seek(n, os.SEEK_CUR)
            return {"array_type": elem, "count": count, "stored": False}
        size = scalar_size(elem)
        if size is None:
            raise SystemExit(f"unsupported gguf array element type {elem}")
        f.seek(size * count, os.SEEK_CUR)
        return {"array_type": elem, "count": count, "stored": False}
    return read_scalar(f, typ)


def want_key(key):
    return key == "general.architecture" or key.endswith(tuple("." + s for s in KEEP_SUFFIXES)) or ".expert" in key


def tensor_nbytes(dims, ggml_type):
    info = GGML_BLOCKS.get(ggml_type)
    if info is None:
        return None, None
    name, type_size, block = info
    n = 1
    for d in dims:
        n *= d
    if block == 0 or n % block != 0:
        return name, None
    return name, (n // block) * type_size


def load_shape(path):
    file_size = os.stat(path).st_size
    with open(path, "rb") as f:
        magic, version = struct.unpack("<II", read_exact(f, 8))
        if magic != 0x46554747:
            raise SystemExit(f"not a gguf file, magic={magic:#x}")
        if version < 2:
            raise SystemExit(f"gguf version {version} is not parsed (need >= 2)")
        tensor_count, kv_count = struct.unpack("<QQ", read_exact(f, 16))
        meta = {}
        for _ in range(kv_count):
            key = read_string(f)
            typ = struct.unpack("<I", read_exact(f, 4))[0]
            keep = want_key(key)
            value = skip_value(f, typ, keep_string=keep)
            if keep:
                meta[key] = value
        tensors = []
        for _ in range(tensor_count):
            name = read_string(f)
            n_dims = struct.unpack("<I", read_exact(f, 4))[0]
            dims = list(struct.unpack("<" + "Q" * n_dims, read_exact(f, 8 * n_dims)))
            ggml_type, offset = struct.unpack("<IQ", read_exact(f, 12))
            if "ffn_" in name or "exps" in name or "expert" in name:
                type_name, nbytes = tensor_nbytes(dims, ggml_type)
                tensors.append(
                    {
                        "name": name,
                        "dims": dims,
                        "ggml_type": ggml_type,
                        "type_name": type_name,
                        "nbytes": nbytes,
                        "offset": offset,
                    }
                )
        header_end = f.tell()
    arch = meta.get("general.architecture")
    if not isinstance(arch, str):
        for key in meta:
            if key.endswith(".expert_count"):
                arch = key[: -len(".expert_count")]
                break
    expert_count = meta.get(f"{arch}.expert_count") if isinstance(arch, str) else None
    if not isinstance(expert_count, int):
        expert_count = None
    per_expert = summarize_experts(tensors, expert_count)
    return {
        "path": path,
        "file_size": file_size,
        "gguf_version": version,
        "tensor_count": tensor_count,
        "kv_count": kv_count,
        "header_end": header_end,
        "meta": meta,
        "expert_tensors_shown": per_expert,
    }


def summarize_experts(tensors, expert_count):
    expert_tensors = [
        t
        for t in tensors
        if "exps" in t["name"] or "shexp" in t["name"] or "expert" in t["name"]
    ]
    if not expert_tensors:
        return {
            "present": False,
            "reason": "no tensor name contains exps, shexp, or expert",
            "dense_ffn_sample": [
                {k: t[k] for k in ("name", "dims", "type_name", "nbytes")}
                for t in tensors
                if t["name"].startswith("blk.0.ffn_")
            ],
        }
    tensors = expert_tensors
    layer0 = [t for t in tensors if t["name"].startswith("blk.0.")]
    shown = layer0 if layer0 else tensors[:12]
    out = []
    for t in shown:
        item = dict(t)
        item.pop("offset", None)
        if expert_count and t["nbytes"] is not None and expert_count in t["dims"]:
            if t["nbytes"] % expert_count == 0:
                item["bytes_per_expert_this_tensor"] = t["nbytes"] // expert_count
            else:
                item["bytes_per_expert_this_tensor"] = None
                item["note"] = "nbytes not divisible by expert_count"
        out.append(item)
    kinds = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")
    packed = []
    for kind in kinds:
        matches = [t for t in tensors if t["name"].endswith(kind)]
        if not matches:
            continue
        sample = matches[0]
        per = None
        if (
            expert_count
            and expert_count in sample["dims"]
            and sample["nbytes"] is not None
            and sample["nbytes"] % expert_count == 0
        ):
            per = sample["nbytes"] // expert_count
        same = all(
            t["dims"] == sample["dims"] and t["nbytes"] == sample["nbytes"] and t["type_name"] == sample["type_name"]
            for t in matches
        )
        groups = []
        if not same:
            grouped = {}
            for t in matches:
                key = (tuple(t["dims"]), t["type_name"], t["nbytes"])
                grouped.setdefault(key, []).append(t["name"])
            for (dims, type_name, nbytes), names in grouped.items():
                per_g = None
                if expert_count and expert_count in dims and nbytes is not None and nbytes % expert_count == 0:
                    per_g = nbytes // expert_count
                groups.append(
                    {
                        "dims": list(dims),
                        "type_name": type_name,
                        "nbytes": nbytes,
                        "bytes_per_expert_this_tensor": per_g,
                        "count": len(names),
                        "names": names,
                    }
                )
        packed.append(
            {
                "suffix": kind,
                "tensors": len(matches),
                "dims": sample["dims"],
                "type_name": sample["type_name"],
                "nbytes": sample["nbytes"],
                "bytes_per_expert_this_tensor": per,
                "same_shape_every_layer": same,
                "shape_groups": groups,
            }
        )
    shared_kinds = ("ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight")
    shared = []
    for kind in shared_kinds:
        matches = [t for t in tensors if t["name"].endswith(kind)]
        if not matches:
            continue
        sample = matches[0]
        shared.append(
            {
                "suffix": kind,
                "tensors": len(matches),
                "dims": sample["dims"],
                "type_name": sample["type_name"],
                "nbytes": sample["nbytes"],
                "same_shape_every_layer": all(
                    t["dims"] == sample["dims"] and t["nbytes"] == sample["nbytes"]
                    for t in matches
                ),
            }
        )
    shared_one = None
    if len(shared) == 3 and all(p["nbytes"] is not None for p in shared):
        shared_one = sum(p["nbytes"] for p in shared)
    one = None
    if (
        len(packed) == 3
        and all(p["bytes_per_expert_this_tensor"] is not None and p["same_shape_every_layer"] for p in packed)
    ):
        one = sum(p["bytes_per_expert_this_tensor"] for p in packed)
    return {
        "present": True,
        "layer0_or_sample": out,
        "packed_projection_summary": packed,
        "one_expert_gate_up_down_bytes": one,
        "shared_expert_summary": shared,
        "one_shared_expert_gate_up_down_bytes": shared_one,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    json.dump(load_shape(args.path), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
