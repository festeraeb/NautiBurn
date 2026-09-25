#!/usr/bin/env python3
"""Time one OLMoE expert GEMM on OpenVINO device CPU.

Gate and up are [M, 2048] x [2048, 1024]. Down is [M, 1024] x [1024, 2048].
Each projection is its own compiled model. The device string is CPU.
"""

import argparse
import json
import socket
import sys
import time

import numpy as np
import openvino as ov

HIDDEN = 2048
FF = 1024
DEVICE = "CPU"


def read_cpu():
    model = None
    flags = []
    with open("/proc/cpuinfo", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("model name") and model is None:
                model = line.split(":", 1)[1].strip()
            elif line.startswith("flags") and not flags:
                flags = line.split(":", 1)[1].split()
            if model is not None and flags:
                break
    if model is None or not flags:
        raise SystemExit("could not read model name and flags from /proc/cpuinfo")
    return model, flags


def matmul_model(m, k, n, name):
    left = ov.op.Parameter(ov.Type.f32, ov.Shape([m, k]))
    right = ov.op.Parameter(ov.Type.f32, ov.Shape([k, n]))
    product = ov.opset13.matmul(left, right, False, False)
    return ov.Model(product, [left, right], name)


def compile_cpu(core, model):
    if DEVICE != "CPU":
        raise SystemExit("OpenVINO device must be CPU")
    return core.compile_model(model, "CPU")


def fill(m, k, n, seed):
    rng = np.random.default_rng(seed)
    left = rng.standard_normal((m, k)).astype(np.float32)
    right = rng.standard_normal((k, n)).astype(np.float32)
    return left, right


def check_against_numpy(compiled, left, right):
    result = compiled([left, right])
    got = np.asarray(result[0])
    ref = left @ right
    if got.shape != ref.shape or not np.allclose(got, ref, rtol=1e-3, atol=1e-3):
        raise SystemExit(
            f"CPU matmul mismatch shape={got.shape} expected={ref.shape}"
        )


def time_ms(compiled, left, right, repeats):
    compiled([left, right])
    start = time.perf_counter()
    for _ in range(repeats):
        compiled([left, right])
    elapsed = time.perf_counter() - start
    return (elapsed / repeats) * 1000.0


def measure(m, repeats):
    if m < 1 or repeats < 1:
        raise SystemExit("m and repeats must be >= 1")
    cpu_model, flags = read_cpu()
    core = ov.Core()
    if "CPU" not in core.available_devices:
        raise SystemExit(f"CPU device missing, available={core.available_devices}")
    full_name = core.get_property("CPU", "FULL_DEVICE_NAME")
    print(full_name, file=sys.stderr)

    specs = (
        ("gate", m, HIDDEN, FF, 1),
        ("up", m, HIDDEN, FF, 2),
        ("down", m, FF, HIDDEN, 3),
    )
    means = {}
    for name, rows, k, n, seed in specs:
        left, right = fill(rows, k, n, seed)
        compiled = compile_cpu(core, matmul_model(rows, k, n, name))
        check_against_numpy(compiled, left, right)
        means[name] = time_ms(compiled, left, right, repeats)

    gate_ms = means["gate"]
    up_ms = means["up"]
    down_ms = means["down"]
    row = {
        "host": socket.gethostname(),
        "cpu_model": cpu_model,
        "openvino_version": ov.__version__,
        "device": "CPU",
        "full_device_name": full_name,
        "avx512f": "avx512f" in flags,
        "avx512_vnni": "avx512_vnni" in flags,
        "avx512_bf16": "avx512_bf16" in flags,
        "m": m,
        "repeats": repeats,
        "gate_ms": gate_ms,
        "up_ms": up_ms,
        "down_ms": down_ms,
        "expert_ms": gate_ms + up_ms + down_ms,
    }
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(measure(args.m, args.repeats), separators=(",", ":")))


if __name__ == "__main__":
    main()
