#!/usr/bin/env python3
"""B6 stall timings on one P100. Starts and stops only the server it spawned."""

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

GPU = "GPU-72784f3e-9536-8d2b-cc49-3081a131de54"
MODEL = "/codebase/models/qwen1.5-moe-a2.7b-chat-q4_k_m.gguf"
SERVER = "/home/cesarops/src/llama.cpp-mainline/build-cuda-60/bin/llama-server"
PORT = 18771
CTX = "512"
PREDICT = 64
OUT = "/home/cesarops/NautiBurn/b6/timings.json"
EXPECTED_BYTES = 9490338464


def gpu_free_mib():
    out = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={GPU}",
            "--query-gpu=memory.free,memory.used,uuid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    free_s, used_s, uuid = [p.strip() for p in out.split(",")]
    return int(free_s), int(used_s), uuid


def wait_health(proc, timeout=600):
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{PORT}/health"
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited {proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(1)
    raise RuntimeError("health timeout")


def complete(prompt):
    body = json.dumps(
        {
            "prompt": prompt,
            "n_predict": PREDICT,
            "temperature": 0,
            "cache_prompt": False,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/completion",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:
        payload = json.loads(resp.read().decode())
    wall_ms = (time.perf_counter() - t0) * 1000.0
    timings = payload.get("timings") or {}
    return {
        "wall_ms": wall_ms,
        "tokens_predicted": payload.get("tokens_predicted"),
        "tokens_evaluated": payload.get("tokens_evaluated"),
        "timings": timings,
        "stop": payload.get("stop"),
    }


def stop_server(proc):
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=15)


def run_mode(name, extra_args, streams):
    free, used, uuid = gpu_free_mib()
    if uuid != GPU:
        raise RuntimeError(f"uuid mismatch {uuid}")
    if free < 12000 and "--cpu-moe" not in extra_args:
        raise RuntimeError(f"resident load needs more free MiB, have {free}")
    if free < 4000:
        raise RuntimeError(f"card too full to start, free_mib={free}")
    log_path = f"/tmp/nautiburn-b6-{name}.log"
    cmd = [
        SERVER,
        "-m",
        MODEL,
        "-ngl",
        "99",
        "-c",
        CTX,
        "-np",
        str(streams),
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--flash-attn",
        "off",
        "-t",
        "8",
        *extra_args,
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPU
    ld = env.get("LD_LIBRARY_PATH", "")
    cuda = "/usr/local/cuda-12.8/lib64"
    env["LD_LIBRARY_PATH"] = cuda + (":" + ld if ld else "")
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    try:
        wait_health(proc)
        # one warmup so the first measured call is not the graph compile
        complete("warmup")
        prompt = "Translate this note into three atomic tasks: check the free P100, keep GEMMs in FP32, and write the stall table."
        if streams == 1:
            rows = [complete(prompt)]
        else:
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futs = [
                    pool.submit(complete, prompt + f" stream {i}")
                    for i in (1, 2)
                ]
                rows = [f.result() for f in futs]
        maps = open(f"/proc/{proc.pid}/maps", encoding="utf-8", errors="replace").read()
        mapped = [
            line.split()[-1]
            for line in maps.splitlines()
            if "libnvrtc" in line or "libcudart" in line or "libcublas.so" in line
        ]
        free_after, used_after, _ = gpu_free_mib()
        return {
            "mode": name,
            "streams": streams,
            "free_mib_before": free,
            "used_mib_before": used,
            "free_mib_during": free_after,
            "used_mib_during": used_after,
            "ctx": int(CTX),
            "n_predict": PREDICT,
            "rows": rows,
            "log": log_path,
            "pid": proc.pid,
            "mapped_libs": mapped,
        }
    finally:
        stop_server(proc)
        log.close()


def main():
    size = os.path.getsize(MODEL)
    if size != EXPECTED_BYTES:
        print(f"size {size} != expected {EXPECTED_BYTES}", file=sys.stderr)
        return 2
    free, used, uuid = gpu_free_mib()
    print(f"gpu={uuid} free_mib={free} used_mib={used} bytes={size}")
    modes = [
        ("resident-batch1", [], 1),
        ("cpu-moe-batch1", ["--cpu-moe"], 1),
        ("resident-two-stream", [], 2),
        ("cpu-moe-two-stream", ["--cpu-moe"], 2),
    ]
    results = []
    for name, extra, streams in modes:
        print(f"start {name}", flush=True)
        results.append(run_mode(name, extra, streams))
        print(f"done {name}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"gpu": GPU, "model": MODEL, "bytes": size, "modes": results}, f, indent=2)
        f.write("\n")
    print(OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
