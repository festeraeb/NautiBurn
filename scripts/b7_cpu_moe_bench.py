#!/usr/bin/env python3
"""B7: Mixtral --cpu-moe single-card P100 timings for hdrhistogram."""
import json, subprocess, time, sys, statistics

HOST = "cesarops4om"
PORT = 5017
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"

PROMPTS = [
    "Explain what a Mixture of Experts model is in 3 sentences.",
    "Write a Rust function to check if a number is prime.",
    "What are the differences between FP16 and FP32 precision?",
    "Describe the NVIDIA P100 GPU architecture in detail.",
    "Write a Python function that merges two sorted lists.",
    "Explain how CUDA shared memory works.",
    "What is quantization in LLM inference context?",
    "Write a shell script to check disk usage and alert if over 80%.",
    "Explain the difference between latency and throughput.",
    "Write a Rust struct for a thread-safe counter.",
    "What is GGUF format and why is it used for LLM weights?",
    "Explain CUDA stream parallelism in detail.",
    "Write a function to compute Fibonacci iteratively.",
    "What is model parallelism vs data parallelism?",
    "Explain KV cache in transformer inference.",
    "Write a Rust enum for HTTP status codes with Display.",
]

def run_prompt(prompt, max_tokens=256):
    body = json.dumps({
        "model": "Mixtral-8x7B-cpu-moe",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.1,
    })
    cmd = ["sshpass", "-p", "cesarops", "ssh", "-o", "BatchMode=yes", HOST,
           f"curl -s --max-time 180 {URL} -H 'Content-Type: application/json' -d '{body}'"]
    t0 = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=190)
    wall = time.monotonic() - t0
    if result.returncode != 0:
        return {"error": result.stderr[:200], "wall_s": wall}
    try:
        data = json.loads(result.stdout)
    except Exception:
        return {"error": result.stdout[:200], "wall_s": wall}
    usage = data.get("usage", {})
    return {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "wall_s": round(wall, 3),
        "finish_reason": data.get("choices", [{}])[0].get("finish_reason", "?"),
    }

def get_vram():
    cmd = ["sshpass", "-p", "cesarops", "ssh", "-o", "BatchMode=yes", HOST,
           "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    return [l.strip() for l in r.stdout.strip().split("\n")]

def main():
    timings = []
    print(f"B7 cpu-moe bench on {HOST}:{PORT}, {len(PROMPTS)} prompts")
    print("  warmup...", end="", flush=True)
    w = run_prompt("Hello warmup.", max_tokens=8)
    print(f" {w.get('wall_s',0):.1f}s")
    vram_before = get_vram()
    for i, p in enumerate(PROMPTS):
        print(f"  [{i+1}/{len(PROMPTS)}] ", end="", flush=True)
        r = run_prompt(p, max_tokens=256)
        timings.append({"index": i, "prompt": p[:80], **r})
        ct = r.get("completion_tokens", 0)
        wall = r.get("wall_s", 0)
        tps = ct / wall if wall > 0 and ct > 0 else 0
        print(f"pt={r.get('prompt_tokens',0)} ct={ct} wall={wall:.1f}s {tps:.1f}tok/s")
    vram_after = get_vram()
    tps_list = [t["completion_tokens"]/t["wall_s"] for t in timings
                if "error" not in t and t.get("wall_s",0) > 0 and t.get("completion_tokens",0) > 0]
    s = sorted(tps_list)
    n = len(s)
    stats = {
        "n": n,
        "mean_tok_s": round(statistics.mean(s), 1) if s else 0,
        "median_tok_s": round(statistics.median(s), 1) if s else 0,
        "p5_tok_s": round(s[int(n*0.05)], 1) if n > 1 else 0,
        "p95_tok_s": round(s[int(n*0.95)], 1) if n > 1 else 0,
        "min_tok_s": round(min(s), 1) if s else 0,
        "max_tok_s": round(max(s), 1) if s else 0,
        "vram_before": vram_before, "vram_after": vram_after,
    }
    result = {"brief": "B7", "test": "cpu-moe single-card Mixtral",
              "host": HOST, "gpu": "Tesla P100-PCIE-16GB (#2)",
              "flags": "-cmoe -ngl 99 -c 2048 -np 1 -t 8 -fa off",
              "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "stats": stats, "timings": timings}
    out = "/tmp/b7_cpu_moe_timings.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n=== SUMMARY ===")
    print(f"N={stats['n']} mean={stats['mean_tok_s']} median={stats['median_tok_s']}")
    print(f"range=[{stats['min_tok_s']}, {stats['max_tok_s']}] p5={stats['p5_tok_s']} p95={stats['p95_tok_s']}")
    print(f"VRAM: {vram_after}")

if __name__ == "__main__":
    main()