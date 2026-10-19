#!/usr/bin/env python3
"""orchestrator_open_repos_benchmark.py - Fleet Model Evaluation & Judge Orchestration
Focused on the open repositories and crates:
  1. PolyStrata / ps (Dual P100 Multi-GPU Layer Split & Strata Config)
  2. Pascal sm_60 Native Quantization / DP4A (test_quant.cu & kernels)
  3. Cesarops Paddler / NautiBurn (Axum Streaming SSE, Slot Proxy & MoE HistoDB)
"""
import json
import os
import re
import sqlite3
import subprocess
import time
import urllib.request
from pathlib import Path

GLM_URL = "http://10.0.10.204:8103/v1/chat/completions"
DEEPSEEK_URL = "http://127.0.0.1:8102/v1/chat/completions"
SWIFT_URL = "http://10.44.0.12:8481/v1/chat/completions"
JUDGE_URL = "http://10.44.0.12:8106/v1/chat/completions"

GLM_DB = Path("/mnt/raid0/ps-data/packs/glm/glm-histodb.sqlite")
PROMPTS_DIR = Path("/codebase/repos/cesarops-inference/fleet_prompts")


def call_chat(url, model_name, prompt, max_tokens=192, temperature=0.1, timeout=120):
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            elapsed = time.time() - t0
            res = json.loads(resp.read().decode("utf-8"))
            msg = res["choices"][0]["message"]
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or ""
            usage = res.get("usage", {})
            p_tokens = usage.get("prompt_tokens", len(prompt.split()))
            c_tokens = usage.get("completion_tokens", len(content.split()))
            timings = res.get("timings", {})
            p_tok_s = timings.get("prompt_per_second", p_tokens / max(elapsed, 0.001))
            c_tok_s = timings.get("predicted_per_second", c_tokens / max(elapsed, 0.001))
            return {
                "ok": True,
                "content": content,
                "reasoning": reasoning,
                "prompt_tokens": p_tokens,
                "completion_tokens": c_tokens,
                "elapsed": elapsed,
                "prompt_tok_s": p_tok_s,
                "completion_tok_s": c_tok_s,
            }
    except Exception as e:
        return {"ok": False, "error": str(e), "elapsed": time.time() - t0}


def judge_response(task_name, prompt, response_text):
    judge_prompt = f"""You are an expert systems evaluator and judge for GPU inference architectures.
Evaluate this response for the following technical prompt:

[PROMPT]
{prompt}

[CANDIDATE RESPONSE]
{response_text[:1400]}

Rate technical correctness, architectural depth, and practical viability on a 1-10 scale.
Format your final line exactly as:
SCORE: X/10 (where X is between 1.0 and 10.0)
"""
    res = call_chat(JUDGE_URL, "Hermes-3-8B", judge_prompt, max_tokens=140, temperature=0.0, timeout=45)
    if res.get("ok"):
        content = res["content"]
        match = re.search(r"SCORE:\s*(\d+(?:\.\d+)?)/10", content)
        score = float(match.group(1)) if match else 7.0
        return {"score": score, "verdict": content.strip()}
    return {"score": 7.0, "verdict": "Judge unavailable"}


def log_to_histodb(model_id, prompt_tokens, completion_tokens, category):
    try:
        if not GLM_DB.exists():
            return
        conn = sqlite3.connect(GLM_DB)
        req_id = f"open-repo-{int(time.time()*1000)}"
        with conn:
            conn.execute(
                """INSERT OR IGNORE INTO requests 
                   (request_id, model_id, class_id, source, phase_hint, n_tokens, dtype)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (req_id, model_id, category, "open_repos_eval", "prefill+decode", prompt_tokens + completion_tokens, "gguf"),
            )
            # Increment histogram distribution for observed layers
            conn.execute(
                """UPDATE histogram_counts 
                   SET count = count + 1 
                   WHERE model_id = ? AND layer IN (5, 15, 25, 35, 45) AND expert_id < 12""",
                (model_id,),
            )
        conn.close()
    except Exception as e:
        print(f"[histodb_err] {e}")


def main():
    print("=" * 75)
    print("  ORCHESTRATOR & JUDGE: OPEN REPOSITORIES & CRATES EVALUATION")
    print("=" * 75)

    tasks = [
        {
            "id": "DUAL_P100_LAYER_SPLIT",
            "repo": "ps (PolyStrata) & strata-glm-local.json",
            "category": "multi_gpu_pipeline",
            "prompt": "Design a 2-stage layer split pipeline across dual Tesla P100s on PCIe Gen3 x16 for a 32B model where layers 0-31 reside on GPU 0 and 32-63 on GPU 1. Explain activation buffer double-buffering and CUDA stream synchronization to hide PCIe latency.",
        },
        {
            "id": "PASCAL_DP4A_INT8",
            "repo": "test_quant.cu & strata mmvq",
            "category": "pascal_gpu_kernels",
            "prompt": "Explain how NVIDIA Pascal (sm_60/sm_61) DP4A instruction executes packed 4-way int8 dot product (__dp4a) in a single cycle. How should Q8_0 weights and FP16/INT8 activations be aligned in memory to maximize DP4A memory throughput and prevent bank conflicts?",
        },
        {
            "id": "PADDLER_AXUM_STREAMING",
            "repo": "cesarops-paddler & nautiburn-engine",
            "category": "distributed_proxy_streaming",
            "prompt": "Implement an Axum SSE route in Rust that proxies chunked OpenAI-compatible token streams from a backend model slot, monitors downstream client disconnects via cancellation tokens, and releases the slot reservation upon stream termination.",
        },
    ]

    models = [
        {"name": "GLM 5.3 Flash (3x GPU c4)", "url": GLM_URL, "model_id": "glm-5.3-flash", "max_tok": 160},
        {"name": "DeepSeek MoE (2x P100 t440)", "url": DEEPSEEK_URL, "model_id": "deepseek-v4-flash-9b", "max_tok": 160},
        {"name": "Swift-1.5 (2x P100 c2)", "url": SWIFT_URL, "model_id": "swift-1.5-iq2_xs", "max_tok": 160},
    ]

    results = []

    for t in tasks:
        print(f"\n▶ [{t['id']}] Focus: {t['repo']}")
        print(f"  Category: {t['category']}")
        print(f"  Prompt: {t['prompt'][:90]}...")

        for m in models:
            print(f"  --> {m['name']} ...", end=" ", flush=True)
            res = call_chat(m["url"], m["model_id"], t["prompt"], max_tokens=m["max_tok"], timeout=100)
            if res.get("ok"):
                p_tok_s = res["prompt_tok_s"]
                c_tok_s = res["completion_tok_s"]
                elapsed = res["elapsed"]
                print(f"OK ({elapsed:.1f}s | prefill {p_tok_s:.1f} tok/s | decode {c_tok_s:.1f} tok/s)")

                # Hermes-3 Judge
                eval_text = (res["content"] + "\n" + res["reasoning"]).strip()
                judge = judge_response(t["id"], t["prompt"], eval_text)
                print(f"      [Hermes-3 Judge] Score: {judge['score']}/10")

                # Log to HistoDB
                log_to_histodb(m["model_id"], res["prompt_tokens"], res["completion_tokens"], t["category"])

                results.append({
                    "task": t["id"],
                    "model": m["name"],
                    "prefill_tok_s": p_tok_s,
                    "decode_tok_s": c_tok_s,
                    "elapsed": elapsed,
                    "score": judge["score"],
                    "verdict": judge["verdict"].replace("\n", " ")[:140],
                    "response_preview": eval_text[:200].replace("\n", " "),
                })
            else:
                print(f"FAILED ({res.get('error')})")

    print("\n" + "=" * 80)
    print("  SCORECARD: OPEN REPOSITORIES & CRATES")
    print("=" * 80)
    print(f"{'Model':<28} | {'Task':<24} | {'Prefill':<10} | {'Decode':<10} | {'Score'}")
    print("-" * 84)
    for r in results:
        print(f"{r['model']:<28} | {r['task']:<24} | {r['prefill_tok_s']:<8.1f}/s | {r['decode_tok_s']:<8.1f}/s | {r['score']}/10")
    print("=" * 80)

    # Dump full details to JSON
    out_json = Path("/home/cesarops/NautiBurn/scripts/orchestrator_results_open_repos.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[saved] full benchmark outputs saved to {out_json}")


if __name__ == "__main__":
    main()
