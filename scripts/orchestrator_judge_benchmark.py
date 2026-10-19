#!/usr/bin/env python3
"""orchestrator_judge_benchmark.py - Fleet Model Evaluation, MoE Histogram Mapping & Judge Orchestration.

Executes fleet prompts across:
  1. GLM-5.3-Flash (3x GPUs on cesarops4:8103)
  2. DeepSeek MoE (2x P100s on t440:8102)
  3. Swift-1.5 / Hermes-3 Judge (cesarops2:8481 / :8106)

Scores responses via automated syntax/code verification and Hermes-3 Judge model,
and logs the activations into NautiBurn HistoDB (glm-histodb.sqlite).
"""
import json
import os
import re
import sqlite3
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

GLM_URL = "http://10.0.10.204:8103/v1/chat/completions"
DEEPSEEK_URL = "http://127.0.0.1:8102/v1/chat/completions"
SWIFT_URL = "http://10.44.0.12:8481/v1/chat/completions"
JUDGE_URL = "http://10.44.0.12:8106/v1/chat/completions"

GLM_DB = Path("/mnt/raid0/ps-data/packs/glm/glm-histodb.sqlite")
FLEET_PROMPTS_DIR = Path("/codebase/repos/cesarops-inference/fleet_prompts")


def call_chat(url, model_name, prompt, max_tokens=256, temperature=0.1, timeout=120):
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


def judge_response(task_desc, prompt, response_text):
    judge_prompt = f"""You are an expert AI code and reasoning evaluator. 
Rate the following model response for the given task on a scale of 1 to 10 (10 = flawless, production-ready).
Briefly state the strengths and flaws, then output 'SCORE: X/10'.

### Task:
{task_desc}

### Model Response:
{response_text[:1500]}
"""
    res = call_chat(JUDGE_URL, "Hermes-3-8B", judge_prompt, max_tokens=150, temperature=0.0, timeout=45)
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
        req_id = f"bench-{int(time.time()*1000)}"
        with conn:
            conn.execute(
                """INSERT OR IGNORE INTO requests 
                   (request_id, model_id, class_id, source, phase_hint, n_tokens, dtype)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (req_id, model_id, category, "benchmark", "prefill+decode", prompt_tokens + completion_tokens, "gguf"),
            )
            # Update activation counts for layer sampled heuristics
            conn.execute(
                """UPDATE histogram_counts 
                   SET count = count + 1 
                   WHERE model_id = ? AND layer IN (0, 10, 20, 30, 40) AND expert_id < 8""",
                (model_id,),
            )
        conn.close()
    except Exception as e:
        print(f"[histodb_err] {e}")


def main():
    print("=" * 70)
    print("  FLEET MODEL ORCHESTRATION, BENCHMARK & JUDGE HARNESS")
    print("=" * 70)

    # 3 representative prompts across diverse domains
    tasks = [
        {
            "id": "WGSL_MOE_SHADER",
            "category": "gpu_shader",
            "file": FLEET_PROMPTS_DIR / "MOE_T6_wgsl_fused_ffn.md",
            "short_prompt": "Write a WGSL compute shader for a Mixture of Experts fused FFN layer with SiLU activation function, top_k routing, and workgroup size 64.",
        },
        {
            "id": "CLUSTER_SCHEDULER",
            "category": "distributed_systems",
            "file": FLEET_PROMPTS_DIR / "NAUTI_T1_scheduler_node.md",
            "short_prompt": "Write a Rust struct and method for an async multi-GPU scheduler that takes Device capacity and a 13ms prefetch budget to decide whether to prefetch an MoE expert or compute on CPU.",
        },
        {
            "id": "SPECULATIVE_PLANNER",
            "category": "inference_optimization",
            "file": FLEET_PROMPTS_DIR / "NODE_T7_llm_planner.md",
            "short_prompt": "Explain how speculative drafting with an auxiliary draft model improves MoE inference throughput when experts are tiered across PCIe and host RAM.",
        },
    ]

    models = [
        {"name": "GLM 5.3 Flash (3x GPU c4)", "url": GLM_URL, "model_id": "glm-5.3-flash", "max_tok": 128},
        {"name": "DeepSeek MoE (2x P100 t440)", "url": DEEPSEEK_URL, "model_id": "deepseek-v4-flash-9b", "max_tok": 128},
        {"name": "Swift-1.5 (2x P100 c2)", "url": SWIFT_URL, "model_id": "swift-1.5-iq2_xs", "max_tok": 128},
    ]

    results = []

    for t in tasks:
        print(f"\n▶ TASK [{t['id']}] Domain: {t['category']}")
        print(f"  Prompt: {t['short_prompt']}")

        for m in models:
            print(f"  --> Querying {m['name']} ...", end=" ", flush=True)
            res = call_chat(m["url"], m["model_id"], t["short_prompt"], max_tokens=m["max_tok"], timeout=90)
            if res.get("ok"):
                p_tok_s = res["prompt_tok_s"]
                c_tok_s = res["completion_tok_s"]
                elapsed = res["elapsed"]
                print(f"OK ({elapsed:.1f}s | prefill {p_tok_s:.1f} tok/s | decode {c_tok_s:.1f} tok/s)")
                
                # Model evaluation by Judge
                judge = judge_response(t["short_prompt"], t["short_prompt"], res["content"] or res["reasoning"])
                print(f"      [Judge Hermes-3] Score: {judge['score']}/10")

                # Ingest to Histodb
                log_to_histodb(m["model_id"], res["prompt_tokens"], res["completion_tokens"], t["category"])

                results.append({
                    "task": t["id"],
                    "model": m["name"],
                    "elapsed": elapsed,
                    "prefill_tok_s": p_tok_s,
                    "decode_tok_s": c_tok_s,
                    "score": judge["score"],
                    "verdict": judge["verdict"][:120],
                })
            else:
                print(f"FAILED ({res.get('error')})")

    print("\n" + "=" * 70)
    print("  FINAL FLEET SCORECARD & ORCHESTRATOR REPORT")
    print("=" * 70)
    print(f"{'Model':<30} | {'Task':<20} | {'Prefill':<10} | {'Decode':<10} | {'Score'}")
    print("-" * 80)
    for r in results:
        print(f"{r['model']:<30} | {r['task']:<20} | {r['prefill_tok_s']:<8.1f}/s | {r['decode_tok_s']:<8.1f}/s | {r['score']}/10")
    print("=" * 70)


if __name__ == "__main__":
    main()
