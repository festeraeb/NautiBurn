#!/usr/bin/env python3
"""dispatch_moe_contest.py - Orchestrator and Judge Dispatch Harness.

Dispatches MoE tasks to fleet models (DeepSeek-v4-Flash, KatCoder-35B),
collects their designs and code, posts to Buzz Board, grades them as Judge,
and outputs the synthesis plan.
"""

import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

DEEPSEEK_URL = "http://127.0.0.1:8102/v1/chat/completions"
KATCODER_URL = "http://10.44.0.12:8481/v1/chat/completions"
BUZZ_JSON = Path("/home/cesarops/agentnotes/buzz_board.json")
BUZZ_MD = Path("/home/cesarops/agentnotes/buzz_board.md")

PROMPT = """You are an expert Rust systems programmer and MoE architecture engineer participating in the NautiBurn MoE Lookahead Contest.
Collaborate with peer MoE agents via the Buzz Board.

You are tasked with two mandatory objectives:

=== TASK A: CONVERT TO WORK AS INTENDED (PURE RUST CONVERSION) ===
Convert the drifted lookahead router predictor and GGUF tensor loading into true, idiomatic, standalone Rust (no C/C++ FFI wrappers, no Python scripts).
Requirements:
1. `RouterLookahead` in Rust:
   - Inputs: Layer L residual hidden state vector x_L (slice of f32, length n_embd), and layer L+1 router weights W_gate (BF16, shape [n_experts, n_embd]).
   - Computes router logits z = W_gate * x_L.
   - Extracts top-k expert IDs (default k=8).
2. Pure Rust GGUF loader:
   - Parse GGUF header and locate `blk.{L}.ffn_gate_inp.weight` in pure Rust without external C libraries.

=== TASK B: SPEED & ACCURACY IMPROVEMENTS ===
Design and implement performance and accuracy enhancements.
Listed enhancement areas to incorporate:
1. RMSNorm Pre-conditioning: Normalize x_L before router GEMV using layer L+1's RMSNorm weight to prevent activation scale drift.
2. SIMD & Cache-Blocked BF16 Dot Product: Optimize GEMV to execute in < 50 microseconds.
3. Bayesian Hybrid Prior Blending: Blend router logits with cross-layer bigram transition probabilities:
   score(e) = z(e) + alpha * ln(P_bigram(e | active_L) + eps).
4. Dynamic Depth & Entropy-Aware Preload: Expand k dynamically or predict L+2 when router confidence is high.
RESEARCH & SUGGEST:
Research and propose at least TWO additional novel enhancements not listed above to further push prediction accuracy and throughput.

=== OUTPUT FORMAT ===
Provide your response structured as:
1. [BUZZ_POST]: Summary for the Buzz Board collaboration feed.
2. [DESIGN_OVERVIEW]: Architecture and your 2+ novel suggested enhancements.
3. [RUST_IMPLEMENTATION]: Complete, self-contained Rust code for `tensor.rs`.
"""

def query_model(url, model_name, prompt, max_tokens=2500):
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a senior Rust systems architect and MoE AI engineer."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            elapsed = time.time() - t0
            msg = data["choices"][0]["message"]
            content = msg.get("content") or msg.get("reasoning_content") or ""
            return {"model": model_name, "content": content, "elapsed": elapsed, "status": "ok"}
    except Exception as e:
        return {"model": model_name, "error": str(e), "status": "error"}

def update_buzz_board(entry):
    if BUZZ_JSON.exists():
        with open(BUZZ_JSON, "r") as f:
            try:
                data = json.load(f)
            except Exception:
                data = {"board_version": "1.0", "entries": []}
    else:
        data = {"board_version": "1.0", "entries": []}
    
    data["entries"].append(entry)
    data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(BUZZ_JSON, "w") as f:
        json.dump(data, f, indent=2)
        
    with open(BUZZ_MD, "a") as f:
        f.write(f"\n---\n\n### [{entry['id']}] {entry['topic']}\n")
        f.write(f"* **Author:** {entry['author']}\n")
        f.write(f"* **Phase:** `{entry['phase']}`\n")
        f.write(f"* **Summary:** {entry['summary']}\n")
        f.write(f"* **Details:**\n{entry['details']}\n")
        if entry.get("open_questions"):
            f.write("* **Open Questions / Suggested Enhancements:**\n")
            for q in entry["open_questions"]:
                f.write(f"  - {q}\n")

def main():
    print("[Orchestrator] Starting Contest Dispatch to MoE fleet...")
    
    # 1. Query KatCoder
    print("[Orchestrator] Querying KatCoder-35B (:8481)...")
    res_kat = query_model(KATCODER_URL, "katcoder-35b-a3b", PROMPT, max_tokens=2048)
    print(f"[Orchestrator] KatCoder returned status: {res_kat['status']}")
    
    # 2. Query DeepSeek
    print("[Orchestrator] Querying DeepSeek-v4-Flash (:8102)...")
    res_deepseek = query_model(DEEPSEEK_URL, "deepseek-v4-flash-9b", PROMPT, max_tokens=2048)
    print(f"[Orchestrator] DeepSeek returned status: {res_deepseek['status']}")
    
    results = {"katcoder": res_kat, "deepseek": res_deepseek}
    out_file = Path("/home/cesarops/NautiBurn/artifacts/moe_contest_raw_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Orchestrator] Saved raw responses to {out_file}")

if __name__ == "__main__":
    main()
