#!/usr/bin/env python3
"""dispatch_moe_contest_round2.py - Round 2 Contest Dispatch & Metrics Harness.

Dispatches MoE advancement challenges to fleet models:
- KatCoder-35B (10.44.0.12:8481)
- Hermes-3-8B / QwQ-32B via Paddler (127.0.0.1:8080)

Collects their kernel designs, records Buzz Board entries, sends telemetry to Telemetry Hub,
and prepares candidate code for Orchestrator/Polisher synthesis.
"""

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

KATCODER_URL = "http://10.44.0.12:8481/v1/chat/completions"
PADDLER_URL = "http://127.0.0.1:8080/v1/chat/completions"
TELEMETRY_URL = "http://127.0.0.1:8090/api/telemetry/submit"

BUZZ_JSON = Path("/home/cesarops/agentnotes/buzz_board.json")
BUZZ_MD = Path("/home/cesarops/agentnotes/buzz_board.md")
ARTIFACTS_DIR = Path("/home/cesarops/NautiBurn/artifacts")

PROMPT = """You are an expert Rust systems architect and MoE acceleration engineer participating in Round 2 of the NautiBurn MoE Lookahead Contest.
Collaborate with peer MoE agents via the Buzz Board.

We have a working pure-Rust lookahead router predictor (`crates/nautiburn-predict/src/tensor.rs`) with 12/12 passing unit tests.
Now we are solving Round 2 Advancement Opportunities:

=== ADVANCEMENT OBJECTIVES ===
1. TRACK 1: Fused Dual-Hop GEMV (L+1 and L+2 in a Single Pass):
   - Design and implement a fused method:
     `predict_dual_fused(&mut self, current_layer: usize, residual_x: &[f32], ...) -> Result<DualPredictionResult, String>`
   - Instead of two separate memory sweeps, process L+1 and L+2 router matrices in shared cache-blocked iterations.
   - Return predicted experts for both horizon L+1 and L+2.

2. TRACK 2: Calibrated / Adaptive Residual EMA Smoothing:
   - Provide a dynamic calculation for the EMA factor lambda based on sequence position or activation variance (e.g. higher stability in decode, agile in prefill).

3. TRACK 3: Dynamic Bayesian Prior Weighting:
   - Adjust `alpha` based on token entropy so confident router logits aren't washed out by bigram priors.

=== OUTPUT FORMAT ===
Provide your response structured as:
1. [BUZZ_POST]: High-level design summary for the Buzz Board feed.
2. [DESIGN_OVERVIEW]: In-depth explanation of cache locality and math derivations.
3. [RUST_IMPLEMENTATION]: Idiomatic Rust code additions/refinements for `tensor.rs`.
"""

def query_model(url, model_name, prompt, max_tokens=2048):
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
        return {"model": model_name, "error": str(e), "elapsed": time.time() - t0, "status": "error"}

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

def submit_telemetry(metric_name, payload):
    data = {
        "metric": metric_name,
        "timestamp": time.time(),
        "source": "nautiburn-round2-contest",
        "data": payload
    }
    try:
        req = urllib.request.Request(
            TELEMETRY_URL,
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200 or resp.status == 201
    except Exception as e:
        print(f"[Telemetry] Warning: failed to submit metric: {e}")
        return False

def main():
    print("[Orchestrator] Starting Round 2 Contest Dispatch...")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Query KatCoder-35B
    print("[Orchestrator] Querying KatCoder-35B (:8481)...")
    res_kat = query_model(KATCODER_URL, "katcoder-35b-a3b", PROMPT, max_tokens=1800)
    print(f"[Orchestrator] KatCoder returned status: {res_kat['status']} in {res_kat.get('elapsed', 0):.2f}s")
    
    # 2. Query Paddler Model (Hermes-3-8B / Slot: judge or hermes)
    print("[Orchestrator] Querying Hermes-3-8B via Paddler (:8080)...")
    res_paddler = query_model(PADDLER_URL, "Hermes-3-8B", PROMPT, max_tokens=1800)
    print(f"[Orchestrator] Paddler model returned status: {res_paddler['status']} in {res_paddler.get('elapsed', 0):.2f}s")
    
    results = {"katcoder": res_kat, "paddler": res_paddler}
    out_file = ARTIFACTS_DIR / "moe_contest_round2_raw_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Orchestrator] Saved Round 2 raw responses to {out_file}")
    
    # Post to Buzz Board
    update_buzz_board({
        "id": "buzz-20261018-round2",
        "author": "Orchestrator-Round2",
        "topic": "moe-lookahead-round2-advancement",
        "phase": "round2_proposals",
        "summary": "Round 2 MoE Lookahead Contest: Fused Dual-Hop GEMV (L+1/L+2), Adaptive EMA, and Dynamic Prior Weighting.",
        "details": f"Dispatched and collected proposals from KatCoder-35B ({res_kat['status']}, {res_kat.get('elapsed', 0):.1f}s) and Hermes-3-8B ({res_paddler['status']}, {res_paddler.get('elapsed', 0):.1f}s). Ready for Orchestrator & Polisher synthesis into tensor.rs.",
        "open_questions": [
            "Evaluation of cache footprint reduction under fused dual-hop GEMV.",
            "Verification of hit rate parity between sequential and fused L+2 lookaheads."
        ]
    })
    
    # Submit Telemetry
    submit_telemetry("moe_round2_dispatch", {
        "katcoder_status": res_kat["status"],
        "katcoder_latency_s": res_kat.get("elapsed", 0),
        "paddler_status": res_paddler["status"],
        "paddler_latency_s": res_paddler.get("elapsed", 0),
        "contest_round": 2
    })
    print("[Orchestrator] Telemetry and Buzz Board updated successfully.")

if __name__ == "__main__":
    main()
