#!/usr/bin/env python3
import json
import os
import re
import subprocess
import tempfile
import time
import urllib.request

# Endpoints
URL_QWQ = "http://127.0.0.1:5240/v1/chat/completions"
URL_TIEL = "http://127.0.0.1:5230/v1/chat/completions"
URL_SWIFT = "http://10.44.0.12:8481/v1/chat/completions"
URL_STRATA = "http://127.0.0.1:8480/v1/chat/completions"
URL_JUDGE = "http://127.0.0.1:5250/v1/chat/completions"

def call_model(url, model, messages, max_tokens=1024, temperature=0.2, timeout=120):
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        elapsed = time.time() - t0
        msg = data["choices"][0]["message"]
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        usage = data.get("usage", {})
        completion_tokens = usage.get("completion_tokens", len((content + " " + reasoning).split()))
        tok_s = completion_tokens / max(elapsed, 0.001)
        return {
            "content": content,
            "reasoning": reasoning,
            "tokens": completion_tokens,
            "elapsed": elapsed,
            "tok_s": tok_s
        }

def extract_code(text):
    matches = re.findall(r"```(?:rust)?\s*(.*?)\s*```", text, re.DOTALL)
    if matches:
        return matches[-1]
    return text

def test_rust_compile(code):
    with tempfile.NamedTemporaryFile("w", suffix=".rs", delete=False) as f:
        wrapped = f"#![allow(dead_code, unused_variables, unused_imports)]\n{code}\nfn main() {{}}\n"
        f.write(wrapped)
        fname = f.name
    try:
        res = subprocess.run(["rustc", "--crate-type", "bin", fname, "-o", f"{fname}.bin"],
                             capture_output=True, text=True, timeout=15)
        passed = (res.returncode == 0)
        return passed, res.stderr
    except Exception as e:
        return False, str(e)
    finally:
        if os.path.exists(fname):
            os.remove(fname)
        if os.path.exists(f"{fname}.bin"):
            os.remove(f"{fname}.bin")

def judge_evaluation(code, task_description):
    prompt = f"""You are the senior code auditor. Review this Rust implementation for:
Task: {task_description}

Code:
```rust
{code}
```

Evaluate correctness, algorithmic efficiency, and memory safety. Provide:
1. Verdict: ACCEPT or REJECT
2. Score: 1 to 10
3. Key reason (1-2 sentences)"""
    
    try:
        res = call_model(URL_JUDGE, "BugTraceAI-Apex-G4-26B-APEX-Mini",
                         [{"role": "user", "content": prompt}],
                         max_tokens=150, temperature=0.1, timeout=180)
        verdict_text = res["content"] if res["content"].strip() else res["reasoning"]
        return verdict_text, res["elapsed"]
    except Exception as e:
        return f"Judge evaluation error: {e}", 0.0

def run_forge_v3_pipeline(round_name, task_prompt):
    print(f"\n==================================================")
    print(f"[{round_name}] STARTING CONTENDER 1: FORGE v3 PIPELINE")
    print(f"==================================================")
    
    t_start = time.time()
    
    # 1. Thinker: QwQ-32B
    print("  -> Step 1: Thinker (QwQ-32B on :5240)...")
    thinker_prompt = f"Analyze the following Rust coding task. Outline key edge cases, constraints, and algorithmic requirements in 2-3 concise bullet points:\n{task_prompt}"
    thinker_res = call_model(URL_QWQ, "QwQ-32B-IQ2_XXS", [{"role": "user", "content": thinker_prompt}], max_tokens=300)
    thinker_notes = thinker_res["content"] or thinker_res["reasoning"]
    print(f"     Thinker finished in {thinker_res['elapsed']:.2f}s ({thinker_res['tok_s']:.1f} tok/s)")
    
    # 2. Context Injector: Tiel
    print("  -> Step 2: Context Injector (Tiel Qwen3.6-35B-A3B on :5230)...")
    injector_prompt = f"Task: {task_prompt}\nThinker Plan:\n{thinker_notes}\nInject the Rust function signatures, types, and constraints for the code specialist:"
    injector_res = call_model(URL_TIEL, "Qwen3.6-35B-A3B-Uncensored", [{"role": "user", "content": injector_prompt}], max_tokens=250)
    injected_context = injector_res["content"] or injector_res["reasoning"]
    print(f"     Injector finished in {injector_res['elapsed']:.2f}s ({injector_res['tok_s']:.1f} tok/s)")
    
    # 3. Code Specialist: Swift 1.5
    print("  -> Step 3: Code Specialist (Swift 1.5 on cesarops2 :8481)...")
    swift_prompt = f"You are the senior Rust specialist. Implement the required Rust code cleanly and efficiently.\nTask: {task_prompt}\nContext:\n{injected_context}\nReturn ONLY the valid Rust code implementation enclosed in ```rust ... ```."
    swift_res = call_model(URL_SWIFT, "swift-1.5-iq2_xs", [{"role": "user", "content": swift_prompt}], max_tokens=512)
    swift_code_raw = swift_res["content"] or swift_res["reasoning"]
    swift_code = extract_code(swift_code_raw)
    print(f"     Code Specialist finished in {swift_res['elapsed']:.2f}s ({swift_res['tok_s']:.1f} tok/s)")
    
    # 4. Reviewer: Tiel
    print("  -> Step 4: Reviewer / Cargo Check (Tiel on :5230)...")
    reviewer_prompt = f"Review this Rust implementation for correctness and syntax:\n```rust\n{swift_code}\n```\nFix any obvious syntax bugs or return unchanged if valid. Return ONLY Rust code:"
    reviewer_res = call_model(URL_TIEL, "Qwen3.6-35B-A3B-Uncensored", [{"role": "user", "content": reviewer_prompt}], max_tokens=400)
    final_forge_code = extract_code(reviewer_res["content"] or reviewer_res["reasoning"]) or swift_code
    print(f"     Reviewer finished in {reviewer_res['elapsed']:.2f}s ({reviewer_res['tok_s']:.1f} tok/s)")
    
    # 5. Compiler Check
    compiled, compiler_output = test_rust_compile(final_forge_code)
    print(f"     rustc compile status: {'PASS' if compiled else 'FAIL'}")
    if not compiled:
        print(f"     compiler error: {compiler_output.strip()[:200]}")
        
    # 6. Judge
    print("  -> Step 5: Judge (BugTrace AI Apex Mini on :5250)...")
    judge_verdict, judge_elapsed = judge_evaluation(final_forge_code, task_prompt)
    print(f"     Judge finished in {judge_elapsed:.2f}s")
    
    total_elapsed = time.time() - t_start
    total_tokens = thinker_res["tokens"] + injector_res["tokens"] + swift_res["tokens"] + reviewer_res["tokens"]
    
    return {
        "name": "Forge v3 Pipeline",
        "total_elapsed": total_elapsed,
        "total_tokens": total_tokens,
        "code": final_forge_code,
        "compiled": compiled,
        "compiler_output": compiler_output,
        "judge_verdict": judge_verdict,
        "steps": {
            "thinker": thinker_res,
            "injector": injector_res,
            "swift": swift_res,
            "reviewer": reviewer_res
        }
    }

def run_strata_qwen_standalone(round_name, task_prompt):
    print(f"\n==================================================")
    print(f"[{round_name}] STARTING CONTENDER 2: STRATA QWEN 3.8 STANDALONE")
    print(f"==================================================")
    
    t_start = time.time()
    
    print("  -> Standalone Coder: Qwen 3.8 Next IQ3 (3 P100s on cesarops4 :8480)...")
    prompt = f"You are an expert Rust systems programmer. Implement the following task with maximum efficiency, zero superfluous allocations, and idiomatic Rust.\n\nTask: {task_prompt}\n\nReturn your response with your reasoning, followed by the complete Rust code implementation enclosed in ```rust ... ```."
    
    strata_res = call_model(URL_STRATA, "qwen3.8-flash-next-iq3_s", [{"role": "user", "content": prompt}], max_tokens=768, timeout=180)
    raw_output = (strata_res["content"] or "") + "\n" + (strata_res["reasoning"] or "")
    code = extract_code(raw_output)
    print(f"     Strata Coder finished in {strata_res['elapsed']:.2f}s ({strata_res['tok_s']:.1f} tok/s)")
    
    # Compiler check
    compiled, compiler_output = test_rust_compile(code)
    print(f"     rustc compile status: {'PASS' if compiled else 'FAIL'}")
    if not compiled:
        print(f"     compiler error: {compiler_output.strip()[:200]}")
        
    # Judge
    print("  -> Step 2: Judge (BugTrace AI Apex Mini on :5250)...")
    judge_verdict, judge_elapsed = judge_evaluation(code, task_prompt)
    print(f"     Judge finished in {judge_elapsed:.2f}s")
    
    total_elapsed = time.time() - t_start
    return {
        "name": "Strata Qwen 3.8 Standalone",
        "total_elapsed": total_elapsed,
        "total_tokens": strata_res["tokens"],
        "code": code,
        "compiled": compiled,
        "compiler_output": compiler_output,
        "judge_verdict": judge_verdict,
        "strata_stats": strata_res
    }

if __name__ == "__main__":
    tasks = [
        {
            "round": "Round 1: Brief 4 Cache Eviction & Replacement",
            "prompt": "Write a Rust function `evaluate_cache_eviction(resident_experts: &[u16], predicted_experts: &[u16], cache_capacity: usize) -> Vec<u16>` that returns the list of expert IDs to evict from `resident_experts` to make room for `predicted_experts`, ensuring that any expert in `resident_experts` that is also in `predicted_experts` is preserved (cache hit) and not evicted. The output list must contain exactly `max(0, resident_experts.len() + new_needed - cache_capacity)` evictions from the resident experts that are NOT predicted. Include full type signatures and zero-allocation logic where possible."
        },
        {
            "round": "Round 2: Brief 5 MoE Activation Filter & Sidecar Dispatch",
            "prompt": "Write a Rust function `filter_active_experts(activations: &[(u8, u16, f32)], threshold: f32, top_k: usize) -> Vec<(u8, Vec<u16>)>` that groups MoE layer activation tuples `(layer_idx, expert_id, score)` by `layer_idx`, retains only experts with `score >= threshold`, sorts each layer descending by score, truncates each layer to at most `top_k` experts, and returns the grouped layers sorted by `layer_idx`. Ensure robust edge-case handling for empty inputs."
        }
    ]
    
    results = []
    for t in tasks:
        r_name = t["round"]
        r_prompt = t["prompt"]
        forge_res = run_forge_v3_pipeline(r_name, r_prompt)
        strata_res = run_strata_qwen_standalone(r_name, r_prompt)
        results.append({
            "round": r_name,
            "prompt": r_prompt,
            "forge": forge_res,
            "strata": strata_res
        })
        
    out_path = "/home/cesarops/NautiBurn/scripts/contest_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nContest complete! Results written to {out_path}")
