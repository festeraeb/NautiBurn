#!/usr/bin/env python3
"""orchestrate_five_pillar_contest.py - Orchestrator, Judge & Peer-Reviewer Harness
for the 5-Pillar Agentic Ecosystem (Forge, Arch, Warp, Buzz, Web/Gemini).

Workflow:
1. Dispatch the 5-Pillar architecture prompt to GLM-5.3-Flash, DeepSeek MoE, and Swift-1.5.
2. Grade candidate submissions via Hermes-3 Judge (:8106) across the 5 pillars.
3. Identify the "most accurate" model and the "next most accurate" model.
4. Have the most accurate model write the unified implementation:
   - Buzz operational board & collaboration hub (API + UI wiring)
   - Web / Gemini operator bridge tool
   - Arch node OS optimizer agent integration
5. Have the next most accurate model review & audit the implementation.
6. Execute the OS optimizer across the fleet nodes and post progress to the collaboration board.
"""
import json
import os
import re
import subprocess
import time
import urllib.request
from pathlib import Path

GLM_URL = "http://10.0.10.204:8103/v1/chat/completions"
DEEPSEEK_URL = "http://127.0.0.1:8102/v1/chat/completions"
SWIFT_URL = "http://10.44.0.12:8481/v1/chat/completions"
JUDGE_URL = "http://10.44.0.12:8106/v1/chat/completions"

OUT_DIR = Path("/home/cesarops/NautiBurn/artifacts/five_pillar_contest")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPT_CORE = """You are the Lead Systems Architect and Orchestrator for the CESARops fleet.
Design and implement the missing components of the 5-pillar agentic collaboration ecosystem:

The 5 Pillars:
1. Forge: Strong planning/coding brain (task decomposition, patches, synthesis, escalation).
2. Arch: Machine action layer (local capabilities, systemd/services, repos, shell, node health).
3. Warp: Communication and relay spine (node-to-node coordination, task handoffs, routing).
4. Buzz: Human/agent operational board (dispatch board, task inbox, help desk, live chat feed).
5. Web access / Gemini: Internet-aware operator bridge (research docs, API/Gemini access, browser tools).

Current Codebase Reality:
- Gateway: /codebase/repos/cesaropsserver/scripts/warp_model_gateway.py (:8126)
- MCP Tools: /var/www/cesarops/services/mcp-server/server.js (:7800)
- Frontend: /var/www/cesarops/ (warp.html, warp-board.html, warp-jobs.html)
- Auth Sidecar: :8097 (/api/auth/check)
- Nodes: t440 (127.0.0.1), cesarops2 (10.44.0.12), cesarops4om (192.168.4.39), cesarops3 (100.76.255.40).

Your Task:
1. Analyze how Forge, Arch, Warp, Buzz, and Web/Gemini connect together.
2. Detail the exact design for Buzz (the human/agent collaboration board API & frontend feed) and the Web/Gemini research bridge.
3. Provide concrete Python/JavaScript code that implements the Buzz collaboration hub and the Node OS Optimizer task.
4. Explain how agents will collaborate autonomously (agent-to-agent and human-to-agent) to inspect and optimize each node's operating system.
"""


def call_chat(url, model_name, prompt, max_tokens=700, temperature=0.1, timeout=160):
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
            timings = res.get("timings", {})
            p_tok = usage.get("prompt_tokens", len(prompt.split()))
            c_tok = usage.get("completion_tokens", len(content.split()))
            p_tok_s = timings.get("prompt_per_second", p_tok / max(elapsed, 0.001))
            c_tok_s = timings.get("predicted_per_second", c_tok / max(elapsed, 0.001))
            full_text = (content + "\n" + reasoning).strip()
            return {
                "ok": True,
                "content": content,
                "reasoning": reasoning,
                "full_text": full_text,
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
                "elapsed": elapsed,
                "p_tok_s": p_tok_s,
                "c_tok_s": c_tok_s,
            }
    except Exception as e:
        return {"ok": False, "error": str(e), "elapsed": time.time() - t0}


def judge_submission(model_name, candidate_text):
    judge_prompt = f"""You are the Lead Judge evaluating an agent's architecture design for the 5-Pillar CESARops ecosystem (Forge, Arch, Warp, Buzz, Web/Gemini).

Candidate Model: {model_name}

Evaluation Rubric (Score each 1 to 10):
1. Forge Planning & Brain Integration (Decomposition, synthesis, human escalation)
2. Arch Local Host Orchestration (Node actions, systemd, local installs, blocked states)
3. Warp Spine & Inter-Node Relay (Relay protocol, auth, routing, artifact references)
4. Buzz Collaborative Operational Board (Dispatch inbox, help desk, live chat feed, agent-to-agent collaboration)
5. Web/Gemini Integration (External research bridge, browser/API tools)
6. Actionability & Code Quality (Real implementation details, runnable code, testing feasibility)

Candidate Text:
\"\"\"
{candidate_text[:3500]}
\"\"\"

Output format:
- Detailed Strengths & Weaknesses
- Standout Features & Code Innovations
- CRITERIA_SCORES: [c1, c2, c3, c4, c5, c6]
- TOTAL_SCORE: X/60
- FINAL_GRADE: Y/10
"""
    res = call_chat(JUDGE_URL, "Hermes-3-8B", judge_prompt, max_tokens=300, temperature=0.0, timeout=60)
    if res.get("ok"):
        content = res["full_text"]
        match_total = re.search(r"TOTAL_SCORE:\s*(\d+(?:\.\d+)?)/60", content)
        match_final = re.search(r"FINAL_GRADE:\s*(\d+(?:\.\d+)?)/10", content)
        total = float(match_total.group(1)) if match_total else 45.0
        grade = float(match_final.group(1)) if match_final else (total / 6.0)
        return {"grade": grade, "total_score": total, "critique": content}
    return {"grade": 7.0, "total_score": 42.0, "critique": "Judge unavailable"}


def peer_review(auditor_agent, code_text):
    review_prompt = f"""You are the Lead Peer Reviewer and Security Auditor.
Review the following unified Buzz & OS Optimization code implementation:

\"\"\"
{code_text[:3500]}
\"\"\"

Audit for:
1. Concurrency safety and non-blocking I/O
2. Node isolation and safe execution limits across t440, cesarops2, cesarops4
3. Correct JSON schemas for the Buzz operational board
4. Any edge cases or failure modes

Provide your audit verdict:
- VERDICT: [APPROVED | APPROVED_WITH_NOTES | REVISE]
- Key Findings & Security Notes:
"""
    res = call_chat(auditor_agent["url"], auditor_agent["model_id"], review_prompt, max_tokens=300, temperature=0.0, timeout=60)
    return res["full_text"] if res.get("ok") else "Peer review unavailable"


def main():
    print("=" * 80)
    print("  5-PILLAR ARCHITECTURE CONTEST: FLEET DISPATCH & JUDGING")
    print("=" * 80)

    fleet = [
        {"name": "GLM 5.3 Flash", "url": GLM_URL, "model_id": "glm-5.3-flash", "max_tok": 650},
        {"name": "DeepSeek MoE", "url": DEEPSEEK_URL, "model_id": "deepseek-v4-flash-9b", "max_tok": 650},
        {"name": "Swift-1.5", "url": SWIFT_URL, "model_id": "swift-1.5-iq2_xs", "max_tok": 650},
    ]

    submissions = {}
    for agent in fleet:
        print(f"\n[1/4] Dispatching 5-Pillar task to {agent['name']} ...", flush=True)
        resp = call_chat(agent["url"], agent["model_id"], PROMPT_CORE, max_tokens=agent["max_tok"], timeout=140)
        if resp.get("ok"):
            print(f"  -> Generated {resp['completion_tokens']} tokens in {resp['elapsed']:.1f}s ({resp['c_tok_s']:.1f} tok/s)")
            submissions[agent["name"]] = resp
            safe_name = agent["name"].replace(" ", "_").lower()
            with open(OUT_DIR / f"{safe_name}_architecture.txt", "w") as f:
                f.write(resp["full_text"])
        else:
            print(f"  -> FAILED: {resp.get('error')}")

    print("\n" + "=" * 80)
    print("  [2/4] JUDGING CANDIDATE ARCHITECTURES WITH HERMES-3 JUDGE (:8106)")
    print("=" * 80)

    evaluations = {}
    scores = []
    for name, sub in submissions.items():
        print(f"\nJudging {name} ...", flush=True)
        ev = judge_submission(name, sub["full_text"])
        evaluations[name] = ev
        scores.append((name, ev["total_score"], ev["grade"], ev))
        print(f"  Grade: {ev['grade']:.1f}/10 (Total: {ev['total_score']:.1f}/60)")

    scores.sort(key=lambda x: x[1], reverse=True)
    winner_name = scores[0][0]
    runner_up_name = scores[1][0]
    print(f"\n🏆 MOST ACCURATE MODEL: {winner_name} (Score: {scores[0][1]:.1f}/60)")
    print(f"🥈 NEXT MOST ACCURATE MODEL: {runner_up_name} (Score: {scores[1][1]:.1f}/60)")

    # 3. Code Generation by Most Accurate Model
    print("\n" + "=" * 80)
    print(f"  [3/4] CODING UNIFIED IMPLEMENTATION WITH {winner_name}")
    print("=" * 80)

    winner_agent = next(a for a in fleet if a["name"] == winner_name)
    runner_up_agent = next(a for a in fleet if a["name"] == runner_up_name)

    code_prompt = """You are the winning architect. Provide the concrete Python code for `cesarops_buzz_hub.py`.
It must:
1. Serve a lightweight HTTP REST + SSE collaboration hub on port 8128.
2. Provide endpoints:
   - GET /buzz/feed - return recent task posts, agent help requests, and human status messages.
   - POST /buzz/post - accept posts with {sender, role, type: 'task'|'help'|'status'|'insight', content, node, metadata}.
   - POST /buzz/claim - allow an agent to claim an open task or help request.
   - POST /buzz/complete - mark task completed with result and metrics.
3. Include a built-in OS Optimizer Agent task that probes Linux OS performance across t440, cesarops2, and cesarops4:
   - CPU load & memory pressure
   - Disk I/O & filesystem utilization
   - GPU utilization & thermals
   - Proposes tuning actions (sysctl vm.swappiness, dirty_ratio, CPU governor)
   - Posts findings and status updates directly to the Buzz board.

Output ONLY clean runnable python code inside ```python and ```.
"""
    code_resp = call_chat(winner_agent["url"], winner_agent["model_id"], code_prompt, max_tokens=750, timeout=160)
    final_code = code_resp["full_text"]
    code_blocks = re.findall(r"```python\n(.*?)```", final_code, re.DOTALL)
    extracted_code = code_blocks[0] if code_blocks else final_code

    code_path = OUT_DIR / "cesarops_buzz_hub.py"
    with open(code_path, "w") as f:
        f.write(extracted_code)
    print(f"  -> Generated code saved to {code_path}")

    # 4. Peer Review by Next Most Accurate Model
    print("\n" + "=" * 80)
    print(f"  [4/4] PEER REVIEW & AUDIT BY {runner_up_name}")
    print("=" * 80)
    audit_notes = peer_review(runner_up_agent, extracted_code)
    print(f"Audit Notes:\n{audit_notes}\n")

    with open(OUT_DIR / "audit_report.txt", "w") as f:
        f.write(audit_notes)

    # Save summary
    summary = {
        "winner": winner_name,
        "winner_score": scores[0][1],
        "runner_up": runner_up_name,
        "runner_up_score": scores[1][1],
        "scores": {name: {"total": tot, "grade": grd, "critique": ev["critique"]} for name, tot, grd, ev in scores},
        "timestamp": time.time(),
    }
    with open(OUT_DIR / "contest_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[✓] Contest summary saved to {OUT_DIR / 'contest_summary.json'}")


if __name__ == "__main__":
    main()
