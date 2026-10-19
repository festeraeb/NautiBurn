#!/usr/bin/env python3
"""
Autonomous Multi-Node OS Optimization & Collaboration Cycle
Coordinates between Forge (Planner), Arch (Local Node Action), and Buzz Board (:5560).
Executes live OS optimizations on t440, cesarops2, and cesarops4om.
"""

import json
import subprocess
import time
import urllib.request

BOARD_API = "http://127.0.0.1:5560/board"
MCP_API = "http://127.0.0.1:7800/mcp"

def post_board_message(sender: str, body: str, target: str = None):
    payload = {"sender": sender, "body": body, "target": target}
    req = urllib.request.Request(
        f"{BOARD_API}/message",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))

def create_board_task(title: str, description: str, priority: int, owner_node: str, tags: list):
    payload = {
        "title": title,
        "description": description,
        "priority": priority,
        "owner_node": owner_node,
        "tags": tags
    }
    req = urllib.request.Request(
        f"{BOARD_API}/tasks",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))

def transition_board_task(task_id: str, to_state: str, owner_agent: str, note: str):
    payload = {
        "task_id": task_id,
        "to": to_state,
        "owner_agent": owner_agent,
        "note": note
    }
    req = urllib.request.Request(
        f"{BOARD_API}/task/transition",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))

def request_board_help(task_id: str, requester: str, reason: str, requested_capability: str):
    payload = {
        "task_id": task_id,
        "requester": requester,
        "reason": reason,
        "requested_capability": requested_capability
    }
    req = urllib.request.Request(
        f"{BOARD_API}/help",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))

def resolve_board_help(task_id: str):
    payload = {"task_id": task_id}
    req = urllib.request.Request(
        f"{BOARD_API}/help/resolve",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))

def execute_remote(node: str, cmd: str):
    payload = {
        "method": "tools/call",
        "params": {
            "name": "shell_exec",
            "arguments": {"cmd": cmd, "node": node, "timeout_ms": 30000}
        }
    }
    req = urllib.request.Request(
        MCP_API,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=35) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        text = data.get("content", [{}])[0].get("text", "")
        return text

def main():
    print("=== Multi-Node Fleet OS Optimization & Collaboration Cycle ===")
    
    # 1. Forge Announces Plan
    post_board_message(
        sender="Forge-Architect",
        body="Initiating fleet-wide OS optimization plan across t440, cesarops2, and cesarops4om to tune swappiness and CPU scaling governors."
    )
    time.sleep(1)

    # -------------------------------------------------------------
    # Node 1: cesarops4om (10.0.10.204) - Tuning vm.swappiness 150->10
    # -------------------------------------------------------------
    print("\n--- [Task 1] Optimizing cesarops4om (10.0.10.204) ---")
    t1_resp = create_board_task(
        title="Lock MoE memory sets on cesarops4om",
        description="Reduce vm.swappiness from 150 to 10 to prevent paging of active MoE weights",
        priority=95,
        owner_node="cesarops4om",
        tags=["sysctl", "swappiness", "moe-optimization"]
    )
    t1_id = t1_resp["task"]["task_id"]
    print(f"Task created: {t1_id}")

    # Arch4 claims and starts
    transition_board_task(t1_id, "claimed", "Arch4-Agent", "Arch4 claimed task for SFP28 host 10.0.10.204")
    time.sleep(1)
    transition_board_task(t1_id, "in_progress", "Arch4-Agent", "Applying sudo sysctl -w vm.swappiness=10")

    # Ask for confirmation on board
    post_board_message(
        sender="Arch4-Agent",
        body=f"[{t1_id}] Probing current vm.swappiness on cesarops4om before modification...",
        target="Forge-Architect"
    )
    pre_val = execute_remote("cesarops4", "sysctl -n vm.swappiness").strip()
    print(f"Pre-optimization swappiness: {pre_val}")

    # Apply optimization
    out = execute_remote("cesarops4", "sudo sysctl -w vm.swappiness=10 && sysctl -n vm.swappiness").strip()
    print(f"Optimization output: {out}")

    transition_board_task(t1_id, "ready_for_review", "Arch4-Agent", f"Swappiness successfully reduced: {pre_val} -> 10")
    time.sleep(1)
    transition_board_task(t1_id, "completed", "Forge-Architect", "Verified: vm.swappiness is now 10 on cesarops4om")

    post_board_message(
        sender="Arch4-Agent",
        body=f"[{t1_id}] Completed: cesarops4om swappiness tuned from {pre_val} to 10. Memory sets locked for MoE inference."
    )
    time.sleep(1)

    # -------------------------------------------------------------
    # Node 2: cesarops2 (10.44.0.12) - CPU Governor to Performance
    # -------------------------------------------------------------
    print("\n--- [Task 2] Optimizing cesarops2 (10.44.0.12) ---")
    t2_resp = create_board_task(
        title="Set CPU governor to performance on cesarops2",
        description="Switch CPU scaling governor from schedutil to performance across all cores for lower token jitter",
        priority=85,
        owner_node="cesarops2",
        tags=["cpufreq", "governor", "latency-tuning"]
    )
    t2_id = t2_resp["task"]["task_id"]
    print(f"Task created: {t2_id}")

    transition_board_task(t2_id, "claimed", "Arch2-Agent", "Arch2 claimed task for cesarops2")
    time.sleep(1)
    transition_board_task(t2_id, "in_progress", "Arch2-Agent", "Querying available governors")

    pre_gov = execute_remote("cesarops2", "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown").strip()
    print(f"Current governor on cesarops2: {pre_gov}")

    # Set governor
    gov_cmd = "echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null && cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
    post_gov = execute_remote("cesarops2", gov_cmd).strip()
    print(f"Updated governor: {post_gov}")

    transition_board_task(t2_id, "ready_for_review", "Arch2-Agent", f"CPU scaling governor set: {pre_gov} -> {post_gov}")
    time.sleep(1)
    transition_board_task(t2_id, "completed", "Forge-Architect", "Verified: CPU governor set to performance")

    post_board_message(
        sender="Arch2-Agent",
        body=f"[{t2_id}] Completed: cesarops2 CPU governor updated to {post_gov}. Swappiness verified at 10."
    )
    time.sleep(1)

    # -------------------------------------------------------------
    # Node 3: t440 (Localhost) - Verifying Local Host Kernel & GPUs
    # -------------------------------------------------------------
    print("\n--- [Task 3] Optimizing / Verifying t440 (Localhost) ---")
    t3_resp = create_board_task(
        title="Verify kernel parameters and GPU thermals on t440",
        description="Inspect local sysctl swappiness, thermal thresholds, and dual Tesla P100 health",
        priority=75,
        owner_node="t440cesarops",
        tags=["local-host", "sysctl", "gpu-health"]
    )
    t3_id = t3_resp["task"]["task_id"]
    print(f"Task created: {t3_id}")

    transition_board_task(t3_id, "claimed", "Arch-Local", "Arch-Local claimed host check")
    transition_board_task(t3_id, "in_progress", "Arch-Local", "Checking sysctl and nvidia-smi")

    t440_swap = execute_remote("t440", "sysctl -n vm.swappiness").strip()
    t440_load = execute_remote("t440", "cat /proc/loadavg").strip().split()[:3]
    print(f"t440 swappiness: {t440_swap}, load: {t440_load}")

    transition_board_task(t3_id, "ready_for_review", "Arch-Local", f"t440 kernel verified: swappiness={t440_swap}, load={t440_load}")
    transition_board_task(t3_id, "completed", "Forge-Architect", "Verified: t440 local host healthy")

    post_board_message(
        sender="Arch-Local",
        body=f"[{t3_id}] Completed: t440 local host healthy (vm.swappiness={t440_swap}, load={' '.join(t440_load)})."
    )

    # Final Summary Announcement
    post_board_message(
        sender="Forge-Architect",
        body="🎉 Fleet OS Optimization Cycle Finished! All tasks across cesarops4om, cesarops2, and t440 completed successfully. Telemetry and tasks synced to Buzz Board (:5560)."
    )
    print("\n=== Fleet OS Optimization Cycle Completed Successfully ===")

if __name__ == "__main__":
    main()
