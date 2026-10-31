#!/usr/bin/env python3
"""cesarops_buzz_hub.py - Buzz Operational Board & Node OS Optimization Engine.

Pillar 4 (Buzz) & Pillar 2 (Arch) Integration for the CESARops Fleet.
Synthesized and Polished from the Fleet Contest:
- Core Architecture: GLM-5.3-Flash (Winner)
- Peer Review & Concurrency/Security Checks: DeepSeek MoE (Runner-up)

Features:
1. Lightweight HTTP REST + SSE server on port 8128.
2. In-memory thread-safe event queue + SQLite persistence.
3. Node OS Optimizer Agent that probes t440, cesarops2, and cesarops4om via SSH.
4. Auto-posts OS performance telemetry, help requests, and optimization advisories.
"""
import json
import os
import queue
import re
import socket
import sqlite3
import subprocess
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Tuple, Dict, List, Any
from urllib.parse import parse_qs, urlparse

PORT = int(os.environ.get("BUZZ_PORT", "8130"))
DB_PATH = Path(os.environ.get("BUZZ_DB", "/var/www/cesarops/data/buzz_board.sqlite"))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Node targeting definitions matching /var/www/cesarops/services/mcp-server/server.js
NODE_TARGETS = {
    "t440": {"name": "t440 (Host)", "is_local": True},
    "cesarops2": {"name": "cesarops2", "is_local": False, "ssh": ["-o", "ConnectTimeout=4", "-i", "/home/cesarops/.ssh/id_ed25519", "cesarops@10.44.0.12"]},
    "cesarops4": {"name": "cesarops4 (c4)", "is_local": False, "ssh": ["-o", "ConnectTimeout=4", "cesarops@10.0.10.204"]},
}


def init_db():
    conn = sqlite3.connect(DB_PATH)
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS buzz_posts (
                id TEXT PRIMARY KEY,
                ts REAL NOT NULL,
                sender TEXT NOT NULL,
                role TEXT NOT NULL,
                type TEXT NOT NULL,
                node TEXT NOT NULL,
                content TEXT NOT NULL,
                status TEXT NOT NULL,
                claimed_by TEXT,
                result TEXT,
                metadata TEXT
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_buzz_ts ON buzz_posts(ts);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_buzz_type ON buzz_posts(type);")
    conn.close()


class BuzzHub:
    def __init__(self):
        init_db()
        self.lock = threading.Lock()
        self.subscribers = []
        self.recent_posts = deque(maxlen=200)
        self._load_recent()

    def _load_recent(self):
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM buzz_posts ORDER BY ts DESC LIMIT 50")
        rows = [dict(r) for r in c.fetchall()]
        conn.close()
        for r in reversed(rows):
            if r.get("metadata"):
                try: r["metadata"] = json.loads(r["metadata"])
                except Exception: pass
            self.recent_posts.append(r)

    def subscribe(self):
        q = queue.Queue(maxsize=100)
        with self.lock:
            self.subscribers.append(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            if q in self.subscribers:
                self.subscribers.remove(q)

    def broadcast(self, event_type: str, data: dict):
        payload = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
        with self.lock:
            dead = []
            for q in self.subscribers:
                try:
                    q.put_nowait(payload)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                if q in self.subscribers:
                    self.subscribers.remove(q)

    def add_post(self, sender: str, role: str, ptype: str, content: str, node: str = "t440", metadata: dict = None) -> dict:
        post_id = f"buzz-{int(time.time()*1000)}-{os.urandom(2).hex()}"
        post = {
            "id": post_id,
            "ts": time.time(),
            "sender": sender,
            "role": role,
            "type": ptype,  # 'task', 'help', 'status', 'insight'
            "node": node,
            "content": content,
            "status": "open" if ptype in ("task", "help") else "info",
            "claimed_by": None,
            "result": None,
            "metadata": metadata or {},
        }
        with self.lock:
            self.recent_posts.append(post)

        # Persist
        conn = sqlite3.connect(DB_PATH)
        with conn:
            conn.execute(
                """INSERT INTO buzz_posts (id, ts, sender, role, type, node, content, status, claimed_by, result, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (post_id, post["ts"], sender, role, ptype, node, content, post["status"], None, None, json.dumps(metadata or {})),
            )
        conn.close()

        self.broadcast("new_post", post)
        return post

    def claim_task(self, post_id: str, claimer: str) -> dict:
        conn = sqlite3.connect(DB_PATH)
        with conn:
            conn.execute("UPDATE buzz_posts SET status = 'claimed', claimed_by = ? WHERE id = ?", (claimer, post_id))
        conn.close()

        updated = None
        with self.lock:
            for p in self.recent_posts:
                if p["id"] == post_id:
                    p["status"] = "claimed"
                    p["claimed_by"] = claimer
                    updated = p
                    break
        if updated:
            self.broadcast("task_claimed", updated)
        return updated or {"ok": False, "error": "not_found"}

    def complete_task(self, post_id: str, completed_by: str, result: str) -> dict:
        conn = sqlite3.connect(DB_PATH)
        with conn:
            conn.execute("UPDATE buzz_posts SET status = 'completed', result = ? WHERE id = ?", (result, post_id))
        conn.close()

        updated = None
        with self.lock:
            for p in self.recent_posts:
                if p["id"] == post_id:
                    p["status"] = "completed"
                    p["result"] = result
                    updated = p
                    break
        if updated:
            self.broadcast("task_completed", updated)
        return updated or {"ok": False, "error": "not_found"}


HUB = BuzzHub()


def run_node_cmd(node_key: str, cmd: str, timeout_sec: int = 6) -> Tuple[bool, str]:
    target = NODE_TARGETS.get(node_key)
    if not target:
        return False, f"Unknown node {node_key}"
    try:
        if target["is_local"]:
            res = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout_sec)
        else:
            res = subprocess.run(["ssh"] + target["ssh"] + [cmd], capture_output=True, text=True, timeout=timeout_sec)
        return (res.returncode == 0), (res.stdout.strip() if res.returncode == 0 else res.stderr.strip())
    except Exception as e:
        return False, str(e)


def probe_node_os(node_key: str) -> dict:
    target = NODE_TARGETS[node_key]
    probe_script = """
echo "===LOAD===" && cat /proc/loadavg
echo "===MEM===" && free -m | grep -E 'Mem|Swap'
echo "===DISK===" && df -h / | tail -n 1
echo "===GPU===" && (nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "NO_GPU")
echo "===GOVERNOR===" && (cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "none")
echo "===SWAP===" && cat /proc/sys/vm/swappiness
"""
    ok, out = run_node_cmd(node_key, probe_script, timeout_sec=8)
    if not ok:
        return {"node": node_key, "online": False, "error": out}

    sections = {}
    cur_sec = None
    lines = out.split("\n")
    for l in lines:
        if l.startswith("===") and l.endswith("==="):
            cur_sec = l.strip("=")
            sections[cur_sec] = []
        elif cur_sec:
            sections[cur_sec].append(l)

    load_raw = " ".join(sections.get("LOAD", ["0 0 0"])).split()
    load1, load5 = (load_raw[0] if len(load_raw) > 0 else "0.0"), (load_raw[1] if len(load_raw) > 1 else "0.0")

    mem_lines = sections.get("MEM", [])
    mem_used, mem_total = 0, 1
    for ml in mem_lines:
        if ml.startswith("Mem:"):
            parts = ml.split()
            if len(parts) >= 3:
                mem_total = int(parts[1])
                mem_used = int(parts[2])

    gpu_lines = [g for g in sections.get("GPU", []) if "NO_GPU" not in g and g.strip()]
    gpus = []
    for gl in gpu_lines:
        gp = [x.strip() for x in gl.split(",")]
        if len(gp) >= 6:
            gpus.append({
                "index": gp[0],
                "name": gp[1],
                "temp_c": int(gp[2]),
                "util_pct": int(gp[3]),
                "vram_used_mb": int(gp[4]),
                "vram_total_mb": int(gp[5]),
            })

    swappiness = sections.get("SWAP", ["60"])[0].strip()
    governor = sections.get("GOVERNOR", ["powersave"])[0].strip()

    # Optimization Recommendations
    advisories = []
    if int(swappiness) > 15:
        advisories.append(f"Reduce vm.swappiness from {swappiness} -> 10 to protect MoE GPU host-RAM caches.")
    if governor != "performance" and governor != "none":
        advisories.append(f"Set CPU scaling_governor from '{governor}' -> 'performance' for predictable 13ms DMA transfers.")
    if float(load1) > 8.0:
        advisories.append(f"High CPU load ({load1}); throttle background batch indexers.")

    return {
        "node": node_key,
        "name": target["name"],
        "online": True,
        "load_1m": float(load1),
        "load_5m": float(load5),
        "mem_used_mb": mem_used,
        "mem_total_mb": mem_total,
        "mem_pct": round((mem_used / max(mem_total, 1)) * 100, 1),
        "gpus": gpus,
        "governor": governor,
        "swappiness": int(swappiness),
        "advisories": advisories,
        "timestamp": time.time(),
    }


def background_os_optimizer_worker():
    """Background Agent task that probes OS across nodes and posts telemetry to Buzz."""
    time.sleep(3)
    HUB.add_post(
        sender="Arch-Optimizer-Agent",
        role="arch",
        ptype="status",
        node="t440",
        content="Arch Node OS Optimizer initialized. Beginning fleet performance survey across t440, cesarops2, and cesarops4.",
    )

    for node_key in ["t440", "cesarops2", "cesarops4"]:
        probe = probe_node_os(node_key)
        if not probe.get("online"):
            HUB.add_post(
                sender=f"Arch-{node_key}",
                role="arch",
                ptype="help",
                node=node_key,
                content=f"Node {node_key} unreachable or timed out during OS probe. Requesting network spine triage.",
                metadata=probe,
            )
            continue

        gpu_summary = ", ".join([f"GPU{g['index']} {g['name']} ({g['temp_c']}°C, {g['vram_used_mb']}MB VRAM)" for g in probe["gpus"]]) or "No discrete GPU"
        status_msg = (
            f"OS Telemetry: Load 1m={probe['load_1m']}, RAM {probe['mem_pct']}% ({probe['mem_used_mb']}/{probe['mem_total_mb']} MB), "
            f"Governor={probe['governor']}, Swappiness={probe['swappiness']}. {gpu_summary}."
        )

        HUB.add_post(
            sender=f"Arch-{node_key}",
            role="arch",
            ptype="status",
            node=node_key,
            content=status_msg,
            metadata=probe,
        )

        # Post optimization task/advisory if found
        if probe.get("advisories"):
            for adv in probe["advisories"]:
                HUB.add_post(
                    sender=f"Arch-{node_key}",
                    role="arch",
                    ptype="task",
                    node=node_key,
                    content=f"[Kernel Tuning] {adv}",
                    metadata={"target_node": node_key, "action": adv},
                )


class BuzzRequestHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/buzz/feed":
            qs = parse_qs(parsed.query)
            if "sse" in qs.get("stream", []) or self.headers.get("Accept") == "text/event-stream":
                # SSE Mode
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                q = HUB.subscribe()
                # Initial snapshot
                self.wfile.write(f"event: snapshot\ndata: {json.dumps(list(HUB.recent_posts))}\n\n".encode("utf-8"))
                self.wfile.flush()
                try:
                    while True:
                        try:
                            msg = q.get(timeout=15)
                            self.wfile.write(msg.encode("utf-8"))
                            self.wfile.flush()
                        except queue.Empty:
                            self.wfile.write(b": keep-alive\n\n")
                            self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    HUB.unsubscribe(q)
                return

            # JSON Snapshot Mode
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({
                "ok": True,
                "posts": list(HUB.recent_posts),
                "total": len(HUB.recent_posts),
                "timestamp": time.time(),
            }).encode("utf-8"))
            return

        elif parsed.path == "/buzz/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({
                "service": "cesarops-buzz-hub",
                "port": PORT,
                "status": "online",
                "subscribers": len(HUB.subscribers),
                "cached_posts": len(HUB.recent_posts),
            }).encode("utf-8"))
            return

        elif parsed.path == "/buzz/probe":
            qs = parse_qs(parsed.query)
            node = qs.get("node", ["t440"])[0]
            probe_res = probe_node_os(node)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(probe_res).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"
        try:
            data = json.loads(body)
        except Exception:
            data = {}

        if parsed.path == "/buzz/post":
            sender = data.get("sender", "anonymous")
            role = data.get("role", "worker")
            ptype = data.get("type", "status")
            content = data.get("content", "")
            node = data.get("node", "t440")
            meta = data.get("metadata", {})
            post = HUB.add_post(sender, role, ptype, content, node, meta)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True, "post": post}).encode("utf-8"))
            return

        elif parsed.path == "/buzz/claim":
            post_id = data.get("post_id")
            claimer = data.get("claimer", "agent")
            res = HUB.claim_task(post_id, claimer)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True, "task": res}).encode("utf-8"))
            return

        elif parsed.path == "/buzz/complete":
            post_id = data.get("post_id")
            completed_by = data.get("completed_by", "agent")
            result = data.get("result", "")
            res = HUB.complete_task(post_id, completed_by, result)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True, "task": res}).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()


def start_server():
    server = ThreadingHTTPServer(("0.0.0.0", PORT), BuzzRequestHandler)
    print(f"[Buzz Hub] Operational Board server listening on 0.0.0.0:{PORT} ...")

    # Start OS optimizer background thread
    t = threading.Thread(target=background_os_optimizer_worker, daemon=True)
    t.start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    start_server()