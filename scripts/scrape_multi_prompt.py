#!/usr/bin/env python3
"""Scrape Mixtral expert routing across10 prompt types."""
import subprocess, json, os, sys

HOST = "cesarops4om"
BIN = "/tmp/moe-histodb-scrape"
MODEL = "/home/cesarops/models/mixtral-8x7b-instruct-v0.1-Q4_K_M-worthdoing.gguf"
OUTDIR = "/tmp/scrape_results"
PROMPTS = {
    "code-rust": "Write a Rust function that validates an IPv4 address string. Return true if valid, false otherwise.",
    "code-python": "Write a Python function to find the longest common subsequence of two strings.",
    "code-cpp": "Write a C++ function that implements a thread-safe lock-free queue using atomics.",
    "math": "Prove that the square root of 2 is irrational. Show each step of the proof.",
    "reasoning": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Explain your reasoning.",
    "chat": "Hey, can you help me plan a weekend trip to the mountains? I like hiking and photography.",
    "translate": "Translate the following to French: The quick brown fox jumps over the lazy dog. The weather is beautiful today.",
    "summarize": "Summarize the key differences between TCP and UDP protocols in networking. Keep it concise.",
    "qa": "What causes the aurora borealis and why is it only visible near the poles?",
    "creative": "Write a short poem about a robot discovering emotions for the first time.",
}

ssh_cmd = lambda c: subprocess.run(
    ["sshpass", "-p", "cesarops", "ssh", "-o", "BatchMode=yes", HOST, c],
    capture_output=True, text=True, timeout=300
)

# mkdir
ssh_cmd(f"mkdir -p {OUTDIR}")

for tag, prompt in PROMPTS.items():
    out = f"{OUTDIR}/{tag}-prefill.json"
    cmd = (f"CUDA_VISIBLE_DEVICES=2 NAUTIBURN_MODEL_ID=mixtral-{tag} NAUTIBURN_OUT={out} NAUTIBURN_DECODE_N=3 "
           f"LD_LIBRARY_PATH=/home/cesarops/llama-cuda60/bin:/home/cesarops/llama-cuda60/cuda12 "
           f"{BIN} -m {MODEL} -ngl 99 -cmoe -c 2048 -t 8 -p '{prompt}' 2>&1 | tail -3")
    print(f"  {tag}...", end="", flush=True)
    r = ssh_cmd(cmd)
    lines = r.stdout.strip().split("\n")
    print(f" {lines[-1] if lines else 'no output'}")

# Pull results
print("\nPulling results...")
ssh_cmd(f"tar czf /tmp/scrape_results.tar.gz -C {OUTDIR} .")
subprocess.run(["scp", "-q", f"{HOST}:/tmp/scrape_results.tar.gz", "/tmp/scrape_results.tar.gz"], check=True)
os.makedirs("/tmp/scrape_results_local", exist_ok=True)
subprocess.run(["tar", "xzf", "/tmp/scrape_results.tar.gz", "-C", "/tmp/scrape_results_local"], check=True)
print("Done. Results in /tmp/scrape_results_local/")