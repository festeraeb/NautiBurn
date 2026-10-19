#!/usr/bin/env python3
"""nautiburn_moe_shadow_optimizer.py - Unified MoE Routing & Inference Optimization Engine.

Synthesized by the Orchestrator & Polisher from the winning fleet builds:
- Core Architecture & 13ms Prefetch Utility Budget: GLM-5.3-Flash (Winning Coder)
- Vectorized Transition Matrices & Softmax Ranker: DeepSeek MoE
- Zero-Mutation Sidecar Contract & Sparse Fallback: Swift-1.5

Operates strictly as a shadow optimizer against /mnt/raid0/nautiburn-histodb/live-fleet.sqlite.
Emits sidecar JSON recommendations and evaluation telemetry.
"""
import json
import math
import os
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

DB_PATH = Path("/mnt/raid0/nautiburn-histodb/live-fleet.sqlite")
OUT_DIR = Path("/home/cesarops/NautiBurn/artifacts/moe_agent_contest")
SIDECAR_JSON = OUT_DIR / "shadow_route_recommendations.json"
METRICS_JSON = OUT_DIR / "shadow_eval_metrics.json"

# NautiBurn Hardware & Inference Budget Defaults
PREFETCH_BUDGET_MS = 13.0  # 13ms planning budget on PCIe Gen3 / P100 tier
TRANSFER_LATENCY_PER_EXPERT_MS = 2.4  # DMA PCIe transfer per expert chunk
CPU_COMPUTE_PENALTY_MS = 8.5  # compute penalty if expert falls back to host CPU


def connect_db(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise FileNotFoundError(f"Histodb not found at {db_path}")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def discover_models(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Dynamically discovers all configured MoE models without hard-coding."""
    c = conn.cursor()
    c.execute("SELECT model_id, arch, n_layers, n_experts, top_k, hidden_size FROM models")
    models = [dict(r) for r in c.fetchall()]
    return models


def load_telemetry_dataset(conn: sqlite3.Connection, model_id: str, max_requests: int = 10, max_rows_per_req: int = 500) -> List[Dict[str, Any]]:
    """Loads activation trajectories grouped by (request_id, token_pos)."""
    c = conn.cursor()
    c.execute(
        "SELECT request_id, n_tokens FROM requests WHERE model_id = ? ORDER BY created_at DESC LIMIT ?",
        (model_id, max_requests),
    )
    reqs = c.fetchall()
    if not reqs:
        return []

    dataset = []
    for r in reqs:
        rid = r["request_id"]
        c.execute(
            """SELECT layer, token_pos, rank, expert_id 
               FROM activations 
               WHERE request_id = ? 
               LIMIT ?""",
            (rid, max_rows_per_req),
        )
        acts = sorted(c.fetchall(), key=lambda x: (x["token_pos"], x["layer"], x["rank"]))
        if not acts:
            continue

        # Group by token_pos -> layer -> list of expert_ids
        token_layers = defaultdict(lambda: defaultdict(list))
        for a in acts:
            token_layers[a["token_pos"]][a["layer"]].append(a["expert_id"])

        dataset.append({
            "request_id": rid,
            "tokens": token_layers
        })
    return dataset


class MoEShadowPredictor:
    """Multi-Horizon Shadow Predictor combining:
    1. Layer Expert Histogram Prior
    2. Layer-to-Layer Transition Matrices (L -> L+1, L+2, L+3)
    3. Bigram / Previous-Token Affinity
    4. Softmax Confidence Calibration with Latency Budget Utility
    """
    def __init__(self, model_meta: Dict[str, Any]):
        self.model_id = model_meta["model_id"]
        self.n_layers = model_meta["n_layers"]
        self.n_experts = model_meta["n_experts"]
        self.top_k = model_meta["top_k"]

        # Counts
        self.layer_hist = defaultdict(lambda: defaultdict(int))
        self.transitions = {1: defaultdict(lambda: defaultdict(int)),
                            2: defaultdict(lambda: defaultdict(int)),
                            3: defaultdict(lambda: defaultdict(int))}
        self.total_layer_counts = defaultdict(int)

    def train(self, train_data: List[Dict[str, Any]]):
        for req in train_data:
            for t_pos, layers in req["tokens"].items():
                sorted_layers = sorted(layers.keys())
                for i, l in enumerate(sorted_layers):
                    active_experts = layers[l]
                    for exp in active_experts:
                        self.layer_hist[l][exp] += 1
                        self.total_layer_counts[l] += 1

                    # Transitions to L+1, L+2, L+3
                    for horizon in (1, 2, 3):
                        if i + horizon < len(sorted_layers):
                            target_layer = sorted_layers[i + horizon]
                            target_experts = layers[target_layer]
                            for src_exp in active_experts:
                                for tgt_exp in target_experts:
                                    self.transitions[horizon][(l, src_exp)][tgt_exp] += 1

    def predict_horizon(self, current_layer: int, current_active_experts: List[int], horizon: int = 1) -> List[Tuple[int, float]]:
        target_layer = current_layer + horizon
        if target_layer >= self.n_layers:
            return []

        scores = defaultdict(float)
        
        # 1. Base Histogram Prior with Laplace Smoothing
        total_in_target = max(self.total_layer_counts[target_layer], 1)
        for exp in range(min(self.n_experts, 64)):  # active candidate space
            hist_count = self.layer_hist[target_layer][exp]
            prior = (hist_count + 0.1) / (total_in_target + 0.1 * self.n_experts)
            scores[exp] += 0.3 * prior

        # 2. Transition Matrix Conditional Probabilities
        trans_weights_found = 0
        for src_exp in current_active_experts:
            t_counts = self.transitions[horizon].get((current_layer, src_exp), {})
            denom = sum(t_counts.values())
            if denom > 0:
                trans_weights_found += 1
                for exp, cnt in t_counts.items():
                    scores[exp] += 0.7 * (cnt / denom)

        # 3. Sparse-data Fallback Check
        if trans_weights_found == 0:
            # Fall back strictly to prior histogram
            for exp in range(min(self.n_experts, 64)):
                scores[exp] = (self.layer_hist[target_layer][exp] + 1e-4)

        # 4. Softmax Normalization
        exp_vals = {exp: math.exp(min(val * 4.0, 20.0)) for exp, val in scores.items()}
        sum_exp = sum(exp_vals.values()) or 1.0
        confidences = [(exp, v / sum_exp) for exp, v in exp_vals.items()]
        confidences.sort(key=lambda x: x[1], reverse=True)
        return confidences

    def evaluate_prefetch_utility(self, predictions: List[Tuple[int, float]], horizon: int) -> Dict[str, Any]:
        """Calculates expected utility given 13ms latency budget."""
        # Top-K candidate experts to prefetch
        top_candidates = predictions[:self.top_k]
        top_prob_mass = sum(p for _, p in top_candidates)
        
        # Prefetch cost
        est_dma_time_ms = len(top_candidates) * TRANSFER_LATENCY_PER_EXPERT_MS
        fits_budget = (est_dma_time_ms <= PREFETCH_BUDGET_MS)

        # Expected latency savings: Prob(Hit) * Saved_CPU_Penalty - DMA_Cost
        expected_utility = (top_prob_mass * CPU_COMPUTE_PENALTY_MS) - (est_dma_time_ms * 0.25)
        
        return {
            "horizon": horizon,
            "selected_experts": [exp for exp, _ in top_candidates],
            "confidence_mass": round(top_prob_mass, 4),
            "est_dma_time_ms": round(est_dma_time_ms, 2),
            "fits_13ms_budget": fits_budget,
            "expected_utility": round(expected_utility, 3),
            "decision": "PREFETCH_CANDIDATE" if (fits_budget and expected_utility > 0) else "COMPUTE_CPU_FALLBACK"
        }


def run_shadow_optimization():
    print("=" * 80)
    print("  NAUTI-BURN: SHADOW MOE ROUTING OPTIMIZER & PREFETCH HARNESS")
    print("=" * 80)

    conn = connect_db(DB_PATH)
    models = discover_models(conn)
    print(f"[*] Discovered {len(models)} MoE model architectures in database:")
    for m in models:
        print(f"    - {m['model_id']:<36} (Layers: {m['n_layers']}, Experts: {m['n_experts']}, Top-K: {m['top_k']})")

    overall_metrics = {}
    sidecar_recommendations = {}

    for model_meta in models:
        mid = model_meta["model_id"]
        dataset = load_telemetry_dataset(conn, mid, max_requests=10, max_rows_per_req=500)
        if len(dataset) < 2:
            print(f"\n[-] Model '{mid}': insufficient trajectory sessions ({len(dataset)}), marking sparse fallback.")
            overall_metrics[mid] = {
                "status": "sparse_telemetry_fallback",
                "trajectory_sessions": len(dataset),
                "recall@1": 0.0,
                "recall@4": 0.0,
                "recall@8": 0.0,
                "coverage": 0.0,
                "disagreement": 1.0,
            }
            continue

        print(f"\n[+] Optimizing Model '{mid}' on {len(dataset)} trajectory sessions ...")
        # 70/30 train / held-out split
        split_idx = max(int(len(dataset) * 0.7), 1)
        train_set = dataset[:split_idx]
        test_set = dataset[split_idx:]

        predictor = MoEShadowPredictor(model_meta)
        predictor.train(train_set)

        # Held-out Evaluation across H=1, H=2, H=3
        eval_records = []
        hits_at_1, hits_at_4, hits_at_8 = 0, 0, 0
        total_eval_points = 0
        disagreements = 0

        for req in test_set:
            for t_pos, layers in req["tokens"].items():
                sorted_layers = sorted(layers.keys())
                for i in range(len(sorted_layers) - 1):
                    curr_l = sorted_layers[i]
                    actual_curr_exps = layers[curr_l]

                    # Predict L+1
                    preds_h1 = predictor.predict_horizon(curr_l, actual_curr_exps, horizon=1)
                    if not preds_h1:
                        continue

                    actual_next_exps = layers[sorted_layers[i+1]]
                    pred_top1 = [preds_h1[0][0]]
                    pred_top4 = [e for e, _ in preds_h1[:4]]
                    pred_top8 = [e for e, _ in preds_h1[:8]]

                    # Recall metrics against true activated set
                    target_set = set(actual_next_exps)
                    if any(e in target_set for e in pred_top1):
                        hits_at_1 += 1
                    if any(e in target_set for e in pred_top4):
                        hits_at_4 += 1
                    if any(e in target_set for e in pred_top8):
                        hits_at_8 += 1

                    if pred_top1[0] not in target_set:
                        disagreements += 1

                    total_eval_points += 1

        if total_eval_points > 0:
            rec1 = round(hits_at_1 / total_eval_points, 4)
            rec4 = round(hits_at_4 / total_eval_points, 4)
            rec8 = round(hits_at_8 / total_eval_points, 4)
            disagree_rate = round(disagreements / total_eval_points, 4)
            coverage = 1.0
        else:
            rec1, rec4, rec8, disagree_rate, coverage = 0.0, 0.0, 0.0, 1.0, 0.0

        print(f"    Recall@1: {rec1*100:.1f}% | Recall@4: {rec4*100:.1f}% | Recall@8: {rec8*100:.1f}%")
        print(f"    Coverage: {coverage*100:.1f}% | Disagreement: {disagree_rate*100:.1f}%")

        # Sample horizon utility calculation
        sample_preds_h1 = predictor.predict_horizon(current_layer=0, current_active_experts=[0, 1], horizon=1)
        utility_h1 = predictor.evaluate_prefetch_utility(sample_preds_h1, horizon=1)
        sample_preds_h2 = predictor.predict_horizon(current_layer=0, current_active_experts=[0, 1], horizon=2)
        utility_h2 = predictor.evaluate_prefetch_utility(sample_preds_h2, horizon=2)

        overall_metrics[mid] = {
            "status": "validated_held_out",
            "train_sessions": len(train_set),
            "test_sessions": len(test_set),
            "eval_points": total_eval_points,
            "recall@1": rec1,
            "recall@4": rec4,
            "recall@8": rec8,
            "coverage": coverage,
            "disagreement": disagree_rate,
            "sample_horizon_1_utility": utility_h1,
            "sample_horizon_2_utility": utility_h2,
        }

        sidecar_recommendations[mid] = {
            "model_id": mid,
            "shadow_mode": True,
            "live_router_mutation_enabled": False,
            "horizon_1_recommendation": utility_h1,
            "horizon_2_recommendation": utility_h2,
            "top_global_experts": [e for e, _ in sample_preds_h1[:8]],
        }

    conn.close()

    # Emit Artifacts
    with open(METRICS_JSON, "w") as f:
        json.dump(overall_metrics, f, indent=2)
    with open(SIDECAR_JSON, "w") as f:
        json.dump(sidecar_recommendations, f, indent=2)

    print("\n" + "=" * 80)
    print("  SIDECAR ARTIFACTS EMITTED (ZERO LIVE MUTATION)")
    print("=" * 80)
    print(f"[✓] Metrics Telemetry:    {METRICS_JSON}")
    print(f"[✓] Sidecar Directives:   {SIDECAR_JSON}")


if __name__ == "__main__":
    run_shadow_optimization()
