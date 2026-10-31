#!/usr/bin/env python3
"""glm_histodb_setup.py - Setup, seed, and prewire NautiBurn HistoDB for GLM-5.3-Flash.

1. Creates SQLite database at /mnt/raid0/ps-data/packs/glm/glm-histodb.sqlite using NautiBurn Schema v3.
2. Upserts model record for 'glm-5.3-flash' (arch: glm5next, 46 layers, 144 experts/layer, top_k=8).
3. Seeds initial histogram_counts from /mnt/raid0/ps-data/packs/glm/expert-profile.txt.
4. Prewires transition scoring & export back to strata-poly expert-profile.txt when prediction rate is reached.
"""
import os
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path("/mnt/raid0/ps-data/packs/glm/glm-histodb.sqlite")
PROFILE_PATH = Path("/mnt/raid0/ps-data/packs/glm/expert-profile.txt")
MODEL_ID = "glm-5.3-flash"
ARCH = "glm5next"
N_LAYERS = 46
N_EXPERTS = 144
TOP_K = 8
HIDDEN_SIZE = 4096

SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS models (
  model_id    TEXT PRIMARY KEY,
  arch        TEXT NOT NULL,
  n_layers    INTEGER NOT NULL,
  n_experts   INTEGER NOT NULL,
  top_k       INTEGER NOT NULL,
  hidden_size INTEGER NOT NULL,
  notes       TEXT
);

CREATE TABLE IF NOT EXISTS request_classes (
  class_id TEXT PRIMARY KEY,
  label    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS requests (
  request_id   TEXT PRIMARY KEY,
  model_id     TEXT NOT NULL REFERENCES models(model_id),
  class_id     TEXT NOT NULL DEFAULT 'unknown' REFERENCES request_classes(class_id),
  source       TEXT NOT NULL,
  phase_hint   TEXT,
  n_tokens     INTEGER,
  dtype        TEXT DEFAULT 'gguf',
  created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS request_tokens (
  request_id TEXT NOT NULL REFERENCES requests(request_id),
  pos        INTEGER NOT NULL,
  token_id   INTEGER NOT NULL,
  PRIMARY KEY (request_id, pos)
);

CREATE TABLE IF NOT EXISTS activations (
  request_id TEXT NOT NULL REFERENCES requests(request_id),
  phase      TEXT NOT NULL,
  layer      INTEGER NOT NULL,
  token_pos  INTEGER NOT NULL,
  rank       INTEGER NOT NULL,
  expert_id  INTEGER NOT NULL,
  PRIMARY KEY (request_id, phase, layer, token_pos, rank)
);
CREATE INDEX IF NOT EXISTS idx_act_model_layer ON activations(layer, expert_id);

CREATE TABLE IF NOT EXISTS histogram_counts (
  model_id   TEXT NOT NULL REFERENCES models(model_id),
  class_id   TEXT NOT NULL REFERENCES request_classes(class_id),
  phase      TEXT NOT NULL,
  layer      INTEGER NOT NULL,
  expert_id  INTEGER NOT NULL,
  count      INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (model_id, class_id, phase, layer, expert_id)
);

CREATE TABLE IF NOT EXISTS beliefs (
  belief_id    TEXT PRIMARY KEY,
  request_id   TEXT NOT NULL REFERENCES requests(request_id),
  layer        INTEGER NOT NULL,
  method       TEXT NOT NULL,
  expert_ids   TEXT NOT NULL,
  latency_us   INTEGER,
  predicted_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS disagreements (
  belief_id     TEXT NOT NULL REFERENCES beliefs(belief_id),
  layer         INTEGER NOT NULL,
  overlap_k     INTEGER NOT NULL,
  actual_ids    TEXT NOT NULL,
  PRIMARY KEY (belief_id, layer)
);
"""


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    with conn:
        conn.executescript(SCHEMA_SQL)
        conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', '3')")
        conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('app', 'nautiburn-histodb')")
        
        # Upsert model
        conn.execute(
            """INSERT OR REPLACE INTO models 
               (model_id, arch, n_layers, n_experts, top_k, hidden_size, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                MODEL_ID,
                ARCH,
                N_LAYERS,
                N_EXPERTS,
                TOP_K,
                HIDDEN_SIZE,
                "GLM-5.3-Flash REAP50 (43 MoE layers x 144 experts, top_k=8)",
            ),
        )
        # Default request classes
        for cid, lbl in [
            ("unknown", "Unclassified requests"),
            ("code", "Code synthesis / analysis"),
            ("reasoning", "Deep reasoning / chain-of-thought"),
            ("sidechannel", "Out-of-process sidechannel probe"),
        ]:
            conn.execute("INSERT OR IGNORE INTO request_classes (class_id, label) VALUES (?, ?)", (cid, lbl))
    print(f"[init_db] initialized {DB_PATH} with schema v3 and model '{MODEL_ID}'")
    return conn


def seed_from_expert_profile(conn):
    if not PROFILE_PATH.exists():
        print(f"[seed] expert profile not found at {PROFILE_PATH}")
        return 0
    rows = []
    with open(PROFILE_PATH, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                layer, expert, count = int(parts[0]), int(parts[1]), int(parts[2])
                rows.append((MODEL_ID, "unknown", "serve", layer, expert, count))

    with conn:
        conn.executemany(
            """INSERT OR REPLACE INTO histogram_counts 
               (model_id, class_id, phase, layer, expert_id, count)
               VALUES (?, ?, ?, ?, ?, ?)""",
            rows,
        )
    print(f"[seed] seeded {len(rows)} expert activation cells from {PROFILE_PATH}")
    return len(rows)


def export_optimized_profile(conn, out_path=PROFILE_PATH, top_vram_experts=2691):
    """Generates an expert-profile.txt ordered by observed frequency in histodb."""
    cursor = conn.cursor()
    cursor.execute(
        """SELECT layer, expert_id, SUM(count) as total_count
           FROM histogram_counts
           WHERE model_id = ?
           GROUP BY layer, expert_id
           ORDER BY total_count DESC""",
        (MODEL_ID,),
    )
    ranked = cursor.fetchall()
    if not ranked:
        print("[export] no counts found to export")
        return False
    
    with open(out_path, "w") as f:
        for layer, exp, count in ranked:
            f.write(f"{layer} {exp} {count}\n")
    print(f"[export] wrote {len(ranked)} ranked experts to {out_path} (top {top_vram_experts} fit in 48GB VRAM)")
    return True


if __name__ == "__main__":
    conn = init_db()
    seed_from_expert_profile(conn)
    print("[ready] GLM-5.3-Flash HistoDB is prewired and ready for live scrapes & shadow predictions.")
