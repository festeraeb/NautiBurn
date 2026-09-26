-- Schema v1. Applied by HistDb::open via schema::MIGRATION_1.
-- Authoritative copy lives in docs/histogram-database-handoff.md.

CREATE TABLE meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE models (
  model_id    TEXT PRIMARY KEY,
  arch        TEXT NOT NULL,
  n_layers    INTEGER NOT NULL,
  n_experts   INTEGER NOT NULL,
  top_k       INTEGER NOT NULL,
  hidden_size INTEGER NOT NULL,
  notes       TEXT
);

CREATE TABLE request_classes (
  class_id TEXT PRIMARY KEY,
  label    TEXT NOT NULL
);

CREATE TABLE requests (
  request_id   TEXT PRIMARY KEY,
  model_id     TEXT NOT NULL REFERENCES models(model_id),
  class_id     TEXT NOT NULL DEFAULT 'unknown' REFERENCES request_classes(class_id),
  source       TEXT NOT NULL,
  phase_hint   TEXT,
  n_tokens     INTEGER,
  created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE request_tokens (
  request_id TEXT NOT NULL REFERENCES requests(request_id),
  pos        INTEGER NOT NULL,
  token_id   INTEGER NOT NULL,
  PRIMARY KEY (request_id, pos)
);

CREATE TABLE activations (
  request_id TEXT NOT NULL REFERENCES requests(request_id),
  phase      TEXT NOT NULL,
  layer      INTEGER NOT NULL,
  token_pos  INTEGER NOT NULL,
  rank       INTEGER NOT NULL,
  expert_id  INTEGER NOT NULL,
  PRIMARY KEY (request_id, phase, layer, token_pos, rank)
);
CREATE INDEX idx_act_model_layer ON activations(layer, expert_id);

CREATE TABLE histogram_counts (
  model_id   TEXT NOT NULL REFERENCES models(model_id),
  class_id   TEXT NOT NULL REFERENCES request_classes(class_id),
  phase      TEXT NOT NULL,
  layer      INTEGER NOT NULL,
  expert_id  INTEGER NOT NULL,
  count      INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (model_id, class_id, phase, layer, expert_id)
);

CREATE TABLE beliefs (
  belief_id  TEXT PRIMARY KEY,
  request_id TEXT NOT NULL REFERENCES requests(request_id),
  layer      INTEGER NOT NULL,
  method     TEXT NOT NULL,
  expert_ids TEXT NOT NULL,
  predicted_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE disagreements (
  belief_id     TEXT NOT NULL REFERENCES beliefs(belief_id),
  layer         INTEGER NOT NULL,
  overlap_k     INTEGER NOT NULL,
  actual_ids    TEXT NOT NULL,
  PRIMARY KEY (belief_id, layer)
);
