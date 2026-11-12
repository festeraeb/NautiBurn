-- 007: the consolidated store (2026-11-12).
-- The DB had grown to 262 GB of per-activation rows that the predictor no longer learned from: for a
-- model with a rule, one earlier request predicted later prefill routing within 1-2 points of the best
-- possible. From here the DB keeps the aggregates a predictor is fitted on (histogram_counts and
-- cofire_counts), a frozen rule per (model, class, phase), one summary row per request, and full
-- activation rows only for requests the rule got wrong, classes without a rule, multi-token decode,
-- and a small random sample.

-- (layer, expert_a) fired together with (the token's next MoE layer, expert_b)
CREATE TABLE IF NOT EXISTS cofire_groups (
  group_id INTEGER PRIMARY KEY,
  model_id TEXT NOT NULL,
  class_id TEXT NOT NULL,
  phase    TEXT NOT NULL,
  UNIQUE (model_id, class_id, phase)
);
CREATE TABLE IF NOT EXISTS cofire_counts (
  group_id INTEGER NOT NULL REFERENCES cofire_groups(group_id),
  layer    INTEGER NOT NULL,
  expert_a INTEGER NOT NULL,
  expert_b INTEGER NOT NULL,
  count    INTEGER NOT NULL,
  PRIMARY KEY (group_id, layer, expert_a, expert_b)
) WITHOUT ROWID;

-- a frozen resident set: the top keep_frac experts of each layer when the rule was built
CREATE TABLE IF NOT EXISTS predict_rules (
  model_id   TEXT NOT NULL,
  class_id   TEXT NOT NULL,
  phase      TEXT NOT NULL,
  version    TEXT NOT NULL,
  keep_frac  REAL NOT NULL,
  selections INTEGER NOT NULL,   -- histogram mass the set was picked from
  requests   INTEGER NOT NULL,   -- requests scored to get median_hit
  median_hit REAL NOT NULL,      -- typical share of a request's selections the set serves
  built_at   TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (model_id, class_id, phase, version)
);
CREATE TABLE IF NOT EXISTS rule_experts (
  model_id  TEXT NOT NULL,
  class_id  TEXT NOT NULL,
  phase     TEXT NOT NULL,
  version   TEXT NOT NULL,
  layer     INTEGER NOT NULL,
  expert_id INTEGER NOT NULL,
  PRIMARY KEY (model_id, class_id, phase, version, layer, expert_id)
) WITHOUT ROWID;

-- every ingested request, whether or not its rows were kept
CREATE TABLE IF NOT EXISTS request_summary (
  request_id   TEXT PRIMARY KEY,
  model_id     TEXT NOT NULL,
  class_id     TEXT NOT NULL,
  phase        TEXT NOT NULL,
  n_tokens     INTEGER,
  selections   INTEGER NOT NULL,
  hits         INTEGER,          -- selections inside the rule's set; NULL when there was no rule
  rule_version TEXT,
  rows_kept    INTEGER NOT NULL, -- 1: its activation rows are in the DB
  reason       TEXT,             -- why they were kept: no_rule | diverged | decode | sample
  created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_summary_group ON request_summary(model_id, class_id, phase);

-- keep policy, tunable in state (not code)
INSERT OR IGNORE INTO meta (key, value) VALUES ('keep_margin', '0.05');
INSERT OR IGNORE INTO meta (key, value) VALUES ('keep_sample_every', '50');
INSERT OR IGNORE INTO meta (key, value) VALUES ('keep_decode', '1');

-- no reader used it, and on the big table every insert paid for it
DROP INDEX IF EXISTS idx_act_model_layer;
