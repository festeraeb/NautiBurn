//! Schema v1 for the NautiBurn histogram store.
//!
//! DDL is transcribed from `docs/histogram-database-handoff.md` ("Schema (v1)").
//! The doc is authoritative; edit it there first, then here.
//!
//! Design rules that the schema itself enforces:
//!
//! - Histograms are keyed by `model_id`, so Mixtral and Qwen never share a
//!   histogram. `histogram_counts` carries no hostname.
//! - `activations` is the source of truth; `histogram_counts` is an aggregate
//!   over it. Importers that only have aggregate counts (continuum) write
//!   `histogram_counts` alone.
//! - Hot / warm / cold is a *derived* query, never a stored flag. See
//!   [`crate::HistDb::histogram_for`].

use serde::{Deserialize, Serialize};

/// Current schema version. Seeded into `meta` at open.
pub const SCHEMA_VERSION: i64 = 1;

/// One scraped activation row: which experts fired, at which rank.
///
/// `rank` is the position in the top-k ordering, so it is always
/// `0 .. top_k-1` for the model. Keeping the rank (rather than only the set)
/// lets a later re-scoring see *where* a belief was wrong, not just that it was.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActivationRow {
    pub layer: i64,
    pub token_pos: i64,
    pub expert_ids: Vec<i64>,
}

/// A v1 scrape record, exactly the JSON the engine posts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScrapeRecord {
    /// Must equal [`SCHEMA_VERSION`]. Rejected otherwise.
    pub schema_version: i64,
    pub model_id: String,
    pub request_id: String,
    /// Missing class is not an error: it files under `unknown`.
    #[serde(default)]
    pub class_id: Option<String>,
    pub source: String,
    pub phase: String,
    pub n_tokens: Option<i64>,
    #[serde(default)]
    pub dtype: Option<String>,
    pub activations: Vec<ActivationRow>,
    /// Optional token ids beside the expert ids. Never embedded.
    #[serde(default)]
    pub token_ids: Option<Vec<i64>>,
}

/// Logical model identity. A `model_id` is weights + arch, never a hostname,
/// so histograms stay separated across machines that run the same weights.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelRecord {
    pub model_id: String,
    pub arch: String,
    pub n_layers: i64,
    pub n_experts: i64,
    pub top_k: i64,
    pub hidden_size: i64,
    #[serde(default)]
    pub notes: Option<String>,
}

/// One cell of a histogram: how many times `expert_id` fired.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HistogramCell {
    pub layer: i64,
    pub expert_id: i64,
    pub count: i64,
}

/// A belief set predicted before the gate, for later disagreement scoring.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BeliefRecord {
    pub request_id: String,
    pub layer: i64,
    /// `frequency` | `prev_token` | `bigram` | `ensemble` | `paper:...`
    pub method: String,
    pub expert_ids: Vec<i64>,
}

/// Errors this crate returns. Every variant names the thing that was wrong, so
/// a failed ingest can be diagnosed from the message alone.
#[derive(Debug, thiserror::Error)]
pub enum HistDbError {
    /// Underlying SQLite or I/O failure.
    #[error("sqlite: {0}")]
    Sqlite(#[from] rusqlite::Error),

    /// The posted `schema_version` is not the one this build understands.
    #[error("unsupported schema_version {got}, this build speaks {expected}")]
    SchemaVersion { got: i64, expected: i64 },

    /// A scrape referenced a `model_id` with no row in `models`.
    #[error("unknown model_id {0}; upsert it first")]
    UnknownModel(String),

    /// A scrape's `expert_ids` length did not match that model's `top_k`.
    ///
    /// This is the rule that stops a Mixtral top-2 trace being filed under a
    /// Qwen model_id: the width has to match the model it claims to be.
    #[error("expert_ids length {got} != top_k {expected} for model {model_id}")]
    TopKMismatch {
        model_id: String,
        got: usize,
        expected: i64,
    },

    /// An `expert_id` fell outside `0 .. n_experts` for that model.
    #[error("expert_id {expert_id} outside 0..{n_experts} for model {model_id}")]
    ExpertOutOfRange {
        model_id: String,
        expert_id: i64,
        n_experts: i64,
    },

    /// Rows carried a `source` that may not be filed under this `model_id`.
    ///
    /// Guards the base/Instruct split: `allenai_mixtral` publishes **base**
    /// Mixtral traces, so they must not land on an Instruct `model_id`.
    #[error("source {ingest_source} is not allowed for model_id {model_id} ({reason})")]
    SourceModelMismatch {
        model_id: String,
        ingest_source: String,
        reason: &'static str,
    },

    /// A JSON payload did not match the v1 scrape shape.
    #[error("malformed scrape json: {0}")]
    Json(#[from] serde_json::Error),

    /// Filesystem failure while reading a scrape or import directory.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

/// DDL for schema v1, applied in order inside one transaction.
pub const MIGRATION_1: &str = r#"
-- meta
CREATE TABLE meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
-- seed: schema_version = 1

-- one logical model (weights + arch), not a hostname
CREATE TABLE models (
  model_id    TEXT PRIMARY KEY,  -- e.g. mixtral-8x7b-instruct-q4_k_m
  arch        TEXT NOT NULL,     -- mixtral | qwen3moe | olmoe | ...
  n_layers    INTEGER NOT NULL,
  n_experts   INTEGER NOT NULL,
  top_k       INTEGER NOT NULL,
  hidden_size INTEGER NOT NULL,
  notes       TEXT
);

-- optional label for a job kind; missing -> class_id 'unknown'
CREATE TABLE request_classes (
  class_id TEXT PRIMARY KEY,
  label    TEXT NOT NULL
);

CREATE TABLE requests (
  request_id   TEXT PRIMARY KEY,
  model_id     TEXT NOT NULL REFERENCES models(model_id),
  class_id     TEXT NOT NULL DEFAULT 'unknown' REFERENCES request_classes(class_id),
  source       TEXT NOT NULL,  -- scrape | allenai_mixtral | allenai_olmoe | continuum_counts | ...
  phase_hint   TEXT,           -- prefill | decode | mixed | null
  n_tokens     INTEGER,        -- context length moved / scored; may be null on count-only imports
  created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

-- optional: token ids beside experts (gigatoken / engine). Not an embedding index.
CREATE TABLE request_tokens (
  request_id TEXT NOT NULL REFERENCES requests(request_id),
  pos        INTEGER NOT NULL,
  token_id   INTEGER NOT NULL,
  PRIMARY KEY (request_id, pos)
);

-- scraped (or imported) activations - source of truth for updates
CREATE TABLE activations (
  request_id TEXT NOT NULL REFERENCES requests(request_id),
  phase      TEXT NOT NULL,      -- prefill | decode
  layer      INTEGER NOT NULL,
  token_pos  INTEGER NOT NULL,
  rank       INTEGER NOT NULL,     -- 0 .. top_k-1
  expert_id  INTEGER NOT NULL,
  PRIMARY KEY (request_id, phase, layer, token_pos, rank)
);
CREATE INDEX idx_act_model_layer ON activations(layer, expert_id);

-- aggregated belief / histogram the scheduler will eventually read
CREATE TABLE histogram_counts (
  model_id   TEXT NOT NULL REFERENCES models(model_id),
  class_id   TEXT NOT NULL REFERENCES request_classes(class_id),
  phase      TEXT NOT NULL,
  layer      INTEGER NOT NULL,
  expert_id  INTEGER NOT NULL,
  count      INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (model_id, class_id, phase, layer, expert_id)
);

-- belief set predicted before the gate (for disagreement scoring)
CREATE TABLE beliefs (
  belief_id  TEXT PRIMARY KEY,
  request_id TEXT NOT NULL REFERENCES requests(request_id),
  layer      INTEGER NOT NULL,
  method     TEXT NOT NULL,       -- frequency | prev_token | bigram | ensemble | paper:...
  expert_ids TEXT NOT NULL,       -- JSON array of ints
  predicted_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE disagreements (
  belief_id     TEXT NOT NULL REFERENCES beliefs(belief_id),
  layer         INTEGER NOT NULL,
  overlap_k     INTEGER NOT NULL,  -- |predicted n actual|
  actual_ids    TEXT NOT NULL,     -- JSON array
  PRIMARY KEY (belief_id, layer)
);
"#;
