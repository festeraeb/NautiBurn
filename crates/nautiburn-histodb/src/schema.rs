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

/// Scrape JSON v1 (`schema_version` on the posted document).
pub const SCHEMA_VERSION: i64 = 1;

/// Database schema after numbered migrations. Scrape JSON stays at [`SCHEMA_VERSION`].
pub const DB_SCHEMA_VERSION: i64 = 3;

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
    /// Missing source is a live scrape. Public traces set this explicitly.
    #[serde(default = "default_scrape_source")]
    pub source: String,
    pub phase: String,
    #[serde(default)]
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
    /// Predict latency in microseconds. Missing on beliefs recorded before
    /// the shadow box. Not a second schema: same `beliefs` row.
    pub latency_us: Option<i64>,
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

    /// `record_disagreement` named a `belief_id` that is not in `beliefs`.
    #[error("unknown belief_id {0}")]
    UnknownBelief(String),

    /// AllenAI `exp_ids` was not `[n_tokens][top_k][n_layers]`.
    #[error("allenai exp_ids {detail}")]
    AllenAiShape { detail: String },

    /// Continuum count file was missing `activation_counts` or had a bad layer key.
    #[error("continuum counts: {0}")]
    Continuum(String),
}

fn default_scrape_source() -> String {
    "scrape".to_string()
}

/// DDL for schema v1. Authoritative numbered file: `migrations/001_schema_v1.sql`.
pub const MIGRATION_1: &str = include_str!("../migrations/001_schema_v1.sql");

/// Adds `requests.dtype`. File: `migrations/002_requests_dtype.sql`.
pub const MIGRATION_2: &str = include_str!("../migrations/002_requests_dtype.sql");

/// Adds `beliefs.latency_us`. File: `migrations/003_beliefs_latency.sql`.
pub const MIGRATION_3: &str = include_str!("../migrations/003_beliefs_latency.sql");
