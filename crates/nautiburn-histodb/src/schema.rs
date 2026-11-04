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
pub const DB_SCHEMA_VERSION: i64 = 5;

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

    // ── prompt awareness (added 2026-11-04) ──
    //
    // All optional, so a v1 producer that knows none of this still imports
    // unchanged. `deny_unknown_fields` is on this struct, so a producer must
    // not emit these until the ingest that understands them is deployed.
    /// The prompt text. A caller unwilling to store user content may omit
    /// this and send `prompt_sha256` alone.
    #[serde(default)]
    pub prompt: Option<String>,
    /// Hash of the prompt **text**. Computed from `prompt` when absent.
    ///
    /// Never hash token ids: they differ per tokenizer, which splits one
    /// prompt into one identity per model and destroys the cross-model
    /// comparison this corpus exists for.
    #[serde(default)]
    pub prompt_sha256: Option<String>,
    /// Forge run that caused this request.
    #[serde(default)]
    pub forge_run_id: Option<String>,
    /// Forge role: coder, reviewer, injector, thinker, ...
    #[serde(default)]
    pub forge_role: Option<String>,
    /// Task domain, when the caller knows it independently of `class_id`.
    #[serde(default)]
    pub task_domain: Option<String>,
}

impl ScrapeRecord {
    /// The prompt hash to store: the explicit one if given, else derived from
    /// the text. `None` when the caller supplied neither.
    pub fn resolved_prompt_sha256(&self) -> Option<String> {
        if let Some(h) = self.prompt_sha256.as_ref().filter(|h| !h.trim().is_empty()) {
            return Some(h.trim().to_lowercase());
        }
        self.prompt
            .as_ref()
            .filter(|p| !p.is_empty())
            .map(|p| sha256_hex(p.as_bytes()))
    }
}

/// SHA-256 as lowercase hex. Small local implementation so the crate does not
/// take a hashing dependency for one field.
pub fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut msg = data.to_vec();
    let bitlen = (data.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bitlen.to_be_bytes());
    for chunk in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                chunk[i * 4], chunk[i * 4 + 1], chunk[i * 4 + 2], chunk[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g; g = f; f = e;
            e = d.wrapping_add(t1);
            d = c; c = b; b = a;
            a = t1.wrapping_add(t2);
        }
        for (i, v) in [a, b, c, d, e, f, g, hh].iter().enumerate() {
            h[i] = h[i].wrapping_add(*v);
        }
    }
    h.iter().map(|x| format!("{x:08x}")).collect()
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
    /// A caller supplied data the envelope cannot accept (empty run_id, a
    /// divergence hop with no predictor version, a hop for an unopened run).
    #[error("invalid data: {0}")]
    InvalidData(String),

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

/// Prompt awareness + forge handoff linkage on `requests`.
pub const MIGRATION_4: &str = include_str!("../migrations/004_request_prompt.sql");

/// The run envelope: per-run provenance with one appended hop per card.
pub const MIGRATION_5: &str = include_str!("../migrations/005_run_envelope.sql");
