//! Parsers for published traces named in the histogram-database handoff.
//!
//! - `allenai/analysis_olmoe` / `allenai/analysis_mixtral`: C4 JSONL with
//!   `input_ids` + `exp_ids` shaped `[n_tokens][top_k][n_layers]`.
//! - continuum Instruct `importance.activation_count.json`: layer → count
//!   vector, into `histogram_counts` only.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::schema::{ActivationRow, HistDbError, ScrapeRecord, SCHEMA_VERSION};

/// Default `model_id` for AllenAI OLMoE C4 traces (not Instruct).
pub const ALLENAI_OLMOE_MODEL_ID: &str = "olmoe-1b-7b-0924";

/// Default `model_id` for AllenAI Mixtral-8x7B **base** C4 traces.
pub const ALLENAI_MIXTRAL_BASE_MODEL_ID: &str = "mixtral-8x7b-v0.1";

/// Default `model_id` when a continuum file omits `model`.
pub const CONTINUUM_MIXTRAL_INSTRUCT_MODEL_ID: &str = "mixtral-8x7b-instruct-v0.1";

#[derive(Debug, Deserialize)]
struct AllenAiLine {
    #[serde(default)]
    input_ids: Option<Vec<i64>>,
    exp_ids: Vec<Vec<Vec<i64>>>,
}

#[derive(Debug, Deserialize)]
struct ContinuumFile {
    #[serde(default)]
    model: Option<String>,
    activation_counts: BTreeMap<String, Vec<i64>>,
}

/// One continuum cell: layer, expert_id, count.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContinuumCell {
    pub layer: i64,
    pub expert_id: i64,
    pub count: i64,
}

/// Parsed continuum Instruct count histogram.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContinuumCounts {
    pub model_id: String,
    pub cells: Vec<ContinuumCell>,
}

/// JSONL files under `path`, or `path` itself when it is a `.jsonl` file.
pub fn allenai_jsonl_paths(path: impl AsRef<Path>) -> Result<Vec<PathBuf>, HistDbError> {
    let path = path.as_ref();
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }
    let mut files: Vec<_> = std::fs::read_dir(path)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "jsonl"))
        .collect();
    files.sort();
    Ok(files)
}

/// Turn one AllenAI C4 JSONL object into a v1 scrape (prefill; class `c4`).
pub fn scrape_from_allenai_line(
    line: &str,
    request_id: String,
    model_id: String,
    source: String,
) -> Result<ScrapeRecord, HistDbError> {
    if let Ok(rec) = serde_json::from_str::<ScrapeRecord>(line) {
        return Ok(rec);
    }

    let row: AllenAiLine = serde_json::from_str(line)?;
    if row.exp_ids.is_empty() {
        return Err(HistDbError::AllenAiShape {
            detail: "empty exp_ids".into(),
        });
    }
    let n_tokens = row.exp_ids.len();
    let top_k = row.exp_ids[0].len();
    if top_k == 0 {
        return Err(HistDbError::AllenAiShape {
            detail: "exp_ids[0] has no ranks".into(),
        });
    }
    let n_layers = row.exp_ids[0][0].len();
    if n_layers == 0 {
        return Err(HistDbError::AllenAiShape {
            detail: "exp_ids[0][0] has no layers".into(),
        });
    }
    for (t, token) in row.exp_ids.iter().enumerate() {
        if token.len() != top_k {
            return Err(HistDbError::AllenAiShape {
                detail: format!("token {t} rank width {} != {top_k}", token.len()),
            });
        }
        for (r, ranks) in token.iter().enumerate() {
            if ranks.len() != n_layers {
                return Err(HistDbError::AllenAiShape {
                    detail: format!("token {t} rank {r} layers {} != {n_layers}", ranks.len()),
                });
            }
        }
    }

    let mut activations = Vec::with_capacity(n_tokens * n_layers);
    for (token_pos, token) in row.exp_ids.iter().enumerate() {
        for layer in 0..n_layers {
            let mut expert_ids = Vec::with_capacity(top_k);
            for rank in token {
                expert_ids.push(rank[layer]);
            }
            activations.push(ActivationRow {
                layer: layer as i64,
                token_pos: token_pos as i64,
                expert_ids,
            });
        }
    }

    Ok(ScrapeRecord {
        schema_version: SCHEMA_VERSION,
        model_id,
        request_id,
        class_id: Some("c4".into()),
        source,
        phase: "prefill".into(),
        n_tokens: Some(n_tokens as i64),
        dtype: None,
        activations,
        token_ids: row.input_ids,
    })
}

/// Parse continuum `importance.activation_count.json`.
pub fn parse_continuum_counts(json: &str) -> Result<ContinuumCounts, HistDbError> {
    let file: ContinuumFile = serde_json::from_str(json)?;
    if file.activation_counts.is_empty() {
        return Err(HistDbError::Continuum("activation_counts is empty".into()));
    }
    let model_id = file
        .model
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| CONTINUUM_MIXTRAL_INSTRUCT_MODEL_ID.to_string());

    let mut cells = Vec::new();
    for (layer_key, counts) in file.activation_counts {
        let layer: i64 = layer_key
            .parse()
            .map_err(|_| HistDbError::Continuum(format!("bad layer key {layer_key}")))?;
        for (expert_id, count) in counts.into_iter().enumerate() {
            if count != 0 {
                cells.push(ContinuumCell {
                    layer,
                    expert_id: expert_id as i64,
                    count,
                });
            }
        }
    }
    Ok(ContinuumCounts { model_id, cells })
}
