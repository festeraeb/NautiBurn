//! Expert-activation scrape.
//!
//! Records which experts fired, and the dtype the device policy scheduled.
//! Hot, warm, and cold placement is not decided here. Example layouts are
//! not constants. A later histogram, per model, is what would name them.
//! `k` is an argument. OLMoE's router width lives on `ModelProfile::OLMOE`
//! as the model config.
//!
//! On-disk JSON matches the histogram-database handoff schema (v1). This
//! crate writes the records. It does not build histodb.

use serde::{Deserialize, Serialize};
use std::fs;
use std::io;
use std::path::Path;

/// Dtype a device is asked to run, or the dtype the policy scheduled.
/// `F16` is not rewritten to `F32`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScheduledDtype {
    #[serde(rename = "f16")]
    F16,
    #[serde(rename = "f32")]
    F32,
}

impl ScheduledDtype {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::F32 => "f32",
        }
    }
}

/// Device class for the dtype policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceClass {
    /// P100, `sm_60`, or a `pascal` tag. This class schedules `f16`.
    Pascal,
    /// Any other device. The requested dtype is returned unchanged.
    Unknown,
}

/// Classifies a device tag. `p100`, `sm_60`, and `pascal` are Pascal.
/// Every other tag is unknown.
pub fn device_class(tag: &str) -> DeviceClass {
    let lower = tag.to_ascii_lowercase();
    let squashed: String = lower
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect();
    if lower.contains("p100") || lower.contains("pascal") || squashed.contains("sm60") {
        DeviceClass::Pascal
    } else {
        DeviceClass::Unknown
    }
}

/// P100 / `sm_60` / `pascal` schedules `f16`.
/// An unknown device schedules `f16` only when `requested` is `f16`.
/// Any other request is returned unchanged, including `f32`.
pub fn schedule_dtype(device: DeviceClass, requested: ScheduledDtype) -> ScheduledDtype {
    match device {
        DeviceClass::Pascal => ScheduledDtype::F16,
        DeviceClass::Unknown => requested,
    }
}

/// One scraped activation: where it fired, which experts, and the dtype
/// that was scheduled for that device.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActivationRecord {
    pub layer: u16,
    pub token_index: u32,
    pub expert_ids: Vec<u16>,
    pub dtype: ScheduledDtype,
}

/// One token's expert ids at one layer. Handoff field name is `token_pos`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActivationRow {
    pub layer: u16,
    pub token_pos: u32,
    pub expert_ids: Vec<u16>,
}

/// Scrape JSON v1. Same shape the histogram database will ingest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScrapeDocument {
    pub schema_version: u32,
    pub model_id: String,
    pub request_id: String,
    pub class_id: String,
    pub source: String,
    pub phase: String,
    pub n_tokens: u32,
    pub dtype: String,
    pub activations: Vec<ActivationRow>,
    pub token_ids: Vec<u32>,
}

impl ScrapeDocument {
    pub fn new(
        model_id: impl Into<String>,
        request_id: impl Into<String>,
        class_id: impl Into<String>,
        phase: impl Into<String>,
        dtype: impl Into<String>,
        activations: Vec<ActivationRow>,
        token_ids: Vec<u32>,
    ) -> Self {
        let token_ids = token_ids;
        let n_tokens = token_ids.len() as u32;
        Self {
            schema_version: 1,
            model_id: model_id.into(),
            request_id: request_id.into(),
            class_id: class_id.into(),
            source: "scrape".to_string(),
            phase: phase.into(),
            n_tokens,
            dtype: dtype.into(),
            activations,
            token_ids,
        }
    }
}

/// Writes one scrape document. Parent directory is created if needed.
pub fn write_scrape(path: impl AsRef<Path>, doc: &ScrapeDocument) -> io::Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let text = serde_json::to_string_pretty(doc)
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
    fs::write(path, text)?;
    Ok(())
}

/// Records an activation. `expert_ids` are stored as given. `dtype` is the
/// dtype that was scheduled, not a later rewrite.
pub fn record_activation(
    layer: u16,
    token_index: u32,
    expert_ids: &[u16],
    dtype: ScheduledDtype,
) -> ActivationRecord {
    ActivationRecord {
        layer,
        token_index,
        expert_ids: expert_ids.to_vec(),
        dtype,
    }
}

/// Expert ids with the highest router scores. `k` is the caller's.
/// Ties break toward the smaller expert id. This is not a placement rule.
pub fn top_k_expert_ids(scores: &[f32], k: usize) -> Vec<u16> {
    let mut ids: Vec<u16> = (0..scores.len())
        .take(u16::MAX as usize)
        .map(|i| i as u16)
        .collect();
    ids.sort_by(|&a, &b| {
        scores[b as usize]
            .partial_cmp(&scores[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    ids.truncate(k.min(ids.len()));
    ids
}

/// Scores a fake or real router vector, keeps the caller's top `k`, and
/// records those ids with the dtype [`schedule_dtype`] selected.
pub fn scrape_router_activation(
    layer: u16,
    token_index: u32,
    scores: &[f32],
    k: usize,
    device: DeviceClass,
    requested: ScheduledDtype,
) -> ActivationRecord {
    let dtype = schedule_dtype(device, requested);
    let expert_ids = top_k_expert_ids(scores, k);
    record_activation(layer, token_index, &expert_ids, dtype)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pascal_schedules_f16_and_a_caller_can_ask_for_f32() {
        assert_eq!(device_class("pascal"), DeviceClass::Pascal);
        assert_eq!(device_class("P100"), DeviceClass::Pascal);
        assert_eq!(device_class("sm_60"), DeviceClass::Pascal);
        assert_eq!(device_class("SM60"), DeviceClass::Pascal);
        assert_eq!(
            schedule_dtype(device_class("pascal"), ScheduledDtype::F32),
            ScheduledDtype::F16
        );
        assert_eq!(
            schedule_dtype(device_class("sm_60"), ScheduledDtype::F16),
            ScheduledDtype::F16
        );
        assert_eq!(
            schedule_dtype(DeviceClass::Unknown, ScheduledDtype::F32),
            ScheduledDtype::F32
        );
        assert_eq!(
            schedule_dtype(device_class("radeon"), ScheduledDtype::F16),
            ScheduledDtype::F16
        );
        assert_eq!(
            schedule_dtype(device_class("radeon"), ScheduledDtype::F32),
            ScheduledDtype::F32
        );
    }

    #[test]
    fn scrape_records_the_callers_top_k() {
        let scores = [0.1_f32, 0.4, 0.05, 0.9, 0.2];
        let recorded =
            scrape_router_activation(3, 4, &scores, 2, DeviceClass::Pascal, ScheduledDtype::F32);
        assert_eq!(recorded.layer, 3);
        assert_eq!(recorded.token_index, 4);
        assert_eq!(recorded.expert_ids, vec![3, 1]);
        assert_eq!(recorded.dtype, ScheduledDtype::F16);

        let wider =
            scrape_router_activation(3, 4, &scores, 4, DeviceClass::Unknown, ScheduledDtype::F32);
        assert_eq!(wider.expert_ids, vec![3, 1, 4, 0]);
        assert_eq!(wider.dtype, ScheduledDtype::F32);
        assert_ne!(wider.expert_ids.len(), recorded.expert_ids.len());
    }

    #[test]
    fn scrape_document_matches_handoff_shape() {
        let doc = ScrapeDocument::new(
            "olmoe-1b-7b-0924-f16",
            "req-1",
            "olmoe-layer0",
            "prefill",
            "f16",
            vec![ActivationRow {
                layer: 0,
                token_pos: 0,
                expert_ids: vec![3, 1, 7, 2, 9, 4, 11, 0],
            }],
            vec![1, 2, 3],
        );
        assert_eq!(doc.schema_version, 1);
        assert_eq!(doc.source, "scrape");
        assert_eq!(doc.dtype, "f16");
        assert_eq!(doc.n_tokens, 3);
        assert_eq!(doc.activations[0].token_pos, 0);
        let text = serde_json::to_string(&doc).unwrap();
        assert!(text.contains("\"token_pos\""));
        assert!(text.contains("\"expert_ids\""));
        assert!(!text.contains("token_index"));
        let back: ScrapeDocument = serde_json::from_str(&text).unwrap();
        assert_eq!(back.activations[0].expert_ids, doc.activations[0].expert_ids);
    }
}
