//! Expert-activation scrape.
//!
//! Records which experts fired, and the dtype the device policy scheduled.
//! Hot, warm, and cold placement is not decided here. Example layouts are
//! not constants. A later histogram, per model, is what would name them.
//! `k` is an argument. OLMoE's router width lives on `ModelProfile::OLMOE`
//! as the model config.

use serde::{Deserialize, Serialize};

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
}
