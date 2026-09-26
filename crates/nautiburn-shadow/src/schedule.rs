//! Shadow job-to-device scheduler.
//!
//! Product rule: send the **job** to a device that already holds the
//! predicted experts. Weights stay put. An idle GPU that does not hold
//! them is not a destination. CPU (OpenVINO miss-path stub) holds the
//! full model.
//!
//! This box **logs** the choice. It does not dispatch, does not SSH, and
//! does not move weights onto an idle GPU.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

/// Why the shadow box picked `chosen_device`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScheduleReason {
    /// Idle device already holds every predicted expert.
    ResidentHit,
    /// No device (busy or idle) holds the full predicted set. CPU miss path.
    CpuMiss,
    /// At least one device holds the set, but every such device is busy.
    /// The job is logged to CPU; it is not sent to the busy card.
    BusySkip,
}

impl ScheduleReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ResidentHit => "resident_hit",
            Self::CpuMiss => "cpu_miss",
            Self::BusySkip => "busy_skip",
        }
    }
}

/// One expert slab a device claims to hold. Layer is optional: a stub may
/// list ids only. That is not a placement policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResidentExpert {
    #[serde(default)]
    pub model_id: Option<String>,
    #[serde(default)]
    pub layer: Option<i64>,
    pub expert_id: i64,
}

/// Predicted experts the scheduler sees. Layer is optional when the belief
/// is a flat id list for the job.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PredictedExpert {
    #[serde(default)]
    pub layer: Option<i64>,
    pub expert_id: i64,
}

/// One row of a device→resident-expert map.
///
/// `holds_all` is the CPU miss path (full model resident). A GPU stub lists
/// whatever slabs it actually claims. Do **not** treat "first N experts" as
/// policy — cuts stay unlocked until scrapes say otherwise.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceRow {
    pub device_id: String,
    #[serde(default)]
    pub kind: String,
    #[serde(default)]
    pub busy: bool,
    #[serde(default)]
    pub busy_reason: Option<String>,
    /// Full-model resident (OpenVINO CPU miss path). Not a live engine.
    #[serde(default)]
    pub holds_all: bool,
    #[serde(default)]
    pub resident: Vec<ResidentExpert>,
    /// Flat expert ids (any layer). Stub convenience, not a lock.
    #[serde(default)]
    pub resident_expert_ids: Vec<i64>,
    #[serde(default)]
    pub note: Option<String>,
}

/// Device map file. `stub: true` means this is an example layout, not live
/// placement.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceMap {
    #[serde(default)]
    pub stub: bool,
    #[serde(default)]
    pub stub_note: Option<String>,
    pub devices: Vec<DeviceRow>,
}

/// Belief the scheduler consumes. Accepts a slim document or the
/// `named_belief` object written by `prepredict`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BeliefInput {
    pub request_id: String,
    #[serde(default)]
    pub model_id: Option<String>,
    #[serde(default)]
    pub method: Option<String>,
    #[serde(default)]
    pub expert_ids: Vec<i64>,
    #[serde(default)]
    pub by_layer: Vec<LayerExperts>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerExperts {
    pub layer: i64,
    pub expert_ids: Vec<i64>,
}

/// Logged shadow decision. `dispatched` is always false.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ScheduleDecision {
    pub request_id: String,
    pub predicted_experts: Vec<i64>,
    pub predicted_by_layer: Vec<PredictedExpert>,
    pub chosen_device: String,
    pub reason: ScheduleReason,
    pub skipped_busy: Vec<String>,
    pub dispatched: bool,
    pub weights_moved: bool,
    pub note: String,
}

const SHADOW_NOTE: &str = "shadow: logged only; not dispatched; weights not moved; no SSH";

/// Pick a device that already holds `belief` experts, else the CPU miss path.
///
/// Never marks `dispatched`. Busy live boxes (Tiel, Mixtral, coder) are
/// skipped; the job is not sent there.
pub fn schedule_shadow(belief: &BeliefInput, map: &DeviceMap) -> ScheduleDecision {
    let predicted = flatten_belief(belief);
    let predicted_ids: Vec<i64> = unique_ids(&predicted);

    let mut skipped_busy = Vec::new();
    let mut resident_idle: Option<&DeviceRow> = None;
    let mut resident_busy: Option<&DeviceRow> = None;
    let mut cpu: Option<&DeviceRow> = None;

    for dev in &map.devices {
        if dev.holds_all || dev.kind == "cpu" {
            if cpu.is_none() {
                cpu = Some(dev);
            }
            continue;
        }
        if !holds_all(dev, belief.model_id.as_deref(), &predicted) {
            continue;
        }
        if dev.busy {
            skipped_busy.push(dev.device_id.clone());
            if resident_busy.is_none() {
                resident_busy = Some(dev);
            }
            continue;
        }
        if resident_idle.is_none() {
            resident_idle = Some(dev);
        }
    }

    let (chosen, reason) = if let Some(dev) = resident_idle {
        (dev.device_id.clone(), ScheduleReason::ResidentHit)
    } else if resident_busy.is_some() {
        let fallback = cpu
            .map(|d| d.device_id.clone())
            .unwrap_or_else(|| "openvino-cpu".into());
        (fallback, ScheduleReason::BusySkip)
    } else {
        let fallback = cpu
            .map(|d| d.device_id.clone())
            .unwrap_or_else(|| "openvino-cpu".into());
        (fallback, ScheduleReason::CpuMiss)
    };

    ScheduleDecision {
        request_id: belief.request_id.clone(),
        predicted_experts: predicted_ids,
        predicted_by_layer: predicted,
        chosen_device: chosen,
        reason,
        skipped_busy,
        dispatched: false,
        weights_moved: false,
        note: SHADOW_NOTE.to_string(),
    }
}

fn flatten_belief(belief: &BeliefInput) -> Vec<PredictedExpert> {
    let mut out = Vec::new();
    for layer in &belief.by_layer {
        for &expert_id in &layer.expert_ids {
            out.push(PredictedExpert {
                layer: Some(layer.layer),
                expert_id,
            });
        }
    }
    if out.is_empty() {
        for &expert_id in &belief.expert_ids {
            out.push(PredictedExpert {
                layer: None,
                expert_id,
            });
        }
    }
    out
}

fn unique_ids(pred: &[PredictedExpert]) -> Vec<i64> {
    let mut set = BTreeSet::new();
    for p in pred {
        set.insert(p.expert_id);
    }
    set.into_iter().collect()
}

fn holds_all(dev: &DeviceRow, model_id: Option<&str>, predicted: &[PredictedExpert]) -> bool {
    if dev.holds_all {
        return true;
    }
    if predicted.is_empty() {
        return false;
    }
    predicted.iter().all(|p| holds_one(dev, model_id, p))
}

fn holds_one(dev: &DeviceRow, model_id: Option<&str>, pred: &PredictedExpert) -> bool {
    if dev.resident_expert_ids.contains(&pred.expert_id) {
        return true;
    }
    dev.resident.iter().any(|r| {
        if r.expert_id != pred.expert_id {
            return false;
        }
        if let (Some(want), Some(have)) = (model_id, r.model_id.as_deref()) {
            if want != have {
                return false;
            }
        }
        match (pred.layer, r.layer) {
            (Some(pl), Some(rl)) => pl == rl,
            _ => true,
        }
    })
}

/// Parse a belief file: slim [`BeliefInput`], or a prepredict report that
/// carries `named_belief`.
pub fn parse_belief_json(text: &str) -> Result<BeliefInput, String> {
    if let Ok(b) = serde_json::from_str::<BeliefInput>(text) {
        if !b.request_id.is_empty()
            && (!b.expert_ids.is_empty() || !b.by_layer.is_empty())
        {
            return Ok(b);
        }
    }
    let v: serde_json::Value =
        serde_json::from_str(text).map_err(|e| format!("belief json: {e}"))?;
    if let Some(named) = v.get("named_belief") {
        return serde_json::from_value(named.clone())
            .map_err(|e| format!("named_belief: {e}"));
    }
    serde_json::from_value(v).map_err(|e| format!("belief: {e}"))
}

pub fn format_decision(d: &ScheduleDecision) -> String {
    let experts = d
        .predicted_experts
        .iter()
        .map(|e| e.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let skipped = d.skipped_busy.join(",");
    format!(
        "request_id\t{}\npredicted_experts\t[{}]\nchosen_device\t{}\nreason\t{}\nskipped_busy\t[{}]\ndispatched\tno\nweights_moved\tno\nnote\t{}\n",
        d.request_id,
        experts,
        d.chosen_device,
        d.reason.as_str(),
        skipped,
        d.note
    )
}

/// Built-in stub map shipped with the crate. Example layout only.
pub fn stub_device_map() -> DeviceMap {
    serde_json::from_str(include_str!("../fixtures/devices.stub.json"))
        .expect("devices.stub.json")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn belief(ids: &[i64]) -> BeliefInput {
        BeliefInput {
            request_id: "req-sched".into(),
            model_id: Some("olmoe-toy".into()),
            method: Some("ensemble".into()),
            expert_ids: ids.to_vec(),
            by_layer: vec![],
        }
    }

    fn map(devices: Vec<DeviceRow>) -> DeviceMap {
        DeviceMap {
            stub: true,
            stub_note: Some("test".into()),
            devices,
        }
    }

    fn gpu(id: &str, busy: bool, resident: &[i64]) -> DeviceRow {
        DeviceRow {
            device_id: id.into(),
            kind: "gpu".into(),
            busy,
            busy_reason: if busy {
                Some("tiel-coder :5230".into())
            } else {
                None
            },
            holds_all: false,
            resident: vec![],
            resident_expert_ids: resident.to_vec(),
            note: None,
        }
    }

    fn cpu() -> DeviceRow {
        DeviceRow {
            device_id: "openvino-cpu".into(),
            kind: "cpu".into(),
            busy: false,
            busy_reason: None,
            holds_all: true,
            resident: vec![],
            resident_expert_ids: vec![],
            note: Some("OpenVINO CPU miss path stub".into()),
        }
    }

    #[test]
    fn resident_hit_when_idle_gpu_holds_the_set() {
        let d = schedule_shadow(
            &belief(&[12, 30]),
            &map(vec![gpu("p100-free", false, &[12, 30, 43]), cpu()]),
        );
        assert_eq!(d.reason, ScheduleReason::ResidentHit);
        assert_eq!(d.chosen_device, "p100-free");
        assert!(!d.dispatched);
        assert!(!d.weights_moved);
        assert_eq!(d.predicted_experts, vec![12, 30]);
    }

    #[test]
    fn busy_skip_does_not_send_to_tiel_or_mixtral() {
        let d = schedule_shadow(
            &belief(&[12, 30]),
            &map(vec![
                gpu("t440-tiel-coder", true, &[12, 30]),
                gpu("cesarops4-mixtral", true, &[12, 30]),
                cpu(),
            ]),
        );
        assert_eq!(d.reason, ScheduleReason::BusySkip);
        assert_eq!(d.chosen_device, "openvino-cpu");
        assert!(d.skipped_busy.contains(&"t440-tiel-coder".into()));
        assert!(d.skipped_busy.contains(&"cesarops4-mixtral".into()));
        assert!(!d.dispatched);
    }

    #[test]
    fn cpu_miss_when_no_gpu_holds_the_set() {
        let d = schedule_shadow(
            &belief(&[99, 100]),
            &map(vec![gpu("p100-partial", false, &[12, 30]), cpu()]),
        );
        assert_eq!(d.reason, ScheduleReason::CpuMiss);
        assert_eq!(d.chosen_device, "openvino-cpu");
        assert!(!d.dispatched);
    }

    #[test]
    fn idle_gpu_without_the_set_is_not_a_destination() {
        let d = schedule_shadow(
            &belief(&[5, 7]),
            &map(vec![gpu("idle-empty", false, &[1, 2]), cpu()]),
        );
        assert_eq!(d.reason, ScheduleReason::CpuMiss);
        assert_eq!(d.chosen_device, "openvino-cpu");
    }

    #[test]
    fn stub_map_is_documented_and_not_first_seven_policy() {
        let m = stub_device_map();
        assert!(m.stub);
        let note = m.stub_note.unwrap_or_default();
        assert!(note.to_ascii_lowercase().contains("stub"));
        assert!(!note.contains("first seven"));
        assert!(m.devices.iter().any(|d| d.holds_all && d.kind == "cpu"));
        assert!(m.devices.iter().any(|d| d.busy && d.device_id.contains("tiel")));
        assert!(m.devices.iter().any(|d| d.busy && d.device_id.contains("mixtral")));
        for d in &m.devices {
            if !d.holds_all {
                let ids = if d.resident_expert_ids.is_empty() {
                    d.resident.iter().map(|r| r.expert_id).collect::<Vec<_>>()
                } else {
                    d.resident_expert_ids.clone()
                };
                assert_ne!(
                    ids,
                    vec![0, 1, 2, 3, 4, 5, 6],
                    "stub must not encode first-seven as policy"
                );
            }
        }
    }

    #[test]
    fn parse_belief_from_named_belief_wrapper() {
        let json = r#"{
            "named_belief": {
                "request_id": "nb-1",
                "model_id": "olmoe-toy",
                "method": "ensemble",
                "expert_ids": [3, 1],
                "by_layer": [{"layer": 1, "expert_ids": [3, 1]}]
            }
        }"#;
        let b = parse_belief_json(json).unwrap();
        assert_eq!(b.request_id, "nb-1");
        assert_eq!(b.expert_ids, vec![3, 1]);
    }
}
