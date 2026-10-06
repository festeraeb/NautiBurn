//! B8 — Fleet predictor behind one interface.
//!
//! One ask → one fragment. Shaped like the fleet's `FleetPlanner` (host_key,
//! slot, model_id, reason) but specialised: the "slot" is the physical device
//! the predicted experts live on, and the fragment carries the prefetch
//! timing decision so the caller can see whether the copy is worth issuing.
//!
//! Mirrors the fleet rule: **no fabricated fragment.** An ask whose prediction
//! finds no resident device and no viable fallback returns `None` rather than
//! a fragment with empty fields to look satisfied.

use serde::Serialize;

use crate::schedule::{
    prefetch_decision, schedule_shadow, BeliefInput, DeviceMap, PrefetchDecision,
    ScheduleDecision,
};

/// Ask for an expert-bearing device. `role` labels the ask only — the shadow
/// box does not select *jobs* by role, it selects *devices by residence*.
#[derive(Debug, Clone)]
pub struct ExpertAsk {
    pub node_id: String,
    pub role: String,
    pub model_id: String,
}

/// One plan fragment for one ask. `host_key` is the chosen device; `slot`
/// names the same device so a fleet planner can hand it on unchanged.
#[derive(Debug, Clone, Serialize)]
pub struct ExpertPlanFragment {
    pub node_id: String,
    pub role: String,
    pub model_id: String,
    /// Chosen device (gpu id or cpu miss-path id).
    pub host_key: String,
    pub slot: String,
    /// Why this fragment was chosen — resident_hit / busy_skip / cpu_miss.
    pub reason: String,
    pub predicted_experts: Vec<i64>,
    /// Prefetch carry-over: did the copy fit the 13 ms planning budget.
    pub prefetch_reason: String,
    pub copy_us: f64,
    pub budget_us: f64,
    pub fits_in_budget: bool,
}

/// Produce the single fragment for `ask`, or `None` when no device can take
/// it. The fragment is a *plan*, not a dispatch — weights are not moved.
pub fn plan_experts(
    ask: &ExpertAsk,
    belief: &BeliefInput,
    map: &DeviceMap,
) -> Option<ExpertPlanFragment> {
    // Schedule first: where do the predicted experts currently live?
    let sched: ScheduleDecision = schedule_shadow(belief, map);

    // Prefetch check on the chosen device: is a copy even worthwhile?
    // Belief with by_layer intact is what prefetch_decision wants.
    let prefetch: PrefetchDecision =
        prefetch_decision(&belief_flat_ids(belief), map, None, None);

    // The empty-prediction case yields no plan — nothing to resident-check.
    if prefetch.reason == crate::schedule::PrefetchReason::EmptyPrediction {
        return None;
    }

    Some(ExpertPlanFragment {
        node_id: ask.node_id.clone(),
        role: ask.role.clone(),
        model_id: ask.model_id.clone(),
        host_key: sched.chosen_device.clone(),
        slot: sched.chosen_device,
        reason: sched.reason.as_str().to_string(),
        predicted_experts: sched.predicted_experts,
        prefetch_reason: prefetch.reason.as_str().to_string(),
        copy_us: prefetch.copy_us,
        budget_us: prefetch.budget_us,
        fits_in_budget: prefetch.fits_in_budget,
    })
}

/// The flat id list prefetch expects: by_layer ids if present, else the
/// top-level belief ids.
fn belief_flat_ids(belief: &BeliefInput) -> Vec<crate::schedule::PredictedExpert> {
    if !belief.by_layer.is_empty() {
        belief
            .by_layer
            .iter()
            .flat_map(|l| l.expert_ids.iter().map(|&id| (l.layer, id)))
            .map(|(layer, expert_id)| crate::schedule::PredictedExpert {
                layer: Some(layer),
                expert_id,
            })
            .collect()
    } else {
        belief
            .expert_ids
            .iter()
            .map(|&expert_id| crate::schedule::PredictedExpert {
                layer: None,
                expert_id,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schedule::{DeviceMap, DeviceRow};

    fn ask() -> ExpertAsk {
        ExpertAsk {
            node_id: "t440".into(),
            role: "coder".into(),
            model_id: "olmoe-toy".into(),
        }
    }

    fn belief(ids: &[i64]) -> BeliefInput {
        BeliefInput {
            request_id: "b8-req".into(),
            model_id: Some("olmoe-toy".into()),
            method: Some("ensemble".into()),
            expert_ids: ids.to_vec(),
            by_layer: vec![],
        }
    }

    fn gpu(id: &str, busy: bool, resident: &[i64]) -> DeviceRow {
        DeviceRow {
            device_id: id.into(),
            kind: "gpu".into(),
            busy,
            busy_reason: None,
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
            note: None,
        }
    }

    fn map(devices: Vec<DeviceRow>) -> DeviceMap {
        DeviceMap { stub: false, stub_note: None, devices }
    }

    #[test]
    fn resident_hit_fragment() {
        let f = plan_experts(&ask(), &belief(&[12, 30]), &map(vec![gpu("p100-free", false, &[12, 30]), cpu()]))
            .unwrap();
        assert_eq!(f.host_key, "p100-free");
        assert_eq!(f.reason, "resident_hit");
        assert!(!f.predicted_experts.is_empty());
    }

    #[test]
    fn busy_skip_falls_to_cpu_fragment() {
        let f = plan_experts(
            &ask(),
            &belief(&[12, 30]),
            &map(vec![gpu("tiel", true, &[12, 30]), cpu()]),
        )
        .unwrap();
        assert_eq!(f.host_key, "openvino-cpu");
        assert_eq!(f.reason, "busy_skip");
    }

    #[test]
    fn cpu_miss_when_nothing_holds() {
        let f = plan_experts(&ask(), &belief(&[99]), &map(vec![gpu("empty", false, &[]), cpu()]))
            .unwrap();
        assert_eq!(f.host_key, "openvino-cpu");
        assert_eq!(f.reason, "cpu_miss");
    }

    #[test]
    fn empty_prediction_yields_no_fragment() {
        let f = plan_experts(&ask(), &belief(&[]), &map(vec![cpu()]));
        assert!(f.is_none(), "empty prediction must not fabricate a fragment");
    }

    #[test]
    fn fragment_is_a_plan_not_a_dispatch() {
        let f = plan_experts(&ask(), &belief(&[1]), &map(vec![gpu("g0", false, &[1]), cpu()]))
            .unwrap();
        // The fragment names a target; schedule_shadow guarantees no dispatch,
        // and the fragment type has no "dispatched" field at all.
        assert!(!f.predicted_experts.is_empty());
        let _ = f.host_key;
    }
}