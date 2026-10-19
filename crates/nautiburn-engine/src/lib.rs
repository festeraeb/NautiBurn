//! Burn-side expert engine.
//!
//! The first backend is WGPU. On a P100 the scheduled compute dtype is FP16.
//! A launch whose shared memory is above 49152 bytes is refused. It is not
//! rewritten as FP32. The GEMM call site stays `Tensor::matmul` in
//! `nautiburn-layer0` (`gemm_cubecl`). This crate is the activation scrape,
//! the hook, and the JSON handoff in front of that call.
//!
//! Router `k` on [`ModelProfile::OLMOE`] is the OLMoE model config. It is
//! not a hot-expert count. The scrape takes `k` from the caller.
//!
//! The hook fields follow MoE-Beyond (`Layer ID`, batch, token, activated
//! expert ids). The layer-by-expert counts are their expert activation
//! matrix, split into prefill and decode. moe-viz reads the same per-token
//! expert list this hook stores.

mod http;
mod scrape;

pub use http::{router, router_with_histodb, serve, serve_with_router, shared, Shared};
pub use scrape::{
    device_class, record_activation, schedule_dtype, scrape_router_activation, top_k_expert_ids,
    write_scrape, ActivationRecord, ActivationRow, DeviceClass, ScheduledDtype, ScrapeDocument,
};

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// P100 shared-memory cap. The CubeCL FP16 expert GEMM asked for 69632 and
/// the card refused it.
pub const P100_SHARED_CAP_BYTES: u32 = 49_152;
/// Stock CubeCL plane/CMMA FP16 blueprint. The selector ignores
/// `max_shared_memory_size` and still asks for this. Do not launch it.
pub const CUBECL_F16_SHARED_BYTES: u32 = 69_632;
/// Forced unit tile: BM=BN=16, BK=32, two FP16 stages.
/// `4 * BK * (BM + BN)` = 4096, under the P100 49152 cap.
/// 64×64×32 (16384) is the paper candidate; this 16×16×32 launch is the
/// one that actually compiles as a CubeCL unit kernel without the plane/CMMA
/// 69632 path.
pub const FORCED_TILE_BM: u32 = 16;
pub const FORCED_TILE_BN: u32 = 16;
pub const FORCED_TILE_BK: u32 = 32;
pub const FORCED_TILE_SHARED_BYTES: u32 =
    4 * FORCED_TILE_BK * (FORCED_TILE_BM + FORCED_TILE_BN);

/// Planning copy rate from the engine plan. Not a measured memcpy.
pub const PLANNING_COPY_BYTES_PER_SEC: f64 = 12e9;
/// Rough Gen3 PCIe x16 budget for moving expert weights onto a card.
/// The prediction has to be this far ahead, or the hot experts have to
/// already be resident so only a small layer is loaded.
pub const TARGET_AHEAD_MS: f64 = 13.0;

/// One OLMoE expert, FP16, three projections: `3 * 2048 * 1024 * 2`.
pub const OLMOE_EXPERT_FP16_BYTES: u64 = 12_582_912;
pub const OLMOE_HIDDEN: usize = 2048;

/// Plan figure for two Mixtral Q4 experts, decimal megabytes, not a measurement.
pub const MIXTRAL_Q4_TWO_EXPERT_BYTES: u64 = 200_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp16Launch {
    Admitted,
    Refused { shared_bytes: u32, cap_bytes: u32 },
}

/// P100 compute stays FP16. Shared memory above the cap refuses the launch.
pub fn admit_fp16_on_p100(shared_bytes: u32) -> Fp16Launch {
    if shared_bytes > P100_SHARED_CAP_BYTES {
        Fp16Launch::Refused {
            shared_bytes,
            cap_bytes: P100_SHARED_CAP_BYTES,
        }
    } else {
        Fp16Launch::Admitted
    }
}

pub fn scheduled_compute_dtype() -> &'static str {
    schedule_dtype(DeviceClass::Pascal, ScheduledDtype::F16).as_str()
}

pub fn copy_ms(bytes: u64) -> f64 {
    bytes as f64 / PLANNING_COPY_BYTES_PER_SEC * 1_000.0
}

pub fn within_target(bytes: u64) -> bool {
    copy_ms(bytes) <= TARGET_AHEAD_MS
}

#[derive(Debug, Clone, Copy)]
pub struct ModelProfile {
    pub name: &'static str,
    pub layers: usize,
    pub experts: usize,
    /// Router top-k from the model config. Not a hot-expert placement count.
    pub k: usize,
    pub hidden: usize,
    pub expert_bytes: u64,
}

impl ModelProfile {
    /// OLMoE-1B-7B. `k` is that model's router width from the model config.
    /// It is not a hot, warm, or cold count.
    pub const OLMOE: Self = Self {
        name: "olmoe-1b-7b",
        layers: 16,
        experts: 64,
        k: 8,
        hidden: OLMOE_HIDDEN,
        expert_bytes: OLMOE_EXPERT_FP16_BYTES,
    };

    pub fn activation_bytes(&self, context_tokens: usize) -> u64 {
        (context_tokens * self.hidden * 2) as u64
    }

    pub fn slab_bytes(&self, n_experts: usize) -> u64 {
        self.expert_bytes.saturating_mul(n_experts as u64)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    pub phase: String,
    pub layer: u16,
    pub batch: u32,
    pub token_index: u32,
    pub token_id: u32,
    pub expert_ids: Vec<u16>,
    #[serde(default)]
    pub expert_weights: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertPass {
    pub job_id: String,
    pub task_class: String,
    pub phase: String,
    pub layer: u16,
    pub token_index: u32,
    pub token_id: u32,
    pub expert_ids: Vec<u16>,
    pub context_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NextExperts {
    pub job_id: String,
    pub compute_dtype: String,
    pub device: String,
    pub known_cubecl_f16_gemm: String,
    pub from_layer: u16,
    pub to_layer: u16,
    pub expert_ids: Vec<u16>,
    pub source: String,
    pub activation_bytes: u64,
    pub slab_bytes: u64,
    pub copy_ms_estimate: f64,
    pub copy_ms_basis: String,
    pub target_ahead_ms: f64,
    pub within_13ms: bool,
    /// Scales for the experts this handoff names, from the OLMoE weight table.
    /// Empty when no calibration is loaded.
    #[serde(default)]
    pub expert_calibration: Vec<ExpertScale>,
}

/// One expert's weight calibration. The checkpoint does not ship this.
/// `router_l2` is the L2 of that expert's `ffn_gate_inp` row.
/// The RMS fields are the FP16 gate, up, and down projections.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertScale {
    pub layer: u16,
    pub expert: u16,
    pub router_l2: f64,
    pub gate_rms: f64,
    pub up_rms: f64,
    pub down_rms: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct CalibrationFile {
    layers: usize,
    experts: usize,
    rows: Vec<ExpertScale>,
}

#[derive(Debug, Clone)]
pub struct Calibration {
    layers: usize,
    experts: usize,
    rows: Vec<ExpertScale>,
}

impl Calibration {
    pub fn from_json(text: &str) -> Result<Self, String> {
        let file: CalibrationFile = serde_json::from_str(text).map_err(|err| err.to_string())?;
        if file.layers == 0 || file.experts == 0 {
            return Err("calibration has no layers or experts".to_string());
        }
        let mut rows = Vec::with_capacity(file.layers * file.experts);
        rows.resize(
            file.layers * file.experts,
            ExpertScale {
                layer: 0,
                expert: 0,
                router_l2: 0.0,
                gate_rms: 0.0,
                up_rms: 0.0,
                down_rms: 0.0,
            },
        );
        let mut seen = vec![false; rows.len()];
        for row in file.rows {
            let layer = row.layer as usize;
            let expert = row.expert as usize;
            if layer >= file.layers || expert >= file.experts {
                return Err(format!("calibration row {layer},{expert} is outside the table"));
            }
            let index = layer * file.experts + expert;
            if seen[index] {
                return Err(format!("duplicate calibration row {layer},{expert}"));
            }
            seen[index] = true;
            rows[index] = row;
        }
        if seen.iter().any(|slot| !*slot) {
            return Err("calibration is missing a layer/expert row".to_string());
        }
        Ok(Self {
            layers: file.layers,
            experts: file.experts,
            rows,
        })
    }

    pub fn get(&self, layer: usize, expert: usize) -> Option<&ExpertScale> {
        if layer >= self.layers || expert >= self.experts {
            return None;
        }
        Some(&self.rows[layer * self.experts + expert])
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    fn top_router(&self, layer: usize, k: usize) -> Vec<u16> {
        let mut order: Vec<u16> = (0..self.experts as u16).collect();
        order.sort_by(|a, b| {
            let la = self.rows[layer * self.experts + *a as usize].router_l2;
            let lb = self.rows[layer * self.experts + *b as usize].router_l2;
            lb.partial_cmp(&la)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(b))
        });
        order.truncate(k);
        order
    }
}

#[derive(Debug, Clone)]
struct Ream {
    counts: Vec<u32>,
    layers: usize,
    experts: usize,
}

impl Ream {
    fn new(layers: usize, experts: usize) -> Self {
        Self {
            counts: vec![0; layers * experts],
            layers,
            experts,
        }
    }

    fn add(&mut self, layer: usize, expert: usize) {
        if layer < self.layers && expert < self.experts {
            self.counts[layer * self.experts + expert] += 1;
        }
    }

    fn row(&self, layer: usize) -> &[u32] {
        let start = layer * self.experts;
        &self.counts[start..start + self.experts]
    }
}

pub struct Engine {
    pub profile: ModelProfile,
    prefill: Ream,
    decode: Ream,
    /// `joint[layer][expert][next_expert]`, predicting layer+1.
    joint: Vec<u32>,
    jobs: BTreeMap<String, Ream>,
    pub observations: Vec<Observation>,
    last_by_token: BTreeMap<(String, u32), (u16, Vec<u16>)>,
    calibration: Option<Calibration>,
}

impl Engine {
    pub fn new(profile: ModelProfile) -> Self {
        let joint_layers = profile.layers.saturating_sub(1);
        Self {
            prefill: Ream::new(profile.layers, profile.experts),
            decode: Ream::new(profile.layers, profile.experts),
            joint: vec![0; joint_layers * profile.experts * profile.experts],
            jobs: BTreeMap::new(),
            observations: Vec::new(),
            last_by_token: BTreeMap::new(),
            calibration: None,
            profile,
        }
    }

    pub fn set_calibration(&mut self, calibration: Calibration) {
        self.calibration = Some(calibration);
    }

    pub fn ingest_trace(&mut self, text: &str, task_class: &str) -> usize {
        let mut n = 0;
        for line in text.lines() {
            if line.is_empty() || line.starts_with('#') || !line.contains('\t') {
                continue;
            }
            let mut parts = line.split('\t');
            let Some(phase) = parts.next() else { continue };
            if phase == "phase" {
                continue;
            }
            let Some(layer) = parts.next().and_then(|s| s.parse::<u16>().ok()) else { continue };
            let Some(token_index) = parts.next().and_then(|s| s.parse::<u32>().ok()) else { continue };
            let Some(token_id) = parts.next().and_then(|s| s.parse::<u32>().ok()) else { continue };
            let Some(_tensor) = parts.next() else { continue };
            let Some(experts) = parts.next() else { continue };
            let expert_ids: Vec<u16> = experts.split(',').filter_map(|s| s.parse().ok()).collect();
            if expert_ids.len() != self.profile.k {
                continue;
            }
            self.observe(Observation {
                phase: phase.to_string(),
                layer,
                batch: 0,
                token_index,
                token_id,
                expert_ids,
                expert_weights: Vec::new(),
            }, task_class);
            n += 1;
        }
        n
    }

    pub fn observe(&mut self, obs: Observation, task_class: &str) -> Option<NextExperts> {
        let layer = obs.layer as usize;
        let ream = if obs.phase == "prefill" {
            &mut self.prefill
        } else {
            &mut self.decode
        };
        for id in &obs.expert_ids {
            ream.add(layer, *id as usize);
        }
        let job = self.jobs.entry(task_class.to_string()).or_insert_with(|| {
            Ream::new(self.profile.layers, self.profile.experts)
        });
        for id in &obs.expert_ids {
            job.add(layer, *id as usize);
        }

        let key = (obs.phase.clone(), obs.token_index);
        if let Some((prev_layer, prev_ids)) = self.last_by_token.get(&key).cloned() {
            if obs.layer == prev_layer + 1 {
                self.add_joint(prev_layer as usize, &prev_ids, &obs.expert_ids);
            }
        }
        self.last_by_token
            .insert(key, (obs.layer, obs.expert_ids.clone()));

        let pass = ExpertPass {
            job_id: task_class.to_string(),
            task_class: task_class.to_string(),
            phase: obs.phase.clone(),
            layer: obs.layer,
            token_index: obs.token_index,
            token_id: obs.token_id,
            expert_ids: obs.expert_ids.clone(),
            context_tokens: 1,
        };
        let next = self.route(&pass);
        self.observations.push(obs);
        Some(next)
    }

    fn add_joint(&mut self, layer: usize, from: &[u16], to: &[u16]) {
        if layer + 1 >= self.profile.layers {
            return;
        }
        let experts = self.profile.experts;
        for e in from {
            for n in to {
                let i = (layer * experts + *e as usize) * experts + *n as usize;
                if i < self.joint.len() {
                    self.joint[i] += 1;
                }
            }
        }
    }

    pub fn route(&self, pass: &ExpertPass) -> NextExperts {
        let to_layer = pass.layer.saturating_add(1);
        let launch = admit_fp16_on_p100(FORCED_TILE_SHARED_BYTES);
        let known_cubecl_f16_gemm = match launch {
            Fp16Launch::Admitted => "admitted_forced_tile",
            Fp16Launch::Refused { .. } => "refused",
        };
        let (ids, source) = self.pick(pass, to_layer as usize);
        let expert_calibration = self.scales_for(to_layer as usize, &ids);
        let slab = self.profile.slab_bytes(ids.len());
        let within_13ms = !ids.is_empty() && within_target(slab);
        let activation = self.profile.activation_bytes(pass.context_tokens.max(1));
        NextExperts {
            job_id: pass.job_id.clone(),
            compute_dtype: schedule_dtype(device_class("p100"), ScheduledDtype::F16)
                .as_str()
                .to_string(),
            device: "p100".to_string(),
            known_cubecl_f16_gemm: known_cubecl_f16_gemm.to_string(),
            from_layer: pass.layer,
            to_layer,
            expert_ids: ids,
            source,
            activation_bytes: activation,
            slab_bytes: slab,
            copy_ms_estimate: copy_ms(slab),
            copy_ms_basis: "planning_12GBps".to_string(),
            target_ahead_ms: TARGET_AHEAD_MS,
            within_13ms,
            expert_calibration,
        }
    }

    fn scales_for(&self, layer: usize, ids: &[u16]) -> Vec<ExpertScale> {
        let Some(calibration) = &self.calibration else {
            return Vec::new();
        };
        ids.iter()
            .filter_map(|id| calibration.get(layer, *id as usize).cloned())
            .collect()
    }

    fn router_l2(&self, layer: usize) -> Option<Vec<f64>> {
        let calibration = self.calibration.as_ref()?;
        if layer >= calibration.layers || calibration.experts != self.profile.experts {
            return None;
        }
        Some(
            (0..calibration.experts)
                .map(|expert| calibration.rows[layer * calibration.experts + expert].router_l2)
                .collect(),
        )
    }

    fn pick(&self, pass: &ExpertPass, to_layer: usize) -> (Vec<u16>, String) {
        let k = self.profile.k;
        let experts = self.profile.experts;
        let l2 = self.router_l2(to_layer);
        if (to_layer as u16) < self.profile.layers as u16 && pass.layer as usize + 1 < self.profile.layers
        {
            let mut scores = vec![0u32; experts];
            let layer = pass.layer as usize;
            for e in &pass.expert_ids {
                for next in 0..experts {
                    let i = (layer * experts + *e as usize) * experts + next;
                    if i < self.joint.len() {
                        scores[next] = scores[next].saturating_add(self.joint[i]);
                    }
                }
            }
            if scores.iter().any(|s| *s > 0) {
                return (top_k(&scores, k, l2.as_deref()), "next_layer_joint".to_string());
            }
        }
        if let Some(job) = self.jobs.get(&pass.task_class) {
            if to_layer < job.layers {
                let row = job.row(to_layer);
                if row.iter().any(|s| *s > 0) {
                    return (top_k(row, k, l2.as_deref()), "job_histogram".to_string());
                }
            }
        }
        let ream = if pass.phase == "prefill" {
            &self.prefill
        } else {
            &self.decode
        };
        if to_layer < ream.layers {
            let row = ream.row(to_layer);
            if row.iter().any(|s| *s > 0) {
                return (top_k(row, k, l2.as_deref()), "frequency".to_string());
            }
        }
        if let Some(calibration) = &self.calibration {
            if to_layer < calibration.layers && calibration.experts == experts {
                return (
                    calibration.top_router(to_layer, k),
                    "weight_calibration".to_string(),
                );
            }
        }
        (Vec::new(), "empty".to_string())
    }

    pub fn layer_counts(&self, phase: &str, layer: usize) -> Vec<u32> {
        let ream = if phase == "prefill" { &self.prefill } else { &self.decode };
        if layer >= ream.layers {
            return Vec::new();
        }
        ream.row(layer).to_vec()
    }
}

fn top_k(scores: &[u32], k: usize, router_l2: Option<&[f64]>) -> Vec<u16> {
    let mut order: Vec<u16> = (0..scores.len() as u16).collect();
    order.sort_by(|a, b| {
        scores[*b as usize]
            .cmp(&scores[*a as usize])
            .then_with(|| match router_l2 {
                Some(l2) => l2[*b as usize]
                    .partial_cmp(&l2[*a as usize])
                    .unwrap_or(std::cmp::Ordering::Equal),
                None => std::cmp::Ordering::Equal,
            })
            .then(a.cmp(b))
    });
    order.truncate(k);
    order
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p100_stays_f16_and_refuses_the_69632_launch() {
        assert_eq!(scheduled_compute_dtype(), "f16");
        assert_eq!(FORCED_TILE_SHARED_BYTES, 4_096);
        assert_eq!(
            admit_fp16_on_p100(CUBECL_F16_SHARED_BYTES),
            Fp16Launch::Refused {
                shared_bytes: 69_632,
                cap_bytes: 49_152,
            }
        );
        assert_eq!(
            admit_fp16_on_p100(FORCED_TILE_SHARED_BYTES),
            Fp16Launch::Admitted
        );
        assert_eq!(admit_fp16_on_p100(49_152), Fp16Launch::Admitted);
        assert_eq!(admit_fp16_on_p100(16_384), Fp16Launch::Admitted);
    }

    #[test]
    fn olmoe_top8_copy_fits_the_13ms_planning_budget() {
        let bytes = ModelProfile::OLMOE.slab_bytes(8);
        assert_eq!(bytes, 8 * OLMOE_EXPERT_FP16_BYTES);
        assert!(within_target(bytes));
        assert!(!within_target(MIXTRAL_Q4_TWO_EXPERT_BYTES));
        let one = ModelProfile::OLMOE.activation_bytes(1);
        assert_eq!(one, 4096);
    }

    #[test]
    fn hook_json_names_the_next_layer_experts() {
        let mut engine = Engine::new(ModelProfile::OLMOE);
        let first = Observation {
            phase: "decode".into(),
            layer: 0,
            batch: 0,
            token_index: 1,
            token_id: 7,
            expert_ids: vec![1, 2, 3, 4, 5, 6, 7, 8],
            expert_weights: vec![],
        };
        let second = Observation {
            phase: "decode".into(),
            layer: 1,
            batch: 0,
            token_index: 1,
            token_id: 7,
            expert_ids: vec![9, 10, 11, 12, 13, 14, 15, 16],
            expert_weights: vec![],
        };
        engine.observe(first, "olmoe");
        engine.observe(second.clone(), "olmoe");
        let pass = ExpertPass {
            job_id: "job-1".into(),
            task_class: "olmoe".into(),
            phase: "decode".into(),
            layer: 0,
            token_index: 2,
            token_id: 8,
            expert_ids: vec![1, 2, 3, 4, 5, 6, 7, 8],
            context_tokens: 4,
        };
        let next = engine.route(&pass);
        assert_eq!(next.compute_dtype, "f16");
        assert_eq!(next.known_cubecl_f16_gemm, "admitted_forced_tile");
        assert_eq!(next.to_layer, 1);
        assert_eq!(next.source, "next_layer_joint");
        assert_eq!(next.expert_ids, vec![9, 10, 11, 12, 13, 14, 15, 16]);
        assert_eq!(next.activation_bytes, 4 * 2048 * 2);
        let text = serde_json::to_string(&next).unwrap();
        let back: NextExperts = serde_json::from_str(&text).unwrap();
        assert_eq!(back.expert_ids, next.expert_ids);
        assert_eq!(engine.observations.len(), 2);
        let _ = second;
    }

    #[test]
    fn b1_trace_fills_the_activation_matrix() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../b1/trace.tsv");
        let text = std::fs::read_to_string(path).expect("b1 trace");
        let mut engine = Engine::new(ModelProfile::OLMOE);
        let n = engine.ingest_trace(&text, "olmoe-b1");
        assert_eq!(n, 227);
        let prefill: u32 = (0..16)
            .flat_map(|layer| engine.layer_counts("prefill", layer))
            .sum();
        let decode: u32 = (0..16)
            .flat_map(|layer| engine.layer_counts("decode", layer))
            .sum();
        assert_eq!(prefill, 211 * 8);
        assert_eq!(decode, 16 * 8);
    }

    #[test]
    fn weight_calibration_fills_an_empty_route() {
        let text = r#"{"layers":2,"experts":4,"rows":[
            {"layer":0,"expert":0,"router_l2":0.1,"gate_rms":0.01,"up_rms":0.01,"down_rms":0.01},
            {"layer":0,"expert":1,"router_l2":0.1,"gate_rms":0.01,"up_rms":0.01,"down_rms":0.01},
            {"layer":0,"expert":2,"router_l2":0.1,"gate_rms":0.01,"up_rms":0.01,"down_rms":0.01},
            {"layer":0,"expert":3,"router_l2":0.1,"gate_rms":0.01,"up_rms":0.01,"down_rms":0.01},
            {"layer":1,"expert":0,"router_l2":0.2,"gate_rms":0.01,"up_rms":0.01,"down_rms":0.01},
            {"layer":1,"expert":1,"router_l2":1.5,"gate_rms":0.02,"up_rms":0.02,"down_rms":0.02},
            {"layer":1,"expert":2,"router_l2":0.4,"gate_rms":0.01,"up_rms":0.01,"down_rms":0.01},
            {"layer":1,"expert":3,"router_l2":0.9,"gate_rms":0.01,"up_rms":0.01,"down_rms":0.01}
        ]}"#;
        let calibration = Calibration::from_json(text).expect("calibration");
        let mut profile = ModelProfile::OLMOE;
        profile.layers = 2;
        profile.experts = 4;
        profile.k = 2;
        let mut engine = Engine::new(profile);
        engine.set_calibration(calibration);
        let pass = ExpertPass {
            job_id: "job-cal".into(),
            task_class: "olmoe".into(),
            phase: "decode".into(),
            layer: 0,
            token_index: 0,
            token_id: 1,
            expert_ids: vec![0],
            context_tokens: 1,
        };
        let next = engine.route(&pass);
        assert_eq!(next.source, "weight_calibration");
        assert_eq!(next.expert_ids, vec![1, 3]);
        assert_eq!(next.expert_calibration.len(), 2);
        assert_eq!(next.expert_calibration[0].expert, 1);
        assert!((next.expert_calibration[0].router_l2 - 1.5).abs() < 1e-12);
    }

    #[test]
    fn olmoe_gguf_calibration_has_every_expert() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../data/olmoe-expert-calibration.json"
        );
        let text = std::fs::read_to_string(path).expect("olmoe calibration");
        let calibration = Calibration::from_json(&text).expect("parse");
        assert_eq!(calibration.len(), 16 * 64);
        let row = calibration.get(0, 0).expect("layer 0 expert 0");
        assert!((row.router_l2 - 1.4803899175344386).abs() < 1e-9);
        assert!(row.gate_rms > 0.0 && row.up_rms > 0.0 && row.down_rms > 0.0);
    }
}
