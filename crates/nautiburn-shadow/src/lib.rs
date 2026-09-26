//! Shadow guesser: B3 predictors run **beside** the real router.
//!
//! They do not change tokens and do not replace the gate. Each method writes
//! a belief (predicted set) and a disagreement (overlap with the scraped
//! top-k) into histodb. Microseconds are stored on `beliefs.latency_us`.
//!
//! Keep the ensemble only when it beats the best **single** method on
//! slot-overlap **and** is still under the 13 ms planning lead.

use std::collections::BTreeMap;
use std::time::Instant;

use nautiburn_histodb::{
    infer_model, ActivationEvent, BeliefRecord, HistDb, HistDbError, ModelRecord, ScrapeRecord,
};
use nautiburn_predict::{
    ensemble_union, overlap_k, parse_trace, CrossLayerBigram, FrequencyBar, PrevTokenCopy,
    METHOD_BIGRAM, METHOD_ENSEMBLE, METHOD_FREQUENCY, METHOD_PREV_TOKEN, PLANNING_LEAD_US,
};
use serde::Serialize;

/// Slot-overlap hit rate: `|pred ∩ actual| / k` averaged over scored positions.
///
/// This is the B3 metric (`hits / slots` with `slots = n * k`). Also reported:
/// `exact_set` (predicted set equals actual set) and `any_overlap` (overlap > 0).
/// A method that never produces a prediction has `n_scored = 0` and no rate.
#[derive(Debug, Clone, Serialize)]
pub struct MethodStats {
    pub method: String,
    pub n_scored: u32,
    pub n_skipped: u32,
    pub overlap_sum: u32,
    pub slots: u32,
    pub exact_set: u32,
    pub any_overlap: u32,
    pub predicted_ids_sum: u32,
    pub latency_sum_us: u64,
    pub latency_max_us: u64,
}

impl MethodStats {
    fn new(method: &str) -> Self {
        Self {
            method: method.to_string(),
            n_scored: 0,
            n_skipped: 0,
            overlap_sum: 0,
            slots: 0,
            exact_set: 0,
            any_overlap: 0,
            predicted_ids_sum: 0,
            latency_sum_us: 0,
            latency_max_us: 0,
        }
    }

    /// `|pred ∩ actual| / k`. `None` when the method could not score.
    pub fn slot_hit(&self) -> Option<f64> {
        if self.slots == 0 {
            None
        } else {
            Some(f64::from(self.overlap_sum) / f64::from(self.slots))
        }
    }

    pub fn exact_set_rate(&self) -> Option<f64> {
        if self.n_scored == 0 {
            None
        } else {
            Some(f64::from(self.exact_set) / f64::from(self.n_scored))
        }
    }

    pub fn any_overlap_rate(&self) -> Option<f64> {
        if self.n_scored == 0 {
            None
        } else {
            Some(f64::from(self.any_overlap) / f64::from(self.n_scored))
        }
    }

    pub fn mean_us(&self) -> Option<f64> {
        if self.n_scored == 0 {
            None
        } else {
            Some(self.latency_sum_us as f64 / f64::from(self.n_scored))
        }
    }

    pub fn fast_enough(&self) -> bool {
        match self.mean_us() {
            Some(us) => us < PLANNING_LEAD_US as f64,
            None => false,
        }
    }

    fn skip(&mut self) {
        self.n_skipped += 1;
    }

    fn add(&mut self, pred: &[i64], actual: &[i64], top_k: usize, latency_us: u64) {
        let overlap = overlap_k(pred, actual) as u32;
        self.n_scored += 1;
        self.overlap_sum += overlap;
        self.slots += top_k as u32;
        self.predicted_ids_sum += pred.len() as u32;
        if sets_equal(pred, actual) {
            self.exact_set += 1;
        }
        if overlap > 0 {
            self.any_overlap += 1;
        }
        self.latency_sum_us += latency_us;
        if latency_us > self.latency_max_us {
            self.latency_max_us = latency_us;
        }
    }
}

fn sets_equal(a: &[i64], b: &[i64]) -> bool {
    let mut x = a.to_vec();
    let mut y = b.to_vec();
    x.sort_unstable();
    y.sort_unstable();
    x == y
}

/// One timed prediction that was (or was not) persisted.
#[derive(Debug, Clone)]
pub struct TimedPred {
    pub method: String,
    pub phase: String,
    pub layer: i64,
    pub token_pos: i64,
    pub expert_ids: Vec<i64>,
    pub actual_ids: Vec<i64>,
    pub overlap: usize,
    pub latency_us: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ShadowReport {
    pub label: String,
    pub request_id: String,
    pub model_id: String,
    pub source: String,
    pub n_events: usize,
    pub layers_seen: Vec<i64>,
    pub phases_seen: Vec<String>,
    pub top_k: i64,
    pub planning_lead_us: u64,
    pub methods: Vec<MethodStats>,
    pub winner: String,
    pub ensemble_kept: bool,
    pub layer0_only: bool,
    pub note: String,
}

/// Walk gate events with the B3 protocol (prefill updates, decode frozen).
pub fn score_events(
    events: &[ActivationEvent],
    n_layers: usize,
    n_experts: usize,
    top_k: usize,
) -> (Vec<TimedPred>, Vec<MethodStats>) {
    let mut freq = FrequencyBar::new(n_layers, n_experts, top_k);
    let mut copy = PrevTokenCopy::new(n_layers);
    let mut bigram = CrossLayerBigram::new(n_layers, n_experts, top_k);

    let mut stats = vec![
        MethodStats::new(METHOD_FREQUENCY),
        MethodStats::new(METHOD_PREV_TOKEN),
        MethodStats::new(METHOD_BIGRAM),
        MethodStats::new(METHOD_ENSEMBLE),
    ];
    let mut preds = Vec::new();
    let mut in_decode = false;

    for ((_ord, phase_key, token_pos), layers) in group_tokens(events) {
        let phase = layers
            .values()
            .next()
            .map(|(_, p, _)| p.clone())
            .unwrap_or_else(|| phase_key.clone());

        if phase == "decode" && !in_decode {
            freq.freeze();
            bigram.freeze();
            copy.begin_decode();
            in_decode = true;
        }

        let layer_map: BTreeMap<i64, Vec<i64>> = layers
            .iter()
            .map(|(&layer, (ids, _, _))| (layer, ids.clone()))
            .collect();

        for (&layer, (actual, _, _)) in &layers {
            let t0 = Instant::now();
            let freq_pred = freq.predict(layer);
            let freq_us = t0.elapsed().as_micros() as u64;
            push_pred(
                &mut preds,
                &mut stats[0],
                METHOD_FREQUENCY,
                &phase,
                layer,
                token_pos,
                Some(freq_pred),
                actual,
                top_k,
                freq_us,
            );

            let t1 = Instant::now();
            let copy_pred = copy.predict(layer, token_pos);
            let copy_us = t1.elapsed().as_micros() as u64;
            push_pred(
                &mut preds,
                &mut stats[1],
                METHOD_PREV_TOKEN,
                &phase,
                layer,
                token_pos,
                copy_pred.clone(),
                actual,
                top_k,
                copy_us,
            );

            let prev_layer = layer_map.get(&(layer - 1));
            let t2 = Instant::now();
            let bigram_pred = match prev_layer {
                Some(prev) => bigram.predict(layer, prev),
                None => None,
            };
            let bigram_us = t2.elapsed().as_micros() as u64;
            push_pred(
                &mut preds,
                &mut stats[2],
                METHOD_BIGRAM,
                &phase,
                layer,
                token_pos,
                bigram_pred.clone(),
                actual,
                top_k,
                bigram_us,
            );

            let t3 = Instant::now();
            let ens = match (copy_pred.as_deref(), bigram_pred.as_deref()) {
                (Some(c), Some(b)) => Some(ensemble_union(c, b)),
                _ => None,
            };
            let ens_us = t3.elapsed().as_micros() as u64;
            // Pipeline time: both singles plus the union. A preload of the
            // combined set cannot start before both predictors have run.
            let ens_pipeline_us = match ens {
                Some(_) => copy_us.saturating_add(bigram_us).saturating_add(ens_us),
                None => ens_us,
            };
            push_pred(
                &mut preds,
                &mut stats[3],
                METHOD_ENSEMBLE,
                &phase,
                layer,
                token_pos,
                ens,
                actual,
                top_k,
                ens_pipeline_us,
            );
        }

        if phase != "decode" {
            for (&layer, (actual, _, _)) in &layers {
                freq.observe(layer, actual);
                copy.observe_prefill(layer, token_pos, actual);
            }
            bigram.observe_token(&layer_map);
        }
    }

    (preds, stats)
}

#[allow(clippy::too_many_arguments)]
fn push_pred(
    preds: &mut Vec<TimedPred>,
    stat: &mut MethodStats,
    method: &str,
    phase: &str,
    layer: i64,
    token_pos: i64,
    pred: Option<Vec<i64>>,
    actual: &[i64],
    top_k: usize,
    latency_us: u64,
) {
    match pred {
        Some(expert_ids) => {
            stat.add(&expert_ids, actual, top_k, latency_us);
            preds.push(TimedPred {
                method: method.to_string(),
                phase: phase.to_string(),
                layer,
                token_pos,
                overlap: overlap_k(&expert_ids, actual),
                expert_ids,
                actual_ids: actual.to_vec(),
                latency_us,
            });
        }
        None => stat.skip(),
    }
}

/// `(layer → (expert_ids, phase, token_pos))` per token, prefill then decode.
fn group_tokens(
    events: &[ActivationEvent],
) -> BTreeMap<(u8, String, i64), BTreeMap<i64, (Vec<i64>, String, i64)>> {
    let mut out: BTreeMap<(u8, String, i64), BTreeMap<i64, (Vec<i64>, String, i64)>> =
        BTreeMap::new();
    for ev in events {
        let ord = match ev.phase.as_str() {
            "prefill" => 0,
            "decode" => 1,
            _ => 2,
        };
        out.entry((ord, ev.phase.clone(), ev.token_pos))
            .or_default()
            .insert(
                ev.layer,
                (ev.expert_ids.clone(), ev.phase.clone(), ev.token_pos),
            );
    }
    out
}

/// Winner and ensemble keep/drop from measured stats.
pub fn decide_keep(stats: &[MethodStats]) -> (String, bool) {
    let singles: Vec<&MethodStats> = stats
        .iter()
        .filter(|s| s.method != METHOD_ENSEMBLE && s.n_scored > 0)
        .collect();
    let best_single = singles.iter().copied().max_by(|a, b| {
        rank_hit(a)
            .partial_cmp(&rank_hit(b))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.mean_us().unwrap_or(f64::MAX).partial_cmp(&a.mean_us().unwrap_or(f64::MAX)).unwrap_or(std::cmp::Ordering::Equal))
    });

    let ensemble = stats.iter().find(|s| s.method == METHOD_ENSEMBLE);
    let ensemble_kept = match (ensemble, best_single) {
        (Some(ens), Some(best)) if ens.n_scored > 0 && best.n_scored > 0 => {
            match (ens.slot_hit(), best.slot_hit()) {
                (Some(e), Some(b)) => e > b && ens.fast_enough(),
                _ => false,
            }
        }
        _ => false,
    };

    let winner = if ensemble_kept {
        METHOD_ENSEMBLE.to_string()
    } else if let Some(best) = best_single {
        best.method.clone()
    } else {
        "none".to_string()
    };
    (winner, ensemble_kept)
}

fn rank_hit(s: &MethodStats) -> f64 {
    s.slot_hit().unwrap_or(f64::NEG_INFINITY)
}

fn layers_and_phases(events: &[ActivationEvent]) -> (Vec<i64>, Vec<String>) {
    let mut layers: Vec<i64> = events.iter().map(|e| e.layer).collect();
    layers.sort_unstable();
    layers.dedup();
    let mut phases: Vec<String> = events.iter().map(|e| e.phase.clone()).collect();
    phases.sort();
    phases.dedup();
    (layers, phases)
}

fn note_for(events: &[ActivationEvent], ensemble_kept: bool) -> (bool, String) {
    let (layers, phases) = layers_and_phases(events);
    let layer0_only = layers == [0];
    let prefill_only = phases == ["prefill"];
    let mut note = String::new();
    if layer0_only && prefill_only {
        note.push_str(
            "Primary scrape is layer-0 prefill only. Cross-layer bigram has no previous layer; \
             previous-token copy has no decode step. Ensemble needs both copy and bigram, so it \
             cannot score here.",
        );
    } else if layer0_only {
        note.push_str("Layer-0 only: cross-layer bigram cannot score (no previous layer).");
    }
    if !ensemble_kept {
        if !note.is_empty() {
            note.push(' ');
        }
        note.push_str(
            "Ensemble kept only if it beats the best single method on slot-hit and mean µs < 13000.",
        );
    }
    if note.is_empty() {
        note = "Scored with B3 protocol (prefill updates the bar/joint; decode is frozen).".into();
    }
    (layer0_only, note)
}

/// Persist scored predictions. Does **not** bump histogram_counts again.
pub fn persist_preds(
    db: &mut HistDb,
    request_id: &str,
    preds: &[TimedPred],
) -> Result<usize, HistDbError> {
    let mut n = 0usize;
    for p in preds {
        let belief_id = db.record_belief(&BeliefRecord {
            request_id: request_id.to_string(),
            layer: p.layer,
            method: p.method.clone(),
            expert_ids: p.expert_ids.clone(),
            latency_us: Some(p.latency_us as i64),
        })?;
        db.record_disagreement_ex(&belief_id, &p.actual_ids, false)?;
        n += 1;
    }
    Ok(n)
}

pub fn events_from_scrape(rec: &ScrapeRecord) -> Vec<ActivationEvent> {
    rec.activations
        .iter()
        .map(|a| ActivationEvent {
            phase: rec.phase.clone(),
            layer: a.layer,
            token_pos: a.token_pos,
            expert_ids: a.expert_ids.clone(),
        })
        .collect()
}

pub fn events_from_trace(text: &str) -> Vec<ActivationEvent> {
    parse_trace(text)
        .into_iter()
        .map(|r| ActivationEvent {
            phase: r.phase,
            layer: i64::from(r.layer),
            token_pos: i64::from(r.token_index),
            expert_ids: r.experts.into_iter().map(i64::from).collect(),
        })
        .collect()
}

fn report_from(
    label: &str,
    request_id: &str,
    model_id: &str,
    source: &str,
    events: &[ActivationEvent],
    top_k: i64,
    stats: Vec<MethodStats>,
) -> ShadowReport {
    let (winner, ensemble_kept) = decide_keep(&stats);
    let (layer0_only, note) = note_for(events, ensemble_kept);
    let (layers_seen, phases_seen) = layers_and_phases(events);
    ShadowReport {
        label: label.to_string(),
        request_id: request_id.to_string(),
        model_id: model_id.to_string(),
        source: source.to_string(),
        n_events: events.len(),
        layers_seen,
        phases_seen,
        top_k,
        planning_lead_us: PLANNING_LEAD_US,
        methods: stats,
        winner,
        ensemble_kept,
        layer0_only,
        note,
    }
}

/// Ingest a v1 scrape (if needed), score all four methods, persist beliefs.
pub fn run_scrape(
    db: &mut HistDb,
    rec: &ScrapeRecord,
    label: &str,
) -> Result<ShadowReport, HistDbError> {
    if db.model(&rec.model_id)?.is_none() {
        if let Some(m) = infer_model(&rec.model_id) {
            db.upsert_model(&m)?;
        } else {
            return Err(HistDbError::UnknownModel(rec.model_id.clone()));
        }
    }
    db.insert_scrape(rec)?;
    let model = db
        .model(&rec.model_id)?
        .ok_or_else(|| HistDbError::UnknownModel(rec.model_id.clone()))?;
    let events = db.activations_for(&rec.request_id)?;
    let events = if events.is_empty() {
        events_from_scrape(rec)
    } else {
        events
    };
    let (preds, stats) = score_events(
        &events,
        model.n_layers as usize,
        model.n_experts as usize,
        model.top_k as usize,
    );
    persist_preds(db, &rec.request_id, &preds)?;
    Ok(report_from(
        label,
        &rec.request_id,
        &rec.model_id,
        &rec.source,
        &events,
        model.top_k,
        stats,
    ))
}

/// Score a B1-style TSV as a **separately labeled** run.
pub fn run_trace(
    db: &mut HistDb,
    text: &str,
    request_id: &str,
    model: &ModelRecord,
    class_id: &str,
    label: &str,
) -> Result<ShadowReport, HistDbError> {
    db.upsert_model(model)?;
    let events = events_from_trace(text);
    let n_tokens = events
        .iter()
        .map(|e| e.token_pos)
        .max()
        .map(|m| m + 1);
    db.upsert_request(
        request_id,
        &model.model_id,
        class_id,
        "b1_trace",
        Some("mixed"),
        n_tokens,
        Some("f16"),
    )?;
    let (preds, stats) = score_events(
        &events,
        model.n_layers as usize,
        model.n_experts as usize,
        model.top_k as usize,
    );
    persist_preds(db, request_id, &preds)?;
    Ok(report_from(
        label,
        request_id,
        &model.model_id,
        "b1_trace",
        &events,
        model.top_k,
        stats,
    ))
}

pub fn format_report(report: &ShadowReport) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "label\t{}\trequest_id\t{}\tmodel_id\t{}\tsource\t{}\tevents\t{}\ttop_k\t{}\n",
        report.label, report.request_id, report.model_id, report.source, report.n_events, report.top_k
    ));
    out.push_str(
        "method\tn_scored\tn_skipped\tslot_hit\texact_set\tany_overlap\thits\tslots\tmean_us\tmax_us\tfast_enough\n",
    );
    for m in &report.methods {
        out.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            m.method,
            m.n_scored,
            m.n_skipped,
            fmt_opt(m.slot_hit()),
            fmt_opt(m.exact_set_rate()),
            fmt_opt(m.any_overlap_rate()),
            m.overlap_sum,
            m.slots,
            fmt_opt(m.mean_us()),
            if m.n_scored == 0 {
                "n/a".into()
            } else {
                m.latency_max_us.to_string()
            },
            if m.n_scored == 0 {
                "n/a"
            } else if m.fast_enough() {
                "yes"
            } else {
                "no"
            },
        ));
    }
    out.push_str(&format!("winner\t{}\n", report.winner));
    out.push_str(&format!(
        "ensemble_kept\t{}\n",
        if report.ensemble_kept { "yes" } else { "no" }
    ));
    out.push_str(&format!("planning_lead_us\t{}\n", report.planning_lead_us));
    out.push_str(&format!(
        "layer0_only\t{}\n",
        if report.layer0_only { "yes" } else { "no" }
    ));
    out.push_str(&format!("note\t{}\n", report.note));
    out
}

fn fmt_opt(v: Option<f64>) -> String {
    match v {
        Some(x) => format!("{x:.4}"),
        None => "n/a".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nautiburn_histodb::{ActivationRow, ScrapeRecord};

    fn toy_scrape_two_layer() -> ScrapeRecord {
        ScrapeRecord {
            schema_version: 1,
            model_id: "olmoe-toy".into(),
            request_id: "shadow-toy".into(),
            class_id: Some("code".into()),
            source: "scrape".into(),
            phase: "prefill".into(),
            n_tokens: Some(2),
            dtype: Some("f16".into()),
            activations: vec![
                ActivationRow {
                    layer: 0,
                    token_pos: 0,
                    expert_ids: vec![0, 1, 2, 3, 4, 5, 6, 7],
                },
                ActivationRow {
                    layer: 1,
                    token_pos: 0,
                    expert_ids: vec![0, 1, 2, 3, 4, 5, 6, 7],
                },
                ActivationRow {
                    layer: 0,
                    token_pos: 1,
                    expert_ids: vec![0, 1, 2, 3, 4, 5, 6, 7],
                },
                ActivationRow {
                    layer: 1,
                    token_pos: 1,
                    expert_ids: vec![8, 9, 10, 11, 12, 13, 14, 15],
                },
            ],
            token_ids: Some(vec![1, 2]),
        }
    }

    fn layer0_only() -> ScrapeRecord {
        ScrapeRecord {
            schema_version: 1,
            model_id: "olmoe-1b-7b-0924-f16".into(),
            request_id: "layer0".into(),
            class_id: Some("olmoe-layer0-forced-tile".into()),
            source: "scrape".into(),
            phase: "prefill".into(),
            n_tokens: Some(2),
            dtype: Some("f16".into()),
            activations: vec![
                ActivationRow {
                    layer: 0,
                    token_pos: 0,
                    expert_ids: vec![43, 5, 23, 30, 12, 44, 56, 32],
                },
                ActivationRow {
                    layer: 0,
                    token_pos: 1,
                    expert_ids: vec![7, 5, 2, 27, 14, 19, 43, 37],
                },
            ],
            token_ids: Some(vec![1, 2]),
        }
    }

    #[test]
    fn layer0_copy_scores_second_token_bigram_skips() {
        let rec = layer0_only();
        let events = events_from_scrape(&rec);
        let (preds, stats) = score_events(&events, 16, 64, 8);
        let freq = stats.iter().find(|s| s.method == METHOD_FREQUENCY).unwrap();
        let copy = stats.iter().find(|s| s.method == METHOD_PREV_TOKEN).unwrap();
        let bg = stats.iter().find(|s| s.method == METHOD_BIGRAM).unwrap();
        let ens = stats.iter().find(|s| s.method == METHOD_ENSEMBLE).unwrap();
        assert_eq!(freq.n_scored, 2);
        assert_eq!(copy.n_scored, 1);
        assert_eq!(copy.n_skipped, 1);
        assert_eq!(bg.n_scored, 0);
        assert_eq!(bg.n_skipped, 2);
        assert_eq!(ens.n_scored, 0);
        assert!(bg.slot_hit().is_none());
        assert!(ens.slot_hit().is_none());
        let (_, kept) = decide_keep(&stats);
        assert!(!kept);
        // second token: copy of first set vs [7,5,2,27,14,19,43,37] → 5 and 43
        let copy_pred = preds
            .iter()
            .find(|p| p.method == METHOD_PREV_TOKEN)
            .unwrap();
        assert_eq!(copy_pred.overlap, 2);
    }

    #[test]
    fn two_layer_all_four_score_and_persist() {
        let mut db = HistDb::open_in_memory().unwrap();
        let rec = toy_scrape_two_layer();
        let report = run_scrape(&mut db, &rec, "toy").unwrap();
        assert_eq!(report.methods.len(), 4);
        for m in &report.methods {
            assert!(m.n_scored > 0, "{} should score", m.method);
        }
        assert!(db.belief_count().unwrap() > 0);
        assert_eq!(db.belief_count().unwrap(), db.disagreement_count().unwrap());
        let after = db
            .histogram_count("olmoe-toy", "code", "prefill", 0, 0)
            .unwrap();
        // persist must not re-bump the ingested scrape
        assert_eq!(after, Some(2));
    }

    #[test]
    fn ensemble_dropped_when_it_cannot_score() {
        let stats = vec![
            MethodStats {
                method: METHOD_FREQUENCY.into(),
                n_scored: 8,
                n_skipped: 0,
                overlap_sum: 20,
                slots: 64,
                exact_set: 0,
                any_overlap: 6,
                predicted_ids_sum: 64,
                latency_sum_us: 80,
                latency_max_us: 20,
            },
            MethodStats {
                method: METHOD_PREV_TOKEN.into(),
                n_scored: 7,
                n_skipped: 1,
                overlap_sum: 14,
                slots: 56,
                exact_set: 0,
                any_overlap: 5,
                predicted_ids_sum: 56,
                latency_sum_us: 7,
                latency_max_us: 1,
            },
            MethodStats {
                method: METHOD_BIGRAM.into(),
                n_scored: 0,
                n_skipped: 8,
                overlap_sum: 0,
                slots: 0,
                exact_set: 0,
                any_overlap: 0,
                predicted_ids_sum: 0,
                latency_sum_us: 0,
                latency_max_us: 0,
            },
            MethodStats {
                method: METHOD_ENSEMBLE.into(),
                n_scored: 0,
                n_skipped: 8,
                overlap_sum: 0,
                slots: 0,
                exact_set: 0,
                any_overlap: 0,
                predicted_ids_sum: 0,
                latency_sum_us: 0,
                latency_max_us: 0,
            },
        ];
        let (winner, kept) = decide_keep(&stats);
        assert!(!kept);
        assert_eq!(winner, METHOD_FREQUENCY);
    }

    #[test]
    fn ensemble_kept_only_if_beats_best_single_and_fast() {
        let mut ens = MethodStats::new(METHOD_ENSEMBLE);
        ens.n_scored = 10;
        ens.overlap_sum = 80;
        ens.slots = 80;
        ens.latency_sum_us = 100;
        ens.latency_max_us = 20;
        let mut copy = MethodStats::new(METHOD_PREV_TOKEN);
        copy.n_scored = 10;
        copy.overlap_sum = 50;
        copy.slots = 80;
        copy.latency_sum_us = 50;
        copy.latency_max_us = 10;
        let (winner, kept) = decide_keep(&[copy.clone(), ens.clone()]);
        assert!(kept);
        assert_eq!(winner, METHOD_ENSEMBLE);

        ens.overlap_sum = 50;
        let (_, kept_eq) = decide_keep(&[copy.clone(), ens.clone()]);
        assert!(!kept_eq);

        ens.overlap_sum = 80;
        ens.latency_sum_us = PLANNING_LEAD_US * 10;
        ens.latency_max_us = PLANNING_LEAD_US;
        let (_, kept_slow) = decide_keep(&[copy, ens]);
        assert!(!kept_slow);
    }

    #[test]
    fn b1_online_decode_matches_evaluate() {
        let b1 = include_str!("../../../b1/trace.tsv");
        let public = include_str!("../../../b1/public-slice.tsv");
        let report = nautiburn_predict::evaluate(b1, public);
        let events = events_from_trace(b1);
        let (preds, _stats) = score_events(&events, 16, 64, 8);

        let decode_hits = |name: &str| -> (u32, u32) {
            let mut hits = 0u32;
            let mut slots = 0u32;
            for p in &preds {
                if p.method == name && p.phase == "decode" {
                    hits += p.overlap as u32;
                    slots += 8;
                }
            }
            (hits, slots)
        };

        let ev = |name: &str| {
            report
                .rows
                .iter()
                .find(|r| r.name == name)
                .unwrap_or_else(|| panic!("missing {name}"))
        };

        let (fh, fs) = decode_hits(METHOD_FREQUENCY);
        assert_eq!(fh, ev("frequency").decode.hits);
        assert_eq!(fs, ev("frequency").decode.slots);

        let (ch, cs) = decode_hits(METHOD_PREV_TOKEN);
        assert_eq!(ch, ev("previous-token copy").decode.hits);
        assert_eq!(cs, ev("previous-token copy").decode.slots);

        let (bh, bs) = decode_hits(METHOD_BIGRAM);
        assert_eq!(bh, ev("cross-layer bigram").decode.hits);
        assert_eq!(bs, ev("cross-layer bigram").decode.slots);

        let (eh, es) = decode_hits(METHOD_ENSEMBLE);
        assert_eq!(eh, report.ensemble_hits);
        assert_eq!(es, report.ensemble_slots);
    }
}
