//! Pre-prediction: name experts from the kept ensemble (L+1 / L+2) before
//! the scrape is treated as truth, then compare and write misses back.
//!
//! Belief comes from the stored histogram / prior activations when those
//! exist (frequency bar + bigram joint frozen). Previous-token copy still
//! walks this run so the ensemble can name the next layer. When the store
//! is empty the walk is in-run causal (same as next-expert scoring).
//!
//! Replay of a JSON scrape measures predict microseconds. It does **not**
//! measure wall-ahead versus a live generate. Do not claim a 13 ms
//! wall-ahead from replay alone.

use std::collections::BTreeSet;

use nautiburn_histodb::{
    infer_model, ActivationEvent, HistDb, HistDbError, HistogramCell, ScrapeRecord,
};
use nautiburn_predict::{
    CrossLayerBigram, FrequencyBar, PrevTokenCopy, HORIZON_LAYER_PLUS_1, HORIZON_LAYER_PLUS_2,
    METHOD_ENSEMBLE, PLANNING_LEAD_US,
};
use serde::Serialize;

use crate::next_expert::{score_next_expert_with, HorizonScore};
use crate::schedule::{BeliefInput, LayerExperts};
use crate::{
    decide_keep, events_from_scrape, group_tokens, persist_next_expert_filtered, HorizonReport,
    MethodStats,
};

/// Where the named set came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BeliefSource {
    /// Frequency + bigram frozen from prior rows; copy still walks this run.
    PriorFrozen,
    /// No prior histogram. Same causal walk as next-expert scoring.
    InRunCausal,
}

impl BeliefSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::PriorFrozen => "prior_frozen",
            Self::InRunCausal => "in_run_causal",
        }
    }
}

/// Named expert set the scheduler can consume.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct NamedBelief {
    pub request_id: String,
    pub model_id: String,
    pub method: String,
    pub expert_ids: Vec<i64>,
    pub by_layer: Vec<LayerExperts>,
}

/// Pre-prediction report. Slot-hit at L+1 / L+2 is the kept ensemble.
#[derive(Debug, Clone, Serialize)]
pub struct PrepredictReport {
    pub label: String,
    pub request_id: String,
    pub model_id: String,
    pub source: String,
    pub n_events: usize,
    pub prior_events: usize,
    pub belief_source: BeliefSource,
    pub replay_only: bool,
    pub wall_ahead_vs_live_generate: bool,
    pub wall_ahead_note: String,
    pub planning_lead_us: u64,
    pub horizons: Vec<HorizonReport>,
    pub ensemble_l1_slot_hit: Option<f64>,
    pub ensemble_l2_slot_hit: Option<f64>,
    pub ensemble_l1_mean_us: Option<f64>,
    pub ensemble_l2_mean_us: Option<f64>,
    pub named_belief: NamedBelief,
    pub misses_written: usize,
    pub histogram_bumped: bool,
    pub note: String,
}

const WALL_AHEAD_REPLAY: &str = "replay-only: predict µs measured; wall-ahead vs a live generate was not measured. 13 ms is the planning lead for an expert-weight copy, not a claim that this replay named the set 13 ms before a live step.";

/// Seed frequency / copy / bigram from prior gate events (prefill updates,
/// decode frozen — same B3 observe rule).
pub fn seed_predictors(
    events: &[ActivationEvent],
    n_layers: usize,
    n_experts: usize,
    top_k: usize,
) -> (FrequencyBar, PrevTokenCopy, CrossLayerBigram) {
    let mut freq = FrequencyBar::new(n_layers, n_experts, top_k);
    let mut copy = PrevTokenCopy::new(n_layers);
    let mut bigram = CrossLayerBigram::new(n_layers, n_experts, top_k);
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
        if phase != "decode" {
            let layer_map: std::collections::BTreeMap<i64, Vec<i64>> = layers
                .iter()
                .map(|(&layer, (ids, _, _))| (layer, ids.clone()))
                .collect();
            for (&layer, (actual, _, _)) in &layers {
                freq.observe(layer, actual);
                copy.observe_prefill(layer, token_pos, actual);
            }
            bigram.observe_token(&layer_map);
        }
    }
    (freq, copy, bigram)
}

/// Add histogram cells into a frequency bar (continuum / empty-activation prior).
pub fn seed_frequency_from_cells(freq: &mut FrequencyBar, cells: &[HistogramCell]) {
    for cell in cells {
        if cell.count <= 0 {
            continue;
        }
        for _ in 0..cell.count {
            freq.observe(cell.layer, &[cell.expert_id]);
        }
    }
}

fn ensemble_stat<'a>(horizons: &'a [HorizonScore], horizon: &str) -> Option<&'a MethodStats> {
    horizons
        .iter()
        .find(|h| h.horizon == horizon)
        .and_then(|h| h.methods.iter().find(|m| m.method == METHOD_ENSEMBLE))
}

fn named_from_horizons(
    request_id: &str,
    model_id: &str,
    horizons: &[HorizonScore],
) -> NamedBelief {
    let mut ids = BTreeSet::new();
    let mut by_layer: Vec<LayerExperts> = Vec::new();
    for hz in horizons {
        if hz.horizon != HORIZON_LAYER_PLUS_1 && hz.horizon != HORIZON_LAYER_PLUS_2 {
            continue;
        }
        for p in &hz.preds {
            if p.method != METHOD_ENSEMBLE {
                continue;
            }
            for e in &p.expert_ids {
                ids.insert(*e);
            }
            // Belief layer is the target (src + hop). L+1 → src+1.
            let hop = if hz.horizon == HORIZON_LAYER_PLUS_2 {
                2
            } else {
                1
            };
            by_layer.push(LayerExperts {
                layer: p.layer + hop,
                expert_ids: p.expert_ids.clone(),
            });
        }
    }
    NamedBelief {
        request_id: request_id.to_string(),
        model_id: model_id.to_string(),
        method: METHOD_ENSEMBLE.to_string(),
        expert_ids: ids.into_iter().collect(),
        by_layer,
    }
}

fn horizon_reports(horizons: &[HorizonScore]) -> Vec<HorizonReport> {
    horizons
        .iter()
        .map(|hz| {
            let (winner, ensemble_kept) = decide_keep(&hz.methods);
            let winner_fast_enough = hz
                .methods
                .iter()
                .find(|m| m.method == winner)
                .map(|m| m.fast_enough())
                .unwrap_or(false);
            HorizonReport {
                horizon: hz.horizon.clone(),
                methods: hz.methods.clone(),
                winner,
                ensemble_kept,
                winner_fast_enough,
            }
        })
        .collect()
}

fn report_from(
    label: &str,
    request_id: &str,
    model_id: &str,
    source: &str,
    events: &[ActivationEvent],
    prior_events: usize,
    belief_source: BeliefSource,
    horizons: &[HorizonScore],
    misses_written: usize,
    extra_note: &str,
) -> PrepredictReport {
    let hz_reports = horizon_reports(horizons);
    let e1 = ensemble_stat(horizons, HORIZON_LAYER_PLUS_1);
    let e2 = ensemble_stat(horizons, HORIZON_LAYER_PLUS_2);
    let named = named_from_horizons(request_id, model_id, horizons);
    let mut note = match belief_source {
        BeliefSource::PriorFrozen => {
            "Named L+1/L+2 from the kept ensemble with frequency+bigram frozen on prior histogram/activations. prev_token still copies this run at the target layer.".to_string()
        }
        BeliefSource::InRunCausal => {
            "No prior histogram for this model (excluding this request). Named L+1/L+2 with the in-run causal ensemble (same walk as next-expert). This names the next layer before that scrape step, not the whole run before it started.".to_string()
        }
    };
    if !extra_note.is_empty() {
        note.push(' ');
        note.push_str(extra_note);
    }
    note.push(' ');
    note.push_str(WALL_AHEAD_REPLAY);
    PrepredictReport {
        label: label.to_string(),
        request_id: request_id.to_string(),
        model_id: model_id.to_string(),
        source: source.to_string(),
        n_events: events.len(),
        prior_events,
        belief_source,
        replay_only: true,
        wall_ahead_vs_live_generate: false,
        wall_ahead_note: WALL_AHEAD_REPLAY.to_string(),
        planning_lead_us: PLANNING_LEAD_US,
        horizons: hz_reports,
        ensemble_l1_slot_hit: e1.and_then(|m| m.slot_hit()),
        ensemble_l2_slot_hit: e2.and_then(|m| m.slot_hit()),
        ensemble_l1_mean_us: e1.and_then(|m| m.mean_us()),
        ensemble_l2_mean_us: e2.and_then(|m| m.mean_us()),
        named_belief: named,
        misses_written,
        histogram_bumped: misses_written > 0,
        note,
    }
}

/// Load prior events for `model_id`, excluding `request_id`.
pub fn prior_events(
    db: &HistDb,
    model_id: &str,
    except_request_id: &str,
) -> Result<Vec<ActivationEvent>, HistDbError> {
    db.activations_for_model_except(model_id, Some(except_request_id))
}

fn predictors_for(
    db: &HistDb,
    model_id: &str,
    class_id: &str,
    except_request_id: &str,
    n_layers: usize,
    n_experts: usize,
    top_k: usize,
) -> Result<(FrequencyBar, PrevTokenCopy, CrossLayerBigram, usize, BeliefSource), HistDbError>
{
    let prior = prior_events(db, model_id, except_request_id)?;
    if !prior.is_empty() {
        let (freq, copy, bigram) = seed_predictors(&prior, n_layers, n_experts, top_k);
        return Ok((freq, copy, bigram, prior.len(), BeliefSource::PriorFrozen));
    }
    let cells = db.histogram_for(model_id, class_id, "prefill")?;
    let mut decode_cells = db.histogram_for(model_id, class_id, "decode")?;
    let mut all = cells;
    all.append(&mut decode_cells);
    if !all.is_empty() {
        let mut freq = FrequencyBar::new(n_layers, n_experts, top_k);
        seed_frequency_from_cells(&mut freq, &all);
        return Ok((
            freq,
            PrevTokenCopy::new(n_layers),
            CrossLayerBigram::new(n_layers, n_experts, top_k),
            0,
            BeliefSource::PriorFrozen,
        ));
    }
    Ok((
        FrequencyBar::new(n_layers, n_experts, top_k),
        PrevTokenCopy::new(n_layers),
        CrossLayerBigram::new(n_layers, n_experts, top_k),
        0,
        BeliefSource::InRunCausal,
    ))
}

/// Ingest scrape (if needed), name L+1/L+2 from the kept ensemble, compare,
/// write ensemble disagreements with a histogram bump on miss.
pub fn run_prepredict(
    db: &mut HistDb,
    rec: &ScrapeRecord,
    label: &str,
) -> Result<PrepredictReport, HistDbError> {
    if db.model(&rec.model_id)?.is_none() {
        if let Some(m) = infer_model(&rec.model_id) {
            db.upsert_model(&m)?;
        } else {
            return Err(HistDbError::UnknownModel(rec.model_id.clone()));
        }
    }
    let model = db
        .model(&rec.model_id)?
        .ok_or_else(|| HistDbError::UnknownModel(rec.model_id.clone()))?;
    let class_id = rec.class_id.as_deref().unwrap_or("unknown");
    let (freq, copy, bigram, prior_n, source) = predictors_for(
        db,
        &rec.model_id,
        class_id,
        &rec.request_id,
        model.n_layers as usize,
        model.n_experts as usize,
        model.top_k as usize,
    )?;
    let freeze = matches!(source, BeliefSource::PriorFrozen);
    let events = events_from_scrape(rec);
    let (horizons, _) = score_next_expert_with(
        &events,
        freq,
        copy,
        bigram,
        model.top_k as usize,
        freeze,
    );
    db.insert_scrape(rec)?;
    let misses = persist_ensemble_disagreements(db, &rec.request_id, &horizons)?;
    Ok(report_from(
        label,
        &rec.request_id,
        &rec.model_id,
        &rec.source,
        &events,
        prior_n,
        source,
        &horizons,
        misses,
        "",
    ))
}

/// Score already-built events (combined prefill+decode) against a seeded bar.
pub fn report_prepredict(
    label: &str,
    request_id: &str,
    model_id: &str,
    source_tag: &str,
    events: &[ActivationEvent],
    freq: FrequencyBar,
    copy: PrevTokenCopy,
    bigram: CrossLayerBigram,
    top_k: usize,
    freeze: bool,
    prior_events: usize,
    belief_source: BeliefSource,
) -> PrepredictReport {
    let (horizons, _) = score_next_expert_with(events, freq, copy, bigram, top_k, freeze);
    report_from(
        label,
        request_id,
        model_id,
        source_tag,
        events,
        prior_events,
        belief_source,
        &horizons,
        0,
        "",
    )
}

/// Persist ensemble@L+1 and ensemble@L+2 only. Misses bump the histogram once
/// per named position (not once per method). Returns how many named sets
/// disagreed with the scrape (the write-backs).
pub fn persist_ensemble_disagreements(
    db: &mut HistDb,
    request_id: &str,
    horizons: &[HorizonScore],
) -> Result<usize, HistDbError> {
    persist_next_expert_filtered(db, request_id, horizons, Some(METHOD_ENSEMBLE), true)?;
    let mut misses = 0usize;
    for hz in horizons {
        if hz.horizon != HORIZON_LAYER_PLUS_1 && hz.horizon != HORIZON_LAYER_PLUS_2 {
            continue;
        }
        for p in &hz.preds {
            if p.method != METHOD_ENSEMBLE {
                continue;
            }
            let mut a = p.expert_ids.clone();
            let mut b = p.actual_ids.clone();
            a.sort_unstable();
            b.sort_unstable();
            if a != b {
                misses += 1;
            }
        }
    }
    Ok(misses)
}

pub fn belief_input_from_named(named: &NamedBelief) -> BeliefInput {
    BeliefInput {
        request_id: named.request_id.clone(),
        model_id: Some(named.model_id.clone()),
        method: Some(named.method.clone()),
        expert_ids: named.expert_ids.clone(),
        by_layer: named.by_layer.clone(),
    }
}

pub fn format_prepredict_report(report: &PrepredictReport) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "label\t{}\trequest_id\t{}\tmodel_id\t{}\tsource\t{}\tevents\t{}\tprior_events\t{}\n",
        report.label,
        report.request_id,
        report.model_id,
        report.source,
        report.n_events,
        report.prior_events
    ));
    out.push_str(&format!(
        "belief_source\t{}\treplay_only\tyes\twall_ahead_vs_live_generate\tno\n",
        report.belief_source.as_str()
    ));
    out.push_str(&format!(
        "ensemble_L+1_slot_hit\t{}\tensemble_L+1_mean_us\t{}\n",
        fmt_opt(report.ensemble_l1_slot_hit),
        fmt_opt(report.ensemble_l1_mean_us)
    ));
    out.push_str(&format!(
        "ensemble_L+2_slot_hit\t{}\tensemble_L+2_mean_us\t{}\n",
        fmt_opt(report.ensemble_l2_slot_hit),
        fmt_opt(report.ensemble_l2_mean_us)
    ));
    out.push_str(&format!("planning_lead_us\t{}\n", report.planning_lead_us));
    out.push_str(&format!(
        "misses_written\t{}\thistogram_bumped\t{}\n",
        report.misses_written,
        if report.histogram_bumped { "yes" } else { "no" }
    ));
    out.push_str(&format!(
        "named_experts\t[{}]\n",
        report
            .named_belief
            .expert_ids
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join(",")
    ));
    for hz in &report.horizons {
        if hz.horizon != HORIZON_LAYER_PLUS_1 && hz.horizon != HORIZON_LAYER_PLUS_2 {
            continue;
        }
        out.push_str(&format!(
            "horizon\t{}\twinner\t{}\tensemble_kept\t{}\n",
            hz.horizon,
            hz.winner,
            if hz.ensemble_kept { "yes" } else { "no" }
        ));
        out.push_str(
            "method\tn_scored\tn_skipped\tslot_hit\thits\tslots\tmean_us\tmax_us\tfast_enough\n",
        );
        for m in &hz.methods {
            out.push_str(&format!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
                m.method,
                m.n_scored,
                m.n_skipped,
                fmt_opt(m.slot_hit()),
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
    }
    out.push_str(&format!("wall_ahead_note\t{}\n", report.wall_ahead_note));
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
    use nautiburn_predict::overlap_k;

    fn two_layer_scrape(request_id: &str) -> ScrapeRecord {
        ScrapeRecord {
            schema_version: 1,
            model_id: "olmoe-toy".into(),
            request_id: request_id.into(),
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
                    expert_ids: vec![8, 9, 10, 11, 12, 13, 14, 15],
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

    #[test]
    fn prepredict_compare_and_write_misses() {
        let mut db = HistDb::open_in_memory().unwrap();
        let seed = two_layer_scrape("seed-req");
        // Different experts on the seed so the frozen bar is a real prior.
        let mut seed = seed;
        seed.activations[1].expert_ids = vec![16, 17, 18, 19, 20, 21, 22, 23];
        seed.activations[3].expert_ids = vec![16, 17, 18, 19, 20, 21, 22, 23];
        let report_seed = run_prepredict(&mut db, &seed, "seed").unwrap();
        assert_eq!(report_seed.belief_source, BeliefSource::InRunCausal);

        let rec = two_layer_scrape("pred-req");
        let before = db
            .histogram_count("olmoe-toy", "code", "prefill", 1, 8)
            .unwrap();
        let report = run_prepredict(&mut db, &rec, "pred").unwrap();
        assert_eq!(report.belief_source, BeliefSource::PriorFrozen);
        assert!(report.replay_only);
        assert!(!report.wall_ahead_vs_live_generate);
        assert!(report.wall_ahead_note.contains("replay-only"));
        assert!(report.horizons.iter().any(|h| h.horizon == "L+1"));
        assert!(report.ensemble_l1_slot_hit.is_some() || report.horizons.iter().any(|h| {
            h.horizon == "L+1"
                && h.methods
                    .iter()
                    .any(|m| m.method == METHOD_ENSEMBLE && m.n_scored + m.n_skipped > 0)
        }));
        // Named set is the ensemble predictions, compared to actuals in persist.
        assert!(!report.named_belief.expert_ids.is_empty() || report.named_belief.by_layer.is_empty());
        assert!(report.misses_written > 0);
        let after = db
            .histogram_count("olmoe-toy", "code", "prefill", 1, 8)
            .unwrap();
        // scrape ingest counts the actuals; a miss also bumps.
        assert!(after.unwrap_or(0) >= before.unwrap_or(0));
        assert_eq!(overlap_k(&[8], &[8, 9]), 1);
    }

    #[test]
    fn empty_prior_is_in_run_causal_and_does_not_claim_wall_ahead() {
        let mut db = HistDb::open_in_memory().unwrap();
        let rec = two_layer_scrape("only");
        let report = run_prepredict(&mut db, &rec, "only").unwrap();
        assert_eq!(report.belief_source, BeliefSource::InRunCausal);
        assert!(!report.wall_ahead_vs_live_generate);
        assert_eq!(report.prior_events, 0);
        let l1 = report
            .horizons
            .iter()
            .find(|h| h.horizon == "L+1")
            .expect("L+1");
        let ens = l1
            .methods
            .iter()
            .find(|m| m.method == METHOD_ENSEMBLE)
            .unwrap();
        // Second token can ensemble: copy of token-0 layer-1 plus bigram.
        assert!(ens.n_scored + ens.n_skipped > 0);
    }
}
