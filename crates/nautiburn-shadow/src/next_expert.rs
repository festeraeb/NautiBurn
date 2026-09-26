//! Next-expert scoring: predict the **next** expert set, not the current gate.
//!
//! Given state at layer `L` / token `t`, the next weight load is layer `L+1`
//! (same token). Accuracy is measured there. `L+2` and `t+1` are further
//! ahead and are scored when the scrape actually has those positions.
//!
//! `prev_token` here copies the previous token's experts at the **target**
//! layer (`(L+1, t-1) → (L+1, t)`). The old same-step copy
//! (`(L, t-1) → (L, t)` at the current gate) is [`METHOD_SAME_STEP`] and is
//! not a next-expert winner.

use std::collections::BTreeMap;
use std::time::Instant;

use nautiburn_histodb::ActivationEvent;
use nautiburn_predict::{
    ensemble_union, CrossLayerBigram, FrequencyBar, PrevTokenCopy, HORIZON_LAYER_PLUS_1,
    HORIZON_LAYER_PLUS_2, HORIZON_TOKEN_PLUS_1, METHOD_BIGRAM, METHOD_ENSEMBLE, METHOD_FREQUENCY,
    METHOD_PREV_TOKEN, METHOD_SAME_STEP,
};

use crate::{
    decide_keep, group_tokens, push_pred, MethodStats, TimedPred,
};

/// One horizon's timed predictions and stats.
#[derive(Debug, Clone)]
pub struct HorizonScore {
    pub horizon: String,
    pub preds: Vec<TimedPred>,
    pub methods: Vec<MethodStats>,
}

impl HorizonScore {
    fn new(horizon: &str) -> Self {
        Self {
            horizon: horizon.to_string(),
            preds: Vec::new(),
            methods: vec![
                MethodStats::new(METHOD_FREQUENCY),
                MethodStats::new(METHOD_PREV_TOKEN),
                MethodStats::new(METHOD_BIGRAM),
                MethodStats::new(METHOD_ENSEMBLE),
            ],
        }
    }
}

/// Walk events and score next-expert horizons plus the old same-step copy.
pub fn score_next_expert(
    events: &[ActivationEvent],
    n_layers: usize,
    n_experts: usize,
    top_k: usize,
) -> (Vec<HorizonScore>, MethodStats) {
    score_next_expert_with(
        events,
        FrequencyBar::new(n_layers, n_experts, top_k),
        PrevTokenCopy::new(n_layers),
        CrossLayerBigram::new(n_layers, n_experts, top_k),
        top_k,
        false,
    )
}

/// Same walk as [`score_next_expert`], starting from already-seeded predictors.
///
/// `freeze_before_walk` holds the frequency bar and bigram joint at the
/// seeded belief (prior histogram). Previous-token copy still updates so
/// the kept ensemble can name `L+1` / `L+2` from this run's last token.
pub fn score_next_expert_with(
    events: &[ActivationEvent],
    mut freq: FrequencyBar,
    mut copy: PrevTokenCopy,
    mut bigram: CrossLayerBigram,
    top_k: usize,
    freeze_before_walk: bool,
) -> (Vec<HorizonScore>, MethodStats) {
    if freeze_before_walk {
        freq.freeze();
        bigram.freeze();
    }

    let mut l1 = HorizonScore::new(HORIZON_LAYER_PLUS_1);
    let mut l2 = HorizonScore::new(HORIZON_LAYER_PLUS_2);
    let mut t1 = HorizonScore::new(HORIZON_TOKEN_PLUS_1);
    let mut same_step = MethodStats::new(METHOD_SAME_STEP);
    let mut same_step_preds: Vec<TimedPred> = Vec::new();

    let mut in_decode = false;
    let grouped = group_tokens(events);
    let token_keys: Vec<(u8, String, i64)> = grouped.keys().cloned().collect();

    for (idx, key) in token_keys.iter().enumerate() {
        let layers = grouped.get(key).expect("grouped key");
        let token_pos = key.2;
        let phase = layers
            .values()
            .next()
            .map(|(_, p, _)| p.clone())
            .unwrap_or_else(|| key.1.clone());

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

        let next_token_layers: Option<BTreeMap<i64, Vec<i64>>> = token_keys.get(idx + 1).and_then(
            |nk| {
                grouped.get(nk).map(|m| {
                    m.iter()
                        .map(|(&layer, (ids, _, _))| (layer, ids.clone()))
                        .collect()
                })
            },
        );

        for (&layer, actual) in &layer_map {
            // Old same-step: copy (L, t-1) vs current (L, t). Not next-expert.
            let t_ss = Instant::now();
            let ss_pred = copy.predict(layer, token_pos);
            let ss_us = t_ss.elapsed().as_micros() as u64;
            push_pred(
                &mut same_step_preds,
                &mut same_step,
                METHOD_SAME_STEP,
                &phase,
                layer,
                token_pos,
                ss_pred,
                actual,
                top_k,
                ss_us,
            );

            if let Some(actual_l1) = layer_map.get(&(layer + 1)) {
                score_horizon(
                    &mut l1,
                    &freq,
                    &copy,
                    &bigram,
                    &phase,
                    layer,
                    token_pos,
                    actual,
                    actual_l1,
                    1,
                    top_k,
                );
            }
            if let Some(actual_l2) = layer_map.get(&(layer + 2)) {
                score_horizon(
                    &mut l2,
                    &freq,
                    &copy,
                    &bigram,
                    &phase,
                    layer,
                    token_pos,
                    actual,
                    actual_l2,
                    2,
                    top_k,
                );
            }
            if let Some(next_map) = &next_token_layers {
                if let Some(actual_t1) = next_map.get(&layer) {
                    score_token_plus_1(
                        &mut t1,
                        &freq,
                        actual,
                        actual_t1,
                        &phase,
                        layer,
                        token_pos,
                        top_k,
                    );
                }
            }
        }

        if phase != "decode" {
            for (&layer, actual) in &layer_map {
                freq.observe(layer, actual);
                copy.observe_prefill(layer, token_pos, actual);
            }
            bigram.observe_token(&layer_map);
        }
    }

    let mut horizons = Vec::new();
    if l1.methods.iter().any(|m| m.n_scored + m.n_skipped > 0) {
        horizons.push(l1);
    }
    if l2.methods.iter().any(|m| m.n_scored + m.n_skipped > 0) {
        horizons.push(l2);
    }
    if t1.methods.iter().any(|m| m.n_scored + m.n_skipped > 0) {
        horizons.push(t1);
    }
    let _ = same_step_preds;
    (horizons, same_step)
}

#[allow(clippy::too_many_arguments)]
fn score_horizon(
    hz: &mut HorizonScore,
    freq: &FrequencyBar,
    copy: &PrevTokenCopy,
    bigram: &CrossLayerBigram,
    phase: &str,
    src_layer: i64,
    token_pos: i64,
    src_experts: &[i64],
    actual: &[i64],
    hops: i64,
    top_k: usize,
) {
    let target = src_layer + hops;

    let t0 = Instant::now();
    let freq_pred = freq.predict(target);
    let freq_us = t0.elapsed().as_micros() as u64;
    push_pred(
        &mut hz.preds,
        &mut hz.methods[0],
        METHOD_FREQUENCY,
        phase,
        src_layer,
        token_pos,
        Some(freq_pred),
        actual,
        top_k,
        freq_us,
    );

    // Next-expert prev_token: previous token at the TARGET layer.
    let t1 = Instant::now();
    let copy_pred = copy.predict(target, token_pos);
    let copy_us = t1.elapsed().as_micros() as u64;
    push_pred(
        &mut hz.preds,
        &mut hz.methods[1],
        METHOD_PREV_TOKEN,
        phase,
        src_layer,
        token_pos,
        copy_pred.clone(),
        actual,
        top_k,
        copy_us,
    );

    let t2 = Instant::now();
    let bigram_pred = bigram.predict_ahead(src_layer, src_experts, hops);
    let bigram_us = t2.elapsed().as_micros() as u64;
    push_pred(
        &mut hz.preds,
        &mut hz.methods[2],
        METHOD_BIGRAM,
        phase,
        src_layer,
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
    let ens_pipeline_us = match ens {
        Some(_) => copy_us.saturating_add(bigram_us).saturating_add(ens_us),
        None => ens_us,
    };
    push_pred(
        &mut hz.preds,
        &mut hz.methods[3],
        METHOD_ENSEMBLE,
        phase,
        src_layer,
        token_pos,
        ens,
        actual,
        top_k,
        ens_pipeline_us,
    );
}

#[allow(clippy::too_many_arguments)]
fn score_token_plus_1(
    hz: &mut HorizonScore,
    freq: &FrequencyBar,
    src_experts: &[i64],
    actual: &[i64],
    phase: &str,
    layer: i64,
    token_pos: i64,
    top_k: usize,
) {
    let t0 = Instant::now();
    let freq_pred = freq.predict(layer);
    let freq_us = t0.elapsed().as_micros() as u64;
    push_pred(
        &mut hz.preds,
        &mut hz.methods[0],
        METHOD_FREQUENCY,
        phase,
        layer,
        token_pos,
        Some(freq_pred),
        actual,
        top_k,
        freq_us,
    );

    // Next-token copy: this token's experts at L predict the next token at L.
    // Already known at (L, t). Bigram does not apply (same layer).
    let t1 = Instant::now();
    let copy_pred = Some(src_experts.to_vec());
    let copy_us = t1.elapsed().as_micros() as u64;
    push_pred(
        &mut hz.preds,
        &mut hz.methods[1],
        METHOD_PREV_TOKEN,
        phase,
        layer,
        token_pos,
        copy_pred.clone(),
        actual,
        top_k,
        copy_us,
    );

    hz.methods[2].skip();
    hz.methods[3].skip();
}

/// Winner per next-expert horizon. `same_step_copy` is never selected.
pub fn decide_next_expert(horizons: &[HorizonScore]) -> (String, bool) {
    let Some(l1) = horizons.iter().find(|h| h.horizon == HORIZON_LAYER_PLUS_1) else {
        return ("none".into(), false);
    };
    decide_keep(&l1.methods)
}

pub fn prev_token_note() -> &'static str {
    "prev_token before this box was same-step copy: (L, t-1) → (L, t) at the \
     current gate (method same_step_copy). That is not a next-expert predictor \
     and is not the winner for this goal. prev_token now means previous-token \
     copy at the TARGET layer: for L+1, (L+1, t-1) → (L+1, t); for t+1, \
     (L, t) → (L, t+1). 13 ms is lookahead vs the next weight load, not a \
     kernel target."
}
