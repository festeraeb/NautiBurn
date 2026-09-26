//! Expert predictors. They do not change the forward.
//!
//! Hit rate is `|predicted ∩ truth| / 8` at OLMoE's own k. A slab is one
//! expert at one layer, 12 MiB (`3 * 2048 * 1024 * 2`). Counts live in
//! [`Pamp`] and are atomic adds. The pre-attention MLP is one learned row.
//! It stays in the table only when its decode hit rate beats previous-token
//! copy. No speculative execution.
//!
//! [`online`] is the same four B3 methods as live state machines for a
//! shadow box beside the gate. The batch bake-off below stays on the B1 TSV.

mod online;

pub use online::{
    ensemble_union, method_at_horizon, overlap_k, top_k_ids, CrossLayerBigram, FrequencyBar,
    PrevTokenCopy, HORIZON_LAYER_PLUS_1, HORIZON_LAYER_PLUS_2, HORIZON_TOKEN_PLUS_1,
    METHOD_BIGRAM, METHOD_ENSEMBLE, METHOD_FREQUENCY, METHOD_PREV_TOKEN, METHOD_SAME_STEP,
    PLANNING_LEAD_US,
};

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU32, Ordering};

use sha2::{Digest, Sha256};

pub const K: usize = 8;
pub const N_EXPERTS: usize = 64;
pub const N_LAYERS: usize = 16;
pub const EXPERT_BYTES: u64 = 3 * 2048 * 1024 * 2;

const PREFILL: &str = "prefill";
const DECODE: &str = "decode";
const PUBLIC: &str = "public";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Row {
    pub phase: String,
    pub layer: u16,
    pub token_index: u32,
    pub token_id: u32,
    pub experts: Vec<u16>,
}

#[derive(Debug, Clone)]
pub struct Split {
    pub hits: u32,
    pub slots: u32,
    pub positions: u32,
    pub slabs: BTreeSet<(u16, u16)>,
}

impl Split {
    fn new() -> Self {
        Self {
            hits: 0,
            slots: 0,
            positions: 0,
            slabs: BTreeSet::new(),
        }
    }

    pub fn hit_rate(&self) -> f64 {
        if self.slots == 0 {
            f64::NAN
        } else {
            f64::from(self.hits) / f64::from(self.slots)
        }
    }

    pub fn mib(&self) -> f64 {
        (self.slabs.len() as u64 * EXPERT_BYTES) as f64 / (1024.0 * 1024.0)
    }

    fn add(&mut self, layer: u16, pred: &[u16], truth: &[u16]) {
        self.positions += 1;
        self.slots += K as u32;
        self.hits += truth.iter().filter(|e| pred.contains(e)).count() as u32;
        for id in pred {
            self.slabs.insert((layer, *id));
        }
    }
}

#[derive(Debug, Clone)]
pub struct PredictorRow {
    pub name: &'static str,
    pub prefill: Split,
    pub decode: Split,
    pub public: Split,
    pub advances: bool,
}

#[derive(Debug, Clone)]
pub struct Report {
    pub rows: Vec<PredictorRow>,
    pub winner: &'static str,
    /// Decode profile scored against prefill. Not one of the five rows.
    pub cosine_reverse_hits: u32,
    pub cosine_reverse_slots: u32,
    pub profile_key: String,
    pub pamp_sha256: String,
    /// Decode hit rate of the pre-attention MLP. Kept only when this beats
    /// previous-token copy on the same trace.
    pub learned_decode_hits: u32,
    pub learned_decode_slots: u32,
    pub learned_kept: bool,
    /// Either-hit combine of previous-token copy and cross-layer bigram on
    /// decode positions both of them score.
    pub ensemble_positions: u32,
    pub ensemble_hits: u32,
    pub ensemble_slots: u32,
    pub ensemble_copy_hits: u32,
    pub ensemble_bigram_hits: u32,
    pub copy_misses: u32,
    pub copy_misses_covered_by_bigram: u32,
    pub bigram_misses: u32,
    pub bigram_misses_covered_by_copy: u32,
    /// Sum of union sizes. Each position still has 8 truth slots.
    pub ensemble_predicted: u32,
    pub ensemble_kept: bool,
}

/// `[phase, layer, expert]` counts and the next-layer joint.
pub struct Pamp {
    counts: Vec<AtomicU32>,
    joint: Vec<AtomicU32>,
}

impl Pamp {
    pub fn new() -> Self {
        Self {
            counts: (0..3 * N_LAYERS * N_EXPERTS).map(|_| AtomicU32::new(0)).collect(),
            joint: (0..(N_LAYERS - 1) * N_EXPERTS * N_EXPERTS)
                .map(|_| AtomicU32::new(0))
                .collect(),
        }
    }

    fn phase_index(phase: &str) -> Option<usize> {
        match phase {
            PREFILL => Some(0),
            DECODE => Some(1),
            PUBLIC => Some(2),
            _ => None,
        }
    }

    pub fn add_count(&self, phase: &str, layer: u16, expert: u16) {
        let Some(p) = Self::phase_index(phase) else {
            return;
        };
        let i = (p * N_LAYERS + layer as usize) * N_EXPERTS + expert as usize;
        self.counts[i].fetch_add(1, Ordering::Relaxed);
    }

    pub fn add_joint(&self, layer: u16, expert: u16, next_expert: u16) {
        let i = ((layer as usize) * N_EXPERTS + expert as usize) * N_EXPERTS + next_expert as usize;
        self.joint[i].fetch_add(1, Ordering::Relaxed);
    }

    pub fn sha256(&self) -> String {
        let mut hasher = Sha256::new();
        for cell in &self.counts {
            hasher.update(cell.load(Ordering::Relaxed).to_le_bytes());
        }
        for cell in &self.joint {
            hasher.update(cell.load(Ordering::Relaxed).to_le_bytes());
        }
        format!("{:x}", hasher.finalize())
    }
}

impl Default for Pamp {
    fn default() -> Self {
        Self::new()
    }
}

pub fn parse_trace(text: &str) -> Vec<Row> {
    let mut rows = Vec::new();
    for line in text.lines() {
        if line.is_empty() || line.starts_with('#') || !line.contains('\t') {
            continue;
        }
        let mut parts = line.split('\t');
        let Some(phase) = parts.next() else { continue };
        let Some(layer) = parts.next().and_then(|s| s.parse().ok()) else { continue };
        let Some(token_index) = parts.next().and_then(|s| s.parse().ok()) else { continue };
        let Some(token_id) = parts.next().and_then(|s| s.parse().ok()) else { continue };
        let Some(_tensor) = parts.next() else { continue };
        let Some(experts) = parts.next() else { continue };
        let experts: Vec<u16> = experts.split(',').filter_map(|s| s.parse().ok()).collect();
        if experts.len() != K {
            continue;
        }
        rows.push(Row {
            phase: phase.to_string(),
            layer,
            token_index,
            token_id,
            experts,
        });
    }
    rows
}

pub fn evaluate(b1_text: &str, public_text: &str) -> Report {
    let b1 = parse_trace(b1_text);
    let public = parse_trace(public_text);
    let pamp = ingest(&b1, &public);
    let mut rows = vec![
        score_frequency(&b1, &public),
        score_previous_token(&b1, &public),
        score_bigram(&b1, &public),
        score_cosine(&b1, &public),
        score_lru(&b1, &public),
    ];
    let (copy_hits, copy_slots) = rows
        .iter()
        .find(|r| r.name == "previous-token copy")
        .map(|r| (r.decode.hits, r.decode.slots))
        .expect("copy row");
    let copy_bar = Split {
        hits: copy_hits,
        slots: copy_slots,
        positions: 0,
        slabs: BTreeSet::new(),
    };
    let learned = score_pre_attention_mlp(&b1);
    let learned_kept = learned_beats_copy(&learned.decode, &copy_bar);
    if learned_kept {
        rows.push(learned.clone());
    }
    let ensemble = score_copy_bigram_ensemble(&b1);
    let ensemble_kept = learned_beats_copy(&ensemble.either, &copy_bar);
    if ensemble_kept {
        rows.push(ensemble.row.clone());
    }
    let freq_hits = rows[0].decode.hits;
    let freq_slots = rows[0].decode.slots;
    for (i, row) in rows.iter_mut().enumerate() {
        row.advances = i == 0 || better_or_equal(row.decode.hits, row.decode.slots, freq_hits, freq_slots);
    }
    let winner = rows
        .iter()
        .enumerate()
        .filter(|(_, row)| row.advances)
        .max_by(|a, b| {
            rate_cmp(&a.1.decode, &b.1.decode)
                .then(rate_cmp(&a.1.prefill, &b.1.prefill))
                .then(b.1.decode.slabs.len().cmp(&a.1.decode.slabs.len()))
                .then(b.0.cmp(&a.0))
        })
        .map(|(_, row)| row.name)
        .unwrap_or("frequency");
    let (cosine_reverse_hits, cosine_reverse_slots) = cosine_reverse(&b1);
    Report {
        rows,
        winner,
        cosine_reverse_hits,
        cosine_reverse_slots,
        profile_key: profile_key(b1_text, &public),
        pamp_sha256: pamp.sha256(),
        learned_decode_hits: learned.decode.hits,
        learned_decode_slots: learned.decode.slots,
        learned_kept,
        ensemble_positions: ensemble.either.positions,
        ensemble_hits: ensemble.either.hits,
        ensemble_slots: ensemble.either.slots,
        ensemble_copy_hits: ensemble.copy_hits,
        ensemble_bigram_hits: ensemble.bigram_hits,
        copy_misses: ensemble.copy_misses,
        copy_misses_covered_by_bigram: ensemble.copy_misses_covered,
        bigram_misses: ensemble.bigram_misses,
        bigram_misses_covered_by_copy: ensemble.bigram_misses_covered,
        ensemble_predicted: ensemble.predicted,
        ensemble_kept,
    }
}

fn learned_beats_copy(learned: &Split, copy: &Split) -> bool {
    learned.slots > 0
        && copy.slots > 0
        && u64::from(learned.hits) * u64::from(copy.slots) > u64::from(copy.hits) * u64::from(learned.slots)
}

pub fn format_report(report: &Report) -> String {
    let mut out = String::new();
    out.push_str("predictor\tprefill_hit\tprefill_n\tprefill_mib\tdecode_hit\tdecode_n\tdecode_mib\tpublic_hit\tpublic_n\tpublic_mib\tadvances\n");
    for row in &report.rows {
        out.push_str(&format!(
            "{}\t{}\t{}\t{:.1}\t{}\t{}\t{:.1}\t{}\t{}\t{:.1}\t{}\n",
            row.name,
            fmt_rate(&row.prefill),
            row.prefill.positions,
            row.prefill.mib(),
            fmt_rate(&row.decode),
            row.decode.positions,
            row.decode.mib(),
            fmt_rate(&row.public),
            row.public.positions,
            row.public.mib(),
            if row.advances { "yes" } else { "no" },
        ));
    }
    let reverse = if report.cosine_reverse_slots == 0 {
        "n/a".to_string()
    } else {
        format!(
            "{:.4}",
            f64::from(report.cosine_reverse_hits) / f64::from(report.cosine_reverse_slots)
        )
    };
    out.push_str(&format!("winner\t{}\n", report.winner));
    out.push_str(&format!(
        "cosine_reverse_decode_to_prefill\t{reverse}\n"
    ));
    out.push_str(&format!("profile_key\t{}\n", report.profile_key));
    out.push_str(&format!("pamp_sha256\t{}\n", report.pamp_sha256));
    let learned_rate = if report.learned_decode_slots == 0 {
        "n/a".to_string()
    } else {
        format!(
            "{:.4}",
            f64::from(report.learned_decode_hits) / f64::from(report.learned_decode_slots)
        )
    };
    out.push_str(&format!(
        "learned_row\tpre-attention-mlp\tdecode_hit\t{learned_rate}\tkept\t{}\n",
        if report.learned_kept { "yes" } else { "no" }
    ));
    let ensemble_rate = if report.ensemble_slots == 0 {
        "n/a".to_string()
    } else {
        format!(
            "{:.4}",
            f64::from(report.ensemble_hits) / f64::from(report.ensemble_slots)
        )
    };
    out.push_str(&format!(
        "ensemble\tcopy-bigram\tpositions\t{}\teither_hit\t{ensemble_rate}\thits\t{}\tslots\t{}\tcopy_hits\t{}\tbigram_hits\t{}\tcopy_misses\t{}\tcopy_misses_covered_by_bigram\t{}\tbigram_misses\t{}\tbigram_misses_covered_by_copy\t{}\tkept\t{}\n",
        report.ensemble_positions,
        report.ensemble_hits,
        report.ensemble_slots,
        report.ensemble_copy_hits,
        report.ensemble_bigram_hits,
        report.copy_misses,
        report.copy_misses_covered_by_bigram,
        report.bigram_misses,
        report.bigram_misses_covered_by_copy,
        if report.ensemble_kept { "yes" } else { "no" },
    ));
    out.push_str(&format!(
        "ensemble_predicted_ids\t{}\n",
        report.ensemble_predicted
    ));
    out.push_str("speculative_execution\tno\n");
    out
}

fn fmt_rate(split: &Split) -> String {
    if split.slots == 0 {
        "n/a".to_string()
    } else {
        format!("{:.4}", split.hit_rate())
    }
}

fn better_or_equal(hits: u32, slots: u32, freq_hits: u32, freq_slots: u32) -> bool {
    slots > 0 && freq_slots > 0 && u64::from(hits) * u64::from(freq_slots) >= u64::from(freq_hits) * u64::from(slots)
}

fn rate_cmp(a: &Split, b: &Split) -> std::cmp::Ordering {
    if a.slots == 0 && b.slots == 0 {
        return std::cmp::Ordering::Equal;
    }
    if a.slots == 0 {
        return std::cmp::Ordering::Less;
    }
    if b.slots == 0 {
        return std::cmp::Ordering::Greater;
    }
    (u64::from(a.hits) * u64::from(b.slots)).cmp(&(u64::from(b.hits) * u64::from(a.slots)))
}

fn ingest(b1: &[Row], public: &[Row]) -> Pamp {
    let pamp = Pamp::new();
    for row in b1.iter().chain(public.iter()) {
        for expert in &row.experts {
            pamp.add_count(&row.phase, row.layer, *expert);
        }
    }
    for rows in [b1, public] {
        for token in tokens(rows, None) {
            let layers: Vec<u16> = token.layers.keys().copied().collect();
            for pair in layers.windows(2) {
                if pair[1] != pair[0] + 1 {
                    continue;
                }
                let Some(left) = token.layers.get(&pair[0]) else { continue };
                let Some(right) = token.layers.get(&pair[1]) else { continue };
                for expert in left {
                    for next in right {
                        pamp.add_joint(pair[0], *expert, *next);
                    }
                }
            }
        }
    }
    pamp
}

struct Token {
    index: u32,
    layers: BTreeMap<u16, Vec<u16>>,
}

fn tokens(rows: &[Row], phase: Option<&str>) -> Vec<Token> {
    let mut map: BTreeMap<u32, BTreeMap<u16, Vec<u16>>> = BTreeMap::new();
    for row in rows {
        if phase.is_some_and(|p| row.phase != p) {
            continue;
        }
        map.entry(row.token_index).or_default().insert(row.layer, row.experts.clone());
    }
    map.into_iter().map(|(index, layers)| Token { index, layers }).collect()
}

fn top_k(counts: &[u32]) -> Vec<u16> {
    let mut ids: Vec<u16> = (0..counts.len() as u16).filter(|&id| counts[id as usize] > 0).collect();
    ids.sort_by(|a, b| counts[*b as usize].cmp(&counts[*a as usize]).then(a.cmp(b)));
    ids.truncate(K);
    ids
}

fn named(name: &'static str, prefill: Split, decode: Split, public: Split) -> PredictorRow {
    PredictorRow {
        name,
        prefill,
        decode,
        public,
        advances: false,
    }
}

fn score_frequency(b1: &[Row], public: &[Row]) -> PredictorRow {
    let mut counts = vec![vec![0u32; N_EXPERTS]; N_LAYERS];
    let mut prefill = Split::new();
    for token in tokens(b1, Some(PREFILL)) {
        for (layer, truth) in &token.layers {
            let pred = top_k(&counts[*layer as usize]);
            prefill.add(*layer, &pred, truth);
            for expert in truth {
                counts[*layer as usize][*expert as usize] += 1;
            }
        }
    }
    let mut decode = Split::new();
    for token in tokens(b1, Some(DECODE)) {
        for (layer, truth) in &token.layers {
            let pred = top_k(&counts[*layer as usize]);
            decode.add(*layer, &pred, truth);
        }
    }
    let mut public_counts = vec![vec![0u32; N_EXPERTS]; N_LAYERS];
    let mut public_split = Split::new();
    for token in tokens(public, Some(PUBLIC)) {
        for (layer, truth) in &token.layers {
            let pred = top_k(&public_counts[*layer as usize]);
            public_split.add(*layer, &pred, truth);
            for expert in truth {
                public_counts[*layer as usize][*expert as usize] += 1;
            }
        }
    }
    named("frequency", prefill, decode, public_split)
}

fn score_previous_token(b1: &[Row], public: &[Row]) -> PredictorRow {
    let mut prev: Vec<Option<(u32, Vec<u16>)>> = vec![None; N_LAYERS];
    let mut prefill = Split::new();
    for token in tokens(b1, Some(PREFILL)) {
        for (layer, truth) in &token.layers {
            if let Some((index, experts)) = &prev[*layer as usize] {
                if *index + 1 == token.index {
                    prefill.add(*layer, experts, truth);
                }
            }
            prev[*layer as usize] = Some((token.index, truth.clone()));
        }
    }
    let last_prefill = tokens(b1, Some(PREFILL)).last().map(|t| t.index);
    let mut decode = Split::new();
    for token in tokens(b1, Some(DECODE)) {
        for (layer, truth) in &token.layers {
            if let (Some(last), Some((index, experts))) = (last_prefill, &prev[*layer as usize]) {
                if *index == last {
                    decode.add(*layer, experts, truth);
                }
            }
        }
    }
    let mut public_split = Split::new();
    let mut public_prev: Vec<Option<(u32, Vec<u16>)>> = vec![None; N_LAYERS];
    for token in tokens(public, Some(PUBLIC)) {
        for (layer, truth) in &token.layers {
            if let Some((index, experts)) = &public_prev[*layer as usize] {
                if *index + 1 == token.index {
                    public_split.add(*layer, experts, truth);
                }
            }
            public_prev[*layer as usize] = Some((token.index, truth.clone()));
        }
    }
    named("previous-token copy", prefill, decode, public_split)
}

fn score_bigram(b1: &[Row], public: &[Row]) -> PredictorRow {
    let mut joint = empty_joint();
    let mut prefill = Split::new();
    for token in tokens(b1, Some(PREFILL)) {
        score_bigram_token(&joint, &token, &mut prefill);
        update_joint(&mut joint, &token);
    }
    let mut decode = Split::new();
    for token in tokens(b1, Some(DECODE)) {
        score_bigram_token(&joint, &token, &mut decode);
    }
    let mut public_split = Split::new();
    let mut public_joint = empty_joint();
    for token in tokens(public, Some(PUBLIC)) {
        score_bigram_token(&public_joint, &token, &mut public_split);
        update_joint(&mut public_joint, &token);
    }
    named("cross-layer bigram", prefill, decode, public_split)
}

fn empty_joint() -> Vec<Vec<Vec<u32>>> {
    vec![vec![vec![0u32; N_EXPERTS]; N_EXPERTS]; N_LAYERS - 1]
}

fn score_bigram_token(joint: &[Vec<Vec<u32>>], token: &Token, split: &mut Split) {
    let layers: Vec<u16> = token.layers.keys().copied().collect();
    for pair in layers.windows(2) {
        if pair[1] != pair[0] + 1 {
            continue;
        }
        let Some(observed) = token.layers.get(&pair[0]) else { continue };
        let Some(truth) = token.layers.get(&pair[1]) else { continue };
        let mut acc = vec![0u32; N_EXPERTS];
        for expert in observed {
            for (next, count) in joint[pair[0] as usize][*expert as usize].iter().enumerate() {
                acc[next] += *count;
            }
        }
        split.add(pair[1], &top_k(&acc), truth);
    }
}

fn update_joint(joint: &mut [Vec<Vec<u32>>], token: &Token) {
    let layers: Vec<u16> = token.layers.keys().copied().collect();
    for pair in layers.windows(2) {
        if pair[1] != pair[0] + 1 {
            continue;
        }
        let Some(left) = token.layers.get(&pair[0]) else { continue };
        let Some(right) = token.layers.get(&pair[1]) else { continue };
        for expert in left {
            for next in right {
                joint[pair[0] as usize][*expert as usize][*next as usize] += 1;
            }
        }
    }
}

fn score_cosine(b1: &[Row], _public: &[Row]) -> PredictorRow {
    let profiles = token_profiles(&tokens(b1, Some(PREFILL)));
    let mut query = vec![0.0; N_LAYERS * N_EXPERTS];
    for profile in &profiles {
        for (i, v) in profile.flat.iter().enumerate() {
            query[i] += *v;
        }
    }
    let retrieved = nearest(&query, &profiles);
    let mut decode = Split::new();
    for token in tokens(b1, Some(DECODE)) {
        for (layer, truth) in &token.layers {
            let pred = retrieved.and_then(|p| p.layers.get(layer)).map(Vec::as_slice).unwrap_or(&[]);
            decode.add(*layer, pred, truth);
        }
    }
    // Prefill is the query, so this row has no prefill column. The public
    // slice is one token: its routing is the label, and there is no earlier
    // prefix to use as a request matrix.
    named("request-matrix cosine", Split::new(), decode, Split::new())
}

fn cosine_reverse(b1: &[Row]) -> (u32, u32) {
    let decode: BTreeMap<u16, Vec<u16>> = tokens(b1, Some(DECODE))
        .into_iter()
        .next()
        .map(|t| t.layers)
        .unwrap_or_default();
    let mut split = Split::new();
    for token in tokens(b1, Some(PREFILL)) {
        for (layer, truth) in &token.layers {
            if let Some(pred) = decode.get(layer) {
                split.add(*layer, pred, truth);
            }
        }
    }
    (split.hits, split.slots)
}

struct Profile {
    layers: BTreeMap<u16, Vec<u16>>,
    flat: Vec<f64>,
}

fn token_profiles(tokens: &[Token]) -> Vec<Profile> {
    tokens
        .iter()
        .map(|token| {
            let mut flat = vec![0.0; N_LAYERS * N_EXPERTS];
            for (layer, experts) in &token.layers {
                for expert in experts {
                    flat[*layer as usize * N_EXPERTS + *expert as usize] = 1.0;
                }
            }
            Profile {
                layers: token.layers.clone(),
                flat,
            }
        })
        .collect()
}

fn nearest<'a>(query: &[f64], profiles: &'a [Profile]) -> Option<&'a Profile> {
    profiles.iter().max_by(|a, b| cosine(query, &a.flat).total_cmp(&cosine(query, &b.flat)))
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

/// Pre-attention MLP. Input is the 64-d multi-hot of layer L-1 on the same
/// token, which is known before layer L attention. One hidden layer, ReLU,
/// 64 logits, top-8. Trained on CPU. Decode weights see prefill only.
const MLP_HIDDEN: usize = 16;
const MLP_EPOCHS: usize = 20;
const MLP_LR: f32 = 0.05;
const MLP_SEED: u64 = 20260924;

struct PreAttentionMlp {
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
}

impl PreAttentionMlp {
    fn new(seed: u64) -> Self {
        let mut rng = seed;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng as f32) / (u64::MAX as f32) * 2.0 - 1.0
        };
        let s1 = (1.0 / 64.0_f32).sqrt();
        let s2 = (1.0 / MLP_HIDDEN as f32).sqrt();
        let w1 = (0..MLP_HIDDEN * N_EXPERTS).map(|_| next() * s1).collect();
        let b1 = vec![0.0; MLP_HIDDEN];
        let w2 = (0..N_EXPERTS * MLP_HIDDEN).map(|_| next() * s2).collect();
        let b2 = vec![0.0; N_EXPERTS];
        Self { w1, b1, w2, b2 }
    }

    fn forward(&self, x: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut pre = vec![0.0; MLP_HIDDEN];
        let mut hidden = vec![0.0; MLP_HIDDEN];
        for i in 0..MLP_HIDDEN {
            let mut z = self.b1[i];
            for j in 0..N_EXPERTS {
                z += self.w1[i * N_EXPERTS + j] * x[j];
            }
            pre[i] = z;
            hidden[i] = z.max(0.0);
        }
        let mut logits = vec![0.0; N_EXPERTS];
        for e in 0..N_EXPERTS {
            let mut z = self.b2[e];
            for i in 0..MLP_HIDDEN {
                z += self.w2[e * MLP_HIDDEN + i] * hidden[i];
            }
            logits[e] = z;
        }
        (pre, hidden, logits)
    }

    fn predict(&self, x: &[f32]) -> Vec<u16> {
        let (_, _, logits) = self.forward(x);
        let mut ids: Vec<u16> = (0..N_EXPERTS as u16).collect();
        ids.sort_by(|a, b| {
            logits[*b as usize]
                .total_cmp(&logits[*a as usize])
                .then(a.cmp(b))
        });
        ids.truncate(K);
        ids
    }

    fn step(&mut self, x: &[f32], truth: &[u16], lr: f32) {
        let (pre, hidden, logits) = self.forward(x);
        let mut target = [0.0_f32; N_EXPERTS];
        for expert in truth {
            target[*expert as usize] = 1.0;
        }
        let mut d_logit = [0.0_f32; N_EXPERTS];
        for e in 0..N_EXPERTS {
            let p = 1.0 / (1.0 + (-logits[e]).exp());
            d_logit[e] = p - target[e];
        }
        let mut d_hidden = vec![0.0; MLP_HIDDEN];
        for e in 0..N_EXPERTS {
            for i in 0..MLP_HIDDEN {
                d_hidden[i] += d_logit[e] * self.w2[e * MLP_HIDDEN + i];
            }
        }
        for i in 0..MLP_HIDDEN {
            if pre[i] <= 0.0 {
                d_hidden[i] = 0.0;
            }
        }
        for e in 0..N_EXPERTS {
            for i in 0..MLP_HIDDEN {
                self.w2[e * MLP_HIDDEN + i] -= lr * d_logit[e] * hidden[i];
            }
            self.b2[e] -= lr * d_logit[e];
        }
        for i in 0..MLP_HIDDEN {
            for j in 0..N_EXPERTS {
                self.w1[i * N_EXPERTS + j] -= lr * d_hidden[i] * x[j];
            }
            self.b1[i] -= lr * d_hidden[i];
        }
    }
}

fn multi_hot(experts: &[u16]) -> Vec<f32> {
    let mut x = vec![0.0; N_EXPERTS];
    for expert in experts {
        x[*expert as usize] = 1.0;
    }
    x
}

fn mlp_examples(token: &Token) -> Vec<(Vec<f32>, Vec<u16>)> {
    let mut out = Vec::new();
    for layer in 1..N_LAYERS as u16 {
        let Some(prev) = token.layers.get(&(layer - 1)) else { continue };
        let Some(truth) = token.layers.get(&layer) else { continue };
        out.push((multi_hot(prev), truth.clone()));
    }
    out
}

fn score_pre_attention_mlp(b1: &[Row]) -> PredictorRow {
    let prefill_tokens = tokens(b1, Some(PREFILL));
    let mut online = PreAttentionMlp::new(MLP_SEED);
    let mut prefill = Split::new();
    for token in &prefill_tokens {
        for (layer, truth) in &token.layers {
            if *layer == 0 {
                continue;
            }
            let Some(prev) = token.layers.get(&(*layer - 1)) else { continue };
            let x = multi_hot(prev);
            let pred = online.predict(&x);
            prefill.add(*layer, &pred, truth);
            online.step(&x, truth, MLP_LR);
        }
    }
    let mut fitted = PreAttentionMlp::new(MLP_SEED);
    for _ in 0..MLP_EPOCHS {
        for token in &prefill_tokens {
            for (x, truth) in mlp_examples(token) {
                fitted.step(&x, &truth, MLP_LR);
            }
        }
    }
    let mut decode = Split::new();
    for token in tokens(b1, Some(DECODE)) {
        for (layer, truth) in &token.layers {
            if *layer == 0 {
                continue;
            }
            let Some(prev) = token.layers.get(&(*layer - 1)) else { continue };
            let pred = fitted.predict(&multi_hot(prev));
            decode.add(*layer, &pred, truth);
        }
    }
    named("pre-attention mlp", prefill, decode, Split::new())
}

struct EnsembleScore {
    either: Split,
    row: PredictorRow,
    copy_hits: u32,
    bigram_hits: u32,
    copy_misses: u32,
    copy_misses_covered: u32,
    bigram_misses: u32,
    bigram_misses_covered: u32,
    predicted: u32,
}

struct ScoredPred {
    pred: Vec<u16>,
    truth: Vec<u16>,
}

fn copy_decode_map(b1: &[Row]) -> BTreeMap<(u32, u16), ScoredPred> {
    let mut prev: Vec<Option<(u32, Vec<u16>)>> = vec![None; N_LAYERS];
    for token in tokens(b1, Some(PREFILL)) {
        for (layer, truth) in &token.layers {
            prev[*layer as usize] = Some((token.index, truth.clone()));
        }
    }
    let last_prefill = tokens(b1, Some(PREFILL)).last().map(|t| t.index);
    let mut out = BTreeMap::new();
    for token in tokens(b1, Some(DECODE)) {
        for (layer, truth) in &token.layers {
            if let (Some(last), Some((index, experts))) = (last_prefill, &prev[*layer as usize]) {
                if *index == last {
                    out.insert(
                        (token.index, *layer),
                        ScoredPred {
                            pred: experts.clone(),
                            truth: truth.clone(),
                        },
                    );
                }
            }
        }
    }
    out
}

fn bigram_decode_map(b1: &[Row]) -> BTreeMap<(u32, u16), ScoredPred> {
    let mut joint = empty_joint();
    for token in tokens(b1, Some(PREFILL)) {
        update_joint(&mut joint, &token);
    }
    let mut out = BTreeMap::new();
    for token in tokens(b1, Some(DECODE)) {
        let layers: Vec<u16> = token.layers.keys().copied().collect();
        for pair in layers.windows(2) {
            if pair[1] != pair[0] + 1 {
                continue;
            }
            let Some(observed) = token.layers.get(&pair[0]) else { continue };
            let Some(truth) = token.layers.get(&pair[1]) else { continue };
            let mut acc = vec![0u32; N_EXPERTS];
            for expert in observed {
                for (next, count) in joint[pair[0] as usize][*expert as usize].iter().enumerate() {
                    acc[next] += *count;
                }
            }
            out.insert(
                (token.index, pair[1]),
                ScoredPred {
                    pred: top_k(&acc),
                    truth: truth.clone(),
                },
            );
        }
    }
    out
}

fn in_pred(pred: &[u16], expert: u16) -> bool {
    pred.contains(&expert)
}

/// Union of the two top-8 sets. A truth expert counts when either predictor named it.
/// Positions are the decode layers both rows score. The denominator stays 8.
fn score_copy_bigram_ensemble(b1: &[Row]) -> EnsembleScore {
    let copy = copy_decode_map(b1);
    let bigram = bigram_decode_map(b1);
    let mut either = Split::new();
    let mut copy_hits = 0u32;
    let mut bigram_hits = 0u32;
    let mut copy_misses = 0u32;
    let mut copy_misses_covered = 0u32;
    let mut bigram_misses = 0u32;
    let mut bigram_misses_covered = 0u32;
    let mut predicted = 0u32;
    for (key, copy_row) in &copy {
        let Some(bigram_row) = bigram.get(key) else { continue };
        let mut union = copy_row.pred.clone();
        for id in &bigram_row.pred {
            if !union.contains(id) {
                union.push(*id);
            }
        }
        predicted += union.len() as u32;
        either.add(key.1, &union, &copy_row.truth);
        for expert in &copy_row.truth {
            let in_copy = in_pred(&copy_row.pred, *expert);
            let in_bigram = in_pred(&bigram_row.pred, *expert);
            if in_copy {
                copy_hits += 1;
            } else {
                copy_misses += 1;
                if in_bigram {
                    copy_misses_covered += 1;
                }
            }
            if in_bigram {
                bigram_hits += 1;
            } else {
                bigram_misses += 1;
                if in_copy {
                    bigram_misses_covered += 1;
                }
            }
        }
    }
    let row = named("copy-bigram ensemble", Split::new(), either.clone(), Split::new());
    EnsembleScore {
        either,
        row,
        copy_hits,
        bigram_hits,
        copy_misses,
        copy_misses_covered,
        bigram_misses,
        bigram_misses_covered,
        predicted,
    }
}

fn score_lru(b1: &[Row], public: &[Row]) -> PredictorRow {
    let mut prefill = Split::new();
    for token in tokens(b1, Some(PREFILL)) {
        score_lru_token(&token, &mut prefill);
    }
    let mut decode = Split::new();
    for token in tokens(b1, Some(DECODE)) {
        score_lru_token(&token, &mut decode);
    }
    let mut public_split = Split::new();
    for token in tokens(public, Some(PUBLIC)) {
        score_lru_token(&token, &mut public_split);
    }
    named("lru next-layer", prefill, decode, public_split)
}

fn score_lru_token(token: &Token, split: &mut Split) {
    let mut previous: Option<(u16, Vec<u16>)> = None;
    for (layer, truth) in &token.layers {
        if let Some((prev_layer, experts)) = &previous {
            if *layer == prev_layer + 1 {
                split.add(*layer, experts, truth);
            }
        }
        previous = Some((*layer, truth.clone()));
    }
}

fn profile_key(b1_text: &str, public: &[Row]) -> String {
    let mut first_tokens = Vec::new();
    for line in b1_text.lines() {
        let mut parts = line.split_whitespace();
        if parts.next() == Some("llama_token:") {
            let Some(index) = parts.next() else { continue };
            let Some(id) = parts.next() else { continue };
            if first_tokens.len() < 4 {
                first_tokens.push(format!("{index}:{id}"));
            }
        }
    }
    let public_ids: Vec<String> = public
        .iter()
        .map(|r| r.token_id.to_string())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    format!(
        "task_class=absent first_tokens={} public_token_ids={}",
        first_tokens.join(","),
        public_ids.join(",")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> (String, String) {
        let b1 = "\
prefill\t0\t0\t1\tffn_moe_topk\t0,1,2,3,4,5,6,7
prefill\t1\t0\t1\tffn_moe_topk\t0,1,2,3,4,5,6,7
prefill\t0\t1\t2\tffn_moe_topk\t0,1,2,3,4,5,6,7
prefill\t1\t1\t2\tffn_moe_topk\t8,9,10,11,12,13,14,15
decode\t0\t0\t3\tffn_moe_topk\t0,1,2,3,4,5,6,7
decode\t1\t0\t3\tffn_moe_topk\t0,1,2,3,4,5,6,7
";
        let public = "\
public\t0\t0\t9\tpublic_expert_indices\t1,2,3,4,5,6,7,8
public\t1\t0\t9\tpublic_expert_indices\t1,2,3,4,5,6,7,8
";
        (b1.to_string(), public.to_string())
    }

    #[test]
    fn previous_token_copies_the_prior_token_only() {
        let (b1, public) = sample();
        let report = evaluate(&b1, &public);
        let row = report.rows.iter().find(|r| r.name == "previous-token copy").unwrap();
        assert_eq!(row.prefill.positions, 2);
        assert_eq!(row.prefill.hits, 8);
        assert_eq!(row.decode.positions, 2);
        assert_eq!(row.decode.hits, 8);
        assert_eq!(row.public.positions, 0);
    }

    #[test]
    fn lru_on_the_public_slice_is_the_previous_layer() {
        let (b1, public) = sample();
        let report = evaluate(&b1, &public);
        let row = report.rows.iter().find(|r| r.name == "lru next-layer").unwrap();
        assert_eq!(row.public.positions, 1);
        assert_eq!(row.public.hits, 8);
    }

    #[test]
    fn real_traces_obey_the_advance_rule() {
        let b1 = include_str!("../../../b1/trace.tsv");
        let public = include_str!("../../../b1/public-slice.tsv");
        let report = evaluate(b1, public);
        assert_eq!(report.rows.len(), 6);
        let freq = report.rows.iter().find(|r| r.name == "frequency").unwrap();
        assert!(freq.advances);
        assert!(freq.decode.positions > 0);
        for row in &report.rows {
            if row.advances {
                assert!(better_or_equal(row.decode.hits, row.decode.slots, freq.decode.hits, freq.decode.slots) || row.name == "frequency");
            } else {
                assert!(u64::from(row.decode.hits) * u64::from(freq.decode.slots) < u64::from(freq.decode.hits) * u64::from(row.decode.slots));
            }
            for split in [&row.prefill, &row.decode, &row.public] {
                assert!(split.hits <= split.slots);
            }
        }
        assert_eq!(report.winner, "copy-bigram ensemble");
        assert!(report.ensemble_kept);
        assert_eq!(report.ensemble_positions, 14);
        assert_eq!(report.ensemble_hits, 73);
        assert_eq!(report.ensemble_slots, 112);
        assert_eq!(report.ensemble_copy_hits, 64);
        assert_eq!(report.ensemble_bigram_hits, 53);
        assert_eq!(report.copy_misses, 48);
        assert_eq!(report.copy_misses_covered_by_bigram, 9);
        assert_eq!(report.bigram_misses, 59);
        assert_eq!(report.bigram_misses_covered_by_copy, 20);
        assert_eq!(report.ensemble_predicted, 163);
        assert!(!report.learned_kept);
        let copy = report.rows.iter().find(|r| r.name == "previous-token copy").unwrap();
        assert!(!learned_beats_copy(
            &Split {
                hits: report.learned_decode_hits,
                slots: report.learned_decode_slots,
                positions: 0,
                slabs: BTreeSet::new(),
            },
            &copy.decode,
        ));
        assert_eq!(report.learned_decode_hits, 26);
        assert_eq!(report.learned_decode_slots, 120);
        let cosine = report.rows.iter().find(|r| r.name == "request-matrix cosine").unwrap();
        assert_eq!(cosine.prefill.positions, 0);
        assert_eq!(cosine.public.positions, 0);
        assert!(!cosine.advances);
        let lru = report.rows.iter().find(|r| r.name == "lru next-layer").unwrap();
        assert!(!lru.advances);
        assert!(report.rows.iter().any(|r| r.name == report.winner && r.advances));
        let text = format_report(&report);
        assert!(text.contains("learned_row\tpre-attention-mlp\tdecode_hit\t"));
        assert!(text.contains("kept\tno"));
        assert!(text.contains("speculative_execution\tno"));
        let parsed = parse_trace(b1);
        assert!(parsed.iter().all(|r| r.experts.iter().all(|e| *e < 64)));
        assert_eq!(parse_trace(public).len(), 16);
    }
}
