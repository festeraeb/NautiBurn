//! Online forms of the B3 predictors, for a shadow box beside the gate.
//!
//! Same four methods as the batch bake-off: frequency bar, previous-token
//! copy, cross-layer bigram, copy+bigram union. They do not change the
//! forward. Sizes (`n_experts`, `top_k`, `n_layers`) come from the model
//! record so a Mixtral top-2 row is not forced through OLMoE's 8.

use std::collections::BTreeMap;

/// Belief `method` strings stored on `beliefs.method` (handoff schema).
pub const METHOD_FREQUENCY: &str = "frequency";
pub const METHOD_PREV_TOKEN: &str = "prev_token";
pub const METHOD_BIGRAM: &str = "bigram";
pub const METHOD_ENSEMBLE: &str = "ensemble";

/// Planning lead for an expert-weight copy over Gen3 PCIe x16 (µs).
///
/// This is a **planning budget**, not a kernel target. A method is fast
/// enough to start a preload when its measured predict time sits under this.
pub const PLANNING_LEAD_US: u64 = 13_000;

/// Top-`k` expert ids by count, ties broken by smaller id.
///
/// Experts with count 0 are omitted. An empty bar therefore yields an empty
/// prediction (overlap 0), not a fabricated ranking.
pub fn top_k_ids(counts: &[u32], k: usize) -> Vec<i64> {
    let mut ids: Vec<i64> = (0..counts.len() as i64)
        .filter(|&id| counts[id as usize] > 0)
        .collect();
    ids.sort_by(|a, b| {
        counts[*b as usize]
            .cmp(&counts[*a as usize])
            .then(a.cmp(b))
    });
    ids.truncate(k);
    ids
}

/// `|predicted ∩ actual|`. Order does not matter.
pub fn overlap_k(pred: &[i64], actual: &[i64]) -> usize {
    pred.iter().filter(|e| actual.contains(e)).count()
}

/// Union of two top-k sets. Length may exceed `k` (B3 either-hit).
pub fn ensemble_union(copy: &[i64], bigram: &[i64]) -> Vec<i64> {
    let mut union = copy.to_vec();
    for id in bigram {
        if !union.contains(id) {
            union.push(*id);
        }
    }
    union
}

/// Phase histogram / bar. Prefill updates after each token; decode is frozen.
pub struct FrequencyBar {
    n_experts: usize,
    top_k: usize,
    counts: Vec<Vec<u32>>,
    updates_frozen: bool,
}

impl FrequencyBar {
    pub fn new(n_layers: usize, n_experts: usize, top_k: usize) -> Self {
        Self {
            n_experts,
            top_k,
            counts: vec![vec![0u32; n_experts]; n_layers],
            updates_frozen: false,
        }
    }

    pub fn predict(&self, layer: i64) -> Vec<i64> {
        let Some(counts) = self.layer_counts(layer) else {
            return Vec::new();
        };
        top_k_ids(counts, self.top_k)
    }

    pub fn observe(&mut self, layer: i64, experts: &[i64]) {
        if self.updates_frozen {
            return;
        }
        let n_experts = self.n_experts;
        let Some(counts) = self.layer_counts_mut(layer) else {
            return;
        };
        for &e in experts {
            if e >= 0 && (e as usize) < n_experts {
                counts[e as usize] = counts[e as usize].saturating_add(1);
            }
        }
    }

    pub fn freeze(&mut self) {
        self.updates_frozen = true;
    }

    fn layer_counts(&self, layer: i64) -> Option<&[u32]> {
        let i = usize::try_from(layer).ok()?;
        self.counts.get(i).map(|c| c.as_slice())
    }

    fn layer_counts_mut(&mut self, layer: i64) -> Option<&mut [u32]> {
        let i = usize::try_from(layer).ok()?;
        self.counts.get_mut(i).map(|c| c.as_mut_slice())
    }
}

/// Previous-token copy. Prefill copies token `t-1` at the same layer.
/// Decode copies the last prefill token (B3), not decode-to-decode.
pub struct PrevTokenCopy {
    n_layers: usize,
    prev: Vec<Option<(i64, Vec<i64>)>>,
    last_prefill_index: Option<i64>,
    in_decode: bool,
}

impl PrevTokenCopy {
    pub fn new(n_layers: usize) -> Self {
        Self {
            n_layers,
            prev: vec![None; n_layers],
            last_prefill_index: None,
            in_decode: false,
        }
    }

    /// `None` when there is no previous token at this layer (cannot score).
    pub fn predict(&self, layer: i64, token_pos: i64) -> Option<Vec<i64>> {
        let i = usize::try_from(layer).ok()?;
        if i >= self.n_layers {
            return None;
        }
        let (index, experts) = self.prev.get(i)?.as_ref()?;
        if self.in_decode {
            if self.last_prefill_index == Some(*index) {
                return Some(experts.clone());
            }
            return None;
        }
        if *index + 1 == token_pos {
            Some(experts.clone())
        } else {
            None
        }
    }

    pub fn observe_prefill(&mut self, layer: i64, token_pos: i64, experts: &[i64]) {
        let Ok(i) = usize::try_from(layer) else {
            return;
        };
        if i >= self.n_layers {
            return;
        }
        self.prev[i] = Some((token_pos, experts.to_vec()));
        self.last_prefill_index = Some(token_pos);
    }

    pub fn begin_decode(&mut self) {
        self.in_decode = true;
    }
}

/// Cross-layer bigram: predict layer `L` from the same token's layer `L-1`.
///
/// Layer 0 has no previous layer — `predict` returns `None`. Prefill updates
/// the joint after the token; decode is frozen (B3).
pub struct CrossLayerBigram {
    n_experts: usize,
    top_k: usize,
    n_layers: usize,
    /// `joint[L][src][dst]` = count of (layer L → layer L+1).
    joint: Vec<Vec<Vec<u32>>>,
    updates_frozen: bool,
}

impl CrossLayerBigram {
    pub fn new(n_layers: usize, n_experts: usize, top_k: usize) -> Self {
        let pairs = n_layers.saturating_sub(1);
        Self {
            n_experts,
            top_k,
            n_layers,
            joint: vec![vec![vec![0u32; n_experts]; n_experts]; pairs],
            updates_frozen: false,
        }
    }

    /// `None` when `layer == 0` or `prev_layer_experts` is empty.
    pub fn predict(&self, layer: i64, prev_layer_experts: &[i64]) -> Option<Vec<i64>> {
        if layer <= 0 || prev_layer_experts.is_empty() {
            return None;
        }
        let src_layer = (layer - 1) as usize;
        if src_layer >= self.joint.len() {
            return None;
        }
        let mut acc = vec![0u32; self.n_experts];
        let mut any = false;
        for &e in prev_layer_experts {
            if e < 0 || (e as usize) >= self.n_experts {
                continue;
            }
            any = true;
            for (next, count) in self.joint[src_layer][e as usize].iter().enumerate() {
                acc[next] = acc[next].saturating_add(*count);
            }
        }
        if !any {
            return None;
        }
        Some(top_k_ids(&acc, self.top_k))
    }

    pub fn observe_token(&mut self, layers: &BTreeMap<i64, Vec<i64>>) {
        if self.updates_frozen {
            return;
        }
        let keys: Vec<i64> = layers.keys().copied().collect();
        for pair in keys.windows(2) {
            if pair[1] != pair[0] + 1 {
                continue;
            }
            let src_layer = pair[0] as usize;
            if src_layer >= self.joint.len() || src_layer + 1 >= self.n_layers {
                continue;
            }
            let Some(left) = layers.get(&pair[0]) else { continue };
            let Some(right) = layers.get(&pair[1]) else { continue };
            for &e in left {
                if e < 0 || (e as usize) >= self.n_experts {
                    continue;
                }
                for &n in right {
                    if n < 0 || (n as usize) >= self.n_experts {
                        continue;
                    }
                    self.joint[src_layer][e as usize][n as usize] =
                        self.joint[src_layer][e as usize][n as usize].saturating_add(1);
                }
            }
        }
    }

    pub fn freeze(&mut self) {
        self.updates_frozen = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_bar_predicts_nothing() {
        let bar = FrequencyBar::new(2, 4, 2);
        assert!(bar.predict(0).is_empty());
    }

    #[test]
    fn frequency_is_causal() {
        let mut bar = FrequencyBar::new(1, 4, 2);
        assert!(bar.predict(0).is_empty());
        bar.observe(0, &[1, 0]);
        assert_eq!(bar.predict(0), vec![0, 1]);
        bar.observe(0, &[1, 2]);
        // 1 seen twice, 0 and 2 once; ties break by id → 1, 0
        assert_eq!(bar.predict(0), vec![1, 0]);
    }

    #[test]
    fn copy_skips_first_token() {
        let mut copy = PrevTokenCopy::new(1);
        assert!(copy.predict(0, 0).is_none());
        copy.observe_prefill(0, 0, &[3, 1]);
        assert_eq!(copy.predict(0, 1).unwrap(), vec![3, 1]);
        assert!(copy.predict(0, 3).is_none());
    }

    #[test]
    fn bigram_skips_layer_zero() {
        let mut bg = CrossLayerBigram::new(2, 4, 2);
        assert!(bg.predict(0, &[]).is_none());
        let mut layers = BTreeMap::new();
        layers.insert(0, vec![0, 1]);
        layers.insert(1, vec![2, 3]);
        // first token: joint empty → top-k of zeros is empty, but we still
        // return Some([]) so the position is scoreable after a prior token.
        assert_eq!(bg.predict(1, &[0, 1]).unwrap(), Vec::<i64>::new());
        bg.observe_token(&layers);
        assert_eq!(bg.predict(1, &[0, 1]).unwrap(), vec![2, 3]);
    }

    #[test]
    fn ensemble_union_is_either_hit() {
        let u = ensemble_union(&[0, 1, 2], &[2, 3, 1]);
        assert_eq!(u, vec![0, 1, 2, 3]);
        assert_eq!(overlap_k(&u, &[1, 9, 3]), 2);
    }

    #[test]
    fn freeze_stops_frequency_updates() {
        let mut bar = FrequencyBar::new(1, 4, 2);
        bar.observe(0, &[0, 1]);
        bar.freeze();
        bar.observe(0, &[2, 3]);
        assert_eq!(bar.predict(0), vec![0, 1]);
    }
}
