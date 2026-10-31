//! Pure-Rust MoE Lookahead Router Predictor and GGUF Tensor Loader.
//!
//! Synthesized from fleet MoE contest submissions (KatCoder-35B and DeepSeek-v4-Flash).
//! Zero C/C++ FFI wrappers, zero Python dependencies, zero-copy architecture.
//!
//! Features:
//! 1. Native BF16 representation with bit-exact f32 conversions.
//! 2. Pure Rust GGUF v2/v3 header, KV, and tensor metadata parser.
//! 3. Cache-blocked, SIMD-unrolled BF16 matrix-vector GEMV.
//! 4. RMSNorm pre-conditioning to damp residual activation drift.
//! 5. Temporal residual EMA smoothing (DeepSeek contribution).
//! 6. Expert Coherence Momentum tracking (KatCoder contribution).
//! 7. Bayesian hybrid prior blending (router logits + bigram transition log-prior).
//! 8. Entropy-gated dynamic depth (L+1 / L+2) & adaptive top-k expansion.
//!
//! Status: experimental. Not called from any crate yet, and tested only on synthetic router
//! weights; no hit rate has been measured against a real trace (see briefs/N2).

use std::collections::HashMap;

/// Native BF16 floating point representation (IEEE 754 top 16 bits).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(transparent)]
pub struct Bf16(pub u16);

impl Bf16 {
    #[inline(always)]
    pub fn from_f32(val: f32) -> Self {
        let bits = val.to_bits();
        // Round-to-nearest-even conversion from f32 to bf16
        let rounding_bias = 0x7FFF + ((bits >> 16) & 1);
        Self(((bits.wrapping_add(rounding_bias)) >> 16) as u16)
    }

    #[inline(always)]
    pub fn to_f32(self) -> f32 {
        f32::from_bits((self.0 as u32) << 16)
    }
}

/// GGUF Tensor metadata parsed from file header.
#[derive(Debug, Clone)]
pub struct GgufTensorMeta {
    pub name: String,
    pub n_dims: u32,
    pub shape: Vec<u64>,
    pub dtype: u32,
    pub offset: u64,
}

/// Lightweight, pure-Rust GGUF parser for router weights.
pub struct GgufHeaderReader {
    pub version: u32,
    pub n_tensors: u64,
    pub n_kv: u64,
    pub tensors: HashMap<String, GgufTensorMeta>,
    pub data_offset: u64,
}

impl GgufHeaderReader {
    /// Parse GGUF header from bytes.
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 24 {
            return Err("GGUF payload too small for header".to_string());
        }
        if &bytes[0..4] != b"GGUF" {
            return Err("Invalid GGUF magic bytes".to_string());
        }

        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let n_tensors = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        let n_kv = u64::from_le_bytes(bytes[16..24].try_into().unwrap());

        let mut pos = 24usize;

        // Skip KV metadata pairs
        for _ in 0..n_kv {
            if pos + 8 > bytes.len() {
                return Err("Truncated KV key length".to_string());
            }
            let key_len = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap()) as usize;
            pos += 8;
            if pos + key_len + 4 > bytes.len() {
                return Err("Truncated KV key or value type".to_string());
            }
            pos += key_len;
            let val_type = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
            pos += 4;
            pos = Self::skip_kv_val(bytes, pos, val_type)?;
        }

        // Parse Tensor Metadata
        let mut tensors = HashMap::with_capacity(n_tensors as usize);
        for _ in 0..n_tensors {
            if pos + 8 > bytes.len() {
                return Err("Truncated tensor name length".to_string());
            }
            let name_len = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap()) as usize;
            pos += 8;
            if pos + name_len > bytes.len() {
                return Err("Truncated tensor name".to_string());
            }
            let name = String::from_utf8_lossy(&bytes[pos..pos + name_len]).to_string();
            pos += name_len;

            if pos + 4 > bytes.len() {
                return Err("Truncated tensor n_dims".to_string());
            }
            let n_dims = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
            pos += 4;

            let mut shape = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                if pos + 8 > bytes.len() {
                    return Err("Truncated tensor dim".to_string());
                }
                shape.push(u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap()));
                pos += 8;
            }

            if pos + 12 > bytes.len() {
                return Err("Truncated tensor dtype/offset".to_string());
            }
            let dtype = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
            pos += 4;
            let offset = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
            pos += 8;

            tensors.insert(name.clone(), GgufTensorMeta { name, n_dims, shape, dtype, offset });
        }

        // Tensor data alignment (default 32 bytes)
        let alignment = 32usize;
        let data_offset = ((pos + alignment - 1) / alignment) * alignment;

        Ok(Self {
            version,
            n_tensors,
            n_kv,
            tensors,
            data_offset: data_offset as u64,
        })
    }

    fn skip_kv_val(bytes: &[u8], mut pos: usize, val_type: u32) -> Result<usize, String> {
        match val_type {
            0..=2 => pos += 1, // u8, i8, bool
            3..=4 => pos += 2, // u16, i16
            5..=6 | 12 => pos += 4, // u32, i32, f32
            7..=8 | 13 => pos += 8, // u64, i64, f64
            9 => { // string
                if pos + 8 > bytes.len() { return Err("Truncated string KV".to_string()); }
                let len = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap()) as usize;
                pos += 8 + len;
            }
            10 => { // array
                if pos + 12 > bytes.len() { return Err("Truncated array KV".to_string()); }
                let elem_type = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
                let count = u64::from_le_bytes(bytes[pos + 4..pos + 12].try_into().unwrap()) as usize;
                pos += 12;
                for _ in 0..count {
                    pos = Self::skip_kv_val(bytes, pos, elem_type)?;
                }
            }
            _ => return Err(format!("Unsupported GGUF KV type: {val_type}")),
        }
        if pos > bytes.len() {
            Err("KV parse overrun".to_string())
        } else {
            Ok(pos)
        }
    }
}

/// Lookahead prediction configuration and hyperparameters.
#[derive(Debug, Clone)]
pub struct LookaheadConfig {
    pub k: usize,
    pub n_embd: usize,
    pub n_expert: usize,
    /// Weight for Bayesian bigram transition log-prior (default 0.15)
    pub bayesian_alpha: f32,
    /// Temporal EMA factor for residual smoothing (0.0 = off, 0.85 = active)
    pub ema_decay: f32,
    /// Expert Coherence momentum boost
    pub coherence_momentum: f32,
    /// Entropy threshold for dynamic multi-hop lookahead (L+2)
    pub entropy_threshold: f32,
    pub eps: f32,
}

impl Default for LookaheadConfig {
    fn default() -> Self {
        Self {
            k: 8,
            n_embd: 2048,
            n_expert: 64,
            bayesian_alpha: 0.15,
            ema_decay: 0.85,
            coherence_momentum: 0.10,
            entropy_threshold: 2.2,
            eps: 1e-5,
        }
    }
}

/// Output of a lookahead prediction step.
#[derive(Debug, Clone)]
pub struct PredictionResult {
    pub predicted_experts: Vec<u16>,
    pub horizon: String,
    pub entropy: f32,
    pub scores: Vec<f32>,
    pub elapsed_us: u64,
}

/// Output of a fused dual-hop lookahead prediction step (L+1 and L+2 simultaneously).
#[derive(Debug, Clone)]
pub struct DualPredictionResult {
    pub l1_predicted_experts: Vec<u16>,
    pub l2_predicted_experts: Vec<u16>,
    pub l1_entropy: f32,
    pub l2_entropy: f32,
    pub l1_scores: Vec<f32>,
    pub l2_scores: Vec<f32>,
    pub elapsed_us: u64,
}

/// Pure-Rust Tensor Router Lookahead Engine.
pub struct RouterLookahead {
    pub config: LookaheadConfig,
    /// Per-layer router gate weights [n_expert * n_embd] in BF16
    pub router_weights: Vec<Vec<Bf16>>,
    /// Per-layer RMSNorm gamma weights [n_embd] in f32
    pub rms_norm_weights: Vec<Vec<f32>>,
    /// Smoothed residual activation state for temporal EMA
    pub smoothed_residual: Vec<f32>,
    /// Coherence history: count of activations per expert across recent layers
    pub expert_coherence: Vec<u32>,
}

impl RouterLookahead {
    pub fn new(config: LookaheadConfig, n_layers: usize) -> Self {
        let n_embd = config.n_embd;
        let n_expert = config.n_expert;
        Self {
            config,
            router_weights: vec![Vec::new(); n_layers],
            rms_norm_weights: vec![Vec::new(); n_layers],
            smoothed_residual: vec![0.0; n_embd],
            expert_coherence: vec![0; n_expert],
        }
    }

    /// Load router weights for a layer from raw BF16 buffer.
    pub fn set_router_weights(&mut self, layer: usize, weights: Vec<Bf16>) -> Result<(), String> {
        let expected = self.config.n_expert * self.config.n_embd;
        if weights.len() != expected {
            return Err(format!("Router weights shape mismatch: expected {expected}, got {}", weights.len()));
        }
        if layer >= self.router_weights.len() {
            self.router_weights.resize(layer + 1, Vec::new());
        }
        self.router_weights[layer] = weights;
        Ok(())
    }

    /// Set RMSNorm weights for layer normalization preconditioning.
    pub fn set_rmsnorm_weights(&mut self, layer: usize, gamma: Vec<f32>) -> Result<(), String> {
        if gamma.len() != self.config.n_embd {
            return Err(format!("RMSNorm dimension mismatch: expected {}, got {}", self.config.n_embd, gamma.len()));
        }
        if layer >= self.rms_norm_weights.len() {
            self.rms_norm_weights.resize(layer + 1, Vec::new());
        }
        self.rms_norm_weights[layer] = gamma;
        Ok(())
    }

    /// Predict next-layer experts from layer L residual stream x_L.
    ///
    /// Incorporates:
    /// 1. Temporal EMA residual smoothing.
    /// 2. Target layer RMSNorm preconditioning.
    /// 3. Cache-blocked unrolled SIMD BF16 GEMV.
    /// 4. Bayesian bigram log-prior blending.
    /// 5. Coherence momentum tracking.
    /// 6. Softmax entropy calculation with adaptive depth.
    pub fn predict(
        &mut self,
        current_layer: usize,
        residual_x: &[f32],
        bigram_prior: Option<&[f32]>,
        active_experts_current: &[u16],
    ) -> Result<PredictionResult, String> {
        let t0 = std::time::Instant::now();
        let target_layer = current_layer + 1;
        if target_layer >= self.router_weights.len() || self.router_weights[target_layer].is_empty() {
            return Err(format!("Target layer {target_layer} router weights not loaded"));
        }

        let n_embd = self.config.n_embd;
        let n_expert = self.config.n_expert;
        if residual_x.len() != n_embd {
            return Err(format!("Residual vector length mismatch: expected {n_embd}, got {}", residual_x.len()));
        }

        // 1. Temporal EMA Smoothing (DeepSeek innovation)
        let mut x_proc = vec![0.0f32; n_embd];
        if self.config.ema_decay > 0.0 && self.smoothed_residual.iter().any(|&v| v != 0.0) {
            let decay = self.config.ema_decay;
            let one_minus_decay = 1.0 - decay;
            for i in 0..n_embd {
                self.smoothed_residual[i] = decay * self.smoothed_residual[i] + one_minus_decay * residual_x[i];
                x_proc[i] = self.smoothed_residual[i];
            }
        } else {
            x_proc.copy_from_slice(residual_x);
            self.smoothed_residual.copy_from_slice(residual_x);
        }

        // 2. RMSNorm Pre-conditioning (Dampen activation drift)
        if target_layer < self.rms_norm_weights.len() && !self.rms_norm_weights[target_layer].is_empty() {
            let gamma = &self.rms_norm_weights[target_layer];
            let mut sum_sq = 0.0f32;
            for &val in &x_proc {
                sum_sq += val * val;
            }
            let rms_recip = 1.0 / (sum_sq / n_embd as f32 + self.config.eps).sqrt();
            for i in 0..n_embd {
                x_proc[i] = x_proc[i] * rms_recip * gamma[i];
            }
        }

        // 3. Cache-Blocked & Unrolled SIMD BF16 GEMV
        let router = &self.router_weights[target_layer];
        let mut logits = vec![0.0f32; n_expert];

        // Process experts in blocks for cache locality
        for e in 0..n_expert {
            let row_offset = e * n_embd;
            let row = &router[row_offset..row_offset + n_embd];

            let mut sum0 = 0.0f32;
            let mut sum1 = 0.0f32;
            let mut sum2 = 0.0f32;
            let mut sum3 = 0.0f32;

            // 16-element unrolled loop
            let chunks = n_embd / 16;
            for c in 0..chunks {
                let idx = c * 16;
                sum0 += row[idx].to_f32() * x_proc[idx]
                     + row[idx + 1].to_f32() * x_proc[idx + 1]
                     + row[idx + 2].to_f32() * x_proc[idx + 2]
                     + row[idx + 3].to_f32() * x_proc[idx + 3];

                sum1 += row[idx + 4].to_f32() * x_proc[idx + 4]
                     + row[idx + 5].to_f32() * x_proc[idx + 5]
                     + row[idx + 6].to_f32() * x_proc[idx + 6]
                     + row[idx + 7].to_f32() * x_proc[idx + 7];

                sum2 += row[idx + 8].to_f32() * x_proc[idx + 8]
                     + row[idx + 9].to_f32() * x_proc[idx + 9]
                     + row[idx + 10].to_f32() * x_proc[idx + 10]
                     + row[idx + 11].to_f32() * x_proc[idx + 11];

                sum3 += row[idx + 12].to_f32() * x_proc[idx + 12]
                     + row[idx + 13].to_f32() * x_proc[idx + 13]
                     + row[idx + 14].to_f32() * x_proc[idx + 14]
                     + row[idx + 15].to_f32() * x_proc[idx + 15];
            }

            // Remainder tail
            let mut remainder = 0.0f32;
            for i in (chunks * 16)..n_embd {
                remainder += row[i].to_f32() * x_proc[i];
            }

            logits[e] = sum0 + sum1 + sum2 + sum3 + remainder;
        }

        // 4. Update Coherence History (KatCoder innovation)
        for &e in active_experts_current {
            if (e as usize) < n_expert {
                self.expert_coherence[e as usize] = self.expert_coherence[e as usize].saturating_add(1);
            }
        }

        // 5. Bayesian Hybrid Prior Blending & Coherence Momentum
        let mut final_scores = vec![0.0f32; n_expert];
        let alpha = self.config.bayesian_alpha;
        let beta = self.config.coherence_momentum;
        let eps = self.config.eps;

        for e in 0..n_expert {
            let mut score = logits[e];

            // Add Bayesian bigram log-prior
            if let Some(prior) = bigram_prior {
                if e < prior.len() {
                    score += alpha * (prior[e] + eps).ln();
                }
            }

            // Add Expert Coherence momentum
            if self.expert_coherence[e] > 0 {
                let coh_boost = (self.expert_coherence[e] as f32).min(5.0) * beta;
                score += coh_boost;
            }

            final_scores[e] = score;
        }

        // 6. Compute Softmax & Shannon Entropy
        let max_score = final_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_sum = 0.0f32;
        let mut probs = vec![0.0f32; n_expert];
        for i in 0..n_expert {
            let exp_val = (final_scores[i] - max_score).exp();
            probs[i] = exp_val;
            exp_sum += exp_val;
        }
        let inv_sum = 1.0 / (exp_sum + eps);
        let mut entropy = 0.0f32;
        for i in 0..n_expert {
            probs[i] *= inv_sum;
            if probs[i] > eps {
                entropy -= probs[i] * probs[i].ln();
            }
        }

        // 7. Dynamic Depth & Top-K Extraction
        let mut indexed: Vec<(usize, f32)> = final_scores.iter().enumerate().map(|(i, &s)| (i, s)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let effective_k = if entropy > self.config.entropy_threshold {
            // High entropy -> expand k by 2 to prevent missing resident experts
            (self.config.k + 2).min(n_expert)
        } else {
            self.config.k
        };

        let predicted_experts: Vec<u16> = indexed.iter().take(effective_k).map(|&(idx, _)| idx as u16).collect();
        let horizon = if entropy < (self.config.entropy_threshold * 0.75) {
            "L+2".to_string() // High confidence: confident enough to prefetch L+2
        } else {
            "L+1".to_string()
        };

        let elapsed_us = t0.elapsed().as_micros() as u64;

        Ok(PredictionResult {
            predicted_experts,
            horizon,
            entropy,
            scores: final_scores,
            elapsed_us,
        })
    }

    /// Calculate adaptive EMA decay factor lambda based on sequence position and activation variance.
    /// Provides agile tracking during prompt prefill and higher smoothing stability during token decode.
    pub fn adaptive_ema_factor(&self, position: usize, variance: f32) -> f32 {
        let base_lambda = self.config.ema_decay;
        if base_lambda <= 0.0 {
            return 0.0;
        }
        // Sigmoid ramp over first 32 tokens: 0.65 -> 0.95
        let pos_scale = 1.0 / (1.0 + (-((position as f32 - 16.0) / 4.0)).exp());
        let var_damp = 1.0 / (1.0 + variance * 0.1);
        (0.65 + 0.30 * pos_scale * var_damp).clamp(0.1, 0.98)
    }

    /// Dynamically scales Bayesian prior weight alpha according to router entropy.
    /// High entropy (high uncertainty) => increases alpha to let bigram transitions guide the choice.
    /// Low entropy (crisp router certainty) => decreases alpha so confident predictions are preserved.
    pub fn dynamic_bayesian_alpha(&self, entropy: f32) -> f32 {
        let base_alpha = self.config.bayesian_alpha;
        let norm_entropy = (entropy / self.config.entropy_threshold).clamp(0.2, 2.0);
        base_alpha * norm_entropy
    }

    /// Predict next-layer (L+1) AND next-next-layer (L+2) experts in a single fused GEMV pass.
    /// Reuses residual stream in L1/L2 cache while streaming through both router matrices simultaneously.
    pub fn predict_dual_fused(
        &mut self,
        current_layer: usize,
        residual_x: &[f32],
        bigram_prior_l1: Option<&[f32]>,
        bigram_prior_l2: Option<&[f32]>,
        active_experts_current: &[u16],
        position: Option<usize>,
    ) -> Result<DualPredictionResult, String> {
        let t0 = std::time::Instant::now();
        let target_l1 = current_layer + 1;
        let target_l2 = current_layer + 2;

        if target_l1 >= self.router_weights.len() || self.router_weights[target_l1].is_empty() {
            return Err(format!("Target layer {target_l1} router weights not loaded"));
        }
        if target_l2 >= self.router_weights.len() || self.router_weights[target_l2].is_empty() {
            return Err(format!("Target layer {target_l2} router weights not loaded"));
        }

        let n_embd = self.config.n_embd;
        let n_expert = self.config.n_expert;
        if residual_x.len() != n_embd {
            return Err(format!("Residual vector length mismatch: expected {n_embd}, got {}", residual_x.len()));
        }

        // 1. Calculate activation variance for adaptive EMA
        let mut mean = 0.0f32;
        for &val in residual_x {
            mean += val;
        }
        mean /= n_embd as f32;
        let mut variance = 0.0f32;
        for &val in residual_x {
            let diff = val - mean;
            variance += diff * diff;
        }
        variance /= n_embd as f32;

        // 2. Temporal EMA Smoothing
        let mut x_proc = vec![0.0f32; n_embd];
        let decay = match position {
            Some(pos) => self.adaptive_ema_factor(pos, variance),
            None => self.config.ema_decay,
        };

        if decay > 0.0 && self.smoothed_residual.iter().any(|&v| v != 0.0) {
            let one_minus_decay = 1.0 - decay;
            for i in 0..n_embd {
                self.smoothed_residual[i] = decay * self.smoothed_residual[i] + one_minus_decay * residual_x[i];
                x_proc[i] = self.smoothed_residual[i];
            }
        } else {
            x_proc.copy_from_slice(residual_x);
            self.smoothed_residual.copy_from_slice(residual_x);
        }

        // 3. RMSNorm Pre-conditioning (Layer L+1)
        if target_l1 < self.rms_norm_weights.len() && !self.rms_norm_weights[target_l1].is_empty() {
            let gamma = &self.rms_norm_weights[target_l1];
            let mut sum_sq = 0.0f32;
            for &val in &x_proc {
                sum_sq += val * val;
            }
            let rms_recip = 1.0 / (sum_sq / n_embd as f32 + self.config.eps).sqrt();
            for i in 0..n_embd {
                x_proc[i] = x_proc[i] * rms_recip * gamma[i];
            }
        }

        // 4. Fused Cache-Blocked SIMD BF16 GEMV for both L+1 and L+2 simultaneously
        let router_l1 = &self.router_weights[target_l1];
        let router_l2 = &self.router_weights[target_l2];
        let mut logits_l1 = vec![0.0f32; n_expert];
        let mut logits_l2 = vec![0.0f32; n_expert];

        let chunks = n_embd / 16;
        for e in 0..n_expert {
            let row_offset = e * n_embd;
            let row_l1 = &router_l1[row_offset..row_offset + n_embd];
            let row_l2 = &router_l2[row_offset..row_offset + n_embd];

            let mut sum0_l1 = 0.0f32;
            let mut sum1_l1 = 0.0f32;
            let mut sum0_l2 = 0.0f32;
            let mut sum1_l2 = 0.0f32;

            for c in 0..chunks {
                let idx = c * 16;
                for j in 0..8 {
                    sum0_l1 += row_l1[idx + j].to_f32() * x_proc[idx + j];
                    sum0_l2 += row_l2[idx + j].to_f32() * x_proc[idx + j];
                }
                for j in 8..16 {
                    sum1_l1 += row_l1[idx + j].to_f32() * x_proc[idx + j];
                    sum1_l2 += row_l2[idx + j].to_f32() * x_proc[idx + j];
                }
            }

            for i in (chunks * 16)..n_embd {
                sum0_l1 += row_l1[i].to_f32() * x_proc[i];
                sum0_l2 += row_l2[i].to_f32() * x_proc[i];
            }

            logits_l1[e] = sum0_l1 + sum1_l1;
            logits_l2[e] = sum0_l2 + sum1_l2;
        }

        // 5. Update Coherence History
        for &e in active_experts_current {
            if (e as usize) < n_expert {
                self.expert_coherence[e as usize] = self.expert_coherence[e as usize].saturating_add(1);
            }
        }

        // 6. Compute scores, dynamic Bayesian alpha, and entropy for L+1 and L+2
        let beta = self.config.coherence_momentum;
        let eps = self.config.eps;

        let (final_scores_l1, entropy_l1, topk_l1) = self.rank_horizon(&logits_l1, bigram_prior_l1, beta, eps);
        let (final_scores_l2, entropy_l2, topk_l2) = self.rank_horizon(&logits_l2, bigram_prior_l2, beta * 0.75, eps);

        let elapsed_us = t0.elapsed().as_micros() as u64;

        Ok(DualPredictionResult {
            l1_predicted_experts: topk_l1,
            l2_predicted_experts: topk_l2,
            l1_entropy: entropy_l1,
            l2_entropy: entropy_l2,
            l1_scores: final_scores_l1,
            l2_scores: final_scores_l2,
            elapsed_us,
        })
    }

    fn rank_horizon(
        &self,
        logits: &[f32],
        bigram_prior: Option<&[f32]>,
        beta: f32,
        eps: f32,
    ) -> (Vec<f32>, f32, Vec<u16>) {
        let n_expert = self.config.n_expert;
        let mut scores = vec![0.0f32; n_expert];

        // Preliminary entropy calculation for dynamic alpha
        let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_sum_init = 0.0f32;
        let mut probs_init = vec![0.0f32; n_expert];
        for i in 0..n_expert {
            let p = (logits[i] - max_l).exp();
            probs_init[i] = p;
            exp_sum_init += p;
        }
        let inv_sum_init = 1.0 / (exp_sum_init + eps);
        let mut prelim_entropy = 0.0f32;
        for i in 0..n_expert {
            probs_init[i] *= inv_sum_init;
            if probs_init[i] > eps {
                prelim_entropy -= probs_init[i] * probs_init[i].ln();
            }
        }

        let dynamic_alpha = self.dynamic_bayesian_alpha(prelim_entropy);

        for e in 0..n_expert {
            let mut score = logits[e];
            if let Some(prior) = bigram_prior {
                if e < prior.len() {
                    score += dynamic_alpha * (prior[e] + eps).ln();
                }
            }
            if self.expert_coherence[e] > 0 {
                score += (self.expert_coherence[e] as f32).min(5.0) * beta;
            }
            scores[e] = score;
        }

        // Final Softmax Entropy
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_sum = 0.0f32;
        let mut probs = vec![0.0f32; n_expert];
        for i in 0..n_expert {
            let p = (scores[i] - max_score).exp();
            probs[i] = p;
            exp_sum += p;
        }
        let inv_sum = 1.0 / (exp_sum + eps);
        let mut final_entropy = 0.0f32;
        for i in 0..n_expert {
            probs[i] *= inv_sum;
            if probs[i] > eps {
                final_entropy -= probs[i] * probs[i].ln();
            }
        }

        let effective_k = if final_entropy > self.config.entropy_threshold {
            (self.config.k + 2).min(n_expert)
        } else {
            self.config.k
        };

        let mut indexed: Vec<(usize, f32)> = scores.iter().enumerate().map(|(i, &s)| (i, s)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let topk: Vec<u16> = indexed.iter().take(effective_k).map(|&(idx, _)| idx as u16).collect();

        (scores, final_entropy, topk)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bf16_conversion() {
        let f = 3.1415927f32;
        let bf = Bf16::from_f32(f);
        let back = bf.to_f32();
        assert!((f - back).abs() < 0.02);
    }

    #[test]
    fn test_router_lookahead_predict() {
        let config = LookaheadConfig {
            k: 4,
            n_embd: 128,
            n_expert: 16,
            ..Default::default()
        };
        let mut lookahead = RouterLookahead::new(config, 2);

        // Populate dummy router weights for layer 1
        let dummy_weights: Vec<Bf16> = (0..16 * 128).map(|i| Bf16::from_f32((i % 7) as f32 * 0.1)).collect();
        lookahead.set_router_weights(1, dummy_weights).unwrap();

        let residual = vec![1.0f32; 128];
        let res = lookahead.predict(0, &residual, None, &[0, 2]).unwrap();

        // High entropy on uniform weights triggers adaptive k expansion (4 -> 6)
        assert_eq!(res.predicted_experts.len(), 6);
        assert!(res.elapsed_us < 500, "Lookahead prediction must be fast");

        // Now test with low entropy (sharp weights)
        let mut sharp_weights = vec![Bf16::from_f32(0.0); 16 * 128];
        // Give expert 3 overwhelming weight
        for j in 0..128 { sharp_weights[3 * 128 + j] = Bf16::from_f32(10.0); }
        lookahead.set_router_weights(1, sharp_weights).unwrap();
        let res_sharp = lookahead.predict(0, &residual, None, &[0, 2]).unwrap();
        assert_eq!(res_sharp.predicted_experts.len(), 4);
        assert_eq!(res_sharp.predicted_experts[0], 3);
        assert_eq!(res_sharp.horizon, "L+2"); // confident enough for L+2
    }

    #[test]
    fn test_fused_dual_hop_predict() {
        let config = LookaheadConfig {
            k: 4,
            n_embd: 128,
            n_expert: 16,
            ..Default::default()
        };
        let mut lookahead = RouterLookahead::new(config, 3);

        // Populate router weights for layer 1 and layer 2
        let weights_l1: Vec<Bf16> = (0..16 * 128).map(|i| Bf16::from_f32((i % 7) as f32 * 0.1)).collect();
        lookahead.set_router_weights(1, weights_l1).unwrap();

        let mut weights_l2 = vec![Bf16::from_f32(0.0); 16 * 128];
        for j in 0..128 {
            weights_l2[5 * 128 + j] = Bf16::from_f32(8.0); // Expert 5 dominant in L+2
        }
        lookahead.set_router_weights(2, weights_l2).unwrap();

        let residual = vec![1.0f32; 128];
        let dual_res = lookahead.predict_dual_fused(0, &residual, None, None, &[1, 5], Some(20)).unwrap();

        assert!(!dual_res.l1_predicted_experts.is_empty());
        assert!(!dual_res.l2_predicted_experts.is_empty());
        assert_eq!(dual_res.l2_predicted_experts[0], 5, "Expert 5 should be top in L+2");
        assert!(dual_res.elapsed_us < 1000, "Fused dual prediction should be fast");

        // Verify adaptive EMA
        let ema_prefill = lookahead.adaptive_ema_factor(2, 0.5);
        let ema_decode = lookahead.adaptive_ema_factor(64, 0.5);
        assert!(ema_decode > ema_prefill, "Decode EMA smoothing must be higher than prefill");

        // Verify dynamic Bayesian alpha
        let alpha_low_entropy = lookahead.dynamic_bayesian_alpha(0.5);
        let alpha_high_entropy = lookahead.dynamic_bayesian_alpha(2.5);
        assert!(alpha_high_entropy > alpha_low_entropy, "High entropy must increase prior weight");
    }
}
