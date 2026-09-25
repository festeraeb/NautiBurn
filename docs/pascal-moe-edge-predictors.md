---
cursor:
  subagentId: "bc-9e12ce95-ff48-53e5-9c56-2540f5c2ab8c"
---

# Pascal MoE edge predictors

Research note for a Burn forward on a Tesla P100 (sm_60, 16 GB, CUDA 12). Nothing here is a build. No new hit rate was measured. Numbers below are either arithmetic on published shapes, a figure copied from a paper with that paper named, or a figure already written in B1, B3, B5, or `crates/nautiburn-engine`.

The engine this note does not build must keep FP16 when the kernel fits in 49152 bytes of shared memory, record which experts fire, hand the next expert set over as JSON, and start the copy about 13 ms before the gate. A miss stays the exact CPU expert. It does not emit a speculative token.

## 1. Pascal limits that change the prefetch

Sources: NVIDIA Pascal Tuning Guide (the 13.3 HTML), the Tesla P100 product table (PCIe 16 GB: 18.7 TFLOP/s FP16, 9.3 TFLOP/s FP32, 732 GB/s HBM2, PCIe x16 listed as 32 GB/s), and the CubeCL CUDA runtime in this tree (`dp4a: arch_version >= 61`, tensor cores only at compute capability 70 and above, `copy_async` only at 80 and above). B5 recorded the refused launch: 69632 bytes requested, 49152 allowed.

### Shared memory

GP100 has 64 KB of shared memory per SM. A thread block is still capped at 48 KB (49152 bytes). There is no opt-in above that. Volta and later raise the per-block opt-in; Pascal does not. CubeCL reads `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN` and refuses the launch before the kernel runs. On this P100 that attribute is 49152. NVIDIA's own occupancy note is stricter: keep a block at or under 32 KB so two blocks can sit in the 64 KB SM.

### FP16 is HFMA2, and only on GP100 is it fast

Peak FP16 on GP100 is a packed `half2` FMA (`hfma2`): two FP16 operations per CUDA core, up to 2× the FP32 rate. The PCIe P100 product number is 18.7 TFLOP/s FP16 against 9.3 TFLOP/s FP32. GP104 (sm_61) is the opposite: FP16 throughput there is 1/64 of FP32, and that die is where `__dp4a` lives. sm_60 has no `__dp4a`, no tensor core, and no BF16 ALU. BF16 tensor-core and `ldmatrix` paths in this CubeCL tree start at sm_75 and sm_80. A BF16 checkpoint has to be cast before it touches this card. OLMoE's published `config.json` is `torch_dtype: bfloat16`; the B1 file is the F16 GGUF.

`cp.async` is sm_80. On Pascal the way to overlap the next K-tile with FMAs is software double buffering, which stores two A/B stages in shared memory and doubles the footprint.

### Why an FP16 GEMM asks for more than 48 KB

The staged bytes are the tile, not the expert matrix. A 1-token expert GEMM still launches a kernel whose shared memory is `stages * (BM*BK + BN*BK) * sizeof(elem)`, plus padding. FP16 makes each element 2 bytes, so a strategy written for chips that can opt into 64–164 KB per block grows BM, BN, and BK until the byte count is back in that band. The smaller dtype does not shrink the launch.

A double-buffered FP16 stage with `BM = 128`, `BK = 64`, `BN = 128`, and 16 columns of N padding (bank-conflict pad to 144) is:

```
2 * (128*64 + 144*64) * 2 = 69632
```

That matches the B5 refusal exactly. The cubek sources are a git dependency and are not in this tree, so this is a footprint that equals the refused size, not a recovered launch config. The same tile without the pad is 65536 bytes and still over 49152. Storing that tile as FP32 doubles it again.

Tiles that fit in 49152 with two FP16 stages and no padding (`bytes = 4 * BK * (BM + BN)`):

| BM | BN | BK | bytes | note |
|---:|---:|---:|---:|---|
| 64 | 64 | 16 | 8192 | |
| 64 | 64 | 32 | 16384 | |
| 64 | 64 | 64 | 32768 | NVIDIA's 32 KB occupancy line |
| 128 | 64 | 32 | 24576 | |
| 128 | 128 | 16 | 16384 | |
| 128 | 128 | 32 | 32768 | same 32 KB line |
| 128 | 128 | 48 | 49152 | no bytes left for padding or a third stage |

A third stage of 128×128×32 FP16 is also exactly 49152, so any pad refuses it. OLMoE's real GEMMs are `M×2048×1024` and `M×1024×2048` (`N_EMBD = 2048`, `N_FF = 1024` in `nautiburn-layer0`). The kernel tile still has to be one of the rows above. The matrix shape does not buy shared memory.

### PCIe copy versus HBM

The planning copy rate in `nautiburn-engine` is `12e9` bytes/s. That is not a measured memcpy. NVIDIA's P100 product table lists PCIe x16 at 32 GB/s (the bidirectional envelope). HBM2 on the 16 GB SKU is 732 GB/s. These cards are the 16 GB SKU (B1, B5, B7).

At 12e9 bytes/s, 13 ms covers `0.013 * 12e9 = 156_000_000` bytes.

| slab | bytes | ms at 12e9 | ms if already in 732 GB/s HBM |
|---|---:|---:|---:|
| one OLMoE FP16 expert | 12,582,912 | 1.05 | 0.017 |
| OLMoE top-8 | 100,663,296 | 8.39 | 0.14 |
| 12 OLMoE experts | 150,994,944 | 12.58 | 0.21 |
| 13 OLMoE experts | 163,577,856 | 13.63 | 0.22 |
| two Mixtral Q4, planning figure `200_000_000` | 200,000,000 | 16.67 | 0.27 |
| one Mixtral FP16 expert | 352,321,536 | 29.36 | 0.48 |

OLMoE top-8 fits inside a 13 ms lead. Thirteen distinct OLMoE experts do not. The planning Mixtral pair does not. One layer of lookahead hides a cold Mixtral pair only if the work between the prediction and the gate is at least 16.67 ms. No P100 layer time was measured in B5 or B7, so that comparison is not closed. Once the bytes are in HBM the same Mixtral pair is 0.27 ms. The stall is the host copy.

## 2. Predictors past frequency, previous-token copy, and the cross-layer bigram

B3 already scored those three, plus request-matrix cosine and LRU, on `b1/trace.tsv`. Decode hit rate there is `|predicted ∩ truth| / 8`. Frequency was 0.3594 on 16 positions and is the bar that scrape used. This section does not replace that table. Paper percentages below were measured by those authors on their models and prompts. They are not OLMoE hit rates and they are not B1 hit rates.

### MoE-Infinity expert activation matrix

arXiv:2401.14361. An EAM is an `L×E` count matrix for one request. After layer `l`'s routing is known, the partial matrix is matched by cosine distance to a collection of past request matrices. The matched rows, reweighted by layer proximity `(1 - (i-l)/L)`, become the prefetch prior for later layers. The signal is expert ids already chosen, not a hidden state and not a token id. It cannot run before the current gate. It can name later layers before their gates. Their prefetch text is "the next layer." On a P100 that is the one-layer lead that loses to a 16.67 ms Mixtral copy and can cover an OLMoE expert (1.05 ms) if the rest of the layer is at least that long, which is unmeasured. They report 3.1–16.7× TPOT against other serving systems on an A5000 over PCIe 4.0 at their stated 24 GB/s. That is not a top-k hit rate. They also write that a Mixtral-8x7B expert is 0.15B parameters (340 MB). The FP16 formula in section 3 is 0.176B parameters and 336 MiB. Both are their wording versus this arithmetic; neither is a measured Q4 size.

B3's request-matrix cosine, the same idea on one B1 prompt, scored decode 0.2500 on 16 positions and did not advance. One prompt is a weak test of a request matcher.

### ExpertFlow, two papers

arXiv:2410.17954. A transformer reads the input token ids and classifies every expert in every layer in one pass, before the MoE forward. No hidden state. It can run before layer 0's gate, so the lead for layer L is whatever compute precedes that gate. They report, on Mixtral-8x7B and Switch-32/64/128, an expert-cache hit ratio up to 91.96% and an average 27.65% hit-ratio gain over LRU, plus peak GPU-memory savings of 93.72% versus GPU-only. Cache hit ratio is not `|predicted ∩ truth| / k`.

arXiv:2510.26730. A different system with the same name. Step size

```
S = (N_e * E_s) / (C_s * T_l)
```

is how many layers ahead to prefetch. `T_l` is per-layer compute time. This session has no `T_l`, so S is not evaluated here. S then moves by a stall counter (too shallow) and an overfetch counter (too deep). The id predictor fuses the current pre-gate (a hidden state run through the next router), token ids, and the expert-id history. The pre-gate term cannot be built from B1. The stall counters can. They report stall time under 0.1% of their baseline. That is their machine, not this P100.

### ProMoE

arXiv:2410.22134. An MLP memorizes the map from a layer's input hidden state to a later layer's gate. Stride prefetching predicts one layer further than the layer just read, so the CPU predictor (they time it at about 200 µs) overlaps the copy. They define `GoodPred = Accuracy × FetchRate`. On the models in their Figure 8 they report 84.7% average accuracy for that learned predictor, 58.3% for a token-id predictor, and a 5% accuracy drop when the stride increases by one. Table 1 lists DeepSeek-MoE, DeepSeek-V2-Lite, Qwen1.5-MoE, Qwen2-MoE, and Mixtral-8x7B; the 84.7% sentence says "both models" in Figure 8 and this note does not guess which pair. The signal is the hidden state. It cannot run before the gate that produces that state. A stride of +1 is the same one-layer lead as above: enough bytes for OLMoE top-8 (8.39 ms) only if the overlapped work is at least 8.39 ms, and short of the Mixtral pair (16.67 ms). B1 has no hidden state, so this row cannot be scored yet.

### SiDA-MoE

MLSys 2024, arXiv:2310.18859. An offline LSTM with a classification head reads the token-embedding sequence and predicts every layer's experts for the batch before that batch's forward. A parallel hash thread is the point: the prediction exists before the gates. They report up to 99% expert-activation prediction accuracy, up to 3.93× throughput, up to 80% GPU-memory saving, and down to 1% task-score drop, on Switch Transformers. The hash replaces the router in their design. That is an accuracy change. It is not an exact CPU miss. The usable piece is the prefetch hint. Embeddings are not in B1. Token ids are. A token-id stand-in is a different model and has no published accuracy here.

### DuoServe-MoE

arXiv:2509.07379. Prefill and decode are separate schedulers. Prefill, on their Mixtral runs, does not predict: they pipeline expert copies on a second CUDA stream because many experts fire. Decode feeds an MLP with per-layer popularity, the cross-layer affinity matrix, and the expert ids selected on earlier layers. No hidden state. It runs after the current gate and names the next layer, so it is not before the current gate. Their Table 3, Mixtral-8x7B, defines "Top-2" as both experts correct and "at least one" separately:

| dataset | both experts | at least one |
|---|---:|---:|
| Orca | 66.85% | 95.45% |
| SQuAD | 60.23% | 92.31% |

That "both experts" number is stricter than `|predicted ∩ truth| / 2`. It is not an OLMoE number. They also say the MLP adds about 0.6 ms and about 300 MB. One-layer lead, same 13 ms verdict as MoE-Infinity: the Mixtral pair needs 16.67 ms and does not fit; OLMoE's 1.05–8.39 ms might, and that layer time was not measured.

### HOBBIT

arXiv:2411.01433. Built on llama.cpp. Three mechanisms, and only one is a predictor. The predictor reuses the current gating input as the next layer's gate input, because residual hidden states stay similar. On Mixtral-8x7B their Figure 7 shows average top-1 accuracy of 96% for the next layer and about 90% for the next two or three layers. That is top-1, not top-2, and it needs the gating hidden state. B1 does not have it. The other two mechanisms change the forward: a low-precision stand-in when the gate score is below `T1 = 0.6`, and a skip when it is above `T2 = 0.9` (their Mixtral thresholds). They also write that loading a Mixtral-8x7B layer from the host over PCIe 4.0 at their stated 32 GB/s takes about 80 ms, against about 3 ms of compute on an RTX 4090. Those times are not this P100. The 9.93× decode speedup is against other offload systems, not a hit rate. Low-precision substitution and the skip are accuracy changes. They are not the miss path.

### Mixtral early-exit and layer-skip

A Mixtral-specific expert skip exists. Lu et al., ACL 2024, "Not All Experts are Equal," drop the second routed expert when `w2 < β w1`, with `β` the per-layer median of that ratio on calibration data. They report 1.2× to 1.3× inference speedup at pruning rate `r = 6`. The gate has already run, the signal is the router weights, and the skipped expert is not computed. That changes the token. It does not prefetch.

LayerSkip (Elhoushi et al., arXiv:2404.16710) is early exit on Llama, trained with layer dropout. It is not a Mixtral method and it is not an expert predictor. No Mixtral layer-skip that keeps the exact top-2 gate was found.

## 3. What moves with the activation

The expert GEMM reads the current hidden state and the expert weights. The combine then scales by the router weight. Expert ids are the address of the slab. They are the JSON handoff (`NextExperts.expert_ids`), not a substitute for the activation. KV is attention state. If attention stays on the card that owns the layer, KV does not cross with the expert. The job's task class is a predictor feature. The GEMM does not read it. B1 does not record it (`task_class=absent` in B3).

Shapes used: OLMoE `config.json` (`hidden_size 2048`, `intermediate_size 1024`, 16 layers, 16 query heads, 16 KV heads, 64 experts, top-8) and the Mixtral config defaults (`hidden_size 4096`, `intermediate_size 14336`, 32 layers, 32 query heads, 8 KV heads, 8 experts, top-2). SwiGLU stores three matrices and no bias. `head_dim = hidden / num_attention_heads` is 128 for both.

```
expert_fp16 = 3 * hidden * intermediate * 2
topk_weights_f32 = k * 4
expert_ids_i32 = k * 4
hidden_fp16 = hidden * 2
kv_per_token_per_layer = 2 * n_kv_heads * head_dim * dtype_bytes
kv_per_token = layers * kv_per_token_per_layer
```

| quantity | OLMoE-1B-7B | Mixtral-8x7B |
|---|---:|---:|
| one expert, FP16 | 12,582,912 (12 MiB) | 352,321,536 (336 MiB) |
| one expert, raw 4-bit, no scales | 3,145,728 | 88,080,384 |
| top-k ids, i32 | 32 | 8 |
| top-k weights, f32 | 32 | 8 |
| hidden state, one token, FP16 | 4,096 | 8,192 |
| hidden state, one token, FP32 | 8,192 | 16,384 |
| KV, one token, one layer, FP16 | 8,192 | 4,096 |
| KV, one token, all layers, FP16 | 131,072 | 131,072 |
| KV, one token, all layers, FP32 | 262,144 | 262,144 |

B7 ran Mixtral with `-ctk f32 -ctv f32`, so the FP32 KV row is the one that matched that server. The Q4 planning pair remains `200_000_000` bytes for two experts (16.67 ms). A raw 4-bit Mixtral expert is 88,080,384 bytes (7.34 ms) and is a lower bound: Q4_K_M adds scales. The per-expert Q4 size was not measured. The GGUF that loaded is 28,448,468,000 bytes for the whole file (B7), which is not split here into experts, attention, and the KV cache.

Relative to one OLMoE expert, the FP16 hidden state is 4096/12,582,912 of the slab. Shipping the activation to a resident expert is the small copy. Shipping the expert to the activation is the 13 ms problem.

## 4. What to try after a live histogram

Score decode only, at the model's own k (8 for OLMoE, 2 for Mixtral): `|predicted ∩ truth| / k`, prefill reported beside it and not allowed to save a decode loss. Keep a row only when that decode rate is at least the frequency row on the same positions, and the copy of the predicted slabs was started early enough that it finishes 13 ms before the gate that consumes them. Drop a row that wins by naming more than k experts. Drop a row whose bytes at 12e9 exceed the measured lead, even if the ids are right. The B3 frequency bar on the current scrape is 0.3594 (16 decode positions). A larger histogram replaces that number; it is not a permanent threshold.

These six can be built from expert ids, token ids, phase, and layer. They do not need a hidden state.

1. Token-id conditional frequency. `P(expert | phase, layer, token_id)` from the histogram. Known as soon as the token id exists, so it is before every gate of that token. Drop if decode hit rate loses to the plain per-layer frequency, or if the union across layers copies more than 156,000,000 bytes inside the 13 ms window.
2. DuoServe MLP on ids only. Popularity, cross-layer affinity, and the expert-id path. Predicts the next layer after the current gate. Keep only if the measured gap from that gate to the next gate is at least the copy time (8.39 ms for OLMoE top-8, 16.67 ms for the Mixtral planning pair) and the decode hit rate beats frequency. Their 66.85% is not this score.
3. Request-matrix cosine, retried once many requests exist. B3's 0.2500 was one prompt. Same two-part keep rule. If it still loses to frequency on decode, it stays dropped.
4. All-layer token-id hash, SiDA's shape without replacing the router. Names every layer before layer 0. Prefetch only the prefix whose bytes finish 13 ms before each gate. Drop if using it as the router, or if layer 0's own experts miss because nothing precedes that gate.
5. Job-class matrix. Same cosine or frequency, keyed by `task_class`. The live log has to grow that field. B1 cannot score it. Drop if task class does not beat the unkeyed frequency row.
6. Horizon controller from ExpertFlow (2510.26730), wrapped around whichever of 1–5 wins. S starts from bytes, the 12e9 planning rate, and a measured `T_l`. Stall increases S, overfetch decreases S. Keep if the fraction of gates whose slabs arrived 13 ms early goes up while the decode hit rate stays at or above frequency. Drop if S grows and the hit rate falls. Do not plug in the RTX 4090's 3 ms as `T_l`.

## Do not build yet

- OpenVINO as the CPU fallback. The miss path is the existing exact CPU GEMM on the host weight copy. A second runtime is a different token.
- A three-card most-used-agents layout. B7's Mixtral is a layer split across three P100s so the model fits. That is not an expert-prefetch result, and there is no histogram of which agents to pin.
- Any predictor that needs a hidden state or a gate weight. B1's columns are phase, layer, token_index, token_id, tensor, expert_ids (`b1/check_b1_trace.py` requires six fields). That list is ProMoE's MLP, HOBBIT's gating-input reuse, ExpertFlow (2510.26730)'s pre-gate delta, Lu et al.'s `w2/w1` skip, and `route_hidden_state`. SiDA's real input is embeddings, which are also absent; only the token-id stand-in in section 4 is in range.
- Speculative execution of a guessed expert, HOBBIT's low-precision substitute, and HOBBIT's `T2` skip. Each one changes the token. The miss stays exact.
