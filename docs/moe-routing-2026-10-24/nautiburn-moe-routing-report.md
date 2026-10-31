# NautiBurn MoE Routing: Findings, Hook Fixes, Research Prompt

Oct 24, 2026 · @Thom Hadfield

## Summary

NautiBurn's routing data is good enough to place experts on cards: a co-activation layout serves 57% of tokens from a single card on a 2-card pair, against under 1% for random placement. The data cannot yet show whether profiles per person or per task type work, because live scrapes carry no session, user or task labels.

- **Placement works.** Grouping experts that fire together, plus copying the 32 hottest experts to every card, keeps a token on one card for 57% of tokens (2 cards) and cuts cards visited per token from 3.6 to 1.9 (4 cards).
- **The histogram is a strong cache tuner.** A fixed hot set from the histogram serves 62% of expert calls with 64 of 256 slots per layer; hot + recent-traffic warm experts reach 66.5%, within 1 point of the best possible fixed set.
- **Next-layer prediction is real.** Layer L's experts predict 61% of layer L+1's with 16 guesses, 2.2x better than popularity alone.
- **Nothing is dead; the tail is cold.** No Tiel expert has never fired in 2M tokens, but the rarest 10.7% carry under 1% of routing and can live in RAM or on disk.

Next: fix the hook so traces carry session and user ids, then build the GGUF expert-reordering tool and use it to spread Tiel across 2, 4 or 8 cards beside other models; GLM-5.3 follows as a separate case.

## Data examined

All analysis used `/var/lib/nautiburn/histogram/sidechannel.sqlite` on t440 (136 GB, read-only queries), centered on Tiel, the model with the most scrapes.

| Model | Scrape rows | Tokens | Layers x experts, top-k | Labelled classes |
| --- | --- | --- | --- | --- |
| tiel-ornith-1.5-35b-a3b | 450,299 | 1.96M | 40 x 256, top-8 | 11 rows only; rest `live-tiel` |
| kat-coder-v2.5-35b-a3b | 158,305 | 1.22M | 40 x 256, top-8 | none |
| qwen3.6-35b-a3b | 8,668 | 97k | 40 x 256, top-8 | 7 probe classes |
| glm-5.3-flash | 1 | 2,048 | 46 x 144, top-8 | none |

Three properties of the data limit what it can prove:

1. **A scrape row is one decode step, not one request.** The hook flushes at every layer-0 wrap, so Tiel averages 4.3 tokens per row and a conversation cannot be rebuilt.
2. **Live traffic has no labels.** 99.99% of Tiel rows are `live-tiel`; there is no session, user, lane or task id, and `token_ids` is always empty.
3. **Probe classes are one prompt each.** `moe_histogram_scrape_loop.sh` replays a single fixed prompt per class (compile, sonar, rust, marine, ml, db, moe), so a class histogram is one \~10-token prompt played hundreds of times.

The simulations used the latest 30,000 live Tiel decode tokens, learned on the first half and scored on the second half.

## Findings

### 1. Experts that fire together can share a card

Co-activation grouping keeps far more tokens on one card than random or load-balanced placement, with card loads staying within 15% of even. Groups were learned per layer on 15k tokens and scored on the next 15k (layers 0, 5, 10, 20, 30, 39).

| Cards | Placement | Cards visited per token | Tokens served by 1 card | Busiest card vs even | Experts per card per layer |
| --- | --- | --- | --- | --- | --- |
| 2 | random | 1.99 | 0.7% | 1.07 | 128 |
| 2 | co-activation groups | 1.58 | 42.0% | 1.10 | 128 |
| 2 | groups + top 32 on both | 1.43 | 56.8% | 1.05 | 144 |
| 3 | random | 2.91 | 0.0% | 1.08 | 85 |
| 3 | groups + top 32 on all | 1.73 | 39.6% | 1.10 | 107 |
| 4 | random | 3.64 | 0.0% | 1.10 | 64 |
| 4 | groups + top 32 on all | 1.92 | 32.5% | 1.15 | 88 |

### 2. The histogram tunes a cache; recent traffic adds a little

With 64 or 128 of 256 expert slots per layer on a card, the share of expert calls already resident:

| What sits on the card | 64 slots | 128 slots | Expert loads per token |
| --- | --- | --- | --- |
| random | 25% | 50% | 0 |
| all-time histogram top set | 62.4% | 85.2% | 0 |
| hot half (histogram) + warm half (recent traffic, refreshed every 64 tokens) | 66.5% | 88.4% | 2.4 to 3.2 |
| best fixed set for this window (ceiling) | 67.5% | 89.4% | n/a |
| plain LRU | 71.5% | 90.4% | 31 to 91 |

LRU wins on hits but moves 60 to 180 MB per token at about 2 MB per expert, which PCIe 3 cannot sustain. Early layers spread wider: layer 0 covers 36% with 64 slots, late layers 58 to 60%.

### 3. The next layer is predictable

A transition table (layer L experts to layer L+1 experts) learned on 15k tokens recalls the next layer's 8 experts at 42.5% with 8 guesses, 60.6% with 16 and 77.6% with 32. Popularity alone gets 16.9%, 27.3% and 42.4%.

### 4. Never-fired experts are a small-sample artifact

Tiel and KAT v2.5 (1.2M to 2M tokens) have 0% never-fired experts; the least-used half of Tiel's experts still carries 33.5% of routing. Models showing 16 to 33% never-fired have only seen 123 to 97k tokens of repeated probe prompts.

| Routing given up | Tiel experts that can go cold | Misses on recent traffic |
| --- | --- | --- |
| 1% | 6.8% | 0.36% |
| 2% | 10.7% | 0.78% |
| 5% | 19.3% | 2.31% |
| 10% | 30.0% | 5.17% |

### 5. Task classes route differently, but generalization is untested

The seven probe prompts route very differently (Jensen-Shannon distance 0.47 to 0.63 between classes), and a per-class profile predicts held-out repeats of the same prompt at 99 to 100% with 64 slots. That is memorization of one prompt, not evidence that a different compile request would use the same experts.

### 6. Prompt routing is a weak guide to generation routing

The prefill hot set covers only 30.3% of decode calls at 64 slots, against 51.7% for the decode histogram itself. Decode needs its own profile.

## Suggestions

Build expert placement for Tiel first, so it runs from slices of 2, 4 or 8 cards shared with other models instead of holding two whole P100s. GLM-5.3 Flash (101.8 GB, 17 layers of experts in RAM today) is a different case and comes later.

Per-card cost of spreading Tiel, at about 2 MB per expert and 40 layers, with the rarest 10% of experts moved to RAM (0.36 to 0.78% misses). Locality was measured on 30k live tokens, layers 0, 5, 10, 20, 30 and 39:

| Cards | Hottest experts copied to every card | Expert VRAM per card | Expert VRAM total | Cards visited per token | Tokens served by 1 card |
| --- | --- | --- | --- | --- | --- |
| 2 (today: 2 whole P100s, about 20.5 GB) | 32 | 10.5 GB | 21.0 GB | 1.43 | 56.8% |
| 4 | 0 | 4.6 GB | 18.4 GB | 2.38 | 16.7% |
| 4 | 32 | 6.5 GB | 26.1 GB | 1.92 | 32.5% |
| 8 | 0 | 2.3 GB | 18.4 GB | 3.52 | 4.6% |
| 8 | 16 | 3.4 GB | 27.4 GB | 2.71 | 12.6% |

Copying hot experts buys locality and costs total VRAM; the cold tier buys back footprint. The cost of visiting 2 to 3.5 cards per layer, 40 times per token, is not measured yet, and the slowest card (the GTX 1070) sets each layer's pace.

1. **Turn on the expert cache the GLM build already has.** `llama.cpp-ggml` has `--moe-cache-mib` (SLRU, decode batches of 32 tokens or fewer); no seat script passes it. It costs no pipeline parallelism because `--n-cpu-moe` already disables that. Try `CPU_MOE=22` with `--moe-cache-mib 3000` against today's `CPU_MOE=17`.
2. **Seed that cache from the histogram.** Warm-start each device's cache with the histogram's hot set instead of an empty cache.
3. **Write a GGUF expert-reordering tool.** Per layer, permute the experts (and the router rows in `ffn_gate_inp`) so each card's co-activation group is contiguous. Outputs are bit-identical; nothing in llama.cpp changes yet.
4. **Split each layer's experts across devices in llama.cpp.** Run one `mul_mat_id` per device on that device's slice, with the hottest experts copied to every device, then sum. This is the real expert-parallel change: the 4-8 KB hidden state moves, not 2 MB weights.
5. **Put the cold tail in RAM or on disk.** The rarest 10 to 20% of experts per layer, chosen from the histogram, served by the CPU on a miss.
6. **Add next-layer prefetch only after measuring PCIe headroom.** Prediction is good (61% at 16 guesses), but published practitioner reports found speculative copies can lose on a shared PCIe link.

[Diagram: router sends the hidden state (4-8 KB) to GPU 1 (32 hottest experts copied + co-activation group A) and GPU 2 (32 hottest + group B); rare experts go to system RAM / CPU (rarest 10-20%); the 8 weighted outputs are added before the next layer.]

On 2 GPUs this layout serves 57% of tokens from a single card per layer; more GPUs each take one co-activation group, and NautiBurn traces relearn the groups when hit rates drift.

## NautiBurn hook improvements

The hook (`common/nautiburn_moe_server_scrape.cpp`, called from `tools/server/server-context.cpp` lines 1101 and 2095) records the right tensor but loses who asked, what was said and which token belongs to which conversation. Ranked by what they unlock:

| # | Change | Why | Where |
| --- | --- | --- | --- |
| 1 | Record slot id, task id and the batch's `seq_id` per token row | With more than one slot, rows from different conversations share a ubatch; `token_pos` today is the row index in the ubatch, not the sequence position | server builds the batch; pass a per-row seq map to the hook before `llama_decode` |
| 2 | Record a user / lane / task id from a request header (e.g. `X-Nauti-User`, `X-Nauti-Task`) | Needed to test per-person and per-task profiles at all | server request parsing, stored on the slot |
| 3 | Flush once per request, keyed by task id | Today a file is written per decode step (4.3 tokens per row on Tiel), so conversations cannot be rebuilt and the scrape dir grew to a 36 MB directory inode | flush in `send_final_response` only; drop the layer-0 wrap flush |
| 4 | Fill `token_ids` (prompt and generated) | Lets traces be grouped by content and replayed | slot prompt tokens + sampled tokens |
| 5 | Record router weights (`ffn_moe_weights`) with the ids | Lets cold-tier and cache decisions weigh how much each expert contributed, not just whether it was picked | match `ffn_moe_weights-*` |
| 6 | Match only one of `ffn_moe_topk` / `ffn_moe_argsort` | Asking for both doubles the per-layer scheduler syncs; the argsort path also reads back the full n\_expert x n\_tokens tensor | prefer topk; argsort only if topk is absent |
| 7 | Fix the argsort fallback's hard-coded top-8 | `k = min(n_exp, 8)` drops 2 experts per token on top-10 models (Qwen3 Coder Next, Qwen3.8 Flash Next) | read top\_k from model hparams |
| 8 | Add sampling (`NAUTIBURN_SAMPLE=N`, scrape 1 request in N) | Every hooked tensor forces a GPU sync; about 80 to 90 per token today, unmeasured cost | gate in `cb_eval` ask phase |
| 9 | Label phase from the batch, not from `max_token_pos > 0` | A 1-token prompt chunk is labelled decode today | use `n_tokens` and slot state |
| 10 | Write the seat's placement (`--n-cpu-moe`, tensor split, cache size) into each scrape | Simulations need to know what was resident when the trace was taken | add to the JSON header |

On t440, two copies of `moe_histogram_consume_loop.sh` (PIDs 1251637 and 2484628) ingest into the same database, which may double-count; the 136 GB database sits on the root filesystem with 20 GB free.

## Test and probe-loop improvements

The tests should answer one question the data cannot answer today: does a profile learned from some requests of a task type (or one person) predict the experts of different, unseen requests of that type?

- [ ] **Many prompts per class.** Replace the one fixed prompt per class in `moe_histogram_scrape_loop.sh` with 50+ distinct prompts per class, generated or taken from real tickets.
- [ ] **Split train and test by prompt, never by repeat.** Hold out whole prompts; repeats of one prompt route identically (JSD 0.0), so a repeat-level split scores 100% and proves nothing.
- [ ] **Generate, not just prefill.** Probe classes have prefill rows only; run 128 to 256 decode tokens per probe, since decode routing differs from prefill (30% overlap at 64 slots).
- [ ] **Probe the model that needs placement.** GLM-5.3 has 2,048 scraped tokens; collect at least 200k decode tokens before learning its co-activation groups.
- [ ] **Measure the hook's cost.** Same prompt set, tokens per second with and without `NAUTIBURN_SCRAPE_DIR`, on Tiel and GLM.
- [ ] **Add a placement replay test.** For each new trace batch, rerun the cache and co-activation simulations (`moe_profile_analysis.py`, `partition.py`) and track hit rate, cards per token and load balance over time; drift means the profile needs relearning.
- [ ] **Score per user once ids exist.** Compare per-user profile vs global profile vs recent-traffic warm set on held-out sessions of that user.
- [ ] **Benchmark the real cache.** GLM seat, `--moe-cache-mib` at 0, 1500, 3000 and 6000, with `CPU_MOE` adjusted to fit; record decode tokens per second.
- [ ] **De-duplicate ingestion.** Stop the second consume loop, then check for request ids ingested twice.

## Research prompt

Paste this into a research agent as-is; it carries the hardware, the data and the findings so far.

```text
GOAL
Find every practical way to use recorded MoE routing data to run Mixture-of-Experts models
larger than GPU memory faster on older GPUs connected by PCIe 3.0. Rank the ideas by expected
decode speed-up per unit of engineering effort on the hardware below.

HARDWARE (fixed, no NVLink anywhere)
- Host A: 2x Tesla P100 16 GB (sm_60), Tesla T10 16 GB (Turing), RTX 2060 12 GB, GTX 1070 8 GB,
  117 GB system RAM, all GPUs on PCIe 3.0 (about 12 GB/s per x16 link in practice).
- Host B (t440): 2x Tesla P100 16 GB, reached from host A over 25 GbE via ggml-rpc.
- Engine: llama.cpp (GGUF, quants from IQ1_M to Q8_0). vLLM/SGLang expert parallelism is not
  an option on sm_60 unless you find a working path.

MODELS
- GLM-5.3 Flash UD-IQ2_XXS: 101.8 GB, 46 layers x 144 routed experts, top-8; today 17 layers of
  experts sit in system RAM (--n-cpu-moe 17).
- Tiel / Qwen3.6-class 35B-A3B: 21 GB, 40 layers x 256 experts, top-8, about 2 MB per expert.
- Also: Qwen3 Coder Next and Qwen3.8 Flash Next (48 layers x 512 experts, top-10), Mixtral 8x7B,
  Gemma 4 26B-A4B, DeepSeek V2/V3-style models.

DATA WE HAVE (NautiBurn)
- Per token, per layer, the top-k expert ids chosen by the router, from live llama-server traffic
  (about 2M tokens for Tiel), plus per-layer expert histograms. Router weights, session ids and
  token ids are being added.

FINDINGS SO FAR (Tiel, 30k live decode tokens, train first half / test second half)
- No expert never fires; the rarest 10.7% carry under 1% of routing.
- Static histogram top-64 of 256 per layer serves 62% of expert calls; hot half + recent-traffic
  warm half serves 66.5% for 2.4 to 3.2 expert loads per token; plain LRU 71.5% but 30 to 90
  loads per token.
- Layer L experts predict layer L+1 experts at 61% recall with 16 guesses (popularity: 27%).
- Co-activation grouping across 2 GPUs (+ top-32 experts copied to both) serves 57% of tokens
  from one GPU per layer (random: under 1%); across 4 GPUs, cards visited per token drop from
  3.6 to 1.9 with loads within 15% of even.
- llama.cpp has --moe-cache-mib (SLRU cache of CPU-resident experts on GPU, batches of 32 tokens
  or fewer) and --n-cpu-moe; it cannot split one layer's experts across GPUs today.

QUESTIONS
1. Expert parallelism without NVLink: how to split a layer's experts across GPUs on PCIe 3
   (and one hop over 25 GbE) so hidden states move instead of weights. Per-layer sync cost on
   Pascal, PCIe peer-to-peer support on P100 and mixed Pascal/Turing, and how others hide it
   (pipelining tokens, micro-batching, overlapping the all-to-all with attention).
2. Placement from traces: graph-partitioning or clustering methods for co-activation, how much
   to replicate hot experts, how to handle unequal GPUs (16/12/8 GB, different speeds), how to run one model from
   partial slices of 2, 4 or 8 cards that other models also use, and how
   often to re-place as traffic drifts.
3. CPU-GPU hybrid execution: when computing a cold expert on the CPU beats copying it
   (kTransformers, Fiddler, MoE-Lightning, PowerInfer-style), with numbers for AVX2/AVX-512 CPUs.
4. Prefetching under a PCIe 3 bandwidth budget: which predictors (next-layer router on the
   current hidden state, transition tables, pre-attention prediction, learned predictors) win
   once speculative copies compete with demand misses; required accuracy to break even.
5. Routing-aware precision: higher-bit hot experts and lower-bit or compressed cold experts
   (HOBBIT-style mixed precision, per-expert GGUF quant types), and the quality cost.
6. Pruning and merging the cold tail (REAP and similar): quality loss versus memory saved at 10%,
   20% and 30% of experts removed.
7. Request-level routing: sending a request to the GPU group whose resident experts best match a
   profile learned from that user or task type, and how quickly a profile must be identified from
   the first prompt tokens.
8. Anything else that uses routing traces to cut decode latency or let a larger model fit:
   speculative decoding interactions, KV-cache placement, expert-aware batching across users.

OUTPUT
- A ranked table: technique, expected decode speed-up on this hardware (with the evidence it
  rests on), engineering effort in llama.cpp (days), what NautiBurn data it needs, risks.
- For the top 3: a concrete implementation sketch against llama.cpp (which files and graph ops
  change) and an experiment we can run with our traces before writing engine code.
- Cite every claim with a link to the paper, PR, issue or benchmark it comes from; mark
  vendor or author-reported numbers as such; prefer results measured on PCIe-only, consumer or
  older datacenter GPUs. Say plainly where evidence is missing.
```

## Appendix: scripts, data and sources

Scripts and data are on cesarops4om in `/tmp/claude-1000/-home-cesarops/6190f69b-bbb3-4ee8-8a7d-bfc72e3e1704/scratchpad/` (temporary; copy them into the NautiBurn repo to keep them).

| File | What it does |
| --- | --- |
| `moe_profile_analysis.py` | Histogram skew, probe-class held-out test, cache simulation, next-layer recall; run as `ssh t440 python3 - 30000 < moe_profile_analysis.py` |
| `never_fired.py` | Never-fired, rare and bottom-half routing share per model |
| `fetch_tiel.py` | Exports the latest N live Tiel decode tokens and the decode histogram to `.npz` |
| `partition.py` | Co-activation placement across 2 to 4 cards, with hot-expert copies, and cold-tier sizing |
| `tiel_decode_30k.npz` | 30,000 x 40 x 8 expert ids plus the all-time decode histogram |
| `res30k.json` | Raw results of the cache and prediction simulations |

Sources:

- [llama.cpp issue #20757: two-tier GPU+RAM expert cache](https://github.com/ggml-org/llama.cpp/issues/20757)
- [llama.cpp discussion #28248: persistent expert slot pool](https://github.com/ggml-org/llama.cpp/discussions/28248)
- [llama.cpp discussion #24528: VRAM caching of hot experts](https://github.com/ggml-org/llama.cpp/discussions/24528)
- [ExLlamaV3 issue #254: predictive expert caching](https://github.com/turboderp-org/exllamav3/issues/254)
- [Speculating Experts Accelerates Inference for MoE (2026)](https://arxiv.org/html/2603.19289v1)
- [Pre-Attention Expert Prediction and Prefetching](https://www.alphaxiv.org/abs/2511.10676)
- [Beyond the Previous Layer: Residual Predictive Structure in MoE Routing](https://pith.science/paper/2609.17940)
- [fMoE: Fine-Grained Expert Offloading](https://arxiv.org/html/2502.05370v1)
- [HOBBIT: Mixed Precision Expert Offloading](https://arxiv.org/html/2411.01433)
- [DALI: Workload-Aware Offloading on Local PCs](https://arxiv.org/html/2602.03495v1)
- [Pre-gated MoE (ISCA 2024)](https://www.microsoft.com/en-us/research/wp-content/uploads/2024/05/isca24_pregated_moe_camera_ready.pdf)
- [MoE-Infinity](https://github.com/EfficientMoE/MoE-Infinity/)
